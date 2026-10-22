"""FLM-specific DIT backbone.

Key difference from models/dit.py DIT:
  - FLMEmbeddingLayer does NOT apply softmax to 3D input.
    FLM's x_t is a linear blend of Gaussian noise and one-hot
    vectors — raw floats, not logits. Applying softmax would
    destroy the continuous interpolation semantics.
  - Supports dual timestep embedding (sigma_prime) for distillation.
  - Supports LearnableLossWeighting.
  - No KV cache (FLM is non-autoregressive).
"""

import math

import einops
import flash_attn
import flash_attn.layers.rotary
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import (
    Rotary,
    TimestepEmbedder,
    DDiTBlock,
    DDiTFinalLayer,
    LayerNorm,
    apply_rotary_pos_emb,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
)


class FLMEmbeddingLayer(nn.Module):
    """Embedding layer for FLM — NO softmax on 3D input.

    When x is 2D (B, L) integer tokens: standard lookup.
    When x is 3D (B, L, V) continuous blend: einsum WITHOUT softmax.
    This is critical because FLM's x_t = (1-t)*noise + t*one_hot
    is a raw continuous tensor, not logits.
    """

    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            x.float(),
            self.embedding.float(),
        ).to(x.dtype)


class LearnableLossWeighting(nn.Module):
    """Learnable per-timestep loss weighting for FLM.

    Parameterizes log(weight) as a small MLP on the timestep embedding.
    Zero-initialized so initially e^{-w} = 1 (no weighting).
    """

    def __init__(self, cond_dim, is_flow=True, hidden_dim=128):
        super().__init__()
        self.s_embed = TimestepEmbedder(cond_dim)
        if not is_flow:
            self.t_embed = TimestepEmbedder(cond_dim)
        else:
            self.t_embed = None
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s, t=None):
        emb = self.s_embed(s)
        if t is not None and self.t_embed is not None:
            emb = emb + self.t_embed(t)
        return self.mlp(emb).squeeze(-1)


class FLMDIT(nn.Module):
    """DIT backbone for Flow Language Models.

    Follows the same forward(x0, xt, sigma, context) API as the
    main DIT, but:
      - Uses FLMEmbeddingLayer (no softmax on 3D input)
      - Reads context.sigma_prime for dual-timestep distillation
      - Always non-causal, always uses adaLN
      - No KV cache
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim

        self.vocab_embed = FLMEmbeddingLayer(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)

        if getattr(config.algo, 'double_temb', False):
            self.sigma_map_prime = TimestepEmbedder(cond_dim)
        else:
            self.sigma_map_prime = None

        if getattr(config.algo, 'learnable_loss_weighting', False):
            is_flow = 'distill' not in config.algo.name
            self.learnable_loss_weighting = LearnableLossWeighting(
                cond_dim=cond_dim, is_flow=is_flow)
        else:
            self.learnable_loss_weighting = None

        self.rotary_emb = Rotary(dim // config.model.n_heads)

        softcap = getattr(config.model, 'softcap', 0.0)
        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(DDiTBlock(
                dim=dim,
                n_heads=config.model.n_heads,
                cond_dim=cond_dim,
                adaLN=True,
                dropout=config.model.dropout,
                softcap=softcap))
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=True)

        # Compatibility with TrainerBase (expects these)
        self.ctx_cached_len = 0

    def reset_kv_cache(self):
        pass  # FLM does not use KV cache

    def forward(self, x0, xt, sigma, context=None):
        del x0  # FLM does not use clean input
        sigma_prime = None
        if context is not None and hasattr(context, 'sigma_prime'):
            sigma_prime = context.sigma_prime

        x = self.vocab_embed(xt)

        t_emb = self.sigma_map(sigma)
        if sigma_prime is not None:
            if self.sigma_map_prime is not None:
                t_prime_emb = self.sigma_map_prime(sigma_prime)
            else:
                t_prime_emb = self.sigma_map(sigma_prime)
            t_emb = t_emb + t_prime_emb
        t_cond = F.silu(t_emb)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(
            device_type=x.device.type, dtype=torch.bfloat16
        ):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c=t_cond)
            return self.output_layer(x, c=t_cond)
