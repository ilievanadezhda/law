"""ContDIT: Continuous Diffusion Transformer for CANDI.

Reuses DDiTFinalLayer, Rotary, TimestepEmbedder from dit.py.
Key differences from DIT:
  1. L2-normalized embeddings (excluding mask token), handles 3D soft inputs.
  2. Conditions on reveal_mask via mixed_coeff blending with mask embedding.
  3. Rescales corrupted positions by c_in = 1/sqrt(1 + sigma^2).
  4. Output excludes the mask token column ([:, :, :-1]).

NOTE: ContDIT uses its own CANDIDDiTBlock instead of the shared DDiTBlock
from dit.py. The only difference is the rotary embedding path:
CANDIDDiTBlock uses split_and_apply_rotary_pos_emb (apply_rotary_emb_torch,
out-of-place) to exactly match the original candi-diffusion codebase, while
dit.py's DDiTBlock uses apply_rotary_pos_emb (apply_rotary_emb_qkv_,
in-place). The two are mathematically equivalent but can produce minor
floating-point differences.
"""

import math

import einops
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import (
  DDiTBlock,
  DDiTFinalLayer,
  LayerNorm,
  Rotary,
  TimestepEmbedder,
  bias_dropout_add_scale_fused_train,
  bias_dropout_add_scale_fused_inference,
  modulate_fused,
  regular_attention_multi_headed,
  split_and_apply_rotary_pos_emb,
)


class ContDITEmbeddingLayer(nn.Module):
  """Embedding with L2-normalized weights; handles 3D soft inputs."""

  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty(vocab_dim, dim))
    nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    # Normalize all rows except the last (mask token)
    normed = F.normalize(self.embedding[:-1], p=2, dim=-1)
    if x.ndim == 2:
      return normed[x]
    assert x.ndim == 3
    return torch.einsum(
      'blv,ve->ble', x.float(), normed.float()
    ).to(x.dtype)


class CANDIDDiTBlock(nn.Module):
  """DDiTBlock matching the original candi-diffusion implementation.

  Uses split_and_apply_rotary_pos_emb (apply_rotary_emb_torch) for
  exact numerical parity with the original codebase.
  """

  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3, h=self.n_heads)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

    x = regular_attention_multi_headed(q, k, v)

    if self.adaLN:
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class ContDIT(nn.Module):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.vocab_size = vocab_size
    self.mixed_coeff = config.algo.mixed_coeff

    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim

    self.vocab_embed = ContDITEmbeddingLayer(dim, vocab_size)
    self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)

    block_cls = CANDIDDiTBlock if config.model.use_original_rope else DDiTBlock
    block_kwargs = dict(
      dim=dim,
      n_heads=config.model.n_heads,
      cond_dim=cond_dim,
      adaLN=True,
      dropout=config.model.dropout,
    )
    if not config.model.use_original_rope:
      block_kwargs['softcap'] = config.model.softcap
    self.blocks = nn.ModuleList([
      block_cls(**block_kwargs)
      for _ in range(config.model.n_blocks)
    ])
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=True,
    )
    self.ctx_cached_len = 0

  def get_embedding(self, x):
    """Embed tokens (2D int or 3D soft probabilities)."""
    return self.vocab_embed(x)

  def reset_kv_cache(self):
    pass  # Non-autoregressive, no KV cache

  def forward(self, x0, xt, sigma, context=None):
    del x0  # Not used
    assert context is not None, (
      'ContDIT requires a CANDITrainingContext')

    # Extract CANDI-specific fields from context
    reveal_mask = context.reveal_mask
    continuous_noise = context.continuous_noise
    is_embed = context.is_embed
    embedding_cache = context.embedding_cache

    # Embed input (skip if already in embedding space)
    x = xt if is_embed else self.vocab_embed(xt)

    # Continuous noise rescaling factor
    if continuous_noise is not None:
      c_in = 1.0 / (1.0 + continuous_noise ** 2) ** 0.5
    else:
      c_in = None

    # Apply reveal_mask conditioning with mixed_coeff
    if reveal_mask is not None:
      special = F.normalize(
        self.vocab_embed.embedding[-1], p=2, dim=-1)
      special = special.view(1, 1, -1).expand_as(x)
      mask = reveal_mask.unsqueeze(-1).float()

      source = embedding_cache if embedding_cache is not None else x
      if c_in is not None:
        scaled = source * c_in[:, None, None]
      else:
        scaled = source

      x = (x * mask
           + (1 - mask) * (
               self.mixed_coeff * special
               + (1 - self.mixed_coeff) * scaled))

    # Time conditioning
    t_cond = F.silu(self.sigma_map(sigma))

    # Transformer blocks
    rotary_cos_sin = self.rotary_emb(x)
    with torch.amp.autocast(
        device_type=x.device.type, dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, rotary_cos_sin, c=t_cond)
      logits = self.output_layer(x, c=t_cond)

    # Exclude mask token column from output
    return logits[:, :, :-1]
