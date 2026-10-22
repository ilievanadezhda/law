import math

import einops
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from .dit import (
  DDiTBlock,
  DDiTFinalLayer,
  TimestepEmbedder,
  Rotary,
  apply_rotary_pos_emb,
  _sdpa_full,
)


def justnorm(x: torch.Tensor, eps: float = 1e-6):
  n = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
  return x / n


class SphereArchBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim,
               use_time_alpha, mlp_type, mlp_ratio=4,
               dropout=0.1, eps=1e-6):
    super().__init__()
    assert dim % n_heads == 0
    self.dim = dim
    self.n_heads = n_heads
    self.d_k = dim // n_heads
    self.base_scale = 1.0 / math.sqrt(dim)
    self.eps = eps
    self.dropout = dropout
    self.mlp_type = mlp_type
    self.use_time_alpha = use_time_alpha

    self.query = nn.Linear(dim, dim, bias=False)
    self.key = nn.Linear(dim, dim, bias=False)
    self.value = nn.Linear(dim, dim, bias=False)
    self.att_out = nn.Linear(dim, dim, bias=False)

    if mlp_type == 'gelu':
      self.c_fc = nn.Linear(dim, mlp_ratio * dim, bias=True)
      self.mlp_out = nn.Linear(mlp_ratio * dim, dim, bias=True)
      self.s_fc_init_value = 1.0
      self.s_fc_init_scaling = 1.0
      self.s_fc = nn.Parameter(
        self.s_fc_init_scaling * torch.ones(mlp_ratio * dim))
    elif mlp_type == 'swiglu':
      self.c_fc = nn.Linear(dim, 2 * mlp_ratio * dim, bias=False)
      self.mlp_out = nn.Linear(mlp_ratio * dim, dim, bias=False)
      self.suv_init_value = 1.0
      self.suv_init_scaling = 1.0
      self.suv = nn.Parameter(
        self.suv_init_scaling * torch.ones(2 * mlp_ratio * dim))
    else:
      raise ValueError(f'Unknown mlp_type: {mlp_type}')

    self.attn_alpha_init_value = 0.05
    self.attn_alpha_init_scaling = self.base_scale
    self.attn_alpha = nn.Parameter(
      self.attn_alpha_init_scaling * torch.ones(dim))

    self.mlp_alpha_init_value = 0.05
    self.mlp_alpha_init_scaling = self.base_scale
    self.mlp_alpha = nn.Parameter(
      self.mlp_alpha_init_scaling * torch.ones(dim))

    self.sqk_init_value = 1.0
    self.sqk_init_scaling = self.base_scale
    self.sqk = nn.Parameter(self.sqk_init_scaling * torch.ones(dim))

    if use_time_alpha:
      self.alpha_modulation = nn.Linear(cond_dim, 2 * dim)
      nn.init.zeros_(self.alpha_modulation.weight)
      nn.init.zeros_(self.alpha_modulation.bias)

  def _effective_alpha(self, base_param, init_value, init_scaling,
                       delta):
    base = base_param * (init_value / init_scaling)          # [d]
    if delta is None:
      return base.abs().view(1, 1, -1)
    return (base + delta).abs().unsqueeze(1)                 # [B, 1, d]

  def forward(self, h, rotary_cos_sin, t_emb):
    B, L, d = h.shape

    # ===== Attention =====
    q = self.query(h)
    k = self.key(h)
    v = self.value(h)
    q = q.view(B, L, self.n_heads, self.d_k)
    k = k.view(B, L, self.n_heads, self.d_k)
    v = v.view(B, L, self.n_heads, self.d_k)

    # RoPE in fp32, exactly like DDiTBlock.
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = torch.stack([q, k, v], dim=2)
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
      1, 1, self.n_heads, self.d_k)
    q = sqk * justnorm(q, self.eps)
    k = sqk * justnorm(k, self.eps)

    q = q.transpose(1, 2)  # [B, H, L, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    softmax_scale = math.sqrt(self.d_k)
    y = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
    y = einops.rearrange(y, 'b h l d -> b l (h d)')

    h_att = self.att_out(y)
    h_att = F.dropout(h_att, p=self.dropout, training=self.training)

    if self.use_time_alpha:
      delta = self.alpha_modulation(t_emb)
      delta_A, delta_M = delta.chunk(2, dim=-1)
    else:
      delta_A = delta_M = None

    alpha_A = self._effective_alpha(
      self.attn_alpha, self.attn_alpha_init_value,
      self.attn_alpha_init_scaling, delta_A)
    A_norm = justnorm(h, self.eps)
    B_norm = justnorm(h_att, self.eps)
    h = justnorm(A_norm + alpha_A * (B_norm - A_norm), self.eps)

    # ===== MLP =====
    if self.mlp_type == 'gelu':
      x_pre = self.c_fc(h)
      s_fc = self.s_fc * (
        (self.s_fc_init_value / self.s_fc_init_scaling)
        * math.sqrt(self.dim))
      x_pre = s_fc * x_pre
      x_mlp = F.gelu(x_pre, approximate='tanh')
      h_mlp = self.mlp_out(x_mlp)
    else:  # swiglu
      uv = self.c_fc(h)
      suv = self.suv * (
        (self.suv_init_value / self.suv_init_scaling)
        * math.sqrt(self.dim))
      uv = suv * uv
      u, v_ = uv.chunk(2, dim=-1)
      x_mlp = u * F.silu(v_)
      h_mlp = self.mlp_out(x_mlp)
    h_mlp = F.dropout(h_mlp, p=self.dropout, training=self.training)

    alpha_M = self._effective_alpha(
      self.mlp_alpha, self.mlp_alpha_init_value,
      self.mlp_alpha_init_scaling, delta_M)
    A_norm = justnorm(h, self.eps)
    B_norm = justnorm(h_mlp, self.eps)
    h = justnorm(A_norm + alpha_M * (B_norm - A_norm), self.eps)

    return h


class SphereArch(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.embed_dim = dim
    self.init_mode = config.model.init
    self.eps = config.model.eps
    self.base_scale = 1.0 / math.sqrt(dim)
    self.n_heads = config.model.n_heads
    self.n_blocks = config.model.n_blocks
    self.use_time_alpha = config.model.use_time_alpha
    self.use_time_token = config.model.use_time_token
    self.mlp_type = config.model.mlp_type
    self.normalize_input_embed = config.model.normalize_input_embed

    self.sphere_embed = nn.Embedding(vocab_size, dim)
    if self.init_mode == 'random':
      nn.init.normal_(self.sphere_embed.weight, std=0.02)
    elif self.init_mode == 'ngpt':
      nn.init.normal_(self.sphere_embed.weight, std=self.base_scale)
    elif self.init_mode == 'pretrained':
      nn.init.zeros_(self.sphere_embed.weight)
    else:
      raise ValueError(self.init_mode)

    self.sigma_map = TimestepEmbedder(cond_dim)

    if self.use_time_token:
      self.time_token_head = nn.Linear(cond_dim, dim, bias=False)

    self.rotary_emb = Rotary(dim // self.n_heads)

    self.blocks = nn.ModuleList([
      SphereArchBlock(
        dim=dim,
        n_heads=self.n_heads,
        cond_dim=cond_dim,
        use_time_alpha=self.use_time_alpha,
        mlp_type=self.mlp_type,
        dropout=config.model.dropout,
        eps=self.eps,
      )
      for _ in range(self.n_blocks)
    ])

    self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    self.sz_init_value = 1.0
    self.sz_init_scaling = self.base_scale
    self.sz = nn.Parameter(self.sz_init_scaling * torch.ones(vocab_size))

    self._init_matrix_weights()

    # Adapt a pretrained AR / DUO checkpoint BEFORE the first renormalize so
    # the loaded weights get snapped onto the sphere like fresh-init ones.
    if config.model.pretrained_ckpt_path is not None:
      self.load_pretrained_from(config.model.pretrained_ckpt_path)

    # Snap all matrices onto the sphere before the first forward pass
    # (except sphere_embed if normalize_input_embed=False). This matches
    # the reimpl's pattern of calling normalize_matrices() once before
    # training starts.
    self.renormalize_weights()

    # Required by TrainerBase.ctx_cached_len property.
    self.ctx_cached_len = 0

  def load_pretrained_from(self, ckpt_path: str) -> None:
    """Adapt an AR / DUO Lightning checkpoint into this backbone on the fly.

    Strips the `backbone.` prefix from source keys and renames
        backbone.vocab_embed.embedding  ->  sphere_embed.weight
    so DIT-trained params map onto SphereArch. DDiTBlock and
    DDiTBlockCausal share parameter names and shapes. Note: AR's
    DDiTBlock(Causal) attention is a standard dot-product attention whereas
    nGPT normalizes Q/K and reads per-head scales from `sqk` — the raw
    Q/K/V/O linear weights still load cleanly, and `renormalize_weights()`
    (called after this) puts them onto the sphere. The normalized-attention
    fine-tune then adapts the scales.

    Target-only params (e.g. sqk, attn_alpha, mlp_alpha, sigma_map,
    time-alpha modulation, lm_head, sz) stay at their fresh-init values.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_sd = (ckpt['state_dict']
              if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)

    own_sd = self.state_dict()
    loaded, sliced, skipped = 0, [], 0
    for k, v in src_sd.items():
      if k.startswith('teacher') or not k.startswith('backbone.'):
        continue
      k = k[len('backbone.'):]
      if k == 'vocab_embed.embedding':
        k = 'sphere_embed.weight'
      if k not in own_sd:
        continue
      tgt = own_sd[k]
      if tgt.shape == v.shape:
        own_sd[k] = v
        loaded += 1
      elif (tgt.ndim == v.ndim and v.ndim >= 1
            and tgt.shape[1:] == v.shape[1:]):
        n = min(tgt.shape[0], v.shape[0])
        new_tgt = tgt.clone()
        new_tgt[:n] = v[:n]
        own_sd[k] = new_tgt
        sliced.append((k, tuple(v.shape), tuple(tgt.shape)))
        loaded += 1
      else:
        skipped += 1
    self.load_state_dict(own_sd, strict=True)
    fresh = len(own_sd) - loaded
    print(
      f'[SphereArch.load_pretrained_from] {ckpt_path}\n'
      f'  loaded   : {loaded}/{len(own_sd)}\n'
      f'  fresh    : {fresh}\n'
      f'  skipped  : {skipped} (not in target or shape mismatch)')
    for k, s_shape, t_shape in sliced:
      print(f'  sliced   : {k}  src{s_shape} -> tgt{t_shape}')

  def _init_matrix_weights(self):
    """Init Q/K/V/O, c_fc, mlp_out, lm_head to N(0, 1/sqrt(d))."""
    std = self.base_scale
    out_std = std / math.sqrt(2 * self.n_blocks)  # GPT-2 residual init
    for block in self.blocks:
      nn.init.normal_(block.query.weight, std=std)
      nn.init.normal_(block.key.weight, std=std)
      nn.init.normal_(block.value.weight, std=std)
      nn.init.normal_(block.att_out.weight, std=out_std)
      nn.init.normal_(block.c_fc.weight, std=std)
      if block.c_fc.bias is not None:
        nn.init.zeros_(block.c_fc.bias)
      nn.init.normal_(block.mlp_out.weight, std=out_std)
      if block.mlp_out.bias is not None:
        nn.init.zeros_(block.mlp_out.bias)
    nn.init.normal_(self.lm_head.weight, std=std)

  def init_sphere_embed_from_pretrained(self, pretrained_weight):
    if pretrained_weight.shape != (self.vocab_size, self.embed_dim):
      raise ValueError(
        f'Expected pretrained_weight shape ({self.vocab_size}, '
        f'{self.embed_dim}), got {tuple(pretrained_weight.shape)}')
    normalized = utils.sphere_normalize(pretrained_weight.float())
    with torch.no_grad():
      self.sphere_embed.weight.copy_(normalized)

  def get_sphere_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
    emb = self.sphere_embed(token_ids)  # [B, L, d]
    return justnorm(emb, self.eps)

  def reset_kv_cache(self):
    self.ctx_cached_len = 0

  def renormalize_weights(self):
    with torch.no_grad():
      if self.normalize_input_embed:
        self.sphere_embed.weight.data.copy_(
          justnorm(self.sphere_embed.weight.data, self.eps))
      # lm_head: shape [V, d], rows are the output embeddings → normalize
      # along dim=1 (the embedding dim).
      self.lm_head.weight.data.copy_(
        justnorm(self.lm_head.weight.data, self.eps))
      for block in self.blocks:
        # Linear(in, out).weight is shape [out, in]; "along embedding dim"
        # means normalize along the d_model axis.
        #   query/key/value: Linear(d, d) → weight [d, d] → dim=1 is d
        #   att_out:         Linear(d, d) → weight [d, d] → dim=0 is d
        #   c_fc:            Linear(d, k) → weight [k, d] → dim=1 is d
        #   mlp_out:         Linear(k, d) → weight [d, k] → dim=0 is d
        block.query.weight.data.copy_(
          justnorm(block.query.weight.data, self.eps))
        block.key.weight.data.copy_(
          justnorm(block.key.weight.data, self.eps))
        block.value.weight.data.copy_(
          justnorm(block.value.weight.data, self.eps))
        # att_out/mlp_out normalize along dim=0; justnorm is dim=-1, so
        # transpose in, normalize, transpose back.
        block.att_out.weight.data.copy_(
          justnorm(block.att_out.weight.data.T, self.eps).T)
        block.c_fc.weight.data.copy_(
          justnorm(block.c_fc.weight.data, self.eps))
        block.mlp_out.weight.data.copy_(
          justnorm(block.mlp_out.weight.data.T, self.eps).T)

  def forward(self, x0, xt: torch.Tensor, sigma: torch.Tensor,
              context=None) -> torch.Tensor:
    del x0, context

    x = xt  # [B, L, d], already on the sphere

    t_emb = F.silu(self.sigma_map(sigma))  # [B, cond_dim]

    if self.use_time_token:
      t_tok = justnorm(self.time_token_head(t_emb), self.eps)
      t_tok = t_tok.unsqueeze(1)  # [B, 1, d]
      x = torch.cat([x, t_tok], dim=1)  # [B, L+1, d]

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, rotary_cos_sin, t_emb)
      if self.use_time_token:
        x = x[:, :-1, :]
      logits = self.lm_head(x)
      sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
      logits = logits * sz
    return logits
