import functools
import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def _compute_cos_sin(self, t: torch.Tensor):
    freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
    emb = torch.cat((freqs, freqs), dim=-1).to(t.device)
    cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    cos[:, :, 2, :, :].fill_(1.)
    sin[:, :, 2, :, :].fill_(0.)
    return cos, sin

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
      self.cos_cached, self.sin_cached = self._compute_cos_sin(t)
    return self.cos_cached, self.sin_cached

  def forward_range(self, start_idx: int, end_idx: int, device=None):
    if device is None:
      device = self.inv_freq.device
    t = torch.arange(start_idx, end_idx, device=device).type_as(self.inv_freq)
    return self._compute_cos_sin(t)


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


def _sdpa_full(q, k, v):
  """Non-causal full-sequence attention. q/k/v: [B, H, S, D]."""
  return F.scaled_dot_product_attention(q, k, v)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


class RMSNorm(nn.Module):
  """Ported from seal-rg/recurrent-pretraining recpre/norms.py."""
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.eps = eps

  def forward(self, x):
    dtype = x.dtype
    with torch.amp.autocast('cuda', enabled=False):
      x = x.float()
      x = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    return x.to(dtype) * self.weight


def make_norm(name='layer', eps=None):
  """dim -> nn.Module. eps=None keeps the class default (1e-5 layer, 1e-6 rms)."""
  try:
    cls = {'layer': LayerNorm, 'rms': RMSNorm}[name]
  except KeyError:
    raise ValueError(f"Unknown model.norm={name!r}; expected 'layer' or 'rms'")
  return cls if eps is None else functools.partial(cls, eps=eps)


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


def apply_spectral_norm(block, n_power_iterations=1):
  """Spectral-normalize the compute-path Linear weights of a DDiT block.

  Constrains the largest singular value of each attention/MLP weight to 1, which
  bounds the block's Lipschitz constant. The point is loop stability: the looped
  DiT feeds its own output back as a carry, so an expansive (spectral radius > 1)
  block map can blow up over passes. Targets attn_qkv, attn_out and the two MLP
  Linears; the adaLN modulation and the final output projection are deliberately
  left untouched -- both are zero-initialized, so SN (weight / sigma) would be a
  0/0 NaN and would also destroy their intended no-op warm-start.
  """
  targets = [block.attn_qkv, block.attn_out, block.mlp[0], block.mlp[2]]
  for linear in targets:
    _spectral_norm(linear, n_power_iterations=n_power_iterations)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

BLOCKS = ('prenorm', 'sandwich', 'sandwich_swin')


class DDiTBlockCausal(nn.Module):
  def __init__(self, dim, n_heads, adaLN: bool,
               cond_dim=None, mlp_ratio=4, dropout=0.1,
               block='prenorm', norm_cls=LayerNorm):
    super().__init__()
    assert block in BLOCKS, f"Unknown model.block={block!r}"
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.sandwich = block == 'sandwich'
    self.swin = block == 'sandwich_swin'

    self.norm1 = norm_cls(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = norm_cls(dim)
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

    if self.sandwich or self.swin:
      self.norm_post_attn = norm_cls(dim)
      self.norm_post_mlp = norm_cls(dim)

    self.k_cache = None
    self.v_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.k_cache = None
    self.v_cache = None

  def forward(self, x, rotary_cos_sin, c=None, *, kv_cache=False):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # attention operation
    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = einops.rearrange(
      qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    if kv_cache:
      k_full = k if self.k_cache is None else torch.cat([self.k_cache, k], dim=1)
      v_full = v if self.v_cache is None else torch.cat([self.v_cache, v], dim=1)
      x = flash_attn.flash_attn_func(q, k_full, v_full, causal=True)
      self.k_cache = k.detach() if self.k_cache is None else torch.cat([self.k_cache, k.detach()], dim=1)
      self.v_cache = v.detach() if self.v_cache is None else torch.cat([self.v_cache, v.detach()], dim=1)
    else:
      x = flash_attn.flash_attn_func(q, k, v, causal=True)
    x = einops.rearrange(x, 'b s h d -> b s (h d)')

    a = self.attn_out(x)
    if self.swin:
      a = self.norm_post_attn(a)

    if self.adaLN:
      x = bias_dropout_scale_fn(a, None, gate_msa, x_skip, self.dropout)
      if self.sandwich:
        x = self.norm_post_attn(x)
      m = self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp))
      if self.swin:
        m = self.norm_post_mlp(m)
      x = bias_dropout_scale_fn(m, None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(a, None, scale, x_skip, self.dropout)
      if self.sandwich:
        x = self.norm_post_attn(x)
      m = self.mlp(self.norm2(x))
      if self.swin:
        m = self.norm_post_mlp(m)
      x = bias_dropout_scale_fn(m, None, scale, x, self.dropout)
    if self.sandwich:
      x = self.norm_post_mlp(x)
    return x



class DDiTBlock(nn.Module):
  """4-norm layouts, with n1=norm1, n2=norm_post_attn, n3=norm2, n4=norm_post_mlp:
    'sandwich' (seal-rg SandwichBlock): x = n2(x + Attn(n1(x))); x = n4(x + MLP(n3(x)))
    'sandwich_swin':                    x = x + n2(Attn(n1(x))); x = x + n4(MLP(n3(x)))
  """
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1, softcap=0.0,
               block='prenorm', norm_cls=LayerNorm):
    super().__init__()
    assert block in BLOCKS, f"Unknown model.block={block!r}"
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.softcap = softcap
    self.sandwich = block == 'sandwich'
    self.swin = block == 'sandwich_swin'

    self.norm1 = norm_cls(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = norm_cls(dim)
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

    if self.sandwich or self.swin:
      self.norm_post_attn = norm_cls(dim)
      self.norm_post_mlp = norm_cls(dim)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def _attn(self, qkv: torch.Tensor) -> torch.Tensor:
    """Standard full-sequence attention. qkv: [B, S, 3, H, D]"""
    if self.softcap > 0:
      x = flash_attn.flash_attn_qkvpacked_func(
        qkv, 0.0, causal=False, softcap=self.softcap)
      return einops.rearrange(x, 'b s h d -> b s (h d)')
    qkv = einops.rearrange(qkv, 'b s three h d -> b h three s d')
    x = _sdpa_full(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
    return einops.rearrange(x, 'b h s d -> b s (h d)')

  def forward(self, x, rotary_cos_sin, c=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x), 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

    x = self._attn(qkv)

    a = self.attn_out(x)
    if self.swin:
      a = self.norm_post_attn(a)

    if self.adaLN:
      x = bias_dropout_scale_fn(a, None, gate_msa, x_skip, self.dropout)
      if self.sandwich:
        x = self.norm_post_attn(x)
      m = self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp))
      if self.swin:
        m = self.norm_post_mlp(m)
      x = bias_dropout_scale_fn(m, None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(a, None, scale, x_skip, self.dropout)
      if self.sandwich:
        x = self.norm_post_attn(x)
      m = self.mlp(self.norm2(x))
      if self.swin:
        m = self.norm_post_mlp(m)
      x = bias_dropout_scale_fn(m, None, scale, x, self.dropout)
    if self.sandwich:
      x = self.norm_post_mlp(x)
    return x


class EmbeddingLayer(nn.Module):
  # normalize=True: rows on a sphere of learnable radius `scale`. Consumers must
  # read the table via `.matrix`, not `.embedding`.
  def __init__(self, dim, vocab_dim, normalize=False, scale_init=None):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
    self.normalize = normalize
    if normalize:
      if scale_init is None:
        scale_init = self.embedding.detach().norm(dim=-1).mean().item()
      self.scale = nn.Parameter(torch.tensor(float(scale_init)))

  @property
  def matrix(self):
    if not self.normalize:
      return self.embedding
    return F.normalize(self.embedding.float(), dim=-1) * self.scale.float()

  def renormalize(self, x):
    if not self.normalize:
      return x
    return (F.normalize(x.float(), dim=-1) * self.scale.float()).to(x.dtype)

  def forward(self, x):
    emb = self.matrix
    if x.ndim == 2:
      return emb[x]
    assert x.ndim == 3
    return torch.einsum(
      "blv,ve->ble",
      torch.nn.functional.softmax(x, dim=-1).float(),
      emb.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim,
               adaLN, norm_cls=LayerNorm):
    super().__init__()
    self.norm_final = norm_cls(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    x = self.norm_final(x)
    if self.adaLN:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = config.algo.causal_attention
    self.adaLN = config.algo.adaLN
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.vocab_embed = EmbeddingLayer(
      dim, vocab_size,
      normalize=getattr(config.model, 'embed_norm', False),
      scale_init=getattr(config.model, 'embed_norm_init', None))
    if self.adaLN:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)

    softcap = getattr(config.model, 'softcap', 0.0)
    # getattr defaults: configs predating these keys keep the original arch.
    block_type = getattr(config.model, 'block', 'prenorm')
    if block_type not in BLOCKS:
      raise ValueError(f"Unknown model.block={block_type!r}")
    norm_cls = make_norm(getattr(config.model, 'norm', 'layer'),
                         getattr(config.model, 'norm_eps', None))
    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          block=block_type,
          norm_cls=norm_cls)
      else:
        block = DDiTBlock(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          softcap=softcap,
          block=block_type,
          norm_cls=norm_cls)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)

    # Spectral normalization (model.spectral_norm): bound each block's Lipschitz
    # constant to stabilize the loop. False => no parametrization is registered
    # and the forward is byte-identical to before.
    if config.model.spectral_norm:
      for block in self.blocks:
        apply_spectral_norm(
          block, n_power_iterations=config.model.spectral_norm_iters)

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      norm_cls=norm_cls)
    self.ctx_cached_len = 0

    self.uses_hidden_carry = (
      config.algo.name in ('lddm_u', 'lddm_m')
      or (config.algo.name == 'law_carry_ablation'
          and config.algo.carry_representation == 'hidden_representation'))
    if self.uses_hidden_carry:
      self.latent_ln = nn.LayerNorm(dim)
      torch.nn.init.zeros_(self.latent_ln.weight)
      torch.nn.init.zeros_(self.latent_ln.bias)

    # LoopDiT enables optional carry-mode conditioning.
    self.loop_mode_embed = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.ctx_cached_len = 0
    if self.causal:
      for block in self.blocks:
        block.reset_kv_cache()

  def _inject_carry(self, xt, context):
    base = self.vocab_embed(xt)
    if context.carry is None:
      # Keep latent_ln in the graph when the hidden carry is absent.
      if self.uses_hidden_carry and context.carry_representation == 'hidden_representation':
        return base + self.latent_ln(torch.zeros_like(base))
      return base
    carry_inj = (self.latent_ln(context.carry)
                 if context.carry_representation == 'hidden_representation'
                 else context.carry)
    if context.injection_mode == 'add':
      return base + carry_inj
    return carry_inj

  def forward(self, x0, xt, sigma, context=None, return_aux=False):
    kv_cache = False if context is None else context.kv_cache

    if context is not None and 'carry' in context:
      x = self._inject_carry(xt, context)
    else:
      x = self.vocab_embed(xt)

    if self.adaLN:
      t_cond = F.silu(self.sigma_map(sigma))
      # Condition adaLN on carry-free versus looped input when enabled.
      if (self.loop_mode_embed is not None
          and context is not None and 'looped' in context):
        t_cond = t_cond + self.loop_mode_embed[int(context.looped)]
    else:
      t_cond = None

    if kv_cache:
      rotary_cos_sin = self.rotary_emb.forward_range(
        self.ctx_cached_len, self.ctx_cached_len + x.shape[1],
        device=x.device)
    else:
      rotary_cos_sin = self.rotary_emb(x)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        if self.causal:
          x = self.blocks[i](x, rotary_cos_sin, c=t_cond, kv_cache=kv_cache)
        else:
          x = self.blocks[i](x, rotary_cos_sin, c=t_cond)
      # Hidden carries use the final block output before projection.
      hidden = x if return_aux else None
      x = self.output_layer(x, c=t_cond)

    if kv_cache:
      self.ctx_cached_len += xt.shape[1]

    if return_aux:
      return x, {'latent': hidden}
    return x
