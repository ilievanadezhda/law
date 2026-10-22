import math

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
)


class SphereDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.vocab_size = vocab_size
    self.adaLN = config.algo.adaLN
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.embed_dim = dim
    self.init_mode = config.model.init
    self.eps = config.model.eps

    self.sphere_embed = nn.Embedding(vocab_size, dim)
    if self.init_mode == 'random':
      nn.init.normal_(self.sphere_embed.weight, std=0.02)
    elif self.init_mode == 'ngpt':
      nn.init.normal_(self.sphere_embed.weight, std=1.0 / math.sqrt(dim))
    elif self.init_mode == 'pretrained':
      nn.init.zeros_(self.sphere_embed.weight)
    else:
      raise ValueError(self.init_mode)

    if self.adaLN:
      self.sigma_map = TimestepEmbedder(cond_dim)

    self.rotary_emb = Rotary(dim // config.model.n_heads)

    self.blocks = nn.ModuleList([
      DDiTBlock(
        dim=dim,
        n_heads=config.model.n_heads,
        cond_dim=cond_dim,
        adaLN=self.adaLN,
        dropout=config.model.dropout)
      for _ in range(config.model.n_blocks)
    ])
    self.out_temperature_scaling = config.model.learn_temperature_scaling
    if self.out_temperature_scaling:
      out_channels = vocab_size + 1
    else:
      out_channels = vocab_size
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=out_channels,
      cond_dim=cond_dim,
      adaLN=self.adaLN)

    # Required by TrainerBase.ctx_cached_len property.
    self.ctx_cached_len = 0

    if config.model.pretrained_ckpt_path is not None:
      self.load_pretrained_from(config.model.pretrained_ckpt_path)

  def load_pretrained_from(self, ckpt_path: str) -> None:
    """Adapt an AR / DUO Lightning checkpoint into this backbone on the fly.

    The source state_dict lives under keys like `backbone.<param>`; we strip
    that prefix so it maps directly onto SphereDiT, and rename the AR / DUO
    embedding parameter to the sphere-embedding name:

        backbone.vocab_embed.embedding  ->  sphere_embed.weight

    DDiTBlock / DDiTBlockCausal / DDiTFinalLayer share parameter names and
    shapes, so block and output-layer weights transfer cleanly. Target-only
    params (e.g. sigma_map and adaLN_modulation when the source is AR) keep
    their fresh-init values. Source-only params (DUO teachers, etc.) and
    shape-mismatched entries are dropped.

    When the source had no adaLN (AR: `algo.adaLN=False`), the target's
    fresh-init block adaLN (zero weight, zero bias) makes gate_msa=gate_mlp=0,
    which silences the attention/MLP outputs at step 0. We patch the bias so
    the modulation starts as an identity: shift=0, scale=0, gate=1 per block
    — matching the non-adaLN forward `x_skip + attn_out(norm1(x))`.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_sd = (ckpt['state_dict']
              if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)

    # Pull source metadata to decide on adaLN identity-init.
    src_hp = ckpt.get('hyper_parameters', {}) if isinstance(ckpt, dict) else {}
    src_cfg = src_hp.get('config', {}) or {}
    src_algo = src_cfg.get('algo', {}) or {}
    src_algo_name = src_algo.get('name', '<unknown>')
    src_had_adaLN = bool(src_algo.get('adaLN', True))

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
        # Dim-0 mismatch only (e.g., AR trains with an extra mask_index
        # row appended; SFM has no mask). Copy the overlapping prefix;
        # the remainder keeps its fresh init.
        n = min(tgt.shape[0], v.shape[0])
        new_tgt = tgt.clone()
        new_tgt[:n] = v[:n]
        own_sd[k] = new_tgt
        sliced.append((k, tuple(v.shape), tuple(tgt.shape)))
        loaded += 1
      else:
        skipped += 1

    # If the source has no adaLN (AR), patch block adaLN biases so the
    # modulation is an identity at step 0: shift=0, scale=0, gate=1. The
    # DDiTFinalLayer's 2-way chunk (shift, scale) is already an identity
    # from fresh init — no patch needed there.
    adaLN_identity_patched = 0
    if self.adaLN and not src_had_adaLN:
      dim = self.embed_dim
      for i, block in enumerate(self.blocks):
        if not block.adaLN:
          continue
        k = f'blocks.{i}.adaLN_modulation.bias'
        if k not in own_sd:
          continue
        b = own_sd[k].clone()
        b.zero_()
        b[2 * dim: 3 * dim] = 1.0
        b[5 * dim: 6 * dim] = 1.0
        own_sd[k] = b
        adaLN_identity_patched += 1

    self.load_state_dict(own_sd, strict=True)
    fresh = len(own_sd) - loaded
    print(
      f'[SphereDiT.load_pretrained_from] {ckpt_path}\n'
      f'  source   : algo.name={src_algo_name!r}  adaLN={src_had_adaLN}\n'
      f'  loaded   : {loaded}/{len(own_sd)}\n'
      f'  fresh    : {fresh}\n'
      f'  skipped  : {skipped} (not in target or shape mismatch)')
    for k, s_shape, t_shape in sliced:
      print(f'  sliced   : {k}  src{s_shape} -> tgt{t_shape}')
    if adaLN_identity_patched:
      print(f'  adaLN    : {adaLN_identity_patched} blocks patched to '
            f'identity (shift=0, scale=0, gate=1)')

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
    return utils.sphere_normalize(emb)

  def reset_kv_cache(self):
    self.ctx_cached_len = 0

  def renormalize_weights(self):
    self.sphere_embed.weight.data = utils.sphere_normalize(
      self.sphere_embed.weight.data)

  def forward(self, x0, xt: torch.Tensor, sigma: torch.Tensor,
              context=None) -> torch.Tensor:
    del x0, context

    x = xt  # [B, L, d], already on the sphere

    if self.adaLN:
      t_cond = F.silu(self.sigma_map(sigma))
    else:
      t_cond = None

    rotary_cos_sin = self.rotary_emb(x)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(x, rotary_cos_sin, c=t_cond)
      x = self.output_layer(x, c=t_cond)
      if self.out_temperature_scaling:
        pre_temperature = x[:, :, [-1]]
        x = x[:, :, :-1]
        t = 1 + self.eps + torch.tanh(pre_temperature) * (1 - self.eps) 
        x = x / t
    return x
