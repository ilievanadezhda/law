"""Implementations of the samplers.

Content:
  - Pure implementations of posteriors, and sampling utils
  - Sampler classes. Looping over step, after state init.
  - Dataclasses naming convention / use case:
      - State: sampler data persisted across steps.
      - Context: data passed to the denoising neural network.
      - View: data derived from state and used only within a
              single step to pass variables around.
"""

import abc
import math
import torch
import torch.nn.functional as F
from dataclass_patch import dataclass
import utils
from temperature_schedules import get_temperature_schedule, sampling_time


def sample_categorical(categorical_probs, gumbel_noise=None):
  """Sample from categorical using the Gumbel trick"""
  if gumbel_noise is None:
    gumbel_noise = (
      1e-10
      - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
  return (categorical_probs / gumbel_noise).argmax(dim=-1)


def _normalize_posterior_inputs(x0_probs, xt, alpha_s, 
                                alpha_t, use_float64=False):
  if use_float64:
    x0_probs = x0_probs.to(torch.float64)
  if alpha_s.ndim == 2:
    alpha_s = alpha_s.unsqueeze(-1)
  if alpha_t.ndim == 2:
    alpha_t = alpha_t.unsqueeze(-1)
  return x0_probs, xt, alpha_s, alpha_t


@torch.compile
def absorbing_posterior_probs(x0_probs, xt, alpha_s, alpha_t,
                              mask_index, vocab_size,
                              use_float64=False):
  """Posterior q(x_s | x_t, x_0) for absorbing diffusion"""
  x0_probs, xt, alpha_s, alpha_t = _normalize_posterior_inputs(
    x0_probs, xt, alpha_s, alpha_t, use_float64=use_float64)
  denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
  xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs)
  is_masked = (xt == mask_index).to(x0_probs.dtype).unsqueeze(-1)
  masked_posterior = (
    xt_one_hot * (1 - denoise_prob)
    + x0_probs * denoise_prob)
  return xt_one_hot * (1 - is_masked) + masked_posterior * is_masked


@torch.compile
def sample_absorbing_posterior(x0_probs, xt, alpha_s, alpha_t,
                               mask_index,
                               noise_removal_step=False):
  """Sample from absorbing posterior without materializing it"""
  sampled_x0 = sample_categorical(x0_probs)

  if noise_removal_step:
    should_denoise = torch.ones_like(sampled_x0, dtype=torch.bool)
  else:
    denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
    should_denoise = torch.rand_like(
      sampled_x0, dtype=torch.float64) < denoise_prob

  is_masked = (xt == mask_index)
  should_denoise_mask = is_masked & should_denoise
  return torch.where(should_denoise_mask, sampled_x0, xt)


def _expand_alpha_like(alpha, ref):
  """Expand alpha to match the dimension of ref. One of
     (B, 1), (B, L), (B, L, 1). Output should have ndim=2"""
  if alpha.ndim == 3:
    alpha = alpha.squeeze(-1)
  if alpha.ndim != 2:
    raise ValueError(f'Unexpected alpha shape: {alpha.shape}')
  if alpha.shape[1] == 1:
    return alpha.expand_as(ref)
  if alpha.shape[1] != ref.shape[1]:
    raise ValueError(
      f'Alpha shape {alpha.shape} not broadcastable to {ref.shape}')
  return alpha


@torch.compile
def uniform_posterior_probs(x0_probs, xt, alpha_s, alpha_t,
                            vocab_size, use_float64=False):
  """Posterior q(x_s | x_t, x_0) for uniform diffusion."""
  x0_probs, xt, alpha_s, alpha_t = _normalize_posterior_inputs(
    x0_probs, xt, alpha_s, alpha_t, use_float64=use_float64)
  alpha_ts = alpha_t / alpha_s
  d_alpha = alpha_s - alpha_t
  xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs)
  p_xt = torch.gather(x0_probs, -1, xt[..., None])
  numerator = (
    alpha_t * vocab_size * x0_probs * xt_one_hot
    + (alpha_ts - alpha_t) * xt_one_hot
    + d_alpha * x0_probs
    + (1 - alpha_ts) * (1 - alpha_s) / vocab_size)
  denominator = alpha_t * vocab_size * p_xt + (1 - alpha_t)
  return numerator / denominator


@torch.compile
def sample_uniform_posterior(x0_probs, xt, alpha_s, alpha_t,
  vocab_size, noise_removal_step=False):
  """Sample from uniform posterior without materializing it."""
  p_xt = torch.gather(x0_probs, -1, xt.unsqueeze(-1)).squeeze(-1)
  alpha_t = _expand_alpha_like(alpha_t, p_xt).to(x0_probs.dtype)
  alpha_s = _expand_alpha_like(alpha_s, p_xt).to(x0_probs.dtype)
  denominator = alpha_t * vocab_size * p_xt + (1 - alpha_t)

  sampled_x0 = sample_categorical(x0_probs)
  sample_threshold = torch.rand_like(p_xt)

  if noise_removal_step:
    keep_xt_prob = (
      alpha_t * vocab_size * p_xt / denominator).clamp(0.0, 1.0)
    return torch.where(sample_threshold < keep_xt_prob,
                       xt, sampled_x0)

  alpha_ts = alpha_t / alpha_s
  sample_uniform_prob = (
    (1 - alpha_ts) * (1 - alpha_s) / denominator).clamp(0.0, 1.0)
  keep_xt_prob = (
    (alpha_t * vocab_size * p_xt + alpha_ts - alpha_t)
    / denominator).clamp(0.0, 1.0)
  uniform_samples = torch.randint(0, vocab_size, xt.shape,
                                  device=xt.device)
  # priority: uniform > xt > x0
  keep_or_resample = torch.where(
    (sample_uniform_prob + keep_xt_prob).clamp(max=1.0)
    > sample_threshold,
    xt,
    sampled_x0)
  return torch.where(sample_uniform_prob > sample_threshold,
                     uniform_samples,
                     keep_or_resample)


def uniform_posterior_probs_mixing(x0_probs, xt, alpha_s, alpha_t,
                                   vocab_size, use_float64=False):
  """Posterior q(x_s | x_t, x_0) for uniform diffusion with a
  distributional (3D) xt, i.e. xt is a probability distribution over
  the vocabulary rather than a one-hot token. Used by MixingDUO."""
  if use_float64:
    x0_probs = x0_probs.to(torch.float64)
  if alpha_s.ndim == 2:
    alpha_s = alpha_s.unsqueeze(-1)
  if alpha_t.ndim == 2:
    alpha_t = alpha_t.unsqueeze(-1)

  alpha_ts = alpha_t / alpha_s
  d_alpha = alpha_s - alpha_t

  xt = xt.to(x0_probs)
  numerator = (
    alpha_t * vocab_size * x0_probs * xt
    + (alpha_ts - alpha_t) * xt
    + d_alpha * x0_probs
    + (1 - alpha_ts) * (1 - alpha_s) / vocab_size)
  denominator = alpha_t * vocab_size * (
    (x0_probs * xt).sum(dim=-1, keepdim=True)) + (1 - alpha_t)

  out = numerator / denominator
  return out / out.sum(dim=-1, keepdim=True)


def sample_posterior(process, x0_probs, xt, alpha_s, alpha_t, *,
  mask_index=None, vocab_size, noise_removal_step=False,
  posterior_sampler='fast', use_float64=False):
  if process == 'absorbing':
    return sample_absorbing_posterior(
      x0_probs=x0_probs, xt=xt,
      alpha_s=alpha_s, alpha_t=alpha_t,
      mask_index=mask_index,
      noise_removal_step=noise_removal_step)
  elif process == 'uniform':
    if posterior_sampler == 'fast':
      return sample_uniform_posterior(
        x0_probs=x0_probs, xt=xt,
        alpha_s=alpha_s, alpha_t=alpha_t,
        vocab_size=vocab_size,
        noise_removal_step=noise_removal_step)
    elif posterior_sampler == 'naive':
      posterior_probs = compute_posterior(
        process, x0_probs, xt, alpha_s, alpha_t,
        mask_index=mask_index,
        vocab_size=vocab_size,
        use_float64=use_float64)
      return sample_categorical(posterior_probs)
    else:
      raise ValueError(posterior_sampler)
  else:
    raise ValueError(process)


def compute_posterior(process, x0_probs, xt, alpha_s, alpha_t, *,
  mask_index=None, vocab_size, use_float64=False):
  if process == 'absorbing':
    return absorbing_posterior_probs(
      x0_probs=x0_probs,
      xt=xt,
      alpha_s=alpha_s,
      alpha_t=alpha_t,
      mask_index=mask_index,
      vocab_size=vocab_size,
      use_float64=use_float64)
  if process == 'uniform':
    if xt.ndim == 2:
      return uniform_posterior_probs(
        x0_probs=x0_probs,
        xt=xt,
        alpha_s=alpha_s,
        alpha_t=alpha_t,
        vocab_size=vocab_size,
        use_float64=use_float64)
    elif xt.ndim == 3:
      return uniform_posterior_probs_mixing(
        x0_probs=x0_probs,
        xt=xt,
        alpha_s=alpha_s,
        alpha_t=alpha_t,
        vocab_size=vocab_size,
        use_float64=use_float64)
    else:
      raise ValueError(f'Wrong dimension for xt: ndim = {xt.ndim}')
  raise ValueError(f'Unknown process: {process}')


def _maybe_cast_log_probs(log_probs, use_float64):
  if use_float64:
    return log_probs.to(torch.float64)
  return log_probs


def _model_mask_index(model):
  if model.diffusion_type != 'absorbing':
    return None
  if not hasattr(model, 'mask_index'):
    raise AttributeError(
      'Absorbing diffusion models must define model.mask_index')
  return model.mask_index


def _decode_direct_tokens(log_probs, greedy):
  if greedy:
    return log_probs.argmax(dim=-1)
  return sample_categorical(log_probs.exp())


def _decode_posterior_tokens(
    model, log_probs, current_tokens, alpha_s, alpha_t, *,
    greedy, noise_removal_step, use_float64):
  probs = log_probs.exp()
  mask_index = _model_mask_index(model)
  if greedy:
    posterior = compute_posterior(
      model.diffusion_type, probs, current_tokens,
      alpha_s, alpha_t,
      mask_index=mask_index,
      vocab_size=model.vocab_size,
      use_float64=use_float64)
    return posterior.argmax(dim=-1)
  return sample_posterior(
    model.diffusion_type, probs, current_tokens,
    alpha_s, alpha_t,
    mask_index=mask_index,
    vocab_size=model.vocab_size,
    noise_removal_step=noise_removal_step)


def _decode_ancestral_update(
    sampler, model, log_probs, current_tokens, alpha_s, alpha_t, *,
    is_last_step):
  """Generic ancestral transition kernel on an arbitrary token window.

  This is the shared black-box update used by the full-sequence ancestral
  sampler. Scheduling and state transitions stay
  outside this function; it only maps:
    current_tokens_t + model log p(x0 | xt) + (alpha_t, alpha_s)
  to:
    next_tokens_s
  """
  if is_last_step and sampler.noise_removal == 'greedy':
    return log_probs.argmax(dim=-1)
  return _decode_posterior_tokens(
    model, log_probs, current_tokens, alpha_s, alpha_t,
    greedy=False,
    noise_removal_step=is_last_step,
    use_float64=sampler.use_float64)


def _early_stop_token(state, xt, idx, tokenizer):
  """Early stopping for token-by-token generation.

  This is useful for tasks where generation should stop at 
  the first EOS. For example, for GSM8K, we only care about 
  generating one answer.

  Called after a single token has been written to xt[:, idx]. 
  Return true when all sequences are done.
  """
  eos_id = tokenizer.eos_token_id
  pad_id = tokenizer.pad_token_id

  newly_finished = (~state.finished) & (xt[:, idx] == eos_id)
  if newly_finished.any():
    xt[newly_finished, idx + 1:] = pad_id
  state.finished = state.finished | newly_finished
  return bool(state.finished.all())


# ----------------------------------------------------------
# Sampler classes, actually used to generate
# ----------------------------------------------------------
@dataclass(kw_only=True)
class BaseState:
  prefix_tokens: torch.Tensor = None   # (B, max_prefix_len) or (B, max_prefix_len, V)
  prefix_lengths: torch.Tensor = None  # (B,) int64


class Sampler(abc.ABC):
  """API that all samplers must comply with."""

  @abc.abstractmethod
  def init_state(self, model, num_samples, *, num_steps,
                 eps, prefix_tokens, prefix_lengths):
    ...

  @abc.abstractmethod
  def step(self, model, state):
    ...

  def _validate_prefix_args(self, prefix_tokens, prefix_lengths):
    assert (prefix_tokens is None) == (prefix_lengths is None), \
      'prefix_tokens and prefix_lengths must both be set or both be None'

  def _project_prefix(self, xt, prefix_tokens, prefix_lengths):
    if prefix_tokens is None:
      return xt
    P = prefix_tokens.shape[1]
    positions = torch.arange(P, device=xt.device)
    mask = positions[None, :] < prefix_lengths[:, None]  # (B, P)
    if xt.ndim == 3:
      mask = mask.unsqueeze(-1)
    xt[:, :P] = torch.where(mask, prefix_tokens, xt[:, :P])
    return xt

  def metadata(self, state):
    return {'nfe': state.nfe}


@dataclass(kw_only=True)
class AncestralState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool


@dataclass
class AncestralContext:
  temperature: float = 1.0
  kv_cache: bool = False


@dataclass
class CarryContext:
  carry: object = None
  carry_representation: str = ''
  injection_mode: str = ''
  kv_cache: bool = False
  temperature: float = 1.0
  looped: bool = False  # Selects carry-free or looped mode conditioning.


class AncestralSampler(Sampler):
  """Full-sequence ancestral sampler for MDLM & DUO"""

  def __init__(self, temperature_schedule, use_float64=False,
               noise_removal='ancestral', steps_policy='full'):
    self.use_float64 = use_float64
    self.noise_removal = noise_removal
    self.steps_policy = steps_policy
    self.temperature_schedule = temperature_schedule
    assert noise_removal in ('ancestral', 'greedy')
    assert steps_policy in ('full', 'proportional')

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = model.prior_sample(num_samples, model.num_tokens)
    if prefix_tokens is not None:
      start_idx = int(prefix_lengths.min())
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
    else:
      start_idx = 0

    if num_steps is None:
      num_steps = model.config.sampler.steps
    # Adapt number of sampling steps to the number of tokens
    #  left to sample.
    if self.steps_policy == 'proportional':
      gen_len = model.num_tokens - start_idx
      num_steps = int(round(num_steps * gen_len))

    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    ones = torch.ones(xt.shape[0], 1, dtype=model.dtype,
                      device=model.device)
    state = AncestralState(
      xt=xt, timesteps=timesteps, ones=ones, start_idx=start_idx,
      step_idx=0, nfe=0, done=False,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)
    return state

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    temperature = self.temperature_schedule(sampling_time(state.step_idx, num_steps))
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones

    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)

    log_probs = _maybe_cast_log_probs(
      model.forward(xt=state.xt, sigma=sigma_t,
                    context=AncestralContext(temperature=temperature)),
      self.use_float64)

    current_tokens = state.xt[:, state.start_idx:]
    log_probs_window = log_probs[:, state.start_idx:]

    current_tokens[:] = _decode_ancestral_update(
      self, model, log_probs_window, current_tokens,
      alpha_s, alpha_t, is_last_step=is_last_step)

    self._project_prefix(state.xt, state.prefix_tokens, 
                         state.prefix_lengths)
    state.nfe += 1
    state.step_idx += 1
    if is_last_step:
      state.done = True
    return state

@dataclass(kw_only=True)
class LDDMState(AncestralState):
  latent: object = None


class LDDMSampler(AncestralSampler):
  """Masked or uniform ancestral sampling with an additive hidden carry."""

  def init_state(self, model, num_samples, *, num_steps=None, eps=1e-5,
                 prefix_tokens=None, prefix_lengths=None):
    base = super().init_state(
      model, num_samples, num_steps=num_steps, eps=eps,
      prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths)
    return LDDMState(**vars(base), latent=None)

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    temperature = self.temperature_schedule(sampling_time(state.step_idx, num_steps))
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones

    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)

    log_probs, aux = model.forward(
      xt=state.xt, sigma=sigma_t,
      context=CarryContext(
        carry=state.latent, carry_representation='hidden_representation',
        injection_mode='add', temperature=temperature), return_aux=True)
    state.latent = aux['latent']
    log_probs = _maybe_cast_log_probs(log_probs, self.use_float64)

    current_tokens = state.xt[:, state.start_idx:]
    log_probs_window = log_probs[:, state.start_idx:]
    current_tokens[:] = _decode_ancestral_update(
      self, model, log_probs_window, current_tokens,
      alpha_s, alpha_t, is_last_step=is_last_step)

    self._project_prefix(state.xt, state.prefix_tokens,
                         state.prefix_lengths)
    state.nfe += 1
    state.step_idx += 1
    if is_last_step:
      state.done = True
    return state


@dataclass(kw_only=True)
class ARState(BaseState):
  xt: torch.Tensor
  ones: torch.Tensor
  zeros: torch.Tensor
  start_idx: int
  token_idx: int
  cached_len: int   # number of tokens committed to the KV cache
  nfe: int
  done: bool
  # Whether each xt has already generated an EOS. Only used 
  #  when early_stopping is True.
  finished: torch.Tensor = None  


@dataclass
class ARContext:
  kv_cache: bool
  temperature: float = 1.0


class ARSampler(Sampler):
  """Autoregressive sampler, generate tokens L2R, feed prefix
     only (suggested to use with causal transformers only)"""

  def __init__(self, use_float64, kv_cache, greedy,
               early_stopping, temperature=1.0):
    self.use_float64 = use_float64
    self.kv_cache = kv_cache
    self.greedy = greedy
    self.early_stopping = early_stopping
    self.temperature = temperature

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    if prefix_lengths is not None and prefix_lengths.unique().numel() > 1:
      raise NotImplementedError(
        'ARSampler does not support variable-length prefixes')
    xt = torch.zeros(num_samples, model.num_tokens,
                     dtype=torch.long, device=model.device)
    if prefix_tokens is not None:
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
      start_idx = int(prefix_lengths.max())
    else:
      xt[:, 0] = model.tokenizer.bos_token_id
      start_idx = 1
    ones = torch.ones(num_samples, 1, dtype=model.dtype,
                      device=model.device)
    zeros = torch.zeros_like(ones)
    if self.kv_cache:
      model.reset_kv_cache()
    if self.early_stopping:
      finished = torch.zeros(num_samples, dtype=torch.bool,
                             device=model.device)
    else:
      finished = None
    state = ARState(xt=xt, ones=ones, zeros=zeros,
      start_idx=start_idx, token_idx=start_idx, cached_len=0,
      nfe=0, done=(start_idx >= xt.shape[1]), finished=finished)
    return state

  def step(self, model, state):
    x = state.xt
    if self.kv_cache:
      start_idx = state.cached_len
    else:
      start_idx = 0
    end_idx = state.token_idx - 1

    x_input = x[:, start_idx: end_idx + 1]
    log_p = model.forward(xt=x_input, sigma=state.zeros,
      context=ARContext(kv_cache=self.kv_cache,
                        temperature=self.temperature))

    log_probs = _maybe_cast_log_probs(log_p[:, -1], self.use_float64)
    new_tok = _decode_direct_tokens(log_probs, greedy=self.greedy)

    x[:, state.token_idx] = new_tok

    if self.early_stopping:
      _early_stop_token(state, x, state.token_idx, model.tokenizer)

    if self.kv_cache:
      state.cached_len = state.token_idx
    state.nfe += 1
    state.token_idx += 1
    state.done = state.token_idx >= x.shape[1]
    if self.early_stopping and bool(state.finished.all()):
      state.done = True
    return state


def run_sampler(sampler, model, num_samples, *,
               num_steps=None, eps=1e-5, prefix_tokens=None,
               prefix_lengths=None):
  state = sampler.init_state(model, num_samples,
                             num_steps=num_steps, eps=eps,
                             prefix_tokens=prefix_tokens,
                             prefix_lengths=prefix_lengths)
  while not state.done:
    state = sampler.step(model, state)
  return state.xt, sampler.metadata(state)


# ══════════════════════════════════════════════════════════════
#  SFM (hyperspherical flow) sampler
# ══════════════════════════════════════════════════════════════

@dataclass(kw_only=True)
class SFMState(BaseState):
  # [B, L, d] float sphere embeddings during integration;
  # replaced by [B, L] int token ids at the final decoding step.
  xt: torch.Tensor
  t_schedule: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool
  prefix_lengths: torch.Tensor = None  # [B] per-sample lengths
  prefix_tokens: torch.Tensor = None  # [B, L]
  prefix_embeds: torch.Tensor = None  # [B, P, d] sphere embeddings of prefix


@dataclass
class SFMContext:
  temperature: float = 1.0

@torch.compile
def sfm_compute_velocity(x, E, log_p, mode, eps):
  p = log_p.exp()
  if mode == 'exact':
    if E.ndim == 2:  # [V, d]
      ein_fwd = 'bld,vd->blv'
      ein_bwd = 'blv,vd->bld'
    else:           # [B, L, k, d]
      ein_fwd = 'bld,blkd->blk'
      ein_bwd = 'blk,blkd->bld'
    cos_omega = torch.einsum(ein_fwd, x, E)
    cos_omega = cos_omega.clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    scale = omega / omega.sin().clamp(min=eps)
    p_scale = p * scale
    term1 = torch.einsum(ein_bwd, p_scale, E)
    term2 = x * (p_scale * cos_omega).sum(dim=-1, keepdim=True)
    return term1 - term2
  elif mode == 'sample':
    target_idx = sample_categorical(p)
    if E.ndim == 2:  # [V, d]
      target = E[target_idx]
    else:            # [B, L, k, d]
      B, L = target_idx.shape
      target = E[
        torch.arange(B, device=E.device)[:, None],
        torch.arange(L, device=E.device)[None, :],
        target_idx]
    return utils.log_map(x, target, eps)
  else:
    raise ValueError(f'Unknown velocity mode: {mode}')


def sfm_step_size(alpha_t, alpha_s, invert_time_convention, eps):
  if invert_time_convention:
    # slerp param = alpha_t, decreasing along schedule
    return (alpha_t - alpha_s) / alpha_t.clamp(min=eps)
  else:
    # slerp param = 1 - alpha_t, decreasing along schedule
    return (alpha_s - alpha_t) / (1 - alpha_t).clamp(min=eps)


class SFMSampler(Sampler):
  def __init__(self, noise_removal, velocity, use_float64,
               slerp_float64, eps, temperature_schedule, p_nucleus,
               top_k,
               top_k_velocity,
               invert_time_convention):
    self.noise_removal = noise_removal
    self.velocity = velocity
    self.use_float64 = use_float64
    self.slerp_float64 = slerp_float64
    self.eps = eps
    self.temperature_schedule = temperature_schedule
    self.p_nucleus = p_nucleus
    self.top_k = top_k
    self.top_k_velocity = top_k_velocity
    self.invert_time_convention = invert_time_convention

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = torch.randn(num_samples, model.num_tokens,
      model.backbone.embed_dim, device=model.device,
      dtype=torch.float32)
    xt = utils.sphere_normalize(xt)

    prefix_embeds = None
    if prefix_tokens is not None:
      prefix_embeds = model.backbone.get_sphere_embeddings(
        prefix_tokens)
      self._project_prefix(xt, prefix_embeds, prefix_lengths)
      start_idx = int(prefix_lengths.min())
    else:
      start_idx = 0

    if num_steps is None:
      num_steps = model.config.sampler.steps

    if self.invert_time_convention:
      t_schedule = torch.linspace(eps, 1.0, num_steps + 1,
                                  device=model.device)
    else:
      t_schedule = torch.linspace(1.0, eps, num_steps + 1,
                                  device=model.device)
    state = SFMState(xt=xt, t_schedule=t_schedule,
      start_idx=start_idx, step_idx=0, nfe=0, done=False,
      prefix_lengths=prefix_lengths,
      prefix_embeds=prefix_embeds,
      prefix_tokens=prefix_tokens)
    return state

  def _last_step_decode(self, state, log_p):
    if self.noise_removal == 'greedy':
      tokens = log_p.argmax(dim=-1)
    elif self.noise_removal == 'ancestral':
      tokens = sample_categorical(log_p.exp())
    else:
      raise ValueError(self.noise_removal)

    if state.prefix_embeds is not None:
      self._project_prefix(tokens, state.prefix_tokens, 
                           state.prefix_lengths)
    state.xt = tokens  # replace continuous [B,L,d] with int [B,L]
    state.done = True
    return state
  
  def _select_topk(self, log_p, E, k):
    log_p_k, top_idxs = torch.topk(log_p, k, dim=-1)
    return torch.log_softmax(log_p_k, dim=-1), F.embedding(top_idxs, E)

  def _compute_velocity(self, x, E, log_p):
    return sfm_compute_velocity(
      x, E, log_p, mode=self.velocity, eps=self.eps)

  def _get_step_size(self, model, state):
    _, alpha_t = model.noise(state.t_schedule[state.step_idx])
    _, alpha_s = model.noise(state.t_schedule[state.step_idx + 1])
    return sfm_step_size(
      alpha_t, alpha_s, self.invert_time_convention, self.eps)

  def step(self, model, state):
    num_steps = len(state.t_schedule) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    temperature = self.temperature_schedule(sampling_time(state.step_idx, num_steps))

    _, alpha_t = model.noise(state.t_schedule[state.step_idx])
    sigma_t = model._sigma_from_alphat(alpha_t).reshape(-1, 1)

    context = SFMContext(temperature=temperature)
    log_p = model.forward(xt=state.xt, sigma=sigma_t,
                          context=context)
    if self.use_float64:
      log_p = log_p.to(torch.float64)
    state.nfe += 1

    if self.p_nucleus != 1.0 or self.top_k != -1:
      log_p = utils.top_k_top_p_filtering(log_p, 
        top_k=self.top_k, top_p=self.p_nucleus).log_softmax(-1)

    if is_last_step:
      return self._last_step_decode(state, log_p)
    # Arguments to compute the velocity field:
    #  v = sum_k p_k * log_{x}(e_k).
    log_p_window = log_p[:, state.start_idx:]  # [B, L, V]
    E = utils.sphere_normalize(
      model.backbone.sphere_embed.weight.detach())  # [V, d]
    x = state.xt[:, state.start_idx:].to(E)  # [B, L, d]

    if self.slerp_float64:
      E = E.to(torch.float64)
      x = x.to(torch.float64)

    if self.top_k_velocity > 0:
      log_p_v, E = self._select_topk(log_p_window, E, self.top_k_velocity)
    else:
      log_p_v = log_p_window

    vel = self._compute_velocity(x, E, log_p_v)

    dt = self._get_step_size(model, state)
    x_new = utils.exp_map(x, dt * vel, self.eps)
    state.xt[:, state.start_idx:] = x_new.to(state.xt.dtype)
    self._project_prefix(
      state.xt, state.prefix_embeds, state.prefix_lengths)
    state.step_idx += 1
    return state


# ════════════════════════════════════════════════════════════════
#  FLM Euler sampler
# ════════════════════════════════════════════════════════════════
def flm_euler_step(z, x_1_pred_probs, gamma_t, gamma_d):
  """One Euler ODE step for FLM.

  z: current state (B, L, V); x_1_pred_probs: exp(log_softmax(out)) (B, L, V);
  gamma_t: current gamma (B,) or scalar; gamma_d: step size in gamma space.
  """
  if not isinstance(gamma_t, (int, float)):
    gamma_t = gamma_t.view(-1, 1, 1)
  if not isinstance(gamma_d, (int, float)):
    gamma_d = gamma_d.view(-1, 1, 1)
  v = (x_1_pred_probs - z) / (1.0 - gamma_t + 1e-5)
  return z + gamma_d * v


@dataclass
class FLMContext:
  temperature: float = 1.0
  skip_softmax: bool = False


@dataclass(kw_only=True)
class FLMState(BaseState):
  # Continuous logits become token ids at the final step.
  xt: torch.Tensor
  t_vals: torch.Tensor    # (num_steps+1,) time grid
  step_idx: int
  nfe: int
  done: bool
  start_idx: int          # prefix length


class FLMEulerSampler(Sampler):
  """Euler ODE sampler for FLM, with token decoding at the final step."""

  def __init__(self, temperature_schedule, use_float64=False):
    self.use_float64 = use_float64
    self.temperature_schedule = temperature_schedule

  def init_state(self, model, num_samples, *, num_steps=None, eps=1e-5,
                 prefix_tokens=None, prefix_lengths=None):
    if num_steps is None:
      num_steps = model.config.sampler.steps
    V = model.vocab_size
    L = model.num_tokens
    device = model.device
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    z = torch.randn(num_samples, L, V, device=device, dtype=model.dtype)
    prefix_onehot = None
    if prefix_tokens is not None:
      prefix_onehot = F.one_hot(prefix_tokens, V).float().to(device)
      start_idx = int(prefix_lengths.min())
      self._project_prefix(z, prefix_onehot, prefix_lengths)
    else:
      start_idx = 0
    t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    return FLMState(xt=z, t_vals=t_vals, step_idx=0, nfe=0, done=False,
                    start_idx=start_idx, prefix_tokens=prefix_onehot,
                    prefix_lengths=prefix_lengths)

  def _model_log_p(self, model, xt, t_in, temperature):
    # Request raw logits only when temperature scaling is needed.
    if temperature == 1.0:
      return model.forward(xt=xt, sigma=t_in.unsqueeze(-1), context=None)
    logits = model.forward(xt=xt, sigma=t_in.unsqueeze(-1),
                           context=FLMContext(skip_softmax=True))
    return (logits / temperature).log_softmax(-1)

  def step(self, model, state):
    num_steps = len(state.t_vals) - 1
    is_last = state.step_idx == num_steps - 1
    temperature = self.temperature_schedule(sampling_time(state.step_idx, num_steps))
    B = state.xt.shape[0]

    t_curr = state.t_vals[state.step_idx]
    t_next = state.t_vals[state.step_idx + 1]
    t_in = t_curr.expand(B)

    gamma_t = model._alpha_t_to_gamma(t_in)
    gamma_next = model._alpha_t_to_gamma(t_next.expand(B))
    gamma_d = gamma_next - gamma_t

    log_p = self._model_log_p(model, state.xt, t_in, temperature)
    state.nfe += 1
    if self.use_float64:
      log_p = log_p.to(torch.float64)
    x_1_pred = log_p.exp()

    if is_last:
      state.xt = log_p.argmax(dim=-1)
      state.done = True
    else:
      state.xt = flm_euler_step(state.xt, x_1_pred, gamma_t, gamma_d)
      self._project_prefix(
        state.xt, state.prefix_tokens, state.prefix_lengths)

    state.step_idx += 1
    return state


@dataclass(kw_only=True)
class LAWState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  trajectory: object
  step_idx: int = 0
  nfe: int = 0
  done: bool = False


class LAWSampler(Sampler):
  def __init__(self, *, steps_policy, noise_removal, temperature_schedule):
    if steps_policy not in ('full', 'proportional'):
      raise ValueError('LAW steps_policy must be full or proportional')
    if noise_removal not in ('ancestral', 'greedy'):
      raise ValueError('LAW noise_removal must be ancestral or greedy')
    self.steps_policy = steps_policy
    self.noise_removal = noise_removal
    self.temperature_schedule = temperature_schedule

  def init_state(self, model, num_samples, *, num_steps=None, eps=1e-5,
                 prefix_tokens=None, prefix_lengths=None):
    from models.dit_loop import LAWTrajectory

    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    if num_steps is None:
      num_steps = model.config.sampler.steps
    if type(num_steps) is not int or num_steps < 1:
      raise ValueError('LAW num_steps must be a positive integer')
    if not math.isfinite(eps) or not 0 < eps < 1:
      raise ValueError('LAW eps must be between zero and one')
    if prefix_tokens is not None:
      if (prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != num_samples
          or prefix_tokens.shape[1] > model.num_tokens
          or prefix_lengths.shape != (num_samples,)
          or prefix_lengths.is_floating_point()
          or (prefix_lengths < 0).any()
          or (prefix_lengths > prefix_tokens.shape[1]).any()):
        raise ValueError('Invalid LAW prefix shapes or lengths')
      start_idx = int(prefix_lengths.min())
    else:
      start_idx = 0
    xt = model.prior_sample(num_samples, model.num_tokens)
    self._project_prefix(xt, prefix_tokens, prefix_lengths)
    if self.steps_policy == 'proportional':
      num_steps = max(1, round(num_steps * (model.num_tokens - start_idx)
                              / model.num_tokens))
    valid_tokens = None
    if prefix_tokens is not None:
      positions = torch.arange(model.num_tokens, device=xt.device)
      valid_tokens = (positions[None, :] >= prefix_lengths[:, None]).long()
    trajectory = LAWTrajectory(xt, valid_tokens, False, None)
    return LAWState(
      xt=xt, timesteps=torch.linspace(1, eps, num_steps + 1, device=model.device),
      ones=torch.ones(num_samples, 1, dtype=model.dtype, device=model.device),
      start_idx=start_idx, trajectory=trajectory,
      prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths)

  def step(self, model, state):
    if state.done:
      return state
    num_steps = len(state.timesteps) - 1
    is_last = state.step_idx == num_steps - 1
    temperature = self.temperature_schedule(sampling_time(state.step_idx, num_steps))
    t = state.timesteps[state.step_idx] * state.ones

    alpha_t = model.noise(t)[1]
    sigma = model._process_sigma(model._sigma_from_alphat(alpha_t))
    log_p, state.trajectory = model.backbone.forward_law(
      state.trajectory, sigma, prediction_grad=False, return_carry=not is_last,
      carry_temperature=temperature, temperature=temperature)
    
    if is_last:
      # Preserve the raw probability dtype and final categorical RNG draw shape.
      probs = log_p[:, state.start_idx:].exp()
      final_window = (probs.argmax(-1) if self.noise_removal == 'greedy'
                      else sample_categorical(probs))
      state.xt = torch.cat([state.xt[:, :state.start_idx], final_window], 1)
      self._project_prefix(state.xt, state.prefix_tokens, state.prefix_lengths)
      state.done = True
    state.step_idx += 1
    state.nfe += 1
    return state


def get_sampler(config):
  s = config.sampler

  if (config.algo.name == 'law') != (s.predictor == 'law'):
    raise ValueError('algo=law requires sampler=law, exclusively')
  if (config.algo.name == 'law_carry_ablation') != (s.predictor == 'law-carry-ablation'):
    raise ValueError('algo=law_carry_ablation requires sampler=law_carry_ablation')
  temperature_schedule = None
  if s.predictor in (
      'ancestral', 'lddm', 'sfm', 'flm_euler', 'law', 'law-carry-ablation'):
    temperature_schedule = get_temperature_schedule(s.temperature_schedule)
  if s.predictor == 'law':
    if s.use_float64 or s.p_nucleus != 1.0:
      raise ValueError('LAW requires use_float64=false and p_nucleus=1.0')
    return LAWSampler(steps_policy=s.steps_policy, noise_removal=s.noise_removal,
                      temperature_schedule=temperature_schedule)
  if s.predictor == 'law-carry-ablation':
    return LAWCarryAblationSampler(
      carry_representation=config.algo.carry_representation,
      injection_mode=config.algo.injection_mode, use_float64=s.use_float64,
      steps_policy=s.steps_policy, noise_removal=s.noise_removal,
      temperature_schedule=temperature_schedule, decoder_mode=s.decoder_mode)

  if s.predictor == 'flm_euler':
    return FLMEulerSampler(use_float64=s.use_float64,
                           temperature_schedule=temperature_schedule)

  if s.predictor == 'sfm':
    return SFMSampler(noise_removal=s.noise_removal,
      velocity=s.velocity, use_float64=s.use_float64,
      slerp_float64=config.algo.slerp_precision=='float64',
      eps=config.algo.eps, temperature_schedule=temperature_schedule,
      p_nucleus=s.p_nucleus, top_k=s.top_k,
      top_k_velocity=s.top_k_velocity,
      invert_time_convention=config.algo.invert_time_convention)

  if s.predictor == 'ancestral':
    return AncestralSampler(
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      temperature_schedule=temperature_schedule)

  if s.predictor == 'lddm':
    return LDDMSampler(
      use_float64=s.use_float64,
      noise_removal=s.noise_removal,
      steps_policy=s.steps_policy,
      temperature_schedule=temperature_schedule)

  if s.predictor == 'ar':
    return ARSampler(
      use_float64=s.use_float64,
      kv_cache=s.use_kv_cache,
      greedy=s.greedy,
      early_stopping=s.early_stopping,
      temperature=s.temperature)

  if s.predictor == 'mixing':
    return MixingDUOSampler(
      use_float64=s.use_float64,
      steps_policy=s.steps_policy,
      noise_removal=s.noise_removal,
      mixing_option=s.MC)

  raise ValueError(f'Unknown sampler predictor: {s.predictor}')


# ----------------------------------------------------------
# MixingDUO sampler
# ----------------------------------------------------------
@dataclass(kw_only=True)
class MixingState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool
  context: object = None


@dataclass
class MixingContext:
  mixed: torch.Tensor
  kv_cache: bool = False


def compute_mixing(probs, q_xs, embedding_matrix, mixing_option):
  if mixing_option == 'probs':
    return torch.matmul(probs.to(embedding_matrix.dtype),
                        embedding_matrix)
  elif mixing_option == 'qxs':
    return torch.matmul(q_xs.to(embedding_matrix.dtype),
                        embedding_matrix)
  else:
    raise ValueError(f'Unknown MC option: {mixing_option}')


class MixingDUOSampler(Sampler):
  def __init__(self, use_float64=False, steps_policy='full',
               mixing_option='probs', noise_removal='ancestral'):
    self.use_float64 = use_float64
    self.steps_policy = steps_policy
    self.mixing_option = mixing_option
    self.noise_removal = noise_removal

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = model.prior_sample(num_samples, model.num_tokens)

    if prefix_tokens is not None:
      start_idx = int(prefix_lengths.min())
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
    else:
      start_idx = 0

    if num_steps is None:
      num_steps = model.config.sampler.steps

    if self.steps_policy == 'proportional':
      gen_len = model.num_tokens - start_idx
      num_steps = int(round(num_steps * (gen_len / model.num_tokens)))

    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    ones = torch.ones(xt.shape[0], 1, dtype=model.dtype,
                      device=model.device)

    return MixingState(
      xt=xt,
      timesteps=timesteps,
      ones=ones,
      start_idx=start_idx,
      step_idx=0,
      context=None,
      nfe=0,
      done=False,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones

    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)

    log_p_x0 = model.forward(xt=state.xt, sigma=sigma_t,
                             context=state.context)
    if self.use_float64:
      log_p_x0 = log_p_x0.to(torch.float64)

    probs = log_p_x0.exp()

    q_xs = compute_posterior(
      process=model.diffusion_type,
      x0_probs=probs,
      xt=state.xt,
      alpha_s=alpha_s.unsqueeze(-1),
      alpha_t=alpha_t.unsqueeze(-1),
      mask_index=getattr(model, 'mask_index', None),
      vocab_size=model.vocab_size,
      use_float64=False)

    if is_last_step:
      window = state.xt[:, state.start_idx:]
      log_p_x0_window = log_p_x0[:, state.start_idx:]

      if self.noise_removal == 'greedy':
        final_window = log_p_x0_window.argmax(dim=-1)
      else:
        posterior = compute_posterior(
          process=model.diffusion_type,
          x0_probs=log_p_x0_window.exp(),
          xt=window,
          alpha_s=alpha_s,
          alpha_t=alpha_t,
          mask_index=getattr(model, 'mask_index', None),
          vocab_size=model.vocab_size,
          use_float64=False)

        final_window = sample_categorical(posterior)

      if state.prefix_tokens is not None:
        out = torch.cat(
          [state.prefix_tokens[:, :state.start_idx], final_window],
          dim=1)
      else:
        out = final_window

      state.xt = out
      self._project_prefix(state.xt, state.prefix_tokens,
                           state.prefix_lengths)
      state.done = True

    else:
      if state.prefix_tokens is not None:
        P = state.prefix_tokens.shape[1]
        pos = torch.arange(P, device=state.xt.device)
        mask = (pos[None, :]
                < state.prefix_lengths[:, None]).unsqueeze(-1)

        prefix_3d = F.one_hot(
          state.prefix_tokens,
          num_classes=model.vocab_size).to(q_xs)

        q_xs = torch.cat(
          [prefix_3d[:, :state.start_idx],
           q_xs[:, state.start_idx:]], dim=1)
        probs = torch.cat(
          [prefix_3d[:, :state.start_idx],
           probs[:, state.start_idx:]], dim=1)

        q_xs[:, :P] = torch.where(mask, prefix_3d, q_xs[:, :P])
        probs[:, :P] = torch.where(mask, prefix_3d, probs[:, :P])

      mixed = compute_mixing(
        probs=probs,
        q_xs=q_xs,
        embedding_matrix=model.backbone.vocab_embed.matrix,
        mixing_option=self.mixing_option)

      state.context = MixingContext(mixed=mixed)
      state.xt = q_xs

    state.nfe += 1
    state.step_idx += 1

    return state


@dataclass(kw_only=True)
class LAWCarryAblationState(BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool
  context: object = None


class LAWCarryAblationSampler(Sampler):
  """Four carry/injection cells, with discrete posterior updates for addition."""

  def __init__(self, *, carry_representation, injection_mode,
               temperature_schedule,
               use_float64=False, steps_policy='full',
               noise_removal='ancestral', decoder_mode='raw'):
    if carry_representation not in ('expected_embedding', 'hidden_representation'):
      raise ValueError('Invalid carry_representation')
    if injection_mode not in ('replace', 'add'):
      raise ValueError('Invalid injection_mode')
    self.carry_representation = carry_representation
    self.injection_mode = injection_mode
    self.use_float64 = use_float64
    self.steps_policy = steps_policy
    self.noise_removal = noise_removal
    self.temperature_schedule = temperature_schedule
    # Decode from the prediction directly or through the diffusion posterior.
    assert decoder_mode in ('raw', 'posterior')
    self.decoder_mode = decoder_mode

  def init_state(self, model, num_samples, *, num_steps=None, eps=1e-5,
                 prefix_tokens=None, prefix_lengths=None):
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = model.prior_sample(num_samples, model.num_tokens)
    if prefix_tokens is not None:
      start_idx = int(prefix_lengths.min())
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
    else:
      start_idx = 0
    if num_steps is None:
      num_steps = model.config.sampler.steps
    if self.steps_policy == 'proportional':
      gen_len = model.num_tokens - start_idx
      num_steps = max(1, round(num_steps * gen_len / model.num_tokens))
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    ones = torch.ones(xt.shape[0], 1, dtype=model.dtype, device=model.device)
    init_context = CarryContext(
      carry=None, carry_representation=self.carry_representation,
      injection_mode=self.injection_mode, looped=False)
    return LAWCarryAblationState(
      xt=xt, timesteps=timesteps, ones=ones, start_idx=start_idx,
      step_idx=0, context=init_context, nfe=0, done=False,
      prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths)

  def _valid_tokens(self, state):
    # Generated positions are 1; prompt positions are 0.
    if state.prefix_tokens is None:
      return None
    pos = torch.arange(state.xt.shape[1], device=state.xt.device)
    return (pos[None, :] >= state.prefix_lengths[:, None]).long()

  def _prompt_x0(self, state):
    # Token ids used to restore prompt embeddings in the carry.
    if state.prefix_tokens is None:
      return state.xt
    P = state.prefix_tokens.shape[1]
    L = state.xt.shape[1]
    if P >= L:
      return state.prefix_tokens[:, :L]
    return torch.cat([state.prefix_tokens, state.xt[:, P:]], dim=1)

  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    temperature = self.temperature_schedule(
      sampling_time(state.step_idx, num_steps))
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones
    _, alpha_t = model.noise(t)
    alpha_s = torch.ones_like(alpha_t) if is_last_step else model.noise(s)[1]
    # forward_carry expects the model-processed sigma.
    sigma_t = model._process_sigma(model._sigma_from_alphat(alpha_t))

    log_p, state.context = model.backbone.forward_carry(
      xt=state.xt, sigma=sigma_t, in_context=state.context,
      carry_representation=self.carry_representation,
      injection_mode=self.injection_mode,
      x0=self._prompt_x0(state), valid_tokens=self._valid_tokens(state),
      detach=False, return_carry=not is_last_step,
      temperature=temperature, carry_temperature=temperature)

    if is_last_step:
      # Preserve log_p's dtype for probabilities and categorical noise.
      probs = log_p[:, state.start_idx:].exp()
      greedy = (self.noise_removal == 'greedy')
      if self.decoder_mode == 'raw':
        final_window = probs.argmax(dim=-1) if greedy else sample_categorical(probs)
      else:
        # Decode through one posterior jump from t=1 to t=0.
        if self.use_float64:
          probs = probs.to(torch.float64)
        z_t = state.xt[:, state.start_idx:]
        alpha_t1 = model.noise(state.ones)[1]          # alpha at t=1
        alpha_s0 = torch.ones_like(alpha_t1)           # alpha at t=0
        mask_index = _model_mask_index(model)
        if greedy:
          post = compute_posterior(
            model.diffusion_type, probs, z_t, alpha_s0, alpha_t1,
            mask_index=mask_index, vocab_size=model.vocab_size,
            use_float64=self.use_float64)
          final_window = post.argmax(dim=-1)
        else:
          final_window = sample_posterior(
            model.diffusion_type, probs, z_t, alpha_s0, alpha_t1,
            mask_index=mask_index, vocab_size=model.vocab_size,
            noise_removal_step=True, use_float64=self.use_float64)
      if state.prefix_tokens is not None:
        state.xt = torch.cat(
          [state.prefix_tokens[:, :state.start_idx], final_window], dim=1)
      else:
        state.xt = final_window
      self._project_prefix(state.xt, state.prefix_tokens, state.prefix_lengths)
      state.done = True
    elif self.injection_mode == 'add':
      # Addition follows a discrete token-space trajectory.
      probs = log_p.exp()
      if self.use_float64:
        probs = probs.to(torch.float64)
      state.xt = sample_posterior(
        process=model.diffusion_type, x0_probs=probs, xt=state.xt,
        alpha_s=alpha_s, alpha_t=alpha_t, mask_index=None,
        vocab_size=model.vocab_size)
      self._project_prefix(state.xt, state.prefix_tokens, state.prefix_lengths)

    state.nfe += 1
    state.step_idx += 1
    return state
