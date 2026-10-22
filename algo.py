import torch
import collections
import math

import models
import samplers
import trainer_base
import utils
import torch.nn.functional as F


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del xt, sigma, context
    model_output[:, :, self.mask_index] = self.neg_infinity
    return torch.log_softmax(model_output, dim=-1)

  def _process_sigma(self, sigma, context=None):
    return None

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step, train_mode,
          valid_tokens=None):
    del train_mode, current_accumulation_step, valid_tokens, context

    x0 = input_tokens
    output = self.forward(xt=x0, sigma=None)
    per_token_nll =  - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]
    return per_token_nll, None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.post_process_mode = config.algo.post_process_mode
    self._validate_configuration()

  def _process_model_output_log_probs(
      self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity

    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def _process_model_output_logits(self, model_output, xt, sigma):
    del sigma
    model_output[..., self.mask_index] = self.neg_infinity
    xt_unsq = xt.unsqueeze(-1)
    unmasked = (xt != self.mask_index)[..., None]
    deterministic = model_output.new_full(
      model_output.shape, self.neg_infinity)
    deterministic.scatter_(-1, xt_unsq, 0.0)
    # For masked positions, return raw logits (unnormalized).
    return torch.where(unmasked, deterministic, model_output)

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.post_process_mode not in {
      'log_probs', 'logits'}:
      raise ValueError(self.post_process_mode)

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del context
    if self.post_process_mode == 'log_probs':
      return self._process_model_output_log_probs(
        model_output, xt, sigma)
    elif self.post_process_mode == 'logits':
      return self._process_model_output_logits(
        model_output, xt, sigma)
    else:
      raise ValueError(self.post_process_mode)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del context, xt, train_mode
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    if low_var:
      loss_coefficient = -1
    else:
      loss_coefficient = dalpha_t / (1 - alpha_t)
    return loss_coefficient * log_p_theta

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    if self.post_process_mode == 'logits':
      raise ValueError(
        'post_process_mode=logits is sampling-only. '
        'Use log_probs mode for training or evaluation.')
    return super().nll(
      x0=x0,
      output_tokens=output_tokens,
      context=context,
      current_accumulation_step=current_accumulation_step,
      train_mode=train_mode,
      valid_tokens=valid_tokens)


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(
      self, model_output, xt, sigma, context=None):
    del xt, sigma, context
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    del context, train_mode
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    if low_var:
      term1 = 0
    else:
      term1 = self.vocab_size * (1 / xbar_xt
                                 - 1 / xbar_theta_xt)

    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss


@torch.compile
def duo_nll_distributional(log_x_theta, xt, x0, alpha_t, dalpha_t, K):
  """Per-token DUO loss for distributional (3D) xt.

  x_bar takes only two values per (B, L): x_bar_one at index x0, x_bar_zero
  elsewhere. Every V-sum involving x_bar collapses to a scalar plus a single
  gather at x0, so we never materialize (B, L, V) tensors for x_bar, S_j, or
  f_i. Only x_bar_theta and log(x_bar_theta) stay full-sized.
  """
  if alpha_t.ndim == 2:
    alpha_t = alpha_t.unsqueeze(-1)
  if dalpha_t.ndim == 2:
    dalpha_t = dalpha_t.unsqueeze(-1)

  xbar_one = (K * alpha_t + (1 - alpha_t)).squeeze(-1)
  xbar_zero = (1 - alpha_t).squeeze(-1)
  log_xbar_one = xbar_one.log()
  log_xbar_zero = xbar_zero.log()

  xbar_theta = K * alpha_t * log_x_theta.exp() + (1 - alpha_t)
  log_xbar_theta = xbar_theta.log()

  x0_idx = x0.unsqueeze(-1)
  xt_at_x0 = xt.gather(-1, x0_idx).squeeze(-1)
  log_xbar_theta_at_x0 = log_xbar_theta.gather(-1, x0_idx).squeeze(-1)

  inv_xbar_theta_sum = (xt / xbar_theta).sum(dim=-1)
  log_xbar_theta_sum = log_xbar_theta.sum(dim=-1)
  xt_log_xbar_theta_sum = (xt * log_xbar_theta).sum(dim=-1)

  A = xt_at_x0 / xbar_one + (1 - xt_at_x0) / xbar_zero
  sum_x_log_x = (xbar_one * log_xbar_one
                 + (K - 1) * xbar_zero * log_xbar_zero)
  sum_x_log_xtheta = ((xbar_one - xbar_zero) * log_xbar_theta_at_x0
                     + xbar_zero * log_xbar_theta_sum)
  sum_w_log_xtheta = (xt_at_x0 * log_xbar_theta_at_x0
                     * (1 / xbar_one - 1 / xbar_zero)
                     + xt_log_xbar_theta_sum / xbar_zero)
  sum_w_log_xbar = ((xt_at_x0 / xbar_one) * log_xbar_one
                   + ((1 - xt_at_x0) / xbar_zero) * log_xbar_zero)

  C = (K * sum_w_log_xtheta
       + (sum_x_log_x - sum_x_log_xtheta) * A
       - K * sum_w_log_xbar)
  coeff = dalpha_t.squeeze(-1) / (K * alpha_t.squeeze(-1))
  return coeff * (K * A - K * inv_xbar_theta_sum - C)


class MixingDUO(DUO_BASE):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    # step_size_policy computes dt = 1 / dt_T, so dt_T must be strictly
    # positive for the multi-step mixing unroll. dt_T defaults to algo.T
    # (duo-mixing.yaml), so set algo.T > 0 OR set algo.dt_T > 0 explicitly.
    # algo.T itself MAY be 0 (continuous-time training) as long as dt_T > 0.
    assert self.config.algo.dt_T > 0, (
      f'MixingDUO requires algo.dt_T > 0 (got dt_T={self.config.algo.dt_T}); '
      'step_size_policy computes dt = 1/dt_T. dt_T defaults to algo.T, so either '
      'set algo.T>0 or set algo.dt_T>0 (e.g. algo.T=0 algo.dt_T=128 for '
      'continuous-time training with a 1/128 unroll step).')

  def on_train_batch_start(self, batch, batch_idx):
    del batch, batch_idx
    if not self.config.algo.reset_during_curriculum:
      return
    if self.config.algo.mixing_depth_policy != 'curriculum':
      return
    boundaries = self.config.algo.curriculum_steps
    if boundaries is None:
      return
    if self.global_step in list(boundaries):
      self.trainer.strategy.setup_optimizers(self.trainer)

  def _diffusion_params_at(self, t):
    dalpha, alpha = self.noise(t)
    alpha = alpha.unsqueeze(-1)
    dalpha = dalpha.unsqueeze(-1)
    sigma = self._sigma_from_alphat(alpha)
    return alpha, dalpha, sigma

  def _project_prefix_embedding(self, mixed, x0, valid_tokens):
    if valid_tokens is None:
      return mixed
    clean = F.embedding(x0, self.backbone.vocab_embed.matrix)
    mask = valid_tokens.unsqueeze(-1).bool()
    return torch.where(mask, mixed, clean)

  def _step_transition(self, probs, xt, x0, t, dt, alpha_t,
                       valid_tokens, step_idx):
    t_next = t - dt
    alpha_s, dalpha_s, sigma_s = self._diffusion_params_at(t_next)

    q_xs = samplers.compute_posterior(
      process=self.diffusion_type,
      x0_probs=probs, xt=xt,
      alpha_s=alpha_s, alpha_t=alpha_t,
      mask_index=None,
      vocab_size=self.vocab_size,
      use_float64=False)

    mixed = samplers.compute_mixing(
      probs=probs, q_xs=q_xs,
      embedding_matrix=self.backbone.vocab_embed.matrix,
      mixing_option=self.config.algo.MC)
    mixed = self._project_prefix_embedding(mixed, x0, valid_tokens)

    if (self.config.algo.mixing_option == 'every-k'
        and self.config.algo.detach_mixing
        and (step_idx % self.config.algo.k == 0)):
      # FULL detach of this transition: cut BOTH cross-step gradient paths.
      #  - `mixed`: the carried soft-embedding fed to the next forward.
      #  - `q_xs` : becomes the next step's `xt` and is used directly in the
      #            next step's NLL (`duo_nll_distributional`). Detaching
      #            only `mixed` would leave a gradient path through q_xs in the
      #            loss, so the unroll would not be truly severed.
      mixed = mixed.detach()
      q_xs = q_xs.detach()

    return (q_xs, samplers.MixingContext(mixed=mixed),
            t_next, alpha_s, dalpha_s, sigma_s)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False, context=None, train_mode=False):
    if xt.ndim == 2:
      return super().nll_per_token(
        log_x_theta, xt, x0, alpha_t, dalpha_t,
        low_var, context, train_mode)
    return duo_nll_distributional(
      log_x_theta, xt, x0, alpha_t, dalpha_t, self.vocab_size)

  def _compute_step_loss(self, step_xt, log_x_theta, x0,
                         step_alpha_t, step_dalpha_t):
    if self.config.algo.loss == 'CE':
      return F.cross_entropy(log_x_theta.transpose(1, 2), x0,
                             reduction='none')
    if self.config.algo.loss == 'nll':
      return self.nll_per_token(
        log_x_theta=log_x_theta,
        xt=step_xt, x0=x0,
        alpha_t=step_alpha_t,
        dalpha_t=step_dalpha_t)
    raise ValueError(f'Unknown loss type: {self.config.algo.loss}')

  def _update_total_loss(self, step_idx, num_steps, total_loss,
                         xt, log_x_theta, x0, alpha_t, dalpha_t,
                         valid_tokens):
    option = self.config.algo.mixing_option
    is_last = (step_idx == num_steps - 1)

    if option == 'last':
      contributes = is_last
    elif option == 'every-k':
      contributes = is_last or (step_idx % self.config.algo.k == 0)
    else:
      raise ValueError(f'Unknown mixing_option: {option}')

    # Skip computing the loss if not used
    if not contributes:
      return total_loss

    step_loss = self._compute_step_loss(
      step_xt=xt, log_x_theta=log_x_theta, x0=x0,
      step_alpha_t=alpha_t, step_dalpha_t=dalpha_t)
    if valid_tokens is not None:
      step_loss = step_loss * valid_tokens

    return total_loss + step_loss

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens, context, train_mode
    x0 = input_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    num_steps = depth_policy(self, self.config)
    dt = step_size_policy(self.config, self.config.algo.dt_T, num_steps)
    # Rescale t so that after (num_steps-1) transitions of size dt the
    # trajectory stays strictly positive. _sample_t already floors t at
    # sampling_eps, so we just lift the lower end by (num_steps-1)*dt.
    t_min = (num_steps - 1) * dt
    assert t_min < 1.0, (
      f't_min={t_min} >= 1; reduce num_steps or step size')
    t = t_min + t * (1.0 - t_min)
    # Initial (rescaled) noise level for this trajectory; returned so
    # trainer_base._loss can feed it to the adaptive noise-schedule hook.
    t_initial = t

    alpha_t, dalpha_t, sigma_t = self._diffusion_params_at(t)
    xt = self.q_xt(x0, alpha_t, use_pure_noise=False,
                   valid_tokens=valid_tokens)
    total_per_token_loss = 0
    mixing_context = None

    for step_idx in range(num_steps):
      log_x_theta = self.forward(xt=xt, sigma=sigma_t,
                                 context=mixing_context)

      total_per_token_loss = self._update_total_loss(
        step_idx=step_idx, num_steps=num_steps,
        total_loss=total_per_token_loss,
        xt=xt, log_x_theta=log_x_theta, x0=x0,
        alpha_t=alpha_t, dalpha_t=dalpha_t,
        valid_tokens=valid_tokens)

      if step_idx < num_steps - 1:
        (xt, mixing_context, t, alpha_t,
         dalpha_t, sigma_t) = self._step_transition(
          probs=log_x_theta.exp(), xt=xt, x0=x0,
          t=t, dt=dt, alpha_t=alpha_t,
          valid_tokens=valid_tokens, step_idx=step_idx)

    if self.config.algo.mixing_option == 'every-k':
      k = self.config.algo.k
      n_contrib = (num_steps - 1) // k + 1
      if (num_steps - 1) % k != 0:
        n_contrib += 1
      total_per_token_loss = total_per_token_loss / n_contrib

    return total_per_token_loss, t_initial


def step_size_policy(config, dt_T, steps):
  # dt_T is the step-size denominator (algo.dt_T), DECOUPLED from algo.T (the
  # number of discrete noise levels used in _sample_t). dt = n / dt_T.
  if config.algo.step_strategy == 'uniform_random_size':
    # pick n in a range in for example U[1, 64], set dt to be n / dt_T
    n = torch.randint(1, config.algo.step_upperbound + 1, (1,)).item()
    dt = n / dt_T
  elif config.algo.step_strategy == 'nonuniform_random' :
    # pick num_steps random steps in the range U[1, dt_T], order them, step from one to the other TODO test and verify this
    step_size_probs = torch.ones(dt_T) / dt_T
    steps = torch.multinomial(step_size_probs, steps, replacements=False)
    steps.sort()
  elif config.algo.step_strategy == 'random_size_decrease':
    # build a probability distribution for the step size in the interval [1, B] where each probability is
    # 1 / i, with i the index in the sequence. i.e. small steps are more likely than large steps
    step_size_probs = 1.0 / torch.arange(1, config.algo.step_upperbound + 1, dtype=torch.float32)
    n = torch.multinomial(step_size_probs, 1).item() + 1
    dt = n / dt_T
  elif config.algo.step_strategy == 'uniform' :
    dt = 1 / dt_T
  else :
    raise ValueError(f"Unknown step_strategy: {config.algo.step_strategy}")
  return dt


def depth_policy(self, config):
  if config.algo.mixing_depth_policy == 'fixed' :
    num_steps = config.algo.mixing_depth
  elif config.algo.mixing_depth_policy == 'random' :
    def parse(string):
      return torch.tensor([float(x) for x in string])

    depth_probabilities = parse(config.algo.mixing_depth_probabilities)
    assert config.algo.mixing_depth == len(depth_probabilities), f"You must give a probability for each number of step from 1 to the K, where K is the maximum depth"
    num_steps = torch.multinomial(depth_probabilities, 1).item() + 1
  elif config.algo.mixing_depth_policy == 'curriculum':
    num_steps = get_current_curriculum_depth(self)
  return num_steps


def get_current_curriculum_depth(self):
  global_step = torch.tensor(self.global_step)
  depth = torch.bucketize(global_step, torch.tensor(self.config.algo.curriculum_steps), right=True).item()
  return max(1, depth)


# ──────────────────────────────────────────────────────────────────────
# SFM — hyperspherical flow matching. Transports random points on the
# sphere to clean embeddings via rotations. 
# ──────────────────────────────────────────────────────────────────────
class SFM(trainer_base.Diffusion):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.eps = config.algo.eps
    self.renormalize_weights = config.algo.renormalize_weights
    self.invert_time_convention = config.algo.invert_time_convention
    self._validate_configuration()

  def _validate_configuration(self):
    if self.invert_time_convention and self.config.noise.adaptive:
      raise ValueError('Adaptive noise schedule requires '
                       'invert_time_convention=false '
                       '(MDLM-like convention).')
    backbone_type = self.config.model.type
    if backbone_type == 'sphere-arch' and not self.renormalize_weights:
      raise ValueError('Backbone sphere-arch requires '
                       'algo.renormalize_weights=True.')

  def _process_model_output(self, model_output, xt, sigma,
                            context=None):
    return model_output.float().log_softmax(-1)

  def _sample_prior(self, e_clean):
    e_noisy = torch.randn_like(e_clean)
    return utils.sphere_normalize(e_noisy)

  def q_xt(self, x, alpha_t, use_pure_noise, valid_tokens=None):
    e_clean = self.backbone.get_sphere_embeddings(x)  # [B, L, d]
    e_noisy = self._sample_prior(e_clean)

    if use_pure_noise:
      x_t = e_noisy
    else:
      slerp_t = alpha_t if self.invert_time_convention else 1 - alpha_t
      x_t = self._slerp(e_clean, e_noisy, slerp_t)

    if valid_tokens is not None:
      x_t = torch.where(valid_tokens.bool().unsqueeze(-1),
                        x_t, e_clean)
    return x_t

  def optimizer_step(self, *args, **kwargs):
    out = super().optimizer_step(*args, **kwargs)
    if self.renormalize_weights:
      self.backbone.renormalize_weights()
    return out

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False, context=None, train_mode=False):
    del xt, alpha_t, dalpha_t, low_var, context

    ce_loss = -log_x_theta.gather(
      -1, x0.unsqueeze(-1)).squeeze(-1)

    return ce_loss

  def _slerp(self, clean, noisy, alpha_t):
    # alpha_t = 0 -> clean, alpha_t = 1 -> noisy
    orig_dtype = None
    if self.config.algo.slerp_precision == 'float64':
      orig_dtype = clean.dtype
      alpha_t = alpha_t.to(torch.float64)
      clean = clean.to(torch.float64)
      noisy = noisy.to(torch.float64)
    out = utils.slerp(
      clean=clean,
      noisy=noisy,
      alpha_t=alpha_t,
      eps=self.eps)

    if orig_dtype is not None:
      out = out.to(orig_dtype)
    return out

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    use_pure_noise = self._use_pure_noise(
      train_mode=train_mode, context=context)
    if use_pure_noise:
      t = torch.ones_like(t)

    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)

    xt = self.q_xt(
      x0, alpha_t, use_pure_noise=use_pure_noise,
      valid_tokens=valid_tokens)

    sigma = self._sigma_from_alphat(alpha_t)
    log_x_theta = self.forward(
      x0=x0, xt=xt, sigma=sigma, context=context)
    utils.print_nans(log_x_theta, 'model_output')

    loss = self.nll_per_token(
      log_x_theta=log_x_theta, xt=xt, x0=x0,
      alpha_t=alpha_t, dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var',
      context=context, train_mode=train_mode)

    return loss, t


class LDDMHiddenCarryMixin:
  """LDDM hidden-state self-conditioning with loss on the final forward."""

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    use_pure_noise = self._use_pure_noise(
      train_mode=train_mode, context=context)
    if use_pure_noise:
      t = torch.ones_like(t)

    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    assert dalpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, use_pure_noise=use_pure_noise,
                   valid_tokens=valid_tokens)

    hidden_state = None
    if torch.rand(1, device=x0.device) < self.config.algo.self_cond_rate:
      # Keep DiT's bf16 active without caching detached weights for the final pass.
      with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32,
                                               cache_enabled=False):
        _, aux = self.forward(
          x0=x0, xt=xt, sigma=sigma,
          context=samplers.CarryContext(
            carry=None, carry_representation='hidden_representation',
            injection_mode='add'),
          return_aux=True)
      hidden_state = aux['latent']
    log_x_theta = self.forward(
      x0=x0, xt=xt, sigma=sigma,
      context=samplers.CarryContext(
        carry=hidden_state, carry_representation='hidden_representation',
        injection_mode='add'))

    utils.print_nans(log_x_theta, 'model_output')
    loss = self.nll_per_token(
      log_x_theta=log_x_theta, xt=xt, x0=x0,
      alpha_t=alpha_t, dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var',
      context=context, train_mode=train_mode)
    return loss, t


class LDDM_M(LDDMHiddenCarryMixin, MDLM):
  """Masked diffusion with hidden-state carry."""


class LDDM_U(LDDMHiddenCarryMixin, DUO_BASE):
  """Uniform diffusion with hidden-state carry."""


class LAW_BASE(DUO_BASE):
  def __init__(self, config, tokenizer):
    self._pass_count_probabilities = torch.as_tensor(
      config.algo.pass_count_probabilities, dtype=torch.float64)
    self._pass_count_probabilities /= self._pass_count_probabilities.sum()
    self._validate_law_config(config, len(self._pass_count_probabilities))
    self._carry_regularization = models.dit_loop.CarryRegularization(
      gaussian_std=config.algo.carry_regularization.gaussian_std,
      rotation_fraction=config.algo.carry_regularization.rotation_fraction)
    self._pass_count_generator = torch.Generator(device='cpu')
    self._pass_count_generator.manual_seed(config.seed)
    super().__init__(config, tokenizer)
    self.loss_type = config.algo.loss

  def on_save_checkpoint(self, checkpoint):
    super().on_save_checkpoint(checkpoint)
    checkpoint['pass_count_generator_state'] = self._pass_count_generator.get_state()

  def on_load_checkpoint(self, checkpoint):
    super().on_load_checkpoint(checkpoint)
    self._pass_count_generator.set_state(checkpoint['pass_count_generator_state'])

  def _sample_pass_count(self):
    return torch.multinomial(self._pass_count_probabilities,
                             1, generator=self._pass_count_generator).item() + 1

  def _pass_schedule(self, t, pass_count):
    dt = self.config.algo.dt if self.config.algo.advance_time_between_passes else 0
    offsets = torch.arange(pass_count, device=t.device, dtype=torch.float32) * dt
    t_min = offsets[-1]
    lower = torch.nextafter(t_min, t_min.new_tensor(1.0))
    t_start = (t_min + t * (1 - t_min)).clamp_min(lower)
    times = t_start.unsqueeze(0) - offsets.unsqueeze(-1)
    dalpha, alpha = self.noise(times)
    sigmas = self._sigma_from_alphat(alpha.unsqueeze(-1))
    sigmas = self._process_sigma(sigmas.flatten(0, 1)).reshape(pass_count, -1)
    return sigmas, alpha[0].unsqueeze(-1), dalpha[0].unsqueeze(-1), t_start

  def _carry_temperatures(self, batch_size, pass_count, device, train_mode):
    if not train_mode or not self.config.algo.carry_regularization.temperature.enabled:
      return 1.0
    draw_count = (pass_count
                  if self.config.algo.carry_regularization.temperature.resample_per_pass
                  else 1)
    draws = torch.rand(draw_count, 3, batch_size, device=device)
    low = (1 / self.config.algo.carry_regularization.temperature.range_scale
           + (1 - 1 / self.config.algo.carry_regularization.temperature.range_scale)
           * draws[:, 0])
    high = (1 + (self.config.algo.carry_regularization.temperature.range_scale - 1)
            * draws[:, 1])
    temperatures = torch.where(draws[:, 2] < 0.5, low, high)
    return temperatures.view(draw_count, batch_size, 1, 1).expand(pass_count, batch_size,
                                                                  1, 1)

  def _pass_loss(self, log_p, xt, x0, alpha, dalpha, valid_tokens):
    if self.config.algo.loss == 'ce':
      loss = -log_p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
      loss = self.nll_per_token(log_x_theta=log_p, xt=xt, x0=x0, alpha_t=alpha,
                                dalpha_t=dalpha)
    return loss if valid_tokens is None else loss * valid_tokens

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step=None, train_mode=False, valid_tokens=None):
    del output_tokens, context
    x0 = input_tokens
    pass_count = self._sample_pass_count() if train_mode else 1

    if train_mode and self._trainer is not None:
      self.log('trainer/pass_count', float(pass_count), on_step=True, on_epoch=False)

    t = self._sample_t(x0.shape[0], current_accumulation_step)
    sigmas, alpha, dalpha, t_start = self._pass_schedule(t, pass_count)
    xt = self.q_xt(x0, alpha, use_pure_noise=False, valid_tokens=valid_tokens)

    temperatures = self._carry_temperatures(x0.shape[0], pass_count, x0.device,
                                            train_mode)
    trajectory, forward = self._start_trajectory(xt, valid_tokens, train_mode)
    total_loss = 0
    contributions = 0
    # LAW computes carry updates and losses in float32. DiT uses bf16 internally.
    # Disable the cache so a no-grad warm-up cannot cache detached bf16 weights.
    with torch.amp.autocast('cuda', dtype=torch.float32, cache_enabled=False):
      for i in range(pass_count):
        is_last = i == pass_count - 1

        contributes = self.config.algo.pass_objective == 'mean' or is_last
        prediction_grad = contributes or not self.config.algo.stop_gradient_between_passes

        log_p, trajectory = forward(
          trajectory, sigmas[i], return_carry=not is_last,
          carry_temperature=temperatures[i] if torch.is_tensor(temperatures)
                            else temperatures,
          prediction_grad=prediction_grad)
        if contributes:
          total_loss = total_loss + self._pass_loss(log_p, xt, x0, alpha, dalpha,
                                                    valid_tokens)
          contributions += 1
    return total_loss / contributions, t_start

  @staticmethod
  def _validate_law_config(config, max_passes):
    if config.model.type != 'dit_loop':
      raise ValueError('LAW requires model.type=dit_loop')
    if config.algo.diffusion_type != 'uniform' or config.algo.T != 0:
      raise ValueError('LAW requires continuous-time uniform diffusion')
    if config.algo.loss not in ('elbo', 'ce'):
      raise ValueError('LAW loss must be elbo or ce')
    if config.algo.pass_objective not in ('mean', 'last_only'):
      raise ValueError('LAW pass_objective must be mean or last_only')
    if type(config.algo.stop_gradient_between_passes) is not bool:
      raise ValueError('stop_gradient_between_passes must be boolean')
    if type(config.algo.advance_time_between_passes) is not bool:
      raise ValueError('advance_time_between_passes must be boolean')
    if not math.isfinite(config.algo.dt) or config.algo.dt <= 0:
      raise ValueError('LAW dt must be finite and positive')
    if config.algo.advance_time_between_passes:
      # Reject schedules whose float32 lower bound rounds to 1.
      t_min = torch.tensor(max_passes - 1, dtype=torch.float32) * config.algo.dt
      if t_min >= 1:
        raise ValueError('LAW requires (K - 1) * dt < 1, including float32 rounding')
    if (type(config.algo.carry_regularization.temperature.enabled) is not bool
        or type(config.algo.carry_regularization.temperature.resample_per_pass) is not bool):
      raise ValueError('temperature enabled and resample_per_pass must be boolean')
    if (not math.isfinite(config.algo.carry_regularization.temperature.range_scale)
        or config.algo.carry_regularization.temperature.range_scale < 1):
      raise ValueError('temperature.range_scale must be finite and >= 1')
    if (not math.isfinite(config.algo.carry_regularization.gaussian_std)
        or config.algo.carry_regularization.gaussian_std < 0):
      raise ValueError('gaussian_std must be finite and non-negative')
    if (not math.isfinite(config.algo.carry_regularization.rotation_fraction)
        or not 0 <= config.algo.carry_regularization.rotation_fraction <= 1):
      raise ValueError('rotation_fraction must be finite and in [0, 1]')

  def on_validation_epoch_end(self):
    self.log(f'val/{self.config.algo.loss}', self.metrics.valid_nlls.nll.compute(),
             on_step=False, on_epoch=True, sync_dist=True)
    super().on_validation_epoch_end()


class LAW(LAW_BASE):
  def _start_trajectory(self, xt, valid_tokens, train_mode):
    trajectory = models.dit_loop.LAWTrajectory(
      xt, valid_tokens, self.config.algo.stop_gradient_between_passes,
      self._carry_regularization if train_mode else None)
    return trajectory, self.backbone.forward_law


class LAWCarryAblation(LAW_BASE):
  def __init__(self, config, tokenizer):
    if config.algo.carry_representation not in ('expected_embedding', 'hidden_representation'):
      raise ValueError('carry_representation must be expected_embedding '
                       'or hidden_representation')
    if config.algo.injection_mode not in ('replace', 'add'):
      raise ValueError('injection_mode must be replace or add')
    if (config.algo.carry_representation == 'hidden_representation'
        and config.algo.carry_regularization.temperature.enabled):
      raise ValueError('temperature regularization requires expected_embedding')
    super().__init__(config, tokenizer)

  def _start_trajectory(self, xt, valid_tokens, train_mode):
    trajectory = models.dit_loop.LAWCarryAblationTrajectory(
      xt, valid_tokens, self.config.algo.stop_gradient_between_passes,
      self._carry_regularization if train_mode else None,
      carry_representation=self.config.algo.carry_representation,
      injection_mode=self.config.algo.injection_mode)
    return trajectory, self.backbone.forward_law_carry_ablation


# ── FLM / CANDI algorithms ──
# Re-exported so main.py's algo.CANDI / algo.FLM resolve.
from algo_flm_candi import FLMBase, FLM, CANDI  # noqa: E402,F401
