"""Noise schedule definitions"""

import abc
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


class NoiseSchedule(torch.nn.Module, abc.ABC):
  def forward(self, t):
    return self.alpha_prime_t(t), self.alpha_t(t)

  @abc.abstractmethod
  def alpha_t(self, t):
    pass

  @abc.abstractmethod
  def alpha_prime_t(self, t):
    pass

  def record_time_loss_pair(self, t, loss, step):
    """For adaptive noise scheduling.

    Args:
      t: (B,) timesteps.
      loss: (B,) per-sample losses.
      step: global training step index (optimizer steps).
    """
    pass


class CosineSquared(NoiseSchedule):
  """alpha_t = (1 - cos^2(pi/2 * (1-t)))"""
  def __init__(self, eps):
    super().__init__()
    self.eps = eps
    self.half_pi = torch.pi / 2

  def alpha_t(self, t):
    angle = self.half_pi * (1 - t)
    base_alpha = torch.sin(angle) ** 2
    return self.eps + (1 - self.eps) * base_alpha

  def alpha_prime_t(self, t):
    angle = self.half_pi * (1 - t)
    return (-(1 - self.eps) * 2 * torch.sin(angle)
            * torch.cos(angle) * self.half_pi)


class LogLinear(NoiseSchedule):
  """alpha_t = 1 - t"""
  def __init__(self, eps):
    super().__init__()
    self.eps = eps

  def alpha_t(self, t):
    return self.eps + (1 - self.eps) * (1 - t)

  def alpha_prime_t(self, t):
    return -(1 - self.eps) * torch.ones_like(t)


class TruncatedScheduleWrapper(NoiseSchedule):
  """Rescale a base schedule to be in [alpha_min, alpha_max]."""
  def __init__(self, base_schedule, alpha_min, alpha_max, eps):
    super().__init__()
    if not 0 <= alpha_min < alpha_max <= 1:
      raise ValueError(
        f'Expected 0 <= alpha_min < alpha_max <= 1, got '
        f'alpha_min={alpha_min}, alpha_max={alpha_max}')
    self.base_schedule = base_schedule
    self.eps = eps
    base_max = base_schedule.alpha_t(torch.tensor(0.0)).item()
    base_min = base_schedule.alpha_t(torch.tensor(1.0)).item()
    self.scale = (alpha_max - alpha_min) / (base_max - base_min)
    self.offset = alpha_min - self.scale * base_min

  def alpha_t(self, t):
    return (self.scale * self.base_schedule.alpha_t(t)
            + self.offset).clamp(min=self.eps)

  def alpha_prime_t(self, t):
    return self.scale * self.base_schedule.alpha_prime_t(t)


class AdaptiveSchedule(NoiseSchedule):
  """
  Collects (t, loss) pairs during training, periodically fits
  a spline to the loss profile, and remaps time to concentrate
  sampling where |dL/dt| is largest.
  """

  def __init__(self, base_schedule, buffer_size,
               refit_every, n_grid, n_knots, spline_degree,
               ridge_alpha, uniform_mix, max_steps, warmup_steps,
               ema, plot_profile=False,
               plot_dir='adaptive_noise_plots'):
    super().__init__()
    self.base_schedule = base_schedule
    self.buffer_size = buffer_size
    self.refit_every = refit_every
    self.n_knots = n_knots
    self.spline_degree = spline_degree
    self.ridge_alpha = ridge_alpha
    self.uniform_mix = uniform_mix
    self.warmup_steps = warmup_steps
    self.ema = ema
    self.plot_profile = plot_profile
    self.plot_dir = plot_dir
    self._step_fmt = f'0{len(str(max_steps))}d'

    # Use buffers to be saved automatically in checkpoints
    self.register_buffer('t_buf', 
      torch.zeros(buffer_size, dtype=torch.float64))
    self.register_buffer('loss_buf', 
      torch.zeros(buffer_size, dtype=torch.float64))
    self.register_buffer('buf_pos', 
      torch.tensor(0, dtype=torch.long))
    self.register_buffer('alpha_vals', 
      torch.zeros(n_grid, dtype=torch.float64))
    self.register_buffer('has_schedule', torch.tensor(False))
    self.register_buffer('ema_alpha_vals',
      torch.zeros(n_grid, dtype=torch.float64))
    self.register_buffer('refit_count',
      torch.tensor(0, dtype=torch.long))

    self._grid = np.linspace(0, 1, n_grid)
    self._alpha_spline = None
    self._dalpha_spline = None

  def record_time_loss_pair(self, t, loss, step):
    if step < self.warmup_steps:
      return
    n = len(t)
    pos = self.buf_pos.item()
    end = pos + n
    t_val = t.detach().to(self.t_buf.dtype)
    l_val = loss.detach().to(self.loss_buf.dtype)
    if end <= self.buffer_size:
      self.t_buf[pos:end] = t_val
      self.loss_buf[pos:end] = l_val
    else:
      # Wrap around: fill end of buffer, spill remainder to start
      first = self.buffer_size - pos
      self.t_buf[pos:] = t_val[:first]
      self.loss_buf[pos:] = l_val[:first]
      self.t_buf[:n - first] = t_val[first:]
      self.loss_buf[:n - first] = l_val[first:]
    self.buf_pos.fill_(end % self.buffer_size)
    # Refit once the buffer has been filled at least once
    buffer_full = end >= self.buffer_size or self.has_schedule.item()
    if (buffer_full and step % self.refit_every == 0):
      self._refit()
      if self.plot_profile:
        self._plot_profile(step)

  def _refit(self):
    # 1. Fit spline to loss profile
    t_np = self.t_buf.cpu().numpy()
    loss_np = self.loss_buf.cpu().numpy()
    model = make_pipeline(
      SplineTransformer(n_knots=self.n_knots,
                        degree=self.spline_degree,
                        extrapolation='continue'),
      Ridge(alpha=self.ridge_alpha))
    model.fit(t_np.reshape(-1, 1), loss_np)

    # 2. Smoothed loss on grid -> gradient -> CDF
    loss_smooth = model.predict(self._grid.reshape(-1, 1))
    dloss_dt = np.gradient(loss_smooth, self._grid)
    # Loss should be always increasing with more noise.
    #  If it is decreasing, it is an artifact -> remove
    importance = np.maximum(dloss_dt, 0)
    # Uniform smoothing. Default: 1e-3. Ensures the CDF is
    #  always strictly increasing, which is needed to invert.
    #  Using uniform_mix = 1 ignores the adaptive schedule.
    importance = (1 - self.uniform_mix) * importance + self.uniform_mix
    cdf = np.cumsum(importance)
    cdf = cdf / cdf[-1]

    # 3. Inverse CDF to get t -> alpha map with high density
    #  on regions where the loss has high derivative.
    t_remapped = PchipInterpolator(cdf, self._grid)(self._grid)
    t_torch = torch.as_tensor(t_remapped, dtype=torch.float32)
    av = self.base_schedule.alpha_t(t_torch).numpy()

    # 4. EMA smoothing with bias correction (like Adam)
    av_torch = torch.from_numpy(av).to(self.ema_alpha_vals.device)
    self.refit_count += 1
    if self.ema > 0:
      self.ema_alpha_vals.mul_(self.ema).add_(
        av_torch, alpha=1 - self.ema)
      # Bias correction: divide by (1 - beta^t)
      correction = 1 - self.ema ** self.refit_count.item()
      av_corrected = (self.ema_alpha_vals / correction).cpu().numpy()
    else:
      av_corrected = av

    # 5. Store schedule as spline + save values for checkpointing
    self._alpha_spline = PchipInterpolator(self._grid, av_corrected)
    self._dalpha_spline = self._alpha_spline.derivative()
    self.alpha_vals.copy_(torch.from_numpy(av_corrected))
    self.has_schedule.fill_(True)

  def load_state_dict(self, sd, strict=True, *args, **kwargs):
    super().load_state_dict(sd, strict=False, *args, **kwargs)
    # Reconstruct splines from the loaded alpha_vals buffer
    if self.has_schedule.item():
      av = self.alpha_vals.cpu().numpy()
      self._alpha_spline = PchipInterpolator(self._grid, av)
      self._dalpha_spline = self._alpha_spline.derivative()

  def _plot_profile(self, step):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(self.plot_dir, exist_ok=True)
    t_grid = self._grid
    t_torch = torch.as_tensor(t_grid, dtype=torch.float32)
    alpha_base = self.base_schedule.alpha_t(t_torch).numpy()
    alpha_adapt = self._alpha_spline(t_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(t_grid, alpha_base, 'k--', label='Base')
    ax1.plot(t_grid, alpha_adapt, 'r-', label='Adapted')
    ax1.set(xlabel='t', ylabel='alpha_t', title=f'Step {step}')
    ax1.legend()

    ax2.scatter(self.t_buf.cpu().numpy()[::4],
                self.loss_buf.cpu().numpy()[::4],
                alpha=0.05, s=1, c='gray')
    ax2.set(xlabel='t', ylabel='Loss', title='Buffer')

    fig.tight_layout()
    fig.savefig(os.path.join(
      self.plot_dir, f'step_{step:{self._step_fmt}}.png'), dpi=100)
    plt.close(fig)

  def _eval_spline(self, spline, base_fn, t):
    if spline is None:
      return base_fn(t)
    vals = spline(t.detach().cpu().numpy())
    return torch.as_tensor(vals, dtype=t.dtype, device=t.device)

  def alpha_t(self, t):
    return self._eval_spline(
      self._alpha_spline, self.base_schedule.alpha_t, t)

  def alpha_prime_t(self, t):
    return self._eval_spline(
      self._dalpha_spline, self.base_schedule.alpha_prime_t, t)


def get_noise(config):
  noise_config = config.noise
  if noise_config.type == 'log-linear':
    noise = LogLinear(noise_config.eps)
  elif noise_config.type == 'cosine-squared':
    noise = CosineSquared(noise_config.eps)
  else:
    raise ValueError(f'Unknown noise type: {noise_config.type}')

  if noise_config.alpha_min is not None or noise_config.alpha_max is not None:
    alpha_min = noise_config.alpha_min
    alpha_max = noise_config.alpha_max
    if alpha_min is None:
      alpha_min = noise.alpha_t(torch.tensor(1.0)).item()
    if alpha_max is None:
      alpha_max = noise.alpha_t(torch.tensor(0.0)).item()
    noise = TruncatedScheduleWrapper(noise, alpha_min,
                                     alpha_max, noise_config.eps)

  if noise_config.adaptive:
    gbs = config.loader.global_batch_size
    buf = noise_config.adaptive_buffer_size
    assert buf % gbs == 0, (
      f'adaptive_buffer_size ({buf}) must be a multiple of '
      f'global_batch_size ({gbs})')
    noise = AdaptiveSchedule(
      noise,
      buffer_size=buf,
      refit_every=noise_config.adaptive_refit_every,
      n_grid=noise_config.adaptive_n_grid,
      n_knots=noise_config.adaptive_n_knots,
      spline_degree=noise_config.adaptive_spline_degree,
      ridge_alpha=noise_config.adaptive_ridge_alpha,
      uniform_mix=noise_config.adaptive_uniform_mix,
      max_steps=config.trainer.max_steps,
      warmup_steps=noise_config.adaptive_warmup_steps,
      ema=noise_config.adaptive_ema,
      plot_profile=noise_config.adaptive_plot_profile,
      plot_dir=noise_config.adaptive_plot_dir)

  return noise
