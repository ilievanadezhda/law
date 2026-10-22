"""Utility functions for CANDI (Continuous and Discrete Diffusion).

Contains noise schedule helpers and categorical sampling used by the
CANDI algorithm. These are pure functions with no side effects.
"""

import torch


def expected_rank(d, a, sigma):
  """Expected rank of the true coordinate in a Gaussian-corrupted one-hot.

  Computes the Bayes error rate: the probability that a random other
  coordinate beats the true coordinate after adding N(0, sigma^2) noise.

  Args:
    d: int, number of other coordinates (vocab_size - 1).
    a: amplitude of the true coordinate (scalar or tensor).
    sigma: noise std (scalar or tensor).

  Returns:
    Expected error rate, same shape as a/sigma.
  """
  a = torch.as_tensor(a, dtype=torch.float32)
  sigma = torch.as_tensor(sigma, dtype=torch.float32)
  Phi = torch.distributions.Normal(0.0, 1.0).cdf
  p = Phi(-a / (sigma * torch.sqrt(torch.tensor(2.0))))
  return (d - 1) * p / d


def training_sigma_ve(t, sigma_min, sigma_max):
  """VE schedule: sigma(t) = sigma_min * (sigma_max / sigma_min) ^ t."""
  t = torch.as_tensor(t, dtype=torch.float32)
  log_ratio = torch.log(
    torch.tensor(sigma_max / sigma_min, dtype=torch.float32).to(t.device))
  return sigma_min * torch.exp(t * log_ratio)


def inference_sigmas(n_steps, sigma_min, sigma_max):
  """Log-linear sigma schedule from sigma_max down to sigma_min."""
  ks = torch.linspace(0, 1, n_steps, dtype=torch.float32)
  return sigma_max * (sigma_min / sigma_max) ** ks


def sigma_from_time_vectorized(t, sigmas, errors):
  """Interpolate: given target error rate t, find corresponding sigma.

  Args:
    t: target error rates (tensor, values in [0, 1]).
    sigmas: 1D tensor of sigma values (ascending).
    errors: 1D tensor of corresponding error rates (ascending).

  Returns:
    Interpolated sigma values, same shape as t.
  """
  t = t.clamp(0.0, 1.0).to(sigmas.device)
  indices = torch.searchsorted(
    errors, t, right=True).clamp(1, len(errors) - 1)
  i0, i1 = indices - 1, indices
  e0, e1 = errors[i0], errors[i1]
  s0, s1 = sigmas[i0], sigmas[i1]
  frac = (t - e0) / (e1 - e0 + 1e-8)
  return s0 + frac * (s1 - s0)


def sample_categorical(categorical_probs):
  """Gumbel-max categorical sampling.

  Args:
    categorical_probs: unnormalized probabilities (..., K).

  Returns:
    Sampled indices (...,).
  """
  gumbel_norm = (
    1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)
