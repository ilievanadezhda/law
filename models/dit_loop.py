"""LoopDiT: a DiT with the denoising loop INSIDE the architecture."""

import contextlib
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F

import samplers
from models.dit import DIT


@dataclass(frozen=True)
class CarryRegularization:
  gaussian_std: float
  rotation_fraction: float


@dataclass(frozen=True)
class LAWTrajectory:
  xt: torch.Tensor
  valid_tokens: torch.Tensor | None
  stop_gradient_between_passes: bool
  regularization: CarryRegularization | None
  carry: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class LAWCarryAblationTrajectory(LAWTrajectory):
  carry_representation: Literal['expected_embedding', 'hidden_representation']
  injection_mode: Literal['replace', 'add']


def _rotate_carry(carry, fraction):
  x = carry.float()
  norm = x.norm(dim=-1, keepdim=True)
  x_hat = x / norm.clamp_min(1e-8)
  r = torch.randn_like(x)
  r_hat = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
  dot = (x_hat * r_hat).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
  omega = dot.arccos()
  sin_omega = omega.sin()
  a = ((1 - fraction) * omega).sin() / sin_omega
  b = (fraction * omega).sin() / sin_omega
  return (norm * (a * x_hat + b * r_hat)).to(carry.dtype)


def apply_temperature(logits, temperature):
  """logits / T. T is a float or a (B,1,1) tensor; None or 1.0 is a no-op."""
  if temperature is None:
    return logits
  if torch.is_tensor(temperature):
    return logits / temperature.to(logits.dtype)  # Prevent bf16 upcast.
  if temperature == 1.0:
    return logits
  return logits / temperature


class LoopDiT(DIT):
  def __init__(self, config, vocab_size):
    super().__init__(config, vocab_size)
    # Zero-initialized adaLN embeddings distinguish carry-free and looped input.
    if config.model.mode_embedding:
      assert self.adaLN, 'model.mode_embedding needs adaLN=True'
      self.loop_mode_embed = torch.nn.Parameter(
        torch.zeros(2, config.model.cond_dim))

  def _project_prefix_embedding(self, carry, x0, valid_tokens):
    # Keep prompt positions at their clean embeddings.
    if valid_tokens is None:
      return carry
    clean = F.embedding(x0, self.vocab_embed.matrix)
    mask = valid_tokens.unsqueeze(-1).bool()
    return torch.where(mask, carry, clean)

  def _regularize_law_carry(self, carry, regularization):
    if regularization.gaussian_std > 0:
      carry = carry + regularization.gaussian_std * torch.randn_like(carry)
    if regularization.rotation_fraction > 0:
      carry = _rotate_carry(carry, regularization.rotation_fraction)
    return carry

  def forward_law(self, trajectory: LAWTrajectory, sigma, *, prediction_grad, return_carry,
                  carry_temperature=1.0, temperature=1.0):
    return self._forward_trajectory(
      trajectory, sigma, carry_representation='expected_embedding',
      injection_mode='replace', prediction_grad=prediction_grad,
      return_carry=return_carry, carry_temperature=carry_temperature,
      temperature=temperature)

  def forward_law_carry_ablation(self, trajectory: LAWCarryAblationTrajectory, sigma, *,
                                 prediction_grad, return_carry, carry_temperature=1.0,
                                 temperature=1.0):
    return self._forward_trajectory(
      trajectory, sigma, carry_representation=trajectory.carry_representation,
      injection_mode=trajectory.injection_mode, prediction_grad=prediction_grad,
      return_carry=return_carry, carry_temperature=carry_temperature,
      temperature=temperature)

  def _forward_trajectory(self, trajectory, sigma, *, carry_representation,
                          injection_mode, prediction_grad, return_carry,
                          carry_temperature, temperature):
    context = samplers.CarryContext(
      carry=trajectory.carry, carry_representation=carry_representation,
      injection_mode=injection_mode, looped=trajectory.carry is not None)
    log_p, context = self.forward_carry(
      trajectory.xt, sigma, context, carry_representation=carry_representation,
      injection_mode=injection_mode, x0=trajectory.xt,
      valid_tokens=trajectory.valid_tokens,
      carry_regularization=trajectory.regularization,
      detach=trajectory.stop_gradient_between_passes,
      prediction_grad=prediction_grad, return_carry=return_carry,
      carry_temperature=carry_temperature, temperature=temperature)
    return log_p, None if context is None else replace(trajectory, carry=context.carry)

  def forward_carry(self, xt, sigma, in_context, *, carry_representation, injection_mode,
                    x0, valid_tokens, detach=False,
                    return_carry=True, temperature=1.0,
                    carry_temperature=None, prediction_grad=True,
                    carry_regularization=None):
    """Return log probabilities and an optional carry.

    temperature affects the prediction; carry_temperature independently affects
    expected-embedding carries and is unused for hidden carries.
    """
    need_hidden = return_carry and carry_representation == 'hidden_representation'
    # Build the expected embedding outside no_grad to preserve vocabulary gradients.
    grad_context = contextlib.nullcontext() if prediction_grad else torch.no_grad()
    with grad_context:
      out = super().forward(x0=None, xt=xt, sigma=sigma,
                            context=in_context,
                            return_aux=need_hidden)
      logits, aux = out if need_hidden else (out, None)
      log_p = apply_temperature(logits, temperature).log_softmax(-1)

    if not return_carry:
      return log_p, None

    carry = self._build_carry(
      logits, aux, xt, carry_representation=carry_representation,
      temperature=carry_temperature,
      detach=detach, x0=x0,
      valid_tokens=valid_tokens, carry_regularization=carry_regularization)
    return log_p, samplers.CarryContext(
      carry=carry, carry_representation=carry_representation,
      injection_mode=injection_mode, looped=True)

  def _build_carry(self, logits, aux, xt, *, carry_representation, temperature,
                   detach, x0, valid_tokens, carry_regularization=None):
    """The signal handed to the next pass; never affects the returned log_p."""
    if carry_representation == 'hidden_representation':
      carry = aux['latent']
    elif carry_representation == 'expected_embedding':
      logits = apply_temperature(logits, temperature)
      embedding = self.vocab_embed.matrix
      probs = logits.softmax(-1).to(embedding.dtype)
      if detach:
        probs = probs.detach()
      carry = torch.matmul(probs, embedding)
      carry = self.vocab_embed.renormalize(carry)
    else:
      raise ValueError(carry_representation)

    if carry_regularization is not None:
      carry = self._regularize_law_carry(carry, carry_regularization)
    carry = self._project_prefix_embedding(carry, x0, valid_tokens)
    if detach and carry_representation == 'hidden_representation':
      carry = carry.detach()
      # Keep the embedding dependency visible to DDP.
      carry = carry + self.vocab_embed(xt) * 0.0
    return carry
