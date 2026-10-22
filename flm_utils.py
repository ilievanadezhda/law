"""FLM lookup-table utilities.

Provides Gauss-Hermite based alpha<->gamma conversion for
continuous-time flow matching on discrete vocabularies.
Adapted from the FLM reference implementation
(https://github.com/david3684/flm).
"""

from typing import Union

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr


def compute_alpha_exact(
    gamma: np.ndarray,
    K: int,
    n_gh: int = 100,
    sigma_floor: float = 1e-12,
    is_diffusion=False,
) -> np.ndarray:
    """Gauss-Hermite integration: maps gamma -> alpha for vocab size K."""
    gamma = np.asarray(gamma)
    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = np.sqrt(sigma)
    sigma = np.maximum(sigma, sigma_floor)
    m_c = gamma / sigma

    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    m_c_expanded = m_c[:, None]
    z_expanded = z_nodes[None, :]

    L_cu = log_ndtr(z_expanded + m_c_expanded)
    log_prod_c = (K - 1) * L_cu
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)

    alpha = K / (K - 1.0) * (q_c - 1.0 / K)
    alpha += (gamma - 1) * 1e-10  # minor trick to ensure monotonicity
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def build_luts(
    K: int, n_points: int = 10000, is_diffusion=False
) -> tuple[CubicSpline, CubicSpline]:
    """Build CubicSpline lookup tables: alpha<->gamma conversion."""
    gamma_vals = np.linspace(0.0, 1.0, n_points)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K, is_diffusion=is_diffusion)

    lut_g2a = CubicSpline(gamma_vals, alpha_vals)

    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]

    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]

    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    return lut_a2g, lut_g2a


def alpha_to_gamma(
    alpha: Union[np.ndarray, torch.Tensor], lut: CubicSpline
) -> Union[np.ndarray, torch.Tensor]:
    """Map alpha -> gamma using the LUT."""
    if isinstance(alpha, torch.Tensor):
        dtype = alpha.dtype
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device, dtype=dtype)
    return np.clip(lut(alpha), 0.0, 1.0)


def gamma_to_alpha(
    gamma: Union[np.ndarray, torch.Tensor], lut: CubicSpline
) -> Union[np.ndarray, torch.Tensor]:
    """Map gamma -> alpha using the LUT."""
    if isinstance(gamma, torch.Tensor):
        dtype = gamma.dtype
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device, dtype=dtype)
    return np.clip(lut(gamma), 0.0, 1.0)
