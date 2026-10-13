"""Derived flow quantities (element-wise; work on owned or global arrays alike)."""
from __future__ import annotations

import numpy as np

GAMMA = 1.4


def speed_of_sound(p, rho, gamma=GAMMA):
    return np.sqrt(gamma * p / rho)


def mach(velocity, p, rho, gamma=GAMMA):
    return np.linalg.norm(velocity, axis=-1) / speed_of_sound(p, rho, gamma)


def total_pressure(p, M, gamma=GAMMA):
    return p * (1.0 + 0.5 * (gamma - 1.0) * M ** 2) ** (gamma / (gamma - 1.0))


def total_temperature(T, M, gamma=GAMMA):
    return T * (1.0 + 0.5 * (gamma - 1.0) * M ** 2)


def entropy(p, rho, gamma=GAMMA, p_ref=1.0, rho_ref=1.0):
    """s/cv = ln((p/p_ref) / (rho/rho_ref)^gamma)."""
    return np.log((p / p_ref) / (rho / rho_ref) ** gamma)


def normal_mach(velocity, grad_p, a):
    """Mach number along the pressure gradient: (V/a) . grad p / |grad p|."""
    g = np.linalg.norm(grad_p, axis=-1)
    return np.einsum("...i,...i->...", velocity, grad_p) / (a * np.where(g > 0, g, np.inf))


# velocity-gradient based (grad_v[..., i, j] = d v_i / d x_j, as returned by Volume.gradient)
def divergence(grad_v):
    return np.trace(grad_v, axis1=-2, axis2=-1)


def vorticity(grad_v):
    g = grad_v
    return np.stack([g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]], -1)


def q_criterion(grad_v):
    """Q = 0.5 (|Omega|^2 - |S|^2)."""
    S = 0.5 * (grad_v + np.swapaxes(grad_v, -1, -2))
    W = 0.5 * (grad_v - np.swapaxes(grad_v, -1, -2))
    return 0.5 * (np.sum(W * W, axis=(-2, -1)) - np.sum(S * S, axis=(-2, -1)))


def lambda2(grad_v):
    """Second eigenvalue of S^2 + Omega^2 (vortex where < 0)."""
    S = 0.5 * (grad_v + np.swapaxes(grad_v, -1, -2))
    W = 0.5 * (grad_v - np.swapaxes(grad_v, -1, -2))
    return np.linalg.eigvalsh(S @ S + W @ W)[..., 1]


def helicity(velocity, grad_v):
    return np.einsum("...i,...i->...", velocity, vorticity(grad_v))


def wind_axes(aoa_deg: float = 0.0, sideslip_deg: float = 0.0, ndim: int = 3):
    """Unit drag, side and lift directions with SU2's conventions."""
    a, b = np.radians(aoa_deg), np.radians(sideslip_deg)
    if ndim == 2:
        return {"drag": np.array([np.cos(a), np.sin(a), 0.0]), "side": np.zeros(3),
                "lift": np.array([-np.sin(a), np.cos(a), 0.0])}
    return {"drag": np.array([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)]),
            "side": np.array([-np.sin(b) * np.cos(a), np.cos(b), -np.sin(b) * np.sin(a)]),
            "lift": np.array([-np.sin(a), 0.0, np.cos(a)])}
