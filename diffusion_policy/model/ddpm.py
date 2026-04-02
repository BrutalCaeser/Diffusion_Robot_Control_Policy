"""
ddpm.py — Denoising Diffusion Probabilistic Model (DDPM) Scheduler
====================================================================

This module handles everything related to the DDPM noise process:
the noise schedule, the forward noising process used during training,
and the reverse denoising process used at inference time.

Mathematical Background
-----------------------
**Forward process** (noising — done during training)::

    q(a_k | a_{k-1}) = N(a_k; √(1-β_k) · a_{k-1},  β_k · I)

A key convenience from the reparameterisation trick lets us jump directly
from a clean action ``a_0`` to any noise level ``k`` in ONE step:

    q(a_k | a_0) = N(a_k;  √ᾱ_k · a_0,  (1 - ᾱ_k) · I)

    ⟹  a_k = √ᾱ_k · a_0  +  √(1-ᾱ_k) · ε,    ε ~ N(0, I)          (*)

where:
    α_k  = 1 - β_k
    ᾱ_k  = ∏_{i=1}^{k} α_i    (cumulative product — signal retention ratio)

As k increases from 0 → K:
    ᾱ_k → 0  →  a_k  looks like pure Gaussian noise.

**Reverse process** (denoising — inference time)::

    p_θ(a_{k-1} | a_k) = N(a_{k-1};  μ_θ(a_k, k),  σ²_k · I)

The mean is derived by substituting the model's noise prediction ε_θ into
the reparameterisation and solving for a_{k-1}:

    μ_θ = (1/√α_k) · [a_k  −  (β_k / √(1-ᾱ_k)) · ε_θ(a_k, k, obs)]

    σ²_k = β_k   (variance choice from Ho et al.)

**Training objective** (simplified ELBO from Ho et al.)::

    L = E_{k, a_0, ε} [ ||ε  −  ε_θ(√ᾱ_k · a_0 + √(1-ᾱ_k) · ε, k)||² ]

i.e. MSE between actual noise and predicted noise.

**Noise schedule options:**

Linear (Ho et al., 2020) — designed for K=1000:

    β_k = β_start + (β_end - β_start) · k/(K-1)

Cosine (Nichol & Dhariwal, 2021) — recommended default for K=100:

    ᾱ_t = cos²(((t/T + s) / (1 + s)) · π/2) / cos²(s / (1+s) · π/2)
    β_k = 1 - ᾱ_k / ᾱ_{k-1}          clipped to [0.0001, 0.9999]

The cosine schedule produces ᾱ_0 ≈ 1 and ᾱ_{K-1} ≈ 0 for ANY choice of K,
making it the standard choice when using K=100 (as in Diffusion Policy).

References
----------
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Nichol & Dhariwal, "Improved DDPM" (ICML 2021) — cosine schedule
- Chi et al., "Diffusion Policy" (RSS 2023)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule from Nichol & Dhariwal (Improved DDPM, ICML 2021).

    Gives ᾱ_0 ≈ 1 and ᾱ_{K-1} ≈ 0 for any K, making it suitable for K=100.

    Args:
        num_steps: Number of diffusion steps K.
        s:         Small offset to prevent β_0 being too small (default 0.008).

    Returns:
        betas: (K,) tensor of noise schedule values.
    """
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps)
    alpha_bar = torch.cos(
        ((t / num_steps) + s) / (1.0 + s) * math.pi * 0.5
    ) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0001, 0.9999)


def _linear_beta_schedule(
    num_steps: int, beta_start: float, beta_end: float
) -> torch.Tensor:
    """Standard linear beta schedule (Ho et al., NeurIPS 2020)."""
    return torch.linspace(beta_start, beta_end, num_steps)


class DDPMScheduler(nn.Module):
    """
    DDPM noise schedule and sampling utilities.

    Inherits from ``nn.Module`` so that the schedule tensors (betas, alphas,
    alphas_cumprod, …) are automatically moved with ``.to(device)`` alongside
    the main model.

    All schedule tensors are registered as non-trainable buffers.

    Args:
        num_diffusion_steps: K — total number of diffusion steps (default 100).
        beta_start:          β₁ — used only with ``beta_schedule="linear"``.
        beta_end:            β_K — used only with ``beta_schedule="linear"``.
        beta_schedule:       ``"cosine"`` (default, works for any K) or
                             ``"linear"`` (Ho et al. 2020, designed for K=1000).
    """

    def __init__(
        self,
        num_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.K = num_diffusion_steps

        # ── Build noise schedule ───────────────────────────────────────────
        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(num_diffusion_steps)
        elif beta_schedule == "linear":
            betas = _linear_beta_schedule(num_diffusion_steps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}. Use 'cosine' or 'linear'.")

        # ── Derived quantities ─────────────────────────────────────────────
        alphas          = 1.0 - betas                                    # α_k
        alphas_cumprod  = torch.cumprod(alphas, dim=0)                   # ᾱ_k

        # Pre-compute square-roots used repeatedly in add_noise and step
        sqrt_alphas_cumprod         = alphas_cumprod.sqrt()              # √ᾱ_k
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()      # √(1-ᾱ_k)

        # Register all as buffers (moved with .to(device))
        self.register_buffer("betas",                        betas)
        self.register_buffer("alphas",                       alphas)
        self.register_buffer("alphas_cumprod",               alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",          sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

        logger.info(
            "DDPMScheduler: K=%d, schedule=%s, final ᾱ_K=%.6f",
            num_diffusion_steps, beta_schedule, alphas_cumprod[-1].item(),
        )

    # ---------------------------------------------------------------------- #
    # Training: forward (noising) process                                     #
    # ---------------------------------------------------------------------- #

    def add_noise(
        self,
        clean_actions: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the forward diffusion process (equation *).

            a_k = √ᾱ_k · a_0 + √(1-ᾱ_k) · ε

        Args:
            clean_actions: (B, T_pred, action_dim) — the ground-truth a_0.
            noise:         (B, T_pred, action_dim) — ε ~ N(0, I).
            timesteps:     (B,) int64 — the diffusion step k for each sample.

        Returns:
            noisy_actions: (B, T_pred, action_dim) — a_k.
        """
        # Index schedule buffers for each sample's timestep
        # Shape: (B,) → (B, 1, 1) for broadcasting over (T_pred, action_dim)
        s = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)          # √ᾱ_k
        n = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1) # √(1-ᾱ_k)

        return s * clean_actions + n * noise

    def sample_timesteps(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample uniformly random diffusion timesteps for a training batch.

        Args:
            batch_size: B — number of timesteps to sample.
            device:     Target device (defaults to scheduler's device).

        Returns:
            timesteps: (B,) int64 tensor with values in {0, ..., K-1}.
        """
        dev = device if device is not None else self.betas.device
        return torch.randint(0, self.K, (batch_size,), device=dev)

    # ---------------------------------------------------------------------- #
    # Inference: reverse (denoising) process                                  #
    # ---------------------------------------------------------------------- #

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        noisy_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one reverse DDPM denoising step.

        Computes a_{k-1} from a_k using the model's noise prediction ε_θ:

            μ_θ = (1/√α_k) · [a_k  −  (β_k / √(1-ᾱ_k)) · ε_θ]

            a_{k-1} = μ_θ  +  √β_k · z     if k > 0   (z ~ N(0,I))
                    = μ_θ                   if k = 0   (final step)

        Args:
            model_output:  (B, T_pred, action_dim) — ε_θ predicted by the U-Net.
            timestep:      int — current diffusion step k.
            noisy_actions: (B, T_pred, action_dim) — a_k.

        Returns:
            denoised: (B, T_pred, action_dim) — a_{k-1} (or a_0 at k=0).
        """
        k = timestep

        beta_k       = self.betas[k]
        alpha_k      = self.alphas[k]
        alpha_bar_k  = self.alphas_cumprod[k]

        # Compute the posterior mean μ_θ
        coeff  = beta_k / (1.0 - alpha_bar_k).sqrt()       # β_k / √(1-ᾱ_k)
        mean   = (1.0 / alpha_k.sqrt()) * (noisy_actions - coeff * model_output)

        if k == 0:
            return mean

        # Add stochastic noise for k > 0
        noise  = torch.randn_like(noisy_actions)
        sigma  = beta_k.sqrt()
        return mean + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        observation: torch.Tensor,
        pred_horizon: int,
        action_dim: int,
    ) -> torch.Tensor:
        """
        Full DDPM reverse sampling: draw a clean action sequence from noise.

        Runs K reverse steps starting from pure Gaussian noise.

        Args:
            model:       The trained ConditionalUnet1D.
            observation: (B, T_obs, obs_dim) — normalised observation history.
            pred_horizon: T_pred — number of action timesteps to generate.
            action_dim:  Dimensionality of each action.

        Returns:
            actions: (B, T_pred, action_dim) — sampled clean action sequence.
        """
        device = observation.device
        B = observation.shape[0]

        # Start from pure Gaussian noise
        a_k = torch.randn(B, pred_horizon, action_dim, device=device)

        for k in reversed(range(self.K)):
            t = torch.full((B,), k, device=device, dtype=torch.long)
            eps_pred = model(a_k, t, observation)
            a_k = self.step(eps_pred, k, a_k)

        return a_k     # a_0 (clean)


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    K, B, T, A = 100, 4, 16, 2
    scheduler = DDPMScheduler(num_diffusion_steps=K)

    a0    = torch.randn(B, T, A)
    eps   = torch.randn(B, T, A)
    ts    = scheduler.sample_timesteps(B)

    # Forward process
    a_k   = scheduler.add_noise(a0, eps, ts)
    assert a_k.shape == (B, T, A), "add_noise shape mismatch"

    # At timestep 0, noisy ≈ clean (ᾱ_0 ≈ 1)
    ts0   = torch.zeros(B, dtype=torch.long)
    a_k0  = scheduler.add_noise(a0, eps, ts0)
    assert torch.allclose(a_k0, a0, atol=0.1), \
        f"At k=0 noisy ≈ clean; max diff {(a_k0-a0).abs().max():.4f}"

    # At timestep K-1, noisy ≈ noise (ᾱ_{K-1} ≈ 0)
    tsK   = torch.full((B,), K - 1, dtype=torch.long)
    a_kK  = scheduler.add_noise(a0, eps, tsK)
    assert torch.allclose(a_kK, eps, atol=0.05), \
        f"At k=K-1 noisy ≈ noise; max diff {(a_kK-eps).abs().max():.4f}"

    # Step (reverse)
    denoised = scheduler.step(eps, 50, a_k)
    assert denoised.shape == (B, T, A), "step shape mismatch"

    # Final ᾱ_K should be near 0
    final_ab = scheduler.alphas_cumprod[-1].item()
    assert final_ab < 0.01, f"ᾱ_K should be near 0, got {final_ab:.4f}"

    print(f"Final ᾱ_K = {final_ab:.6f}")
    print("All DDPM sanity checks passed!")
