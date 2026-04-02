"""
ddim.py — Denoising Diffusion Implicit Models (DDIM) Scheduler
================================================================

DDIM is a **drop-in replacement for DDPM at inference time**.  No retraining
is required — the same U-Net weights are reused.  Only the sampling loop
changes.

Key insight: why can DDIM skip steps?
--------------------------------------
DDPM's reverse update adds explicit Gaussian noise at every step::

    a_{k-1} = μ_θ(a_k, k)  +  √β_k · z

Because of this injected noise, you *must* take every step; skipping any
would accumulate noise from un-cancelled stochastic terms.

DDIM rewrites the reverse update as a **deterministic mapping** (when η=0)::

    â_0    = (a_k  −  √(1-ᾱ_k) · ε_θ) / √ᾱ_k          ← predicted clean action

    a_{k-1} = √ᾱ_{k-1} · â_0  +  √(1-ᾱ_{k-1}) · ε_θ   ← project to k-1 level

This is a direct calculation from ᾱ_k and ᾱ_{k-1}.  Because it is
deterministic and only depends on the *cumulative* schedule values (not on
the step delta), the step from k → k-1 is valid even when k-1 is not the
adjacent integer.  You can jump from k=99 to k=89 without loss of validity.

The full update rule (with stochasticity parameter η):
------------------------------------------------------

    σ_k = η · √((1-ᾱ_{k-1})/(1-ᾱ_k)) · √(1 - ᾱ_k/ᾱ_{k-1})

    a_{k-1} = √ᾱ_{k-1} · â_0
              + √(1-ᾱ_{k-1} - σ²_k) · ε_θ
              + σ_k · z,      z ~ N(0, I)

    η = 0  →  fully deterministic (recommended for evaluation, reproducible)
    η = 1  →  recovers standard DDPM sampling
    0<η<1  →  partial stochasticity

Speed comparison:
    DDPM:  100 network evaluations  (K=100 steps)
    DDIM:  10  network evaluations  (ddim_steps=10)  ≈ 10× faster

References
----------
- Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
- Chi et al., "Diffusion Policy" (RSS 2023) — uses DDIM for fast inference
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from diffusion_policy.model.ddpm import DDPMScheduler

logger = logging.getLogger(__name__)


class DDIMScheduler:
    """
    DDIM sampler — fast deterministic inference on a trained DDPM model.

    This class does NOT hold trainable parameters; it is a pure utility.
    It borrows the noise schedule tensors from a fitted ``DDPMScheduler``.

    Args:
        ddpm_scheduler: A fitted DDPMScheduler (provides betas, alphas_cumprod).
        ddim_steps:     Number of denoising steps to use at inference.
                        10 gives ~10× speedup over DDPM with negligible loss.
        eta:            Stochasticity parameter η ∈ [0, 1].
                        0 = fully deterministic (default, best for eval).
    """

    def __init__(
        self,
        ddpm_scheduler: DDPMScheduler,
        ddim_steps: int = 10,
        eta: float = 0.0,
    ) -> None:
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}")

        self.ddpm = ddpm_scheduler
        self.ddim_steps = ddim_steps
        self.eta = eta

        # Compute the DDIM timestep subsequence
        self.timestep_seq: np.ndarray = self._compute_timestep_seq()

        logger.info(
            "DDIMScheduler: ddim_steps=%d, eta=%.2f, timestep_seq=%s",
            ddim_steps, eta, self.timestep_seq,
        )

    # ---------------------------------------------------------------------- #
    # Timestep subsequence                                                    #
    # ---------------------------------------------------------------------- #

    def _compute_timestep_seq(self) -> np.ndarray:
        """
        Select an evenly-spaced subsequence of K timesteps.

        For K=100, ddim_steps=10:
            [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        Reversed during sampling: [99, 89, 79, ..., 9, 0]

        Returns:
            np.ndarray of shape (ddim_steps+1,), sorted ascending.
            The extra entry is 0 (the "clean" target level).
        """
        # Evenly spaced in {1, ..., K}, then shift by -1 to be 0-indexed
        step = self.ddpm.K // self.ddim_steps
        seq  = np.arange(0, self.ddpm.K, step)  # e.g. [0,10,20,...,90] for K=100,s=10
        # Make sure last timestep K-1 is included
        if seq[-1] != self.ddpm.K - 1:
            seq = np.append(seq, self.ddpm.K - 1)
        return seq.astype(np.int64)

    # ---------------------------------------------------------------------- #
    # Single DDIM step                                                        #
    # ---------------------------------------------------------------------- #

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        t_prev: int,
        noisy_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one DDIM denoising step from level t to level t_prev.

        Args:
            model_output:  (B, T_pred, action_dim) — ε_θ from the U-Net.
            t:             Current timestep index (higher = more noise).
            t_prev:        Previous (less noisy) timestep index. -1 means
                           we are going to the clean level (ᾱ = 1).
            noisy_actions: (B, T_pred, action_dim) — a_t.

        Returns:
            a_prev: (B, T_pred, action_dim) — a_{t_prev}.
        """
        ab_t = self.ddpm.alphas_cumprod[t]                    # ᾱ_t

        if t_prev >= 0:
            ab_prev = self.ddpm.alphas_cumprod[t_prev]        # ᾱ_{t-1}
        else:
            # t_prev == -1 means "clean" level; treat ᾱ = 1
            ab_prev = torch.ones_like(ab_t)

        # ── Predict clean action â_0 ──────────────────────────────────────
        # â_0 = (a_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
        a0_pred = (
            noisy_actions - (1.0 - ab_t).sqrt() * model_output
        ) / ab_t.sqrt()

        # ── Compute σ (stochasticity) ─────────────────────────────────────
        # σ_k = η · √((1-ᾱ_{k-1})/(1-ᾱ_k)) · √(1 - ᾱ_k / ᾱ_{k-1})
        # Guard against ᾱ_{t-1} = 1 (first step) to avoid division issues
        ratio  = ((1.0 - ab_prev) / (1.0 - ab_t)).clamp(min=0.0)
        factor = (1.0 - ab_t / (ab_prev + 1e-8)).clamp(min=0.0)
        sigma  = self.eta * ratio.sqrt() * factor.sqrt()

        # ── "Pointing direction" component of ε_θ ────────────────────────
        # √(1 - ᾱ_{t-1} - σ²)  ·  ε_θ
        dir_coeff = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt()

        # ── DDIM update ───────────────────────────────────────────────────
        a_prev = ab_prev.sqrt() * a0_pred + dir_coeff * model_output

        if self.eta > 0.0:
            a_prev = a_prev + sigma * torch.randn_like(noisy_actions)

        return a_prev

    # ---------------------------------------------------------------------- #
    # Full sampling loop                                                      #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        observation: torch.Tensor,
        pred_horizon: int,
        action_dim: int,
    ) -> torch.Tensor:
        """
        Full DDIM reverse sampling: draw a clean action from noise in
        ``ddim_steps`` network evaluations (vs. K for DDPM).

        Args:
            model:        The trained ConditionalUnet1D.
            observation:  (B, T_obs, obs_dim) — normalised observation history.
            pred_horizon: T_pred — action sequence length to generate.
            action_dim:   Action dimensionality.

        Returns:
            actions: (B, T_pred, action_dim) — sampled clean action sequence.
        """
        device = observation.device
        B = observation.shape[0]

        # Start from pure noise
        a_t = torch.randn(B, pred_horizon, action_dim, device=device)

        # Iterate over the DDIM timestep subsequence in reverse (high → low noise)
        seq_reversed = list(reversed(self.timestep_seq))
        for i, t in enumerate(seq_reversed):
            t_prev = seq_reversed[i + 1] if i + 1 < len(seq_reversed) else -1

            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = model(a_t, t_tensor, observation)
            a_t = self.step(eps_pred, int(t), int(t_prev), a_t)

        return a_t     # a_0 (clean action sequence)


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    K, B, T, A = 100, 2, 16, 2
    ddpm = DDPMScheduler(K)
    ddim = DDIMScheduler(ddpm, ddim_steps=10, eta=0.0)

    # Verify timestep sequence length
    assert len(ddim.timestep_seq) >= 10, "Too few timesteps"
    print(f"DDIM timestep sequence: {ddim.timestep_seq}")

    # Deterministic: same noise → same output
    torch.manual_seed(0)
    obs   = torch.randn(B, 2, 5)

    # Dummy model that returns zeros (denoised at every step)
    class ZeroModel(nn.Module):
        def forward(self, a, t, obs): return torch.zeros_like(a)

    model = ZeroModel()
    torch.manual_seed(0); a1 = ddim.sample(model, obs, T, A)
    torch.manual_seed(0); a2 = ddim.sample(model, obs, T, A)
    assert torch.allclose(a1, a2), "DDIM (η=0) should be deterministic"
    print("DDIM deterministic check passed")

    # Stochastic: η=1 → outputs differ
    ddim_stochastic = DDIMScheduler(ddpm, ddim_steps=10, eta=1.0)
    torch.manual_seed(1); a3 = ddim_stochastic.sample(model, obs, T, A)
    torch.manual_seed(2); a4 = ddim_stochastic.sample(model, obs, T, A)
    assert not torch.allclose(a3, a4), "DDIM (η=1) should be stochastic"
    print("DDIM stochastic check passed")

    print("All DDIM sanity checks passed!")
