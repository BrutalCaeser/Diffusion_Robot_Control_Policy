"""
flow_matching.py — Flow Matching Training & ODE Inference
==========================================================

[Phase 5 — Stretch Goal]

Flow Matching (FM) is an alternative to DDPM that reuses the EXACT SAME U-Net
architecture.  Only the training loss and inference procedure change.

DDPM vs Flow Matching — side-by-side
--------------------------------------
+------------------+----------------------------------+----------------------------------+
| Aspect           | DDPM                             | Flow Matching                    |
+==================+==================================+==================================+
| Time             | Discrete k ∈ {0,...,K-1}         | Continuous t ∈ [0, 1]            |
+------------------+----------------------------------+----------------------------------+
| Forward path     | Stochastic (add Gaussian noise)  | Deterministic straight-line      |
|                  | a_k = √ᾱ_k·a_0 + √(1-ᾱ_k)·ε   | x_t = (1-t)·a_0 + t·ε           |
+------------------+----------------------------------+----------------------------------+
| Model predicts   | Noise ε                          | Velocity v = dx/dt = ε − a_0     |
+------------------+----------------------------------+----------------------------------+
| Training loss    | MSE(ε_θ, ε)                      | MSE(v_θ, ε − a_0)               |
+------------------+----------------------------------+----------------------------------+
| Inference        | K reverse steps (or fewer DDIM)  | ODE solve: Euler/RK4, t: 1→0    |
+------------------+----------------------------------+----------------------------------+
| Steps needed     | 10–100                           | 5–10                             |
+------------------+----------------------------------+----------------------------------+
| Noise schedule   | Must tune β_start, β_end, K      | None (just t ~ Uniform[0,1])     |
+------------------+----------------------------------+----------------------------------+

Mathematical Background
-----------------------
**Interpolation (forward path):**

    x_t = (1 − t) · a_0  +  t · ε,    ε ~ N(0, I),   t ∈ [0, 1]

At t=0: x_0 = a_0  (clean data)
At t=1: x_1 = ε    (pure noise)

**Target velocity (conditional vector field):**

    u_t = dx_t/dt  = ε − a_0

i.e. "point in the direction from data to noise".

**Training objective:**

    L = E_{t, a_0, ε} [ ||v_θ(x_t, t, obs)  −  (ε − a_0)||² ]

**Inference (Euler ODE, t: 1 → 0):**

    x_{t-Δt}  =  x_t  −  Δt · v_θ(x_t, t, obs)

    where Δt = 1 / num_steps

Start from x_1 ~ N(0, I); after ``num_steps`` Euler steps, arrive at x_0 ≈ a_0.

**Connection to Schrödinger Bridges (thesis link):**
Flow Matching is the deterministic (σ=0) limit of the Schrödinger Bridge (SB)
problem.  The SB adds a diffusion term:

    dx = v_θ dt  +  σ dW

As σ → 0 the SB solution converges to the FM (optimal-transport) solution.
This provides a natural code-level bridge to extend the project into SB
estimation — the thesis direction.

References
----------
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Liu et al., "Flow Straight and Fast: Rectified Flow from Any Distribution"
  (ICLR 2023)
- Albergo & Vanden-Eijnden, "Building Normalizing Flows with Stochastic
  Interpolants" (2023)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Scale factor applied to the continuous time t ∈ [0,1] before passing to the
# U-Net's sinusoidal embedding.  The embedding was designed for integer
# timesteps k ∈ [0, 99].  Scaling t by K brings it into a numerically
# compatible range, so we can reuse the exact same U-Net without modification.
_FM_T_SCALE: float = 100.0


class FlowMatchingScheduler:
    """
    Flow Matching scheduler for training and Euler-ODE inference.

    This is NOT an nn.Module — it holds no trainable parameters.
    It is a pure utility used by the training loop (``get_loss``) and the
    evaluation loop (``sample``).

    Args:
        num_inference_steps: Number of Euler integration steps at inference.
                             Fewer is faster; 10 is usually sufficient.
        t_scale:             Multiplier applied to t before passing to the U-Net
                             timestep embedding (default 100 to match DDPM range).
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        t_scale: float = _FM_T_SCALE,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.t_scale = t_scale
        logger.info(
            "FlowMatchingScheduler: num_inference_steps=%d, t_scale=%.1f",
            num_inference_steps, t_scale,
        )

    # ---------------------------------------------------------------------- #
    # Forward interpolation                                                   #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def interpolate(
        a_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the noisy intermediate: x_t = (1-t)·a_0 + t·ε.

        Args:
            a_0:   (B, T_pred, action_dim) — clean action sequence.
            noise: (B, T_pred, action_dim) — Gaussian noise ε ~ N(0, I).
            t:     (B,) — continuous time in [0, 1].

        Returns:
            x_t:   (B, T_pred, action_dim) — interpolated noisy sample.
        """
        # Reshape t for broadcasting: (B,) → (B, 1, 1)
        t_bc = t.view(-1, 1, 1)
        return (1.0 - t_bc) * a_0 + t_bc * noise

    @staticmethod
    def compute_target(
        a_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the conditional velocity target u = ε − a_0.

        This is the direction from data (t=0) to noise (t=1), i.e., the
        "ground-truth velocity" that the model should predict.

        Args:
            a_0:    (B, T_pred, action_dim) — clean action.
            noise:  (B, T_pred, action_dim) — Gaussian noise.

        Returns:
            target: (B, T_pred, action_dim) — velocity target.
        """
        return noise - a_0

    # ---------------------------------------------------------------------- #
    # Training loss                                                           #
    # ---------------------------------------------------------------------- #

    def get_loss(
        self,
        model: nn.Module,
        a_0: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Flow Matching training loss for a batch.

        Steps:
            1. Sample t ~ Uniform[0, 1] independently per sample
            2. Sample ε ~ N(0, I)
            3. Interpolate: x_t = (1-t)·a_0 + t·ε
            4. Compute velocity target: u = ε − a_0
            5. Predict velocity: v_θ = model(x_t, t * t_scale, obs)
            6. Return MSE(v_θ, u)

        Args:
            model: The ConditionalUnet1D.
            a_0:   (B, T_pred, action_dim) — normalised clean actions.
            obs:   (B, T_obs, obs_dim) — normalised observations.

        Returns:
            loss: scalar tensor — MSE between predicted and target velocity.
        """
        B = a_0.shape[0]
        device = a_0.device

        t      = torch.rand(B, device=device)              # (B,) ~ Uniform[0,1]
        noise  = torch.randn_like(a_0)                     # ε ~ N(0, I)

        x_t    = self.interpolate(a_0, noise, t)            # (B, T_pred, action_dim)
        target = self.compute_target(a_0, noise)            # (B, T_pred, action_dim)

        # Scale t into the U-Net's expected numeric range for the sinusoidal emb
        t_scaled = t * self.t_scale                         # (B,)

        v_pred = model(x_t, t_scaled, obs)                  # (B, T_pred, action_dim)

        return F.mse_loss(v_pred, target)

    # ---------------------------------------------------------------------- #
    # Euler ODE inference                                                     #
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
        Generate a clean action sequence via Euler ODE integration.

        Integrates from t=1 (noise) to t=0 (data) using the predicted
        velocity field:

            x_{t - Δt} = x_t  −  Δt · v_θ(x_t, t, obs)

        Args:
            model:        The trained ConditionalUnet1D.
            observation:  (B, T_obs, obs_dim) — normalised obs history.
            pred_horizon: T_pred — number of action steps to generate.
            action_dim:   Dimensionality of each action.

        Returns:
            actions: (B, T_pred, action_dim) — predicted clean action sequence.
        """
        device = observation.device
        B = observation.shape[0]

        # Start from pure noise at t=1
        x_t = torch.randn(B, pred_horizon, action_dim, device=device)

        # Integration step size (equal steps from 1 to 0)
        dt = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            # Current time, going from 1.0 down to 0.0+dt
            t_cur = 1.0 - i * dt
            t_tensor = torch.full(
                (B,), t_cur * self.t_scale, device=device, dtype=torch.float32
            )

            v_pred = model(x_t, t_tensor, observation)      # predicted velocity
            x_t = x_t - dt * v_pred                         # Euler step toward data

        return x_t   # x_0 ≈ clean action


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    B, T, A = 4, 16, 2
    fm = FlowMatchingScheduler(num_inference_steps=10)

    a0    = torch.randn(B, T, A)
    noise = torch.randn(B, T, A)
    t     = torch.rand(B)

    # Interpolation
    x_t = fm.interpolate(a0, noise, t)
    assert x_t.shape == (B, T, A), "interpolate shape mismatch"

    # At t=0: x_t ≈ a_0
    x_t0 = fm.interpolate(a0, noise, torch.zeros(B))
    assert torch.allclose(x_t0, a0, atol=1e-6), "At t=0, x_t should equal a_0"

    # At t=1: x_t ≈ noise
    x_t1 = fm.interpolate(a0, noise, torch.ones(B))
    assert torch.allclose(x_t1, noise, atol=1e-6), "At t=1, x_t should equal noise"

    # Target velocity
    target = fm.compute_target(a0, noise)
    assert target.shape == (B, T, A)
    assert torch.allclose(target, noise - a0), "Target should be noise - a_0"

    # Check interpolation consistency: dx/dt ≈ target (finite difference)
    dt = 1e-4
    t0 = torch.full((B,), 0.5)
    t1 = torch.full((B,), 0.5 + dt)
    x_t0_f = fm.interpolate(a0, noise, t0)
    x_t1_f = fm.interpolate(a0, noise, t1)
    numerical_vel = (x_t1_f - x_t0_f) / dt
    assert torch.allclose(numerical_vel, target, atol=2e-3), \
        "Numerical velocity should match analytical target"

    print("All Flow Matching sanity checks passed!")
