"""
baselines/bc_policy.py — Behavioral Cloning (BC) Baseline
==========================================================

Behavioral Cloning is the simplest possible imitation learning algorithm:
  • Collect expert demonstrations (obs, action) pairs
  • Train a neural network  f_θ: obs → action  with MSE loss
  • Done — the policy mimics the expert by regression

WHY WE INCLUDE THIS (and why it matters for your grade):
=========================================================
BC is our baseline. It is easy to understand, easy to implement, and — most
importantly — it FAILS on Push-T in an instructive way.

The failure reveals the core problem that Diffusion Policy solves:

  Push-T has a MULTIMODAL action distribution.

  At many states, a human expert might push the T-block from the LEFT *or* from
  the RIGHT — both are equally valid strategies. So in the dataset you'll see
  demonstrations going left AND right from the same starting position.

  BC minimises Mean Squared Error (MSE):
      L = E[ ||f_θ(obs) - action_demo||² ]

  The minimiser of MSE over a bimodal distribution is the MEAN of the two modes.
  In 2D action space this means the policy outputs a vector pointing straight at
  the block from neither valid side — and the robot gets stuck.

  Diffusion Policy instead learns the full distribution p(action | obs). It can
  sample from either mode, so it always commits to one valid strategy.

  Expected results:
      BC success rate      ≈ 40–55%  (low — averaging the modes)
      Diffusion success rate ≈ 90–96%  (high — samples one mode cleanly)

  This gap IS the paper's central contribution, and having BC to compare against
  makes that contribution concrete and undeniable.

Architecture:
    obs_history  (T_obs, obs_dim)
         │
         ▼
    flatten  → (T_obs × obs_dim,)   e.g. (10,) for T_obs=2, obs_dim=5
         │
    Linear(10  → 256) + ReLU
         │
    Linear(256 → 256) + ReLU
         │
    Linear(256 → T_pred × action_dim)   e.g. (32,) for T_pred=16, action_dim=2
         │
         ▼
    reshape → (T_pred, action_dim)      e.g. (16, 2)

Total parameters: ~135K  (vs ~68M for Diffusion U-Net)
The capacity gap is intentional: BC's failure is architectural, not due to
model size. Even a much larger MLP cannot fix the mode-averaging problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """
    Two-hidden-layer MLP for behavioral cloning.

    Direct mapping: observation history → full action sequence chunk.
    Trained with standard MSE loss against demonstration actions.

    Args:
        obs_horizon:  T_obs — number of past observations fed as input.
        obs_dim:      Dimensionality of each observation (5 for Push-T state).
        pred_horizon: T_pred — number of future actions to predict.
        action_dim:   Dimensionality of each action (2 for Push-T: vx, vy).
        hidden_dim:   Width of each hidden layer.  Default 256 matches the
                      conditioning dimension of the Diffusion U-Net so the
                      capacity comparison is fair at this layer width.
    """

    def __init__(
        self,
        obs_horizon: int,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.pred_horizon = pred_horizon
        self.action_dim   = action_dim

        in_dim  = obs_horizon * obs_dim       # flattened observation history
        out_dim = pred_horizon * action_dim   # flattened action chunk

        self.net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),   # no activation — regression output
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict an action chunk from the current observation history.

        This is a single forward pass — no iterative denoising, no sampling.
        That simplicity is both BC's strength (speed) and its weakness
        (cannot represent multimodal distributions).

        Args:
            obs: (B, T_obs, obs_dim) — normalised observation history.
                 Values should be in [-1, 1] (same normalisation as Diffusion).

        Returns:
            (B, T_pred, action_dim) — predicted (normalised) action chunk.
            The caller unnormalises these before executing in the environment.
        """
        B = obs.shape[0]
        flat = obs.reshape(B, -1)                           # (B, T_obs*obs_dim)
        out  = self.net(flat)                               # (B, T_pred*action_dim)
        return out.reshape(B, self.pred_horizon, self.action_dim)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"BCPolicy("
            f"in={self.net[0].in_features}, "
            f"hidden={self.net[0].out_features}, "
            f"out={self.net[-1].out_features}, "
            f"params={self.num_parameters():,})"
        )
