"""
unet1d.py — Conditional 1D Temporal U-Net
==========================================

This module implements the core neural network for the Diffusion Policy:
a **U-Net that operates on 1D action sequences** conditioned on both the
diffusion timestep and the robot's observation history.

Architecture overview
---------------------
Think of the action trajectory (T_pred, action_dim) as a 1D "audio signal"
with ``action_dim`` channels and ``T_pred`` time positions.  The U-Net is an
encoder-decoder that processes this signal at multiple temporal resolutions,
exactly like a WaveNet-style architecture, but with skip connections.

The three conditioning inputs are:
    1. **Noisy actions** — the primary 1D signal being denoised (the input).
    2. **Diffusion timestep k** — tells the model "how noisy" the input is.
       Encoded via sinusoidal positional embedding + 2-layer MLP.
    3. **Observation history** — the robot's state (last T_obs frames).
       Encoded via a small MLP → conditioning vector.

FiLM conditioning (Feature-wise Linear Modulation)
----------------------------------------------------
At every residual block the conditioning vector modulates the intermediate
features via an affine transform applied AFTER GroupNorm:

    x_out = γ(cond) * GroupNorm(x) + β(cond)

where γ (scale) and β (bias) are learned linear projections of the combined
conditioning vector (timestep_emb ‖ obs_emb).  This is more expressive than
simple concatenation because it can *gate* features multiplicatively.

Tensor shape convention
-----------------------
PyTorch Conv1d expects (Batch, Channels, Length).
We receive actions as (B, T_pred, action_dim) and immediately permute to
(B, action_dim, T_pred) before processing.  The final output is permuted back.

References
----------
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  RSS 2023 — architecture design
- Ho et al., "DDPM" NeurIPS 2020 — U-Net backbone idea
- Perez et al., "FiLM" AAAI 2018 — conditioning mechanism
- Vaswani et al., "Attention Is All You Need" — sinusoidal embeddings
"""

from __future__ import annotations

import math
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==============================================================================
# Building Blocks
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Converts a scalar timestep k (or continuous t ∈ [0,1] for Flow Matching)
    into a dense vector representation using sine and cosine at different
    frequencies — exactly the same idea as Transformer positional encodings.

    For a dimension index ``i`` in [0, dim//2):

        emb[2i]   = sin(k / 10000^(2i/dim))
        emb[2i+1] = cos(k / 10000^(2i/dim))

    Low-frequency components capture coarse information (is this near the
    start or end of diffusion?); high-frequency components capture fine
    distinctions between adjacent timesteps.

    This representation generalises to BOTH integer DDPM timesteps and
    continuous-time Flow Matching values without any code change.

    Args:
        dim: Output embedding dimensionality.  Must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, f"SinusoidalPosEmb dim must be even, got {dim}"
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Timestep tensor, shape (B,).  May be int (DDPM) or float (FM).

        Returns:
            Embedding tensor, shape (B, dim).
        """
        device = x.device
        half = self.dim // 2
        # log-space frequency: 10000^(2i/dim) for i in [0, half)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )                                                       # (half,)
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)    # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)       # (B, dim)
        return emb


class Conv1dBlock(nn.Module):
    """
    Conv1d → GroupNorm → Mish activation.

    This is the atomic building block used inside every residual block.
    GroupNorm normalises features within a single sample (independent of
    batch size), making it well-suited for small batches typical in RL/robot
    learning tasks.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Convolutional kernel size (default 5).
        n_groups:     Number of groups for GroupNorm (must divide out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # same-padding to preserve temporal length
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, T) → (B, out_channels, T)"""
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    1D Residual block with FiLM conditioning.

    Processing pipeline::

        x ──► Conv1dBlock ──► FiLM(cond) ──► Conv1dBlock ──► + ──► output
        │                                                      ▲
        └──────────────────── residual_proj ───────────────────┘

    The FiLM layer applies after the first Conv1dBlock:

        out = γ(cond) * block1(x)  +  β(cond)

    where γ, β are learned linear projections of the conditioning vector.
    This modulates *every channel* of the intermediate feature map, allowing
    the model to selectively amplify or suppress features based on the current
    timestep and observation.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        cond_dim:     Dimensionality of the conditioning vector.
        kernel_size:  Conv kernel size (default 5).
        n_groups:     GroupNorm groups (default 8).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.block1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.block2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)

        # FiLM projection: cond_dim → 2 * out_channels (scale γ and bias β)
        self.film_proj = nn.Linear(cond_dim, 2 * out_channels)

        # 1×1 conv to match channels on the residual path (if necessary)
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Input features, shape (B, in_channels, T).
            cond: Conditioning vector, shape (B, cond_dim).

        Returns:
            Output features, shape (B, out_channels, T).
        """
        residual = self.residual_proj(x)        # (B, out_channels, T)

        out = self.block1(x)                    # (B, out_channels, T)

        # Compute FiLM scale and bias from conditioning vector
        film = self.film_proj(cond)             # (B, 2 * out_channels)
        gamma, beta = film.chunk(2, dim=-1)     # each (B, out_channels)
        # Broadcast over temporal dimension
        gamma = gamma.unsqueeze(-1)             # (B, out_channels, 1)
        beta  = beta.unsqueeze(-1)              # (B, out_channels, 1)

        out = gamma * out + beta                # FiLM modulation
        out = self.block2(out)                  # (B, out_channels, T)

        return out + residual


# ==============================================================================
# Down / Upsample helpers
# ==============================================================================

class Downsample1d(nn.Module):
    """Stride-2 Conv1d to halve temporal resolution. Channels stay the same."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, C, T//2)"""
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed Conv1d to double temporal resolution. Channels stay the same."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, C, T*2)"""
        return self.conv(x)


# ==============================================================================
# Full U-Net
# ==============================================================================

class ConditionalUnet1D(nn.Module):
    """
    Conditional 1D Temporal U-Net for Diffusion Policy.

    This network takes a *noisy* action sequence and predicts the noise that
    was added to it (ε-prediction, standard DDPM objective).  For Flow
    Matching, it predicts the velocity field instead — no code change needed,
    only the loss function differs.

    Architecture (for action_dim=2, down_dims=[256,512,1024], T_pred=16):

        Input:  (B, 16, 2)  ← noisy action sequence
        Permute: (B, 2, 16)

        ── Encoder ──────────────────────────────────────────────
        Level 0: ResBlock(2→256) + ResBlock(256→256)     T=16  → skip0
                 Downsample2x                             T=8
        Level 1: ResBlock(256→512) + ResBlock(512→512)   T=8   → skip1
                 Downsample2x                             T=4
        Level 2: ResBlock(512→1024) + ResBlock(1024→1024) T=4  → skip2

        ── Bottleneck ───────────────────────────────────────────
        Mid1: ResBlock(1024→1024)   T=4
        Mid2: ResBlock(1024→1024)   T=4

        ── Decoder ──────────────────────────────────────────────
        cat(mid, skip2) = (B, 2048, 4)
        Up 0: ResBlock(2048→512) + ResBlock(512→512)     T=4
              Upsample2x                                  T=8
        cat(x, skip1) = (B, 1024, 8)
        Up 1: ResBlock(1024→256) + ResBlock(256→256)     T=8
              Upsample2x                                  T=16

        ── Output ───────────────────────────────────────────────
        Conv1dBlock(256→256) + Conv1d(256→2)             T=16

        Permute: (B, 16, 2)
        Output: (B, 16, 2)  ← predicted noise (or velocity)

    Conditioning:
        - Timestep k encoded via SinusoidalPosEmb(256) → MLP → (B, 256)
        - Obs (T_obs × obs_dim) flattened → MLP → (B, 256)
        - cond = cat([timestep_emb, obs_emb]) → (B, 512)
        - cond injected into EVERY ResBlock via FiLM

    Args:
        action_dim:               Dimensionality of actions (2 for PushT).
        obs_horizon:              T_obs — number of obs frames per sample.
        obs_dim:                  Dimensionality of each observation frame (5).
        diffusion_step_embed_dim: Timestep embedding dim (default 256).
        down_dims:                Channel counts per U-Net level (default (256,512,1024)).
        cond_dim:                 Obs encoder output dim (default 256).
        kernel_size:              Conv1d kernel size in residual blocks (default 5).
        n_groups:                 GroupNorm group count (default 8).
    """

    def __init__(
        self,
        action_dim: int,
        obs_horizon: int,
        obs_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        cond_dim: int = 256,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Conditioning encoders                                               #
        # ------------------------------------------------------------------ #
        obs_feature_dim = obs_horizon * obs_dim   # e.g. 2*5 = 10

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feature_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Combined conditioning dimensionality
        total_cond_dim = diffusion_step_embed_dim + cond_dim   # 512

        # ------------------------------------------------------------------ #
        # U-Net parameters                                                    #
        # ------------------------------------------------------------------ #
        # all_dims: channels at each level, starting from the input
        # e.g. [2, 256, 512, 1024] for action_dim=2, down_dims=(256,512,1024)
        all_dims = (action_dim,) + tuple(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        # [(2, 256), (256, 512), (512, 1024)]

        num_levels = len(in_out)

        # ------------------------------------------------------------------ #
        # Encoder                                                             #
        # ------------------------------------------------------------------ #
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind == num_levels - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, total_cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, total_cond_dim, kernel_size, n_groups),
                nn.Identity() if is_last else Downsample1d(dim_out),
            ]))

        # ------------------------------------------------------------------ #
        # Bottleneck                                                          #
        # ------------------------------------------------------------------ #
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, total_cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, total_cond_dim, kernel_size, n_groups),
        ])

        # ------------------------------------------------------------------ #
        # Decoder                                                             #
        # ------------------------------------------------------------------ #
        # Decode using reversed in_out pairs starting from index 1
        # (skip the first pair whose skip connection has the raw input scale)
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # dim_out is the larger dimension coming from the skip
            # input to ResBlock = dim_out (from previous level) + dim_out (skip)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, total_cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, total_cond_dim, kernel_size, n_groups),
                Upsample1d(dim_in),
            ]))

        # ------------------------------------------------------------------ #
        # Output                                                              #
        # ------------------------------------------------------------------ #
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

        logger.info(
            "ConditionalUnet1D: action_dim=%d, obs_horizon=%d, obs_dim=%d, "
            "down_dims=%s, total_cond_dim=%d, params=%.2fM",
            action_dim, obs_horizon, obs_dim,
            str(down_dims), total_cond_dim,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the conditional U-Net.

        Args:
            noisy_actions: Shape (B, T_pred, action_dim).
                           The partially-denoised action sequence at step k.
            timestep:      Shape (B,).
                           Diffusion step k ∈ {0,...,K-1} (int) or
                           continuous t ∈ [0,1] (float for Flow Matching).
            obs:           Shape (B, T_obs, obs_dim).
                           Observation history (normalized to [-1, 1]).

        Returns:
            noise_pred:    Shape (B, T_pred, action_dim).
                           Predicted noise ε (DDPM) or velocity v (Flow Matching).
        """
        B = noisy_actions.shape[0]

        # ---- Permute: (B, T_pred, action_dim) → (B, action_dim, T_pred)
        x = noisy_actions.permute(0, 2, 1)              # (B, action_dim, T_pred)

        # ---- Encode conditioning ------------------------------------------
        # Flatten obs history: (B, T_obs, obs_dim) → (B, T_obs*obs_dim)
        obs_emb  = self.obs_encoder(obs.reshape(B, -1)) # (B, cond_dim)
        t_emb    = self.timestep_encoder(timestep)       # (B, diffusion_step_embed_dim)
        cond     = torch.cat([t_emb, obs_emb], dim=-1)  # (B, total_cond_dim)

        # ---- Encoder (save skip connections before downsampling) ----------
        skips: list[torch.Tensor] = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            skips.append(x)    # save skip before spatially compressing
            x = downsample(x)

        # ---- Bottleneck --------------------------------------------------
        for mid in self.mid_modules:
            x = mid(x, cond)

        # ---- Decoder (consume skips in LIFO order) -----------------------
        for resnet1, resnet2, upsample in self.up_modules:
            skip = skips.pop()                              # matching encoder level
            # Handle potential ±1 size mismatch from integer division
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1])
            x = torch.cat([x, skip], dim=1)                # concat along channel dim
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        # ---- Output projection -------------------------------------------
        x = self.final_conv(x)                             # (B, action_dim, T_pred)

        # ---- Permute back: (B, action_dim, T_pred) → (B, T_pred, action_dim)
        return x.permute(0, 2, 1)

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Quick sanity check — run this file directly to verify shapes
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    B, T_pred, action_dim = 4, 16, 2
    T_obs, obs_dim = 2, 5

    model = ConditionalUnet1D(
        action_dim=action_dim,
        obs_horizon=T_obs,
        obs_dim=obs_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        cond_dim=256,
        kernel_size=5,
        n_groups=8,
    )

    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    noisy_actions = torch.randn(B, T_pred, action_dim)
    timestep      = torch.randint(0, 100, (B,))
    obs           = torch.randn(B, T_obs, obs_dim)

    with torch.no_grad():
        out = model(noisy_actions, timestep, obs)

    assert out.shape == (B, T_pred, action_dim), \
        f"Expected ({B}, {T_pred}, {action_dim}), got {out.shape}"
    print(f"Forward pass OK: {noisy_actions.shape} → {out.shape}")

    # Test with float timestep (Flow Matching)
    t_float = torch.rand(B)
    with torch.no_grad():
        out_fm = model(noisy_actions, t_float, obs)
    assert out_fm.shape == (B, T_pred, action_dim)
    print(f"Flow Matching timestep (float) OK: {t_float.shape}")

    print("All unet1d sanity checks passed!")
