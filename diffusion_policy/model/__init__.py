"""
diffusion_policy/model/ — Neural network and diffusion components.
==================================================================

This sub-package contains:

    unet1d.py         — The Conditional 1D Temporal U-Net. This is the shared
                         backbone used by DDPM, DDIM, and Flow Matching. It takes
                         noisy actions, a timestep, and observations, and outputs
                         a prediction (noise for DDPM/DDIM, velocity for FM).

    ddpm.py           — DDPM noise schedule, forward process (adding noise),
                         and reverse sampling loop (iteratively removing noise).

    ddim.py           — DDIM sampling scheduler. Uses the SAME trained model as
                         DDPM but replaces the stochastic reverse process with a
                         deterministic one, enabling fewer inference steps.

    flow_matching.py  — [Phase 5] Flow Matching training loss and ODE-based
                         inference. Reuses the same U-Net architecture.

    ema.py            — Exponential Moving Average helper for model weights.
                         Maintains a smoothed copy of weights for stable inference.

    vision_encoder.py — ResNet-18 visual encoder for visuomotor diffusion policy.
                         Encodes a stack of T_obs RGB frames (96×96×3) into a
                         fixed-size conditioning vector for the 1D temporal U-Net.
"""

from diffusion_policy.model.unet1d        import ConditionalUnet1D
from diffusion_policy.model.ema           import EMA
from diffusion_policy.model.ddpm          import DDPMScheduler
from diffusion_policy.model.ddim          import DDIMScheduler
from diffusion_policy.model.flow_matching import FlowMatchingScheduler
from diffusion_policy.model.vision_encoder import ResNetEncoder

__all__ = [
    "ConditionalUnet1D",
    "EMA",
    "DDPMScheduler",
    "DDIMScheduler",
    "FlowMatchingScheduler",
    "ResNetEncoder",
]
