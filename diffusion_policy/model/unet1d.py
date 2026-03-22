"""
unet1d.py — Conditional 1D Temporal U-Net
==========================================

This module implements the core neural network for the Diffusion Policy.

Architecture overview
---------------------
The U-Net processes action sequences along their TEMPORAL dimension using 1D
convolutions. This is NOT an image U-Net — the "spatial" dimension is time
steps in the action trajectory.

Input layout (after permutation for Conv1d):
    (Batch, Channels=action_dim, Length=T_pred)
    e.g., (256, 2, 16) for PushT with batch_size=256, 2D actions, 16-step horizon.

The network has three types of conditioning inputs:
    1. The noisy action sequence itself (the main input being denoised)
    2. The diffusion timestep k (or continuous time t for Flow Matching)
    3. The observation history (robot state / environment state)

Conditioning mechanism — FiLM (Feature-wise Linear Modulation):
    Rather than concatenating the conditioning information to the input (which
    would require the first convolution layer to "figure out" which channels
    are signal vs. conditioning), FiLM injects conditioning at EVERY residual
    block by modulating the features AFTER GroupNorm:

        x_out = γ(cond) * GroupNorm(x) + β(cond)

    where γ (scale) and β (bias) are learned linear projections of the
    conditioning vector. This is more expressive than concatenation because
    it can multiplicatively gate features — amplifying relevant channels and
    suppressing irrelevant ones based on the current timestep and observation.

Key design decisions
--------------------
- The timestep embedding accepts BOTH integer and float inputs, so the same
  U-Net works for DDPM (discrete k) and Flow Matching (continuous t ∈ [0,1]).
- The observation encoder is a simple MLP that flattens the observation history
  and projects it to a fixed-size conditioning vector. For state-based PushT
  this is sufficient; for image-based observations, this would be replaced
  with a CNN or Vision Transformer encoder.
- Skip connections follow the standard U-Net pattern: features from each encoder
  level are concatenated with the corresponding decoder level, allowing the
  decoder to recover fine-grained temporal details lost during downsampling.

Components to implement (Phase 2):
    - SinusoidalPosEmb: timestep → embedding vector
    - FiLMConditionedResBlock: the core building block
    - ConditionalUnet1D: the full encoder-decoder architecture

References:
    - Chi et al., "Diffusion Policy" (RSS 2023) — architecture details
    - Ho et al., "DDPM" (NeurIPS 2020) — U-Net backbone design
    - Perez et al., "FiLM" (AAAI 2018) — conditioning mechanism
"""

# Implementation will be added in Phase 2.
