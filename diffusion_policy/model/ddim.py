"""
ddim.py — Denoising Diffusion Implicit Models (DDIM) Scheduler
================================================================

This module implements DDIM sampling, which is an ALTERNATIVE INFERENCE METHOD
that uses the SAME trained DDPM model. No retraining is needed.

Key insight
-----------
DDPM's reverse process is stochastic — each step adds noise (the σ_k · z term).
This means you MUST take all K steps; you can't skip steps because the stochastic
transitions are calibrated for adjacent timesteps.

DDIM reformulates the reverse process as a DETERMINISTIC mapping (when η=0):

    a_{k-1} = √(ᾱ_{k-1}) · â_0  +  √(1 - ᾱ_{k-1}) · ε_θ

where â_0 is the "predicted clean action" computed from the current noisy
action and the model's noise prediction:

    â_0 = (a_k  -  √(1 - ᾱ_k) · ε_θ) / √(ᾱ_k)

Because this is deterministic, the mapping from a_k to a_{k-1} doesn't depend
on the step SIZE — you can jump from timestep 99 directly to timestep 89
(skipping 9 intermediate steps) and get a valid result.

This allows DDIM to use a SUBSEQUENCE of the full K timesteps:
    e.g., [99, 89, 79, 69, 59, 49, 39, 29, 19, 9, 0]  (10 steps instead of 100)

The result is ~10x faster inference with minimal quality loss.

The η parameter
---------------
DDIM has a stochasticity parameter η ∈ [0, 1]:
    - η = 0: fully deterministic (default, recommended for reproducible eval)
    - η = 1: equivalent to DDPM (fully stochastic)
    - 0 < η < 1: partially stochastic

The full DDIM update rule is:

    a_{k-1} = √(ᾱ_{k-1}) · â_0
              + √(1 - ᾱ_{k-1} - σ²) · ε_θ
              + σ · z

where σ = η · √((1 - ᾱ_{k-1})/(1 - ᾱ_k)) · √(β_k)  and  z ~ N(0, I).

Implementation plan (Phase 3):
    - DDIMScheduler class with:
        - __init__: takes the same noise schedule as DDPM, plus ddim_steps and eta
        - compute_timestep_subsequence(): select evenly spaced timesteps
        - step(model_output, timestep, noisy_actions, prev_timestep): one DDIM step
        - full_loop(model, observation): complete DDIM sampling from noise to actions

    The DDIMScheduler is a DROP-IN REPLACEMENT for DDPMScheduler at inference time.
    Training always uses DDPM; only the inference loop changes.

References:
    - Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
    - Chi et al., "Diffusion Policy" (RSS 2023) — uses DDIM for fast inference
"""

# Implementation will be added in Phase 3.
