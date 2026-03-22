"""
ddpm.py — Denoising Diffusion Probabilistic Model (DDPM) Scheduler
====================================================================

This module handles everything related to the DDPM noise process:

1. NOISE SCHEDULE — computing β_k, α_k, and ᾱ_k
2. FORWARD PROCESS — adding noise to clean actions (used during training)
3. REVERSE PROCESS — iteratively denoising to generate clean actions (inference)

Mathematical background
-----------------------
The forward (noising) process is a Markov chain that gradually adds Gaussian
noise to the data over K steps:

    q(a_k | a_{k-1}) = N(a_k; √(1 - β_k) · a_{k-1},  β_k · I)

By the reparameterization trick, we can jump directly from clean data a_0 to
any noise level k in one step:

    q(a_k | a_0) = N(a_k; √(ᾱ_k) · a_0,  (1 - ᾱ_k) · I)

    ⟹  a_k = √(ᾱ_k) · a_0 + √(1 - ᾱ_k) · ε,    ε ~ N(0, I)

where:
    α_k = 1 - β_k
    ᾱ_k = ∏_{i=1}^{k} α_i   (cumulative product — signal retention ratio)

The reverse process (denoising) inverts this:

    p_θ(a_{k-1} | a_k) = N(a_{k-1}; μ_θ(a_k, k),  σ²_k · I)

where the mean μ_θ is computed from the model's noise prediction ε_θ:

    μ_θ = (1 / √α_k) · (a_k  -  (β_k / √(1 - ᾱ_k)) · ε_θ(a_k, k, obs))

and σ²_k = β_k  (the simplest variance choice from Ho et al., 2020).

Implementation plan (Phase 2):
    - DDPMScheduler class with:
        - __init__: compute and store β, α, ᾱ, √ᾱ, √(1-ᾱ) as registered buffers
        - add_noise(clean_actions, noise, timesteps): forward process
        - step(model_output, timestep, noisy_actions): one reverse step
        - sample_timesteps(batch_size): uniform random timestep sampling

References:
    - Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
    - Chi et al., "Diffusion Policy" (RSS 2023) — application to action sequences
"""

# Implementation will be added in Phase 2.
