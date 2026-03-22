"""
flow_matching.py — Flow Matching Training & ODE Inference
==========================================================

[Phase 5 — Stretch Goal / Post-Deadline Extension]

This module implements Flow Matching (FM) as an alternative to DDPM for training
the action generation policy. FM reuses the EXACT SAME U-Net architecture; only
the training loss and inference procedure change.

Conceptual comparison: DDPM vs. Flow Matching
----------------------------------------------
    DDPM:
        - Discrete timesteps k ∈ {0, 1, ..., K-1}
        - Forward process adds noise via a fixed schedule: a_k = √ᾱ_k · a_0 + √(1-ᾱ_k) · ε
        - Model predicts the NOISE ε that was added
        - Reverse process: stochastic chain of K denoising steps
        - Inference: iterate K (or fewer via DDIM) steps to remove noise

    Flow Matching:
        - Continuous time t ∈ [0, 1]
        - Forward process is STRAIGHT-LINE interpolation: x_t = (1-t) · a_0 + t · ε
        - Model predicts the VELOCITY v = dx/dt = ε - a_0 (direction from data to noise)
        - No noise schedule to tune (β, α, ᾱ are gone!)
        - Inference: solve an ODE from t=1 (noise) to t=0 (data) using Euler or RK4

Why Flow Matching is relevant to this project
----------------------------------------------
1. Simpler training: the loss is just MSE(v_θ, target_velocity). No noise schedule,
   no ᾱ computations, no variance terms.

2. Faster inference: straight transport paths → fewer solver steps needed.
   Typically 5-10 Euler steps suffice, compared to 10-100 for DDIM.

3. **Connection to Yashvardhan's thesis:** Flow Matching is the DETERMINISTIC
   (zero-diffusion-coefficient) limit of Schrödinger Bridges. Specifically:
   - FM solves: min_v ∫ E[||v_θ(x_t, t) - u_t||²] dt
   - Schrödinger Bridge adds a diffusion term: dx = v dt + σ dW
   - As σ → 0, the SB solution converges to the FM (OT) solution

   So building FM here creates a direct code-level bridge: add σ back in and
   you're doing Schrödinger Bridge estimation, which is the thesis topic.

Implementation plan (Phase 5):
    - FlowMatchingScheduler class with:
        - interpolate(a_0, noise, t): compute x_t = (1-t)*a_0 + t*noise
        - compute_target(a_0, noise): return velocity target u = noise - a_0
        - euler_step(model, x_t, t, obs, dt): one Euler integration step
        - sample(model, observation, num_steps): full ODE solve from noise to data
        - [Optional] ot_pair(batch_a0, batch_noise): mini-batch OT coupling

References:
    - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
    - Liu et al., "Flow Straight and Fast" (ICLR 2023)
    - Albergo & Vanden-Eijnden, "Stochastic Interpolants" (2023)
"""

# Implementation will be added in Phase 5.
