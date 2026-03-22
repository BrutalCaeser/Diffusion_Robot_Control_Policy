"""
ema.py — Exponential Moving Average (EMA) for Model Weights
=============================================================

This module provides an EMA helper that maintains a shadow copy of the model's
parameters, updated as an exponential moving average of the training weights.

What is EMA?
------------
During training, model weights θ_train are updated by the optimizer at every
step. These updates are inherently noisy — each gradient is computed from a
random mini-batch and may push weights in slightly wrong directions.

EMA maintains a separate set of weights θ_ema that smooths out this noise:

    θ_ema ← decay · θ_ema + (1 - decay) · θ_train

With decay = 0.995, the EMA weights are approximately a weighted average of
the last 1/(1 - 0.995) = 200 training iterations. Recent weights contribute
more, older weights exponentially less.

Why use EMA for Diffusion Policy?
---------------------------------
Diffusion models are especially sensitive to weight stability because the
denoising process is ITERATIVE — small errors in the predicted noise compound
over multiple reverse steps. The EMA weights produce more consistent noise
predictions, leading to:
    - Higher quality generated action trajectories
    - More stable evaluation metrics across runs
    - Better generalization (less overfitting to recent batches)

CRITICAL: The EMA model is used ONLY for inference. Training always updates
the original model weights. The typical workflow:

    1. optimizer.step()           # update θ_train
    2. ema.update(model)          # update θ_ema ← decay * θ_ema + (1-decay) * θ_train
    3. ...
    4. ema.apply(model)           # temporarily replace θ_train with θ_ema
    5. evaluate(model)            # inference with smooth weights
    6. ema.restore(model)         # put θ_train back for continued training

Implementation plan (Phase 2):
    - EMA class with:
        - __init__(model, decay): register shadow params
        - update(model): one EMA step
        - apply(model): swap in EMA weights
        - restore(model): swap back training weights
        - state_dict() / load_state_dict(): for checkpointing

References:
    - Polyak & Juditsky (1992) — original averaging technique
    - Ho et al., "DDPM" (NeurIPS 2020) — uses EMA for diffusion models
"""

# Implementation will be added in Phase 2.
