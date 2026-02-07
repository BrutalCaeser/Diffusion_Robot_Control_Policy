"""
evaluate.py — Evaluation / Rollout Script
==========================================

This script loads a trained checkpoint and runs evaluation rollouts in the
PushT environment to measure policy performance.

Evaluation procedure:
    1. Load trained model checkpoint (uses EMA weights)
    2. Load the normalizer (same stats used during training)
    3. For each evaluation episode (default: 50-100 episodes):
        a. Reset the PushT environment
        b. Receding-horizon control loop:
            i.   Collect observation history (T_obs frames)
            ii.  Run diffusion sampling (DDPM or DDIM) to generate action chunk
            iii. Unnormalize the predicted actions
            iv.  Execute first T_action steps in the environment
            v.   Repeat until episode ends
        c. Record episode reward / success

    4. Report aggregate metrics:
        - Success rate (% episodes where T-block reaches target)
        - Mean reward / score
        - Inference time per step

Usage:
    # Evaluate with DDPM sampling (100 steps):
    python evaluate.py --checkpoint checkpoints/best.pt --sampler ddpm

    # Evaluate with DDIM sampling (10 steps, faster):
    python evaluate.py --checkpoint checkpoints/best.pt --sampler ddim

    # Evaluate with Flow Matching (Phase 5):
    python evaluate.py --checkpoint checkpoints/best_fm.pt --sampler flow

    # Save rollout GIFs:
    python evaluate.py --checkpoint checkpoints/best.pt --save_gifs

The evaluation loop will be implemented in Phase 3.
"""

# Implementation will be added in Phase 3.
