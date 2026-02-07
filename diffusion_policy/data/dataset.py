"""
dataset.py — PushT Demonstration Dataset with Sliding Window Extraction
=========================================================================

This module loads the expert demonstration dataset (Zarr format) and converts
it into training samples using a sliding window approach.

Dataset format (Zarr)
---------------------
The PushT Zarr dataset contains ~200 expert episodes with this structure:

    root.zarr/
    ├── data/
    │   ├── state     (N, obs_dim)   — observations for all episodes concatenated
    │   └── action    (N, action_dim) — actions for all episodes concatenated
    └── meta/
        └── episode_ends  (num_episodes,) — index where each episode ends

For example, if episode 0 has 150 steps and episode 1 has 200 steps:
    episode_ends = [150, 350, ...]
    state[0:150] belongs to episode 0
    state[150:350] belongs to episode 1

Sliding window extraction
-------------------------
For each timestep t in each episode, we extract a (observation, action) pair:

    obs:    [o_{t-T_obs+1}, ..., o_t]          shape: (T_obs, obs_dim)
    action: [a_t, a_{t+1}, ..., a_{t+T_pred-1}]  shape: (T_pred, action_dim)

    ┌─ episode ──────────────────────────────────────────┐
    │  ... o_{t-1}  o_t  │  a_t  a_{t+1}  ...  a_{t+15} │
    │  ◄── T_obs=2 ──►   │  ◄──── T_pred=16 ────────►   │
    └─────────────────────┴──────────────────────────────┘

At episode boundaries:
    - Beginning: pad observations by repeating the first observation
    - End: pad actions by repeating the last action

This ensures we can extract a sample from every timestep without index errors.

Why sliding window?
-------------------
Each episode is ~300 steps long. Rather than using each episode as one sample
(which would give only ~200 training samples), the sliding window extracts
one sample per timestep, giving ~200 × 300 = ~60,000 training samples.
This is essential for training a diffusion model, which is data-hungry.

Implementation plan (Phase 1):
    - PushTStateDataset(Dataset):
        - __init__: load Zarr, compute per-episode indices, create normalizer
        - __len__: total number of valid sliding window positions
        - __getitem__: extract and normalize one (obs, action) sample
        - get_normalizer(): return the fitted normalizer for use at inference

References:
    - Chi et al., "Diffusion Policy" — dataset format and preprocessing
    - Official repo: github.com/real-stanford/diffusion_policy
"""

# Implementation will be added in Phase 1.
