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
    │   ├── state     (N, obs_dim)    — observations for all episodes concatenated
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

    obs:    [o_{t-T_obs+1}, ..., o_t]           shape: (T_obs, obs_dim)
    action: [a_t, a_{t+1}, ..., a_{t+T_pred-1}] shape: (T_pred, action_dim)

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

References:
    - Chi et al., "Diffusion Policy" — dataset format and preprocessing
    - Official repo: github.com/real-stanford/diffusion_policy
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from diffusion_policy.data.normalizer import MinMaxNormalizer


class PushTStateDataset(Dataset):
    """
    PyTorch Dataset for state-based PushT expert demonstrations.

    Loads the full Zarr dataset into memory at construction time, fits
    per-dimension min-max normalizers, and exposes one sliding-window
    (obs, action) sample per __getitem__ call.

    Args:
        dataset_path: path to the ``*.zarr`` directory.
        obs_horizon:  T_obs — number of past observation frames to include.
        pred_horizon: T_pred — number of future action steps to predict.

    Returns (per sample):
        A dict with keys:
            "obs":    FloatTensor of shape (T_obs, obs_dim), in [-1, 1]
            "action": FloatTensor of shape (T_pred, action_dim), in [-1, 1]
    """

    def __init__(
        self,
        dataset_path: str | Path,
        obs_horizon: int,
        pred_horizon: int,
    ) -> None:
        super().__init__()
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}.\n"
                "Download the PushT expert demonstration dataset with:\n"
                "  pip install gdown\n"
                "  gdown --fuzzy 'https://drive.google.com/file/d/1ALQyEgMoVPBFxDgMVHWDCekqEFBQfGqs/view' -O pusht.zip\n"
                "  unzip pusht.zip -d data/\n"
                "Or clone the dataset from the diffusion_policy repo:\n"
                "  git clone https://github.com/real-stanford/diffusion_policy\n"
                "  cp -r diffusion_policy/data/pusht_cchi_v7_replay.zarr <dataset_path>"
            )

        # ------------------------------------------------------------------ #
        # Load raw data from Zarr                                             #
        # ------------------------------------------------------------------ #
        import zarr
        root = zarr.open(str(dataset_path), mode="r")

        # All observations across all episodes, concatenated
        # shape: (N_total, obs_dim)
        all_obs: np.ndarray = root["data"]["state"][:]
        # shape: (N_total, action_dim)
        all_actions: np.ndarray = root["data"]["action"][:]
        # Cumulative episode end indices, shape: (num_episodes,)
        episode_ends: np.ndarray = root["meta"]["episode_ends"][:]

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # ------------------------------------------------------------------ #
        # Fit normalizers on the full dataset                                 #
        # ------------------------------------------------------------------ #
        self.obs_normalizer = MinMaxNormalizer()
        self.obs_normalizer.fit(all_obs)

        self.action_normalizer = MinMaxNormalizer()
        self.action_normalizer.fit(all_actions)

        # ------------------------------------------------------------------ #
        # Normalize and cache data in memory                                  #
        # ------------------------------------------------------------------ #
        # We store raw (unnormalized) data and normalize in __getitem__ so
        # that the normalizer state_dict stays the ground truth for inference.
        self._obs: np.ndarray = all_obs.astype(np.float32)
        self._actions: np.ndarray = all_actions.astype(np.float32)
        self._episode_ends: np.ndarray = episode_ends.astype(np.int64)

        # ------------------------------------------------------------------ #
        # Build sliding-window index table                                    #
        # ------------------------------------------------------------------ #
        # Each entry: (episode_start, episode_end, t)  — t is the current
        # timestep whose observation frame we want to predict from.
        self._indices: list[tuple[int, int, int]] = []
        ep_start = 0
        for ep_end in episode_ends:
            for t in range(ep_start, ep_end):
                self._indices.append((ep_start, int(ep_end), t))
            ep_start = int(ep_end)

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_start, ep_end, t = self._indices[idx]

        # ---- Observation window [t - T_obs + 1, t] ---- #
        obs_start = max(ep_start, t - self.obs_horizon + 1)
        obs_slice = self._obs[obs_start : t + 1]         # (≤T_obs, obs_dim)

        # Left-pad with the first frame if near the start of an episode
        if obs_slice.shape[0] < self.obs_horizon:
            pad_len = self.obs_horizon - obs_slice.shape[0]
            pad = np.tile(obs_slice[:1], (pad_len, 1))   # repeat first frame
            obs_slice = np.concatenate([pad, obs_slice], axis=0)

        # ---- Action window [t, t + T_pred) ---- #
        act_end = min(ep_end, t + self.pred_horizon)
        act_slice = self._actions[t : act_end]            # (≤T_pred, action_dim)

        # Right-pad with the last frame if near the end of an episode
        if act_slice.shape[0] < self.pred_horizon:
            pad_len = self.pred_horizon - act_slice.shape[0]
            pad = np.tile(act_slice[-1:], (pad_len, 1))  # repeat last frame
            act_slice = np.concatenate([act_slice, pad], axis=0)

        # ---- Normalize ---- #
        obs_norm = self.obs_normalizer.normalize(obs_slice)       # (T_obs, obs_dim)
        act_norm = self.action_normalizer.normalize(act_slice)    # (T_pred, action_dim)

        return {
            "obs":    torch.from_numpy(obs_norm),
            "action": torch.from_numpy(act_norm),
        }

    # ---------------------------------------------------------------------- #
    # Accessor for normalizer (needed at inference time)                      #
    # ---------------------------------------------------------------------- #

    def get_normalizers(self) -> tuple[MinMaxNormalizer, MinMaxNormalizer]:
        """
        Return the fitted (obs_normalizer, action_normalizer) pair.

        These must be saved alongside the model checkpoint and restored at
        inference time so that predictions are unnormalized with the same
        statistics used during training.

        Returns:
            (obs_normalizer, action_normalizer)
        """
        return self.obs_normalizer, self.action_normalizer

    def __repr__(self) -> str:
        ep_count = len(self._episode_ends)
        return (
            f"PushTStateDataset("
            f"samples={len(self)}, "
            f"episodes={ep_count}, "
            f"obs_horizon={self.obs_horizon}, "
            f"pred_horizon={self.pred_horizon})"
        )


# ==============================================================================
# Quick sanity check — run directly to verify dataset loads correctly
# ==============================================================================

if __name__ == "__main__":
    import sys

    dataset_path = "data/pusht_cchi_v7_replay.zarr"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    print(f"Loading dataset from: {dataset_path}")
    ds = PushTStateDataset(dataset_path, obs_horizon=2, pred_horizon=16)
    print(ds)

    # Check shapes
    sample = ds[0]
    print(f"  obs shape:    {sample['obs'].shape}")    # (2, 5)
    print(f"  action shape: {sample['action'].shape}") # (16, 2)

    # Check normalization range
    print(f"  obs min/max:    [{sample['obs'].min():.3f}, {sample['obs'].max():.3f}]")
    print(f"  action min/max: [{sample['action'].min():.3f}, {sample['action'].max():.3f}]")

    # Check a boundary sample (first timestep of the dataset)
    sample0 = ds[0]
    assert sample0["obs"].shape == (2, 5)
    assert sample0["action"].shape == (16, 2)
    assert sample0["obs"].min() >= -1.0 - 1e-4
    assert sample0["obs"].max() <= 1.0 + 1e-4

    print("All dataset sanity checks passed!")
