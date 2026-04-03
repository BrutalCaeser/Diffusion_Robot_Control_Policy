"""
image_dataset.py — PushT Image Dataset for Visuomotor Diffusion Policy
=======================================================================

This module provides ``PushTImageDataset``, the image-observation counterpart
to ``PushTStateDataset``.  Instead of returning a low-dimensional state vector
(5-D: agent xy, block xy, angle), it returns a stack of T_obs RGB frames
(each 96×96×3).  These are fed to ``ResNetEncoder`` in the training loop to
produce the conditioning vector for the 1D temporal U-Net.

Why image observations?
-----------------------
Real robots typically lack access to perfect, low-noise state estimates.
Cameras are cheap, ubiquitous, and capture rich scene information.  The
image-based variant of Diffusion Policy (the "visuomotor" policy) is therefore
the version closest to real-world deployment.

The Zarr dataset ``pusht_cchi_v7_replay.zarr`` contains both:
    data/state   (25650, 5)         — low-dimensional state (used by PushTStateDataset)
    data/img     (25650, 96, 96, 3) — rendered RGB frames as float32 [0, 255]

This module uses ``data/img``.

Data pipeline
-------------
    Zarr img  (N, 96, 96, 3) float32 [0, 255]
        │   /255.0
        ▼
    normalised (N, 96, 96, 3) float32 [0, 1]
        │   HWC → CHW
        ▼
    (N, 3, 96, 96)
        │   sliding window  [t-T_obs+1 .. t]
        ▼
    obs_stack  (T_obs, 3, 96, 96)  — left-padded at episode boundaries
        │
    action_normalizer.normalize(actions)
        ▼
    act_norm   (T_pred, 2)

``__getitem__`` returns a dict::

    {
        "obs":    torch.Tensor  (T_obs, 3, H, W)   float32 [0, 1]
        "action": torch.Tensor  (T_pred, action_dim) float32 [-1, 1]
    }

The action normalizer is exposed via ``get_action_normalizer()`` and must be
saved with the model checkpoint for correct unnormalization at inference time.

Note on augmentation
--------------------
The original Diffusion Policy paper applies random crop augmentation during
training (crops a 76×76 patch from the 96×96 frame and resizes back to 96×96).
We do NOT apply augmentation here to keep this dataset class simple and
deterministic.  Augmentation can be added as a ``transform`` argument if needed.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.data.normalizer import MinMaxNormalizer

logger = logging.getLogger(__name__)


class PushTImageDataset(Dataset):
    """
    PyTorch Dataset for PushT with image observations.

    Each sample is a sliding-window slice of a demonstration episode:
    - ``obs``    — T_obs consecutive RGB frames (already divided by 255)
    - ``action`` — T_pred consecutive actions normalised to [-1, 1]

    Args:
        dataset_path:   Path to ``pusht_cchi_v7_replay.zarr``.
        obs_horizon:    T_obs — number of consecutive frames per sample (default 2).
        pred_horizon:   T_pred — number of future actions per sample (default 16).
        transform:      Optional callable applied to each (T_obs, 3, H, W) tensor.
                        Use this for data augmentation (random crop, color jitter …).
    """

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        transform=None,
    ) -> None:
        super().__init__()
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}.\n"
                "Download with:\n"
                "  wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip\n"
                "  unzip pusht.zip"
            )

        # ------------------------------------------------------------------ #
        # Load raw data from Zarr                                              #
        # ------------------------------------------------------------------ #
        import zarr
        root = zarr.open(str(dataset_path), mode="r")

        # Images: (N, 96, 96, 3) float32 in [0, 255]
        imgs_raw: np.ndarray = root["data"]["img"][:]         # load all into RAM
        # Actions: (N, action_dim)
        all_actions: np.ndarray = root["data"]["action"][:]
        # Episode boundary indices
        episode_ends: np.ndarray = root["meta"]["episode_ends"][:]

        self.obs_horizon   = obs_horizon
        self.pred_horizon  = pred_horizon
        self.transform     = transform
        self.image_height  = imgs_raw.shape[1]   # 96
        self.image_width   = imgs_raw.shape[2]   # 96
        self.action_dim    = all_actions.shape[1] # 2

        # ------------------------------------------------------------------ #
        # Pre-process images: [0,255] float32 → [0,1] float32 + HWC→CHW      #
        # ------------------------------------------------------------------ #
        # Divide by 255 once and store as float32 in (N, 3, H, W) layout.
        # Keeping images in memory (25650 × 3 × 96 × 96 × 4 bytes ≈ 284 MB)
        # is fine for a single machine.
        imgs_01 = (imgs_raw / 255.0).astype(np.float32)       # (N, H, W, 3)
        # HWC → CHW
        self._imgs: np.ndarray = imgs_01.transpose(0, 3, 1, 2)  # (N, 3, H, W)

        logger.info(
            "PushTImageDataset: loaded %d frames, image shape %s",
            len(self._imgs), self._imgs.shape[1:],
        )

        # ------------------------------------------------------------------ #
        # Fit action normalizer on the full dataset                            #
        # ------------------------------------------------------------------ #
        # We normalise actions (not images — the CNN handles image normalisation).
        self.action_normalizer = MinMaxNormalizer()
        self.action_normalizer.fit(all_actions)

        # Store raw actions for __getitem__
        self._actions: np.ndarray = all_actions.astype(np.float32)
        self._episode_ends: np.ndarray = episode_ends.astype(np.int64)

        # ------------------------------------------------------------------ #
        # Build sliding-window index table                                     #
        # ------------------------------------------------------------------ #
        # Each entry: (ep_start, ep_end, t)
        self._indices: list = []
        ep_start = 0
        for ep_end in episode_ends:
            for t in range(ep_start, int(ep_end)):
                self._indices.append((ep_start, int(ep_end), t))
            ep_start = int(ep_end)

        logger.info(
            "PushTImageDataset: %d episodes, %d samples (obs_horizon=%d, pred_horizon=%d)",
            len(episode_ends), len(self._indices), obs_horizon, pred_horizon,
        )

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        ep_start, ep_end, t = self._indices[idx]

        # ---- Observation image window [t - T_obs + 1, t] ---- #
        obs_start = max(ep_start, t - self.obs_horizon + 1)
        obs_slice = self._imgs[obs_start : t + 1]           # (≤T_obs, 3, H, W)

        # Left-pad with the first frame if near episode start
        if obs_slice.shape[0] < self.obs_horizon:
            pad_len   = self.obs_horizon - obs_slice.shape[0]
            pad       = np.tile(obs_slice[:1], (pad_len, 1, 1, 1))  # repeat
            obs_slice = np.concatenate([pad, obs_slice], axis=0)

        # obs_slice: (T_obs, 3, H, W) float32 in [0, 1]

        # ---- Action window [t, t + T_pred) ---- #
        act_end   = min(ep_end, t + self.pred_horizon)
        act_slice = self._actions[t : act_end]              # (≤T_pred, 2)

        if act_slice.shape[0] < self.pred_horizon:
            pad_len   = self.pred_horizon - act_slice.shape[0]
            pad       = np.tile(act_slice[-1:], (pad_len, 1))
            act_slice = np.concatenate([act_slice, pad], axis=0)

        # ---- Normalise actions ---- #
        act_norm = self.action_normalizer.normalize(act_slice)  # (T_pred, 2) [-1,1]

        # ---- Convert to tensors ---- #
        obs_tensor = torch.from_numpy(obs_slice.copy())          # (T_obs, 3, H, W)
        act_tensor = torch.from_numpy(act_norm)                  # (T_pred, 2)

        # ---- Optional transform (e.g. random crop augmentation) ---- #
        if self.transform is not None:
            obs_tensor = self.transform(obs_tensor)

        return {"obs": obs_tensor, "action": act_tensor}

    # ---------------------------------------------------------------------- #
    # Accessor for normalizer (needed at inference time)                       #
    # ---------------------------------------------------------------------- #

    def get_action_normalizer(self) -> MinMaxNormalizer:
        """
        Return the fitted action normalizer.

        Save this alongside the model checkpoint and restore at inference time
        so that predicted actions are correctly unnormalized.
        """
        return self.action_normalizer

    def __repr__(self) -> str:
        return (
            f"PushTImageDataset("
            f"samples={len(self)}, "
            f"episodes={len(self._episode_ends)}, "
            f"obs_horizon={self.obs_horizon}, "
            f"pred_horizon={self.pred_horizon}, "
            f"image_shape=(3,{self.image_height},{self.image_width}))"
        )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys
    ZARR_PATH = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Volumes/Crucial_X9/Projects/ML_6140/pusht/pusht_cchi_v7_replay.zarr"
    )
    ds = PushTImageDataset(ZARR_PATH, obs_horizon=2, pred_horizon=16)
    print(ds)

    sample = ds[0]
    print(f"  obs    shape: {sample['obs'].shape}   (T_obs=2, 3, 96, 96)")
    print(f"  action shape: {sample['action'].shape} (T_pred=16, 2)")
    print(f"  obs    range: [{sample['obs'].min():.3f}, {sample['obs'].max():.3f}]  (should be [0,1])")
    print(f"  action range: [{sample['action'].min():.3f}, {sample['action'].max():.3f}]  (should be ≈[-1,1])")

    # Check over 500 random samples
    idx = torch.randperm(len(ds))[:500]
    obs_min, obs_max = float('inf'), float('-inf')
    act_min, act_max = float('inf'), float('-inf')
    for i in idx:
        s = ds[int(i)]
        obs_min = min(obs_min, s['obs'].min().item())
        obs_max = max(obs_max, s['obs'].max().item())
        act_min = min(act_min, s['action'].min().item())
        act_max = max(act_max, s['action'].max().item())
    print(f"\n500-sample scan:  obs=[{obs_min:.4f},{obs_max:.4f}]  act=[{act_min:.4f},{act_max:.4f}]")
    print("PushTImageDataset sanity check passed!")
