"""
diffusion_policy/data/ — Data loading and preprocessing.
=========================================================

This sub-package handles everything between "raw demonstration data on disk"
and "ready-to-train (obs, action) tensor batches."

Modules:
    dataset.py       — PushTStateDataset: sliding-window samples from state observations
    image_dataset.py — PushTImageDataset: sliding-window samples from RGB image observations
    normalizer.py    — Min-max normalization to [-1, 1]
"""

from diffusion_policy.data.normalizer     import MinMaxNormalizer
from diffusion_policy.data.dataset        import PushTStateDataset
from diffusion_policy.data.image_dataset  import PushTImageDataset

__all__ = ["MinMaxNormalizer", "PushTStateDataset", "PushTImageDataset"]
