"""
diffusion_policy/data/ — Data loading and preprocessing.
=========================================================

This sub-package handles everything between "raw demonstration data on disk"
and "ready-to-train (obs, action) tensor batches."

Modules:
    dataset.py     — PushT dataset class with sliding-window sample extraction
    normalizer.py  — Min-max normalization to [-1, 1]
"""

from diffusion_policy.data.normalizer import MinMaxNormalizer
from diffusion_policy.data.dataset    import PushTStateDataset

__all__ = ["MinMaxNormalizer", "PushTStateDataset"]
