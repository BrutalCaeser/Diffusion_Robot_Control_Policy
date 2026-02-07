"""
normalizer.py — Min-Max Normalization to [-1, 1]
==================================================

This module normalizes observations and actions to the range [-1, 1] using
dataset-wide min/max statistics.

Why normalize?
--------------
Diffusion models assume the data distribution is approximately Gaussian-like
and bounded. The reverse process starts from N(0, I) noise and denoises toward
the data. If the data has a very different scale (e.g., positions in [0, 512],
angles in [0, 2π]), the model struggles because:

1. The noise scale doesn't match the data scale — at the last diffusion step,
   the noisy actions should be ~N(0, I), but if actions range [0, 512], the
   signal-to-noise ratio is completely off.

2. Different dimensions have different scales, so the model must implicitly
   learn to weight them differently. Normalization removes this burden.

3. The MSE loss treats all dimensions equally. If position errors are ~O(100)
   but angle errors are ~O(1), the loss is dominated by position, and the
   model under-optimizes for angle accuracy.

Min-max normalization to [-1, 1] solves all three issues:
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1

    Inverse: x = (x_norm + 1) / 2 * (x_max - x_min) + x_min

The statistics (x_min, x_max) are computed ONCE from the full training dataset
and stored. The same statistics must be used for both training and inference.

Implementation plan (Phase 1):
    - MinMaxNormalizer:
        - fit(data): compute min/max from training data (per dimension)
        - normalize(data): map to [-1, 1]
        - unnormalize(data): map back to original scale
        - state_dict() / load_state_dict(): for saving/loading with checkpoints
"""

# Implementation will be added in Phase 1.
