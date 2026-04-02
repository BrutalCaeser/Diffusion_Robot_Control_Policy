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
"""

import numpy as np
import torch


class MinMaxNormalizer:
    """
    Per-dimension min-max normalizer mapping data to [-1, 1].

    Fitting is done from raw numpy arrays (the full dataset). Normalization and
    unnormalization accept both numpy arrays and PyTorch tensors, returning the
    same type as the input.

    Usage::

        normalizer = MinMaxNormalizer()
        normalizer.fit(all_actions)           # shape (N, action_dim)
        normed   = normalizer.normalize(x)    # [-1, 1]
        unormed  = normalizer.unnormalize(y)  # original scale

        # Save alongside a model checkpoint
        torch.save({'normalizer': normalizer.state_dict(), ...}, path)

        # Reload
        normalizer2 = MinMaxNormalizer()
        normalizer2.load_state_dict(ckpt['normalizer'])
    """

    def __init__(self):
        self._x_min: np.ndarray | None = None   # (D,) per-dimension minimum
        self._x_max: np.ndarray | None = None   # (D,) per-dimension maximum
        self._range: np.ndarray | None = None   # (D,) x_max - x_min (safe: ≥ 1)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> "MinMaxNormalizer":
        """
        Compute per-dimension statistics from training data.

        Args:
            data: array of shape (N, D) or (N, T, D).  Any shape where the
                  last axis is the feature dimension. Statistics are computed
                  over all sample / time positions.

        Returns:
            self, so you can chain: normalizer.fit(data).normalize(x)
        """
        data = np.asarray(data, dtype=np.float64)
        flat = data.reshape(-1, data.shape[-1])   # (N*T, D)

        self._x_min = flat.min(axis=0).astype(np.float32)  # (D,)
        self._x_max = flat.max(axis=0).astype(np.float32)  # (D,)

        # Protect against zero-variance dimensions (e.g. a constant feature).
        # Setting range=1 for those dims means they map to x_norm = -1, which
        # is a stable constant the model will simply learn to ignore.
        raw_range = self._x_max - self._x_min
        self._range = np.where(raw_range < 1e-8, 1.0, raw_range).astype(np.float32)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Normalization / unnormalization
    # ------------------------------------------------------------------

    def normalize(self, data):
        """
        Map data from its original scale to [-1, 1].

        Args:
            data: np.ndarray or torch.Tensor, any shape (..., D) where D
                  matches the dimension fitted.
        Returns:
            Normalized data, same type and shape as input.
        """
        self._check_fitted()
        is_tensor = isinstance(data, torch.Tensor)
        device = data.device if is_tensor else None
        arr = data.cpu().numpy() if is_tensor else np.asarray(data, dtype=np.float32)

        out = 2.0 * (arr - self._x_min) / self._range - 1.0

        if is_tensor:
            return torch.from_numpy(out.astype(np.float32)).to(device)
        return out.astype(np.float32)

    def unnormalize(self, data):
        """
        Map data from [-1, 1] back to its original scale.

        Args:
            data: np.ndarray or torch.Tensor in [-1, 1], any shape (..., D).
        Returns:
            Unnormalized data, same type and shape as input.
        """
        self._check_fitted()
        is_tensor = isinstance(data, torch.Tensor)
        device = data.device if is_tensor else None
        arr = data.cpu().numpy() if is_tensor else np.asarray(data, dtype=np.float32)

        out = (arr + 1.0) / 2.0 * self._range + self._x_min

        if is_tensor:
            return torch.from_numpy(out.astype(np.float32)).to(device)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serializable dictionary of statistics for checkpointing."""
        self._check_fitted()
        return {
            "x_min": self._x_min.tolist(),
            "x_max": self._x_max.tolist(),
        }

    def load_state_dict(self, state: dict) -> "MinMaxNormalizer":
        """Restore statistics from a previously saved state dict."""
        self._x_min = np.array(state["x_min"], dtype=np.float32)
        self._x_max = np.array(state["x_max"], dtype=np.float32)
        raw_range = self._x_max - self._x_min
        self._range = np.where(raw_range < 1e-8, 1.0, raw_range).astype(np.float32)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "MinMaxNormalizer is not fitted yet. "
                "Call fit(data) before normalize/unnormalize."
            )

    def __repr__(self) -> str:
        if not self._fitted:
            return "MinMaxNormalizer(not fitted)"
        return (
            f"MinMaxNormalizer("
            f"x_min={self._x_min.round(3)}, "
            f"x_max={self._x_max.round(3)})"
        )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate PushT actions: (N, 2) in [0, 512]
    data = rng.uniform(low=[0, 0], high=[512, 512], size=(10000, 2)).astype(np.float32)

    norm = MinMaxNormalizer()
    norm.fit(data)
    print(norm)

    normed = norm.normalize(data)
    assert normed.min() >= -1.0 - 1e-6 and normed.max() <= 1.0 + 1e-6, \
        "Normalized values out of [-1, 1]"

    recovered = norm.unnormalize(normed)
    assert np.allclose(data, recovered, atol=1e-5), \
        "Round-trip error too large"

    # Test with torch tensors
    t = torch.from_numpy(data)
    normed_t = norm.normalize(t)
    recovered_t = norm.unnormalize(normed_t)
    assert torch.allclose(t, recovered_t, atol=1e-4), \
        "Torch round-trip error too large"

    print("All normalizer sanity checks passed!")
