"""
tests/test_normalizer.py — Unit tests for MinMaxNormalizer.

Run with:  pytest tests/test_normalizer.py -v
"""

import numpy as np
import pytest
import torch

from diffusion_policy.data.normalizer import MinMaxNormalizer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fitted_normalizer(rng):
    data = rng.uniform(low=[0.0, 0.0, -np.pi], high=[512.0, 512.0, np.pi],
                       size=(5000, 3)).astype(np.float32)
    norm = MinMaxNormalizer()
    norm.fit(data)
    return norm, data


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMinMaxNormalizerFit:
    def test_fit_stores_stats(self, rng):
        data = rng.uniform(0, 10, (100, 4)).astype(np.float32)
        norm = MinMaxNormalizer()
        norm.fit(data)
        assert norm._fitted
        assert norm._x_min is not None
        assert norm._x_max is not None
        assert norm._x_min.shape == (4,)
        assert norm._x_max.shape == (4,)

    def test_fit_returns_self(self, rng):
        data = rng.uniform(0, 1, (50, 2)).astype(np.float32)
        norm = MinMaxNormalizer()
        result = norm.fit(data)
        assert result is norm   # supports chaining

    def test_fit_accepts_3d_input(self, rng):
        """fit() should work with (N, T, D) shaped inputs."""
        data = rng.uniform(0, 1, (100, 16, 2)).astype(np.float32)
        norm = MinMaxNormalizer()
        norm.fit(data)                  # should not raise
        assert norm._x_min.shape == (2,)

    def test_zero_variance_dimension(self):
        """Dimensions with zero variance should not cause division by zero."""
        data = np.ones((100, 3), dtype=np.float32)
        data[:, 0] = np.linspace(0, 1, 100)   # dim 0 has variance
        # dim 1 and 2 are constant → zero variance
        norm = MinMaxNormalizer()
        norm.fit(data)
        out = norm.normalize(data)
        assert np.all(np.isfinite(out))


class TestMinMaxNormalizerNormalize:
    def test_output_in_range(self, fitted_normalizer):
        norm, data = fitted_normalizer
        out = norm.normalize(data)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <=  1.0 + 1e-5

    def test_round_trip_numpy(self, fitted_normalizer):
        norm, data = fitted_normalizer
        recovered = norm.unnormalize(norm.normalize(data))
        np.testing.assert_allclose(recovered, data, atol=1e-5)

    def test_round_trip_torch(self, fitted_normalizer):
        norm, data = fitted_normalizer
        t = torch.from_numpy(data)
        recovered = norm.unnormalize(norm.normalize(t))
        assert isinstance(recovered, torch.Tensor)
        torch.testing.assert_close(t, recovered, atol=1e-4, rtol=0)

    def test_returns_same_type_numpy(self, fitted_normalizer):
        norm, data = fitted_normalizer
        out = norm.normalize(data)
        assert isinstance(out, np.ndarray)

    def test_returns_same_type_tensor(self, fitted_normalizer):
        norm, data = fitted_normalizer
        t = torch.from_numpy(data)
        out = norm.normalize(t)
        assert isinstance(out, torch.Tensor)

    def test_preserves_device(self, fitted_normalizer):
        """Tensor device should be preserved through normalize/unnormalize."""
        norm, data = fitted_normalizer
        # CPU test only (no GPU guarantee in CI)
        t = torch.from_numpy(data).cpu()
        out = norm.normalize(t)
        assert out.device.type == "cpu"


class TestMinMaxNormalizerCheckpoint:
    def test_state_dict_round_trip(self, fitted_normalizer, rng):
        norm, data = fitted_normalizer
        sd = norm.state_dict()

        norm2 = MinMaxNormalizer()
        norm2.load_state_dict(sd)

        np.testing.assert_array_equal(norm._x_min, norm2._x_min)
        np.testing.assert_array_equal(norm._x_max, norm2._x_max)

        out1 = norm.normalize(data)
        out2 = norm2.normalize(data)
        np.testing.assert_allclose(out1, out2, atol=1e-7)

    def test_not_fitted_raises(self):
        norm = MinMaxNormalizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            norm.normalize(np.zeros((5, 2)))

    def test_repr_unfitted(self):
        norm = MinMaxNormalizer()
        assert "not fitted" in repr(norm)

    def test_repr_fitted(self, fitted_normalizer):
        norm, _ = fitted_normalizer
        r = repr(norm)
        assert "MinMaxNormalizer" in r
        assert "x_min" in r
