"""
tests/test_ddpm.py — Unit tests for DDPMScheduler.

Run with:  pytest tests/test_ddpm.py -v
"""

import pytest
import torch

from diffusion_policy.model.ddpm import DDPMScheduler


B, T, A = 4, 16, 2


@pytest.fixture
def scheduler():
    return DDPMScheduler(num_diffusion_steps=100, beta_start=1e-4, beta_end=0.02)


class TestNoiseSchedule:
    def test_betas_shape(self, scheduler):
        assert scheduler.betas.shape == (100,)

    def test_betas_increasing(self, scheduler):
        """Linear schedule: betas should be monotonically non-decreasing."""
        diffs = scheduler.betas[1:] - scheduler.betas[:-1]
        assert (diffs >= 0).all()

    def test_alphas_cumprod_decreasing(self, scheduler):
        """ᾱ_k should decrease from ~1 at k=0 to ~0 at k=K (cosine schedule)."""
        ab = scheduler.alphas_cumprod
        assert ab[0] > 0.99
        assert ab[-1] < 0.01   # cosine schedule guarantees this for K=100

    def test_all_buffers_on_same_device(self, scheduler):
        dev = scheduler.betas.device
        for buf in [scheduler.alphas, scheduler.alphas_cumprod,
                    scheduler.sqrt_alphas_cumprod,
                    scheduler.sqrt_one_minus_alphas_cumprod]:
            assert buf.device == dev


class TestAddNoise:
    def test_output_shape(self, scheduler):
        a0  = torch.randn(B, T, A)
        eps = torch.randn(B, T, A)
        ts  = scheduler.sample_timesteps(B)
        a_k = scheduler.add_noise(a0, eps, ts)
        assert a_k.shape == (B, T, A)

    def test_k0_is_clean(self, scheduler):
        """At timestep 0 the noisy action should be nearly identical to clean."""
        a0  = torch.randn(B, T, A)
        eps = torch.randn(B, T, A)
        ts  = torch.zeros(B, dtype=torch.long)
        a_k = scheduler.add_noise(a0, eps, ts)
        torch.testing.assert_close(a_k, a0, atol=0.1, rtol=0)  # cosine: sqrt(1-ab_0)≈0.025, 4-sigma

    def test_kK_is_noise(self, scheduler):
        """At timestep K-1 the noisy action should be nearly identical to noise."""
        K   = scheduler.K
        a0  = torch.randn(B, T, A)
        eps = torch.randn(B, T, A)
        ts  = torch.full((B,), K - 1, dtype=torch.long)
        a_k = scheduler.add_noise(a0, eps, ts)
        torch.testing.assert_close(a_k, eps, atol=0.05, rtol=0)

    def test_noise_increases_with_k(self, scheduler):
        """Higher timesteps should result in more deviation from clean."""
        a0  = torch.zeros(1, T, A)
        eps = torch.ones(1, T, A)
        def get_noise_level(k):
            ts = torch.tensor([k])
            return (scheduler.add_noise(a0, eps, ts) - a0).abs().mean().item()

        n_low  = get_noise_level(0)
        n_mid  = get_noise_level(50)
        n_high = get_noise_level(99)
        assert n_low < n_mid < n_high


class TestStep:
    def test_step_output_shape(self, scheduler):
        a_k      = torch.randn(B, T, A)
        eps_pred = torch.randn(B, T, A)
        a_prev   = scheduler.step(eps_pred, timestep=50, noisy_actions=a_k)
        assert a_prev.shape == (B, T, A)

    def test_step_k0_is_deterministic(self, scheduler):
        """At k=0 step() should return the deterministic mean (no noise added)."""
        a_k      = torch.randn(B, T, A)
        eps_pred = torch.randn(B, T, A)
        out1 = scheduler.step(eps_pred, 0, a_k)
        out2 = scheduler.step(eps_pred, 0, a_k)
        torch.testing.assert_close(out1, out2)

    def test_step_k_nonzero_stochastic(self, scheduler):
        """At k>0 step() should produce different outputs on repeated calls."""
        a_k      = torch.randn(B, T, A)
        eps_pred = torch.randn(B, T, A)
        out1 = scheduler.step(eps_pred, 50, a_k)
        out2 = scheduler.step(eps_pred, 50, a_k)
        # With high probability, stochastic outputs differ
        assert not torch.allclose(out1, out2)


class TestSampleTimesteps:
    def test_shape(self, scheduler):
        ts = scheduler.sample_timesteps(B)
        assert ts.shape == (B,)

    def test_range(self, scheduler):
        ts = scheduler.sample_timesteps(1000)
        assert ts.min() >= 0
        assert ts.max() < scheduler.K

    def test_dtype(self, scheduler):
        ts = scheduler.sample_timesteps(B)
        assert ts.dtype == torch.int64
