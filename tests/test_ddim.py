"""
tests/test_ddim.py — Unit tests for DDIMScheduler.

Run with:  pytest tests/test_ddim.py -v
"""

import pytest
import torch
import torch.nn as nn

from diffusion_policy.model.ddpm import DDPMScheduler
from diffusion_policy.model.ddim import DDIMScheduler


B, T, A = 2, 16, 2


class ZeroModel(nn.Module):
    """Dummy model that always predicts zero noise."""
    def forward(self, a, t, obs):
        return torch.zeros_like(a)


@pytest.fixture
def ddpm():
    return DDPMScheduler(num_diffusion_steps=100)


@pytest.fixture
def ddim(ddpm):
    return DDIMScheduler(ddpm, ddim_steps=10, eta=0.0)


class TestTimestepSeq:
    def test_length(self, ddim):
        assert len(ddim.timestep_seq) >= 10

    def test_sorted_ascending(self, ddim):
        assert all(
            ddim.timestep_seq[i] <= ddim.timestep_seq[i+1]
            for i in range(len(ddim.timestep_seq)-1)
        )

    def test_max_within_K(self, ddim):
        assert ddim.timestep_seq[-1] <= 99


class TestStep:
    def test_output_shape(self, ddim):
        eps_pred = torch.randn(B, T, A)
        a_t      = torch.randn(B, T, A)
        a_prev   = ddim.step(eps_pred, t=50, t_prev=40, noisy_actions=a_t)
        assert a_prev.shape == (B, T, A)

    def test_eta0_deterministic(self, ddim):
        """With eta=0 the step is fully deterministic."""
        eps_pred = torch.randn(B, T, A)
        a_t      = torch.randn(B, T, A)
        out1 = ddim.step(eps_pred, 50, 40, a_t)
        out2 = ddim.step(eps_pred, 50, 40, a_t)
        torch.testing.assert_close(out1, out2)


class TestSample:
    def test_output_shape(self, ddim):
        obs = torch.randn(B, 2, 5)
        out = ddim.sample(ZeroModel(), obs, pred_horizon=T, action_dim=A)
        assert out.shape == (B, T, A)

    def test_deterministic_with_eta0(self, ddim):
        obs = torch.randn(B, 2, 5)
        torch.manual_seed(0)
        out1 = ddim.sample(ZeroModel(), obs, T, A)
        torch.manual_seed(0)
        out2 = ddim.sample(ZeroModel(), obs, T, A)
        torch.testing.assert_close(out1, out2)

    def test_stochastic_with_eta1(self, ddpm):
        ddim_stoch = DDIMScheduler(ddpm, ddim_steps=10, eta=1.0)
        obs = torch.randn(B, 2, 5)
        torch.manual_seed(0)
        out1 = ddim_stoch.sample(ZeroModel(), obs, T, A)
        torch.manual_seed(1)
        out2 = ddim_stoch.sample(ZeroModel(), obs, T, A)
        assert not torch.allclose(out1, out2)

    def test_eta_out_of_range(self, ddpm):
        with pytest.raises(ValueError, match="eta"):
            DDIMScheduler(ddpm, ddim_steps=10, eta=1.5)
