"""
tests/test_flow_matching.py — Unit tests for FlowMatchingScheduler.

Run with:  pytest tests/test_flow_matching.py -v
"""

import pytest
import torch
import torch.nn as nn

from diffusion_policy.model.flow_matching import FlowMatchingScheduler


B, T, A = 4, 16, 2


@pytest.fixture
def fm():
    return FlowMatchingScheduler(num_inference_steps=10)


class ZeroVelocityModel(nn.Module):
    """Always predicts zero velocity → ODE goes nowhere → output = input."""
    def forward(self, a, t, obs):
        return torch.zeros_like(a)


class IdentityVelocityModel(nn.Module):
    """Predicts velocity = input → trivial dynamics."""
    def forward(self, a, t, obs):
        return a


class TestInterpolate:
    def test_output_shape(self, fm):
        a0    = torch.randn(B, T, A)
        noise = torch.randn(B, T, A)
        t     = torch.rand(B)
        x_t   = fm.interpolate(a0, noise, t)
        assert x_t.shape == (B, T, A)

    def test_t0_gives_data(self, fm):
        a0    = torch.randn(B, T, A)
        noise = torch.randn(B, T, A)
        x_t   = fm.interpolate(a0, noise, torch.zeros(B))
        torch.testing.assert_close(x_t, a0)

    def test_t1_gives_noise(self, fm):
        a0    = torch.randn(B, T, A)
        noise = torch.randn(B, T, A)
        x_t   = fm.interpolate(a0, noise, torch.ones(B))
        torch.testing.assert_close(x_t, noise)

    def test_linearity(self, fm):
        """x_{0.5} should be the midpoint of a_0 and noise."""
        a0    = torch.zeros(B, T, A)
        noise = torch.ones(B, T, A) * 2.0
        x_t   = fm.interpolate(a0, noise, torch.full((B,), 0.5))
        expected = torch.ones(B, T, A)  # midpoint
        torch.testing.assert_close(x_t, expected)


class TestComputeTarget:
    def test_output_shape(self, fm):
        a0    = torch.randn(B, T, A)
        noise = torch.randn(B, T, A)
        target = fm.compute_target(a0, noise)
        assert target.shape == (B, T, A)

    def test_value(self, fm):
        a0    = torch.zeros(B, T, A)
        noise = torch.ones(B, T, A)
        target = fm.compute_target(a0, noise)
        # target should be noise - a0 = 1 - 0 = 1
        torch.testing.assert_close(target, torch.ones(B, T, A))

    def test_velocity_consistent_with_interpolation(self, fm):
        """Finite-difference velocity should match analytical target."""
        a0    = torch.randn(1, T, A)
        noise = torch.randn(1, T, A)
        target = fm.compute_target(a0, noise)

        dt = 1e-4
        t0 = torch.full((1,), 0.5)
        t1 = torch.full((1,), 0.5 + dt)
        numerical_vel = (fm.interpolate(a0, noise, t1) - fm.interpolate(a0, noise, t0)) / dt
        torch.testing.assert_close(numerical_vel, target, atol=1e-3, rtol=0)


class TestGetLoss:
    def test_returns_scalar(self, fm):
        model = ZeroVelocityModel()
        a0  = torch.randn(B, T, A)
        obs = torch.randn(B, 2, 5)
        loss = fm.get_loss(model, a0, obs)
        assert loss.shape == ()   # scalar

    def test_loss_non_negative(self, fm):
        model = ZeroVelocityModel()
        a0  = torch.randn(B, T, A)
        obs = torch.randn(B, 2, 5)
        loss = fm.get_loss(model, a0, obs)
        assert loss.item() >= 0.0

    def test_loss_differentiable(self, fm):
        """Loss should be differentiable w.r.t. model parameters."""
        import torch.nn.functional as F

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(T * A, T * A)
            def forward(self, a, t, obs):
                b = a.shape[0]
                return self.fc(a.reshape(b, -1)).reshape(b, T, A)

        model = SmallModel()
        a0  = torch.randn(B, T, A)
        obs = torch.randn(B, 2, 5)
        loss = fm.get_loss(model, a0, obs)
        loss.backward()
        assert model.fc.weight.grad is not None


class TestSample:
    def test_output_shape(self, fm):
        obs = torch.randn(B, 2, 5)
        out = fm.sample(ZeroVelocityModel(), obs, T, A)
        assert out.shape == (B, T, A)
