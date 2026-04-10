"""
tests/test_bc_policy.py — Unit tests for the BC baseline policy.

Run with:  pytest tests/test_bc_policy.py -v

These tests verify:
  1. Output shape is correct for various input shapes
  2. Forward pass is differentiable (gradients flow)
  3. Parameter count is reasonable (much smaller than diffusion U-Net)
  4. The policy handles the exact PushT configuration
"""

import pytest
import torch
import torch.nn as nn

from baselines.bc_policy import BCPolicy


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pusht_policy():
    """BC policy with exact PushT config: T_obs=2, obs_dim=5, T_pred=16, action_dim=2."""
    return BCPolicy(obs_horizon=2, obs_dim=5, pred_horizon=16, action_dim=2, hidden_dim=256)


@pytest.fixture
def generic_policy():
    """BC policy with arbitrary dimensions for shape testing."""
    return BCPolicy(obs_horizon=3, obs_dim=8, pred_horizon=10, action_dim=4, hidden_dim=128)


# ── Shape tests ───────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_pusht_single_batch(self, pusht_policy):
        obs = torch.randn(1, 2, 5)
        out = pusht_policy(obs)
        assert out.shape == (1, 16, 2), f"Expected (1, 16, 2), got {out.shape}"

    def test_pusht_batch_256(self, pusht_policy):
        obs = torch.randn(256, 2, 5)
        out = pusht_policy(obs)
        assert out.shape == (256, 16, 2)

    def test_generic_shape(self, generic_policy):
        obs = torch.randn(8, 3, 8)
        out = generic_policy(obs)
        assert out.shape == (8, 10, 4)

    def test_obs_horizon_1(self):
        """Single observation frame (no history) should still work."""
        policy = BCPolicy(obs_horizon=1, obs_dim=5, pred_horizon=16, action_dim=2)
        obs    = torch.randn(4, 1, 5)
        out    = policy(obs)
        assert out.shape == (4, 16, 2)

    def test_obs_horizon_4(self):
        """Longer observation history."""
        policy = BCPolicy(obs_horizon=4, obs_dim=5, pred_horizon=16, action_dim=2)
        obs    = torch.randn(4, 4, 5)
        out    = policy(obs)
        assert out.shape == (4, 16, 2)


# ── Gradient tests ────────────────────────────────────────────────────────────

class TestGradients:
    def test_gradients_flow(self, pusht_policy):
        """MSE loss must be differentiable through the BC policy."""
        obs    = torch.randn(4, 2, 5, requires_grad=False)
        target = torch.randn(4, 16, 2)
        pred   = pusht_policy(obs)
        loss   = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        for name, param in pusht_policy.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_zero_grad_clears(self, pusht_policy):
        """Zero-ing gradients should clear them."""
        obs  = torch.randn(2, 2, 5)
        pred = pusht_policy(obs)
        pred.sum().backward()
        pusht_policy.zero_grad()
        for param in pusht_policy.parameters():
            assert param.grad is None or param.grad.abs().max() == 0


# ── Architecture tests ────────────────────────────────────────────────────────

class TestArchitecture:
    def test_is_nn_module(self, pusht_policy):
        assert isinstance(pusht_policy, nn.Module)

    def test_net_is_sequential(self, pusht_policy):
        assert isinstance(pusht_policy.net, nn.Sequential)

    def test_three_linear_layers(self, pusht_policy):
        linears = [m for m in pusht_policy.net if isinstance(m, nn.Linear)]
        assert len(linears) == 3, "BC should have exactly 3 Linear layers"

    def test_correct_input_dim(self, pusht_policy):
        """First layer input: T_obs * obs_dim = 2 * 5 = 10."""
        first_linear = [m for m in pusht_policy.net if isinstance(m, nn.Linear)][0]
        assert first_linear.in_features == 10

    def test_correct_output_dim(self, pusht_policy):
        """Last layer output: T_pred * action_dim = 16 * 2 = 32."""
        last_linear = [m for m in pusht_policy.net if isinstance(m, nn.Linear)][-1]
        assert last_linear.out_features == 32

    def test_hidden_dim(self, pusht_policy):
        """Hidden layers should be 256."""
        linears = [m for m in pusht_policy.net if isinstance(m, nn.Linear)]
        assert linears[0].out_features == 256
        assert linears[1].out_features == 256

    def test_param_count_reasonable(self, pusht_policy):
        """BC should have far fewer parameters than the diffusion U-Net (~68M).
        Expected: ~135K params for hidden_dim=256, in=10, out=32."""
        n_params = pusht_policy.num_parameters()
        assert n_params < 500_000, f"BC has {n_params:,} params — unexpectedly large"
        assert n_params > 10_000,  f"BC has {n_params:,} params — unexpectedly small"

    def test_repr_contains_key_info(self, pusht_policy):
        s = repr(pusht_policy)
        assert "BCPolicy" in s
        assert "params" in s


# ── Determinism tests ─────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self, pusht_policy):
        """BC forward pass is deterministic (no stochastic sampling)."""
        pusht_policy.eval()
        obs  = torch.randn(2, 2, 5)
        out1 = pusht_policy(obs)
        out2 = pusht_policy(obs)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self, pusht_policy):
        pusht_policy.eval()
        obs1 = torch.randn(1, 2, 5)
        obs2 = torch.randn(1, 2, 5)
        # With overwhelming probability two random inputs produce different outputs
        assert not torch.allclose(pusht_policy(obs1), pusht_policy(obs2))


# ── Device tests ──────────────────────────────────────────────────────────────

class TestDevice:
    def test_cpu_forward(self, pusht_policy):
        policy = pusht_policy.cpu()
        obs    = torch.randn(2, 2, 5)
        out    = policy(obs)
        assert out.device.type == "cpu"

    def test_to_device_and_back(self, pusht_policy):
        """Model should move cleanly to CPU and produce same-shaped output."""
        pusht_policy.to("cpu")
        obs = torch.randn(1, 2, 5)
        out = pusht_policy(obs)
        assert out.shape == (1, 16, 2)
