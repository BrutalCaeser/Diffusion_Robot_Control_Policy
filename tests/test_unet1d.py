"""
tests/test_unet1d.py — Unit tests for ConditionalUnet1D.

Run with:  pytest tests/test_unet1d.py -v
"""

import pytest
import torch
import torch.nn as nn

from diffusion_policy.model.unet1d import (
    SinusoidalPosEmb,
    Conv1dBlock,
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

B, T, A, T_OBS, OBS_DIM = 4, 16, 2, 2, 5


@pytest.fixture
def model():
    return ConditionalUnet1D(
        action_dim=A,
        obs_horizon=T_OBS,
        obs_dim=OBS_DIM,
        diffusion_step_embed_dim=64,
        down_dims=(64, 128, 256),
        cond_dim=64,
        kernel_size=5,
        n_groups=8,
    )


# ── SinusoidalPosEmb ─────────────────────────────────────────────────────────

class TestSinusoidalPosEmb:
    def test_output_shape(self):
        emb = SinusoidalPosEmb(dim=32)
        x = torch.randint(0, 100, (B,))
        out = emb(x)
        assert out.shape == (B, 32)

    def test_float_input(self):
        """Accepts float timesteps for Flow Matching."""
        emb = SinusoidalPosEmb(dim=32)
        t = torch.rand(B)
        out = emb(t)
        assert out.shape == (B, 32)
        assert out.dtype == torch.float32

    def test_deterministic(self):
        emb = SinusoidalPosEmb(dim=32)
        t = torch.arange(B, dtype=torch.float32)
        out1 = emb(t)
        out2 = emb(t)
        torch.testing.assert_close(out1, out2)

    def test_different_timesteps_differ(self):
        emb = SinusoidalPosEmb(dim=32)
        t1 = torch.zeros(B)
        t2 = torch.ones(B) * 50
        assert not torch.allclose(emb(t1), emb(t2))

    def test_odd_dim_raises(self):
        with pytest.raises(AssertionError):
            SinusoidalPosEmb(dim=33)


# ── Conv1dBlock ───────────────────────────────────────────────────────────────

class TestConv1dBlock:
    def test_output_shape_same_channels(self):
        block = Conv1dBlock(32, 32, kernel_size=5, n_groups=8)
        x = torch.randn(B, 32, T)
        assert block(x).shape == (B, 32, T)

    def test_output_shape_diff_channels(self):
        block = Conv1dBlock(32, 64, kernel_size=5, n_groups=8)
        x = torch.randn(B, 32, T)
        assert block(x).shape == (B, 64, T)

    def test_temporal_length_preserved(self):
        """Same-padding should preserve temporal dimension."""
        for T_test in [8, 16, 32]:
            block = Conv1dBlock(16, 16, kernel_size=5, n_groups=8)
            x = torch.randn(B, 16, T_test)
            assert block(x).shape[-1] == T_test


# ── ConditionalResidualBlock1D ────────────────────────────────────────────────

class TestConditionalResidualBlock1D:
    @pytest.fixture
    def block(self):
        return ConditionalResidualBlock1D(
            in_channels=32, out_channels=64,
            cond_dim=128, kernel_size=5, n_groups=8,
        )

    def test_output_shape(self, block):
        x    = torch.randn(B, 32, T)
        cond = torch.randn(B, 128)
        assert block(x, cond).shape == (B, 64, T)

    def test_same_channels_no_proj(self):
        block = ConditionalResidualBlock1D(
            in_channels=32, out_channels=32,
            cond_dim=64, kernel_size=5, n_groups=8,
        )
        assert isinstance(block.residual_proj, nn.Identity)

    def test_diff_channels_has_proj(self, block):
        assert isinstance(block.residual_proj, nn.Conv1d)

    def test_film_modulation_changes_output(self, block):
        """Different conditioning vectors should produce different outputs."""
        x     = torch.randn(B, 32, T)
        cond1 = torch.zeros(B, 128)
        cond2 = torch.ones(B, 128)
        out1  = block(x, cond1)
        out2  = block(x, cond2)
        assert not torch.allclose(out1, out2)


# ── ConditionalUnet1D ─────────────────────────────────────────────────────────

class TestConditionalUnet1D:
    def test_output_shape(self, model):
        noisy  = torch.randn(B, T, A)
        ts     = torch.randint(0, 100, (B,))
        obs    = torch.randn(B, T_OBS, OBS_DIM)
        out    = model(noisy, ts, obs)
        assert out.shape == (B, T, A)

    def test_float_timestep(self, model):
        """Model should accept float timesteps (Flow Matching)."""
        noisy = torch.randn(B, T, A)
        t     = torch.rand(B)
        obs   = torch.randn(B, T_OBS, OBS_DIM)
        out   = model(noisy, t, obs)
        assert out.shape == (B, T, A)

    def test_output_dtype_float32(self, model):
        noisy = torch.randn(B, T, A)
        ts    = torch.randint(0, 100, (B,))
        obs   = torch.randn(B, T_OBS, OBS_DIM)
        out   = model(noisy, ts, obs)
        assert out.dtype == torch.float32

    def test_gradients_flow(self, model):
        """Backprop should not raise and gradients should be non-zero."""
        noisy = torch.randn(B, T, A, requires_grad=False)
        ts    = torch.randint(0, 100, (B,))
        obs   = torch.randn(B, T_OBS, OBS_DIM)
        out   = model(noisy, ts, obs)
        loss  = out.mean()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_num_parameters(self, model):
        n = model.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_different_batch_sizes(self, model):
        for bs in [1, 2, 8]:
            noisy = torch.randn(bs, T, A)
            ts    = torch.randint(0, 100, (bs,))
            obs   = torch.randn(bs, T_OBS, OBS_DIM)
            out   = model(noisy, ts, obs)
            assert out.shape == (bs, T, A)

    def test_no_grad_context(self, model):
        with torch.no_grad():
            noisy = torch.randn(B, T, A)
            ts    = torch.randint(0, 100, (B,))
            obs   = torch.randn(B, T_OBS, OBS_DIM)
            out   = model(noisy, ts, obs)
        assert out.shape == (B, T, A)

    def test_conditioning_changes_output(self, model):
        """Different observations should produce different noise predictions."""
        noisy = torch.randn(B, T, A)
        ts    = torch.randint(0, 100, (B,))
        obs1  = torch.zeros(B, T_OBS, OBS_DIM)
        obs2  = torch.ones(B, T_OBS, OBS_DIM)
        with torch.no_grad():
            out1 = model(noisy, ts, obs1)
            out2 = model(noisy, ts, obs2)
        assert not torch.allclose(out1, out2)
