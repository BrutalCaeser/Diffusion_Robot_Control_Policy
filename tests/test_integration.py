"""
tests/test_integration.py — End-to-end integration tests.

These tests verify that the full pipeline (model + scheduler + normalizer)
works without any shape errors.  They do NOT require the PushT dataset.

Run with:  pytest tests/test_integration.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.data.normalizer import MinMaxNormalizer
from diffusion_policy.model.unet1d import ConditionalUnet1D
from diffusion_policy.model.ddpm import DDPMScheduler
from diffusion_policy.model.ddim import DDIMScheduler
from diffusion_policy.model.ema import EMA
from diffusion_policy.model.flow_matching import FlowMatchingScheduler


# Minimal config for tests (smaller than real training for speed)
ACTION_DIM  = 2
OBS_DIM     = 5
OBS_HORIZON = 2
PRED_HORIZON= 16
BATCH_SIZE  = 4
DEVICE      = "cpu"


@pytest.fixture
def model():
    return ConditionalUnet1D(
        action_dim=ACTION_DIM, obs_horizon=OBS_HORIZON, obs_dim=OBS_DIM,
        diffusion_step_embed_dim=64, down_dims=(64, 128, 256), cond_dim=64,
    )


@pytest.fixture
def ddpm():
    return DDPMScheduler(num_diffusion_steps=20)


@pytest.fixture
def normalizers():
    rng = np.random.default_rng(0)
    obs_data    = rng.uniform(0, 512, (1000, OBS_DIM)).astype(np.float32)
    action_data = rng.uniform(0, 512, (1000, ACTION_DIM)).astype(np.float32)
    obs_norm    = MinMaxNormalizer().fit(obs_data)
    action_norm = MinMaxNormalizer().fit(action_data)
    return obs_norm, action_norm


class TestDDPMTrainStep:
    """One training step with DDPM should not raise and return a finite loss."""

    def test_single_step(self, model, ddpm):
        obs    = torch.randn(BATCH_SIZE, OBS_HORIZON, OBS_DIM)
        action = torch.randn(BATCH_SIZE, PRED_HORIZON, ACTION_DIM)

        ts          = ddpm.sample_timesteps(BATCH_SIZE)
        noise       = torch.randn_like(action)
        noisy_action= ddpm.add_noise(action, noise, ts)
        noise_pred  = model(noisy_action, ts, obs)
        loss        = F.mse_loss(noise_pred, noise)

        assert torch.isfinite(loss)
        loss.backward()
        # Gradients should be non-None
        assert any(p.grad is not None for p in model.parameters())


class TestDDPMInference:
    """Full DDPM reverse sampling should produce the right shape."""

    def test_full_sample(self, model, ddpm):
        obs = torch.randn(BATCH_SIZE, OBS_HORIZON, OBS_DIM)
        with torch.no_grad():
            actions = ddpm.sample(model, obs, PRED_HORIZON, ACTION_DIM)
        assert actions.shape == (BATCH_SIZE, PRED_HORIZON, ACTION_DIM)
        assert torch.isfinite(actions).all()


class TestDDIMInference:
    """DDIM should produce the same shape with fewer steps."""

    def test_ddim_sample(self, model, ddpm):
        ddim = DDIMScheduler(ddpm, ddim_steps=5, eta=0.0)
        obs  = torch.randn(BATCH_SIZE, OBS_HORIZON, OBS_DIM)
        with torch.no_grad():
            actions = ddim.sample(model, obs, PRED_HORIZON, ACTION_DIM)
        assert actions.shape == (BATCH_SIZE, PRED_HORIZON, ACTION_DIM)


class TestFlowMatchingTrainStep:
    """One FM training step should return a finite loss."""

    def test_fm_loss(self, model):
        fm     = FlowMatchingScheduler(num_inference_steps=5)
        obs    = torch.randn(BATCH_SIZE, OBS_HORIZON, OBS_DIM)
        action = torch.randn(BATCH_SIZE, PRED_HORIZON, ACTION_DIM)
        loss   = fm.get_loss(model, action, obs)
        assert torch.isfinite(loss)
        loss.backward()


class TestFlowMatchingInference:
    def test_fm_sample(self, model):
        fm  = FlowMatchingScheduler(num_inference_steps=5)
        obs = torch.randn(BATCH_SIZE, OBS_HORIZON, OBS_DIM)
        with torch.no_grad():
            actions = fm.sample(model, obs, PRED_HORIZON, ACTION_DIM)
        assert actions.shape == (BATCH_SIZE, PRED_HORIZON, ACTION_DIM)


class TestEMAIntegration:
    """EMA workflow: update, apply, evaluate, restore."""

    def test_full_cycle(self, model, ddpm):
        ema       = EMA(model, decay=0.9)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Simulate 3 training steps
        for _ in range(3):
            obs    = torch.randn(BATCH_SIZE, OBS_HORIZON, OBS_DIM)
            action = torch.randn(BATCH_SIZE, PRED_HORIZON, ACTION_DIM)
            ts     = ddpm.sample_timesteps(BATCH_SIZE)
            noise  = torch.randn_like(action)
            noisy  = ddpm.add_noise(action, noise, ts)
            loss   = F.mse_loss(model(noisy, ts, obs), noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ema.update(model)

        # Evaluation cycle
        train_weights_before = {
            k: v.clone() for k, v in model.named_parameters()
        }
        ema.apply(model)
        obs = torch.randn(1, OBS_HORIZON, OBS_DIM)
        with torch.no_grad():
            out = ddpm.sample(model, obs, PRED_HORIZON, ACTION_DIM)
        assert out.shape == (1, PRED_HORIZON, ACTION_DIM)
        ema.restore(model)

        # Verify training weights restored
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.data, train_weights_before[name])


class TestNormalizerPipeline:
    """Normalizer should round-trip through training → inference."""

    def test_round_trip(self, normalizers):
        obs_norm, action_norm = normalizers

        rng = np.random.default_rng(1)
        raw_actions = rng.uniform(0, 512, (16, ACTION_DIM)).astype(np.float32)

        normed   = action_norm.normalize(raw_actions)
        unormed  = action_norm.unnormalize(normed)

        np.testing.assert_allclose(raw_actions, unormed, atol=1e-5)

    def test_state_dict_preserves_pipeline(self, normalizers):
        _, action_norm = normalizers
        sd = action_norm.state_dict()

        action_norm2 = MinMaxNormalizer()
        action_norm2.load_state_dict(sd)

        raw = np.random.rand(5, ACTION_DIM).astype(np.float32)
        np.testing.assert_allclose(
            action_norm.normalize(raw),
            action_norm2.normalize(raw),
            atol=1e-7,
        )
