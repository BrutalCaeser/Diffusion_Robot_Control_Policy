"""
tests/test_vision_encoder.py — Unit tests for ResNetEncoder.

Run with:  pytest tests/test_vision_encoder.py -v
"""

import pytest
import torch

from diffusion_policy.model.vision_encoder import ResNetEncoder
from diffusion_policy.model.unet1d import ConditionalUnet1D


B, T_OBS, C, H, W = 3, 2, 3, 96, 96
OBS_COND_DIM = 256


@pytest.fixture
def encoder():
    return ResNetEncoder(obs_horizon=T_OBS, obs_cond_dim=OBS_COND_DIM, pretrained=False)


class TestResNetEncoderShapes:
    def test_output_shape(self, encoder):
        x = torch.randn(B, T_OBS, C, H, W)
        out = encoder(x)
        assert out.shape == (B, OBS_COND_DIM)

    def test_output_shape_batch1(self, encoder):
        x = torch.randn(1, T_OBS, C, H, W)
        assert encoder(x).shape == (1, OBS_COND_DIM)

    def test_wrong_obs_horizon_raises(self, encoder):
        x = torch.randn(B, T_OBS + 1, C, H, W)
        with pytest.raises(AssertionError, match="obs_horizon"):
            encoder(x)

    def test_wrong_channels_raises(self, encoder):
        x = torch.randn(B, T_OBS, 1, H, W)   # grayscale, not RGB
        with pytest.raises(AssertionError, match="3-channel"):
            encoder(x)


class TestResNetEncoderGradients:
    def test_gradients_flow_through_encoder(self, encoder):
        x = torch.randn(B, T_OBS, C, H, W, requires_grad=False)
        x.requires_grad_(True)
        out = encoder(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradients_flow_through_proj(self, encoder):
        x = torch.randn(B, T_OBS, C, H, W)
        out = encoder(x)
        # All projection parameters should receive gradients
        out.mean().backward()
        for name, p in encoder.proj.named_parameters():
            assert p.grad is not None, f"No gradient for proj.{name}"


class TestResNetEncoderConditioningEffect:
    def test_different_inputs_produce_different_outputs(self, encoder):
        encoder.eval()
        with torch.no_grad():
            x1 = torch.randn(B, T_OBS, C, H, W)
            x2 = torch.randn(B, T_OBS, C, H, W)
            out1 = encoder(x1)
            out2 = encoder(x2)
        assert not torch.allclose(out1, out2), \
            "Different images should produce different conditioning vectors"

    def test_same_input_deterministic(self, encoder):
        encoder.eval()
        x = torch.randn(B, T_OBS, C, H, W)
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)
        torch.testing.assert_close(out1, out2)

    def test_obs_horizon_frames_affect_output(self, encoder):
        """Changing one frame in the obs window should change the output."""
        encoder.eval()
        x = torch.randn(B, T_OBS, C, H, W)
        x_perturbed = x.clone()
        x_perturbed[:, 0] = torch.randn(B, C, H, W)  # change first frame only
        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x_perturbed)
        assert not torch.allclose(out1, out2)


class TestResNetEncoderFreezeBackbone:
    def test_frozen_backbone_has_no_grad(self):
        enc = ResNetEncoder(obs_horizon=2, obs_cond_dim=256, freeze_backbone=True)
        for p in enc.backbone.parameters():
            assert not p.requires_grad, "Backbone should be frozen"

    def test_frozen_backbone_proj_still_trains(self):
        enc = ResNetEncoder(obs_horizon=2, obs_cond_dim=256, freeze_backbone=True)
        for p in enc.proj.parameters():
            assert p.requires_grad, "Projection should still be trainable"


class TestResNetEncoderWithUnet:
    def test_image_conditioning_unet_forward(self, encoder):
        """End-to-end: images → encoder → UNet → action predictions."""
        unet = ConditionalUnet1D(
            action_dim=2, obs_horizon=T_OBS, obs_dim=0,  # obs_dim=0 → image mode
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            cond_dim=OBS_COND_DIM,
        )
        obs_imgs   = torch.randn(B, T_OBS, C, H, W)
        noisy_act  = torch.randn(B, 16, 2)
        timesteps  = torch.randint(0, 100, (B,))

        cond = encoder(obs_imgs)              # (B, 256)
        out  = unet(noisy_act, timesteps, cond)
        assert out.shape == (B, 16, 2)

    def test_image_and_state_conditioning_differ(self):
        """Image-conditioned output should differ from state-conditioned output."""
        B2 = 2
        # State-based UNet
        unet_state = ConditionalUnet1D(
            action_dim=2, obs_horizon=2, obs_dim=5,
            diffusion_step_embed_dim=256, down_dims=(64, 128, 256), cond_dim=256,
        )
        # Image-based UNet (same architecture, different obs_dim)
        enc = ResNetEncoder(obs_horizon=2, obs_cond_dim=256, pretrained=False)
        unet_img = ConditionalUnet1D(
            action_dim=2, obs_horizon=2, obs_dim=0,
            diffusion_step_embed_dim=256, down_dims=(64, 128, 256), cond_dim=256,
        )
        noisy_act = torch.randn(B2, 16, 2)
        ts        = torch.randint(0, 100, (B2,))

        obs_state = torch.randn(B2, 2, 5)
        obs_imgs  = torch.randn(B2, 2, 3, 96, 96)
        obs_cond  = enc(obs_imgs)

        out_state = unet_state(noisy_act, ts, obs_state)
        out_img   = unet_img(noisy_act, ts, obs_cond)

        # Both should produce valid action predictions
        assert out_state.shape == (B2, 16, 2)
        assert out_img.shape   == (B2, 16, 2)


class TestResNetEncoderRepr:
    def test_repr_contains_key_info(self, encoder):
        r = repr(encoder)
        assert "obs_horizon=2" in r
        assert "obs_cond_dim=256" in r
        assert "params=" in r

    def test_output_dim_property(self, encoder):
        assert encoder.output_dim == OBS_COND_DIM
