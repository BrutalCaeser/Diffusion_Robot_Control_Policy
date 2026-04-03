"""
vision_encoder.py — ResNet-18 Visual Encoder for Visuomotor Diffusion Policy
=============================================================================

This module implements the visual observation encoder described in:

    Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
    RSS 2023. https://arxiv.org/abs/2303.04137

The visuomotor variant replaces the low-dimensional state vector (agent_pos,
block_pos, angle) with raw RGB images from the environment camera.  A ResNet-18
backbone encodes each frame into a 512-dim feature; the features from all
T_obs frames are concatenated and projected into the conditioning vector that
FiLM-modulates every residual block of the 1D temporal U-Net.

Intuition
---------
Instead of telling the model "the agent is at (222, 97) and the block is at
(223, 382)" we show it a 96×96 RGB snapshot of the scene.  The CNN must
extract the relevant spatial information (positions, angles, contact state)
from pixels.  Using T_obs=2 consecutive frames gives the network implicit
velocity information — the same role it played in the state-based variant.

Architecture (per frame)
------------------------
    Input (3, 96, 96)
        │
        ▼
    ResNet-18 body (conv1 → bn1 → relu → maxpool → layer1–4)
        │  Output: (512, 3, 3)   for 96×96 input
        ▼
    AdaptiveAvgPool2d(1)   →   (512,)    global feature vector
        │
        ▼  (concatenate T_obs frames)
    (T_obs × 512,)
        │
        ▼
    Linear(T_obs×512, obs_cond_dim) → ReLU → Linear(obs_cond_dim, obs_cond_dim)
        │
        ▼
    cond  (B, obs_cond_dim)

Mathematical formulation
------------------------
Let φ(·) denote the ResNet-18 body + global average pool.

Single frame:
    f_i = φ(x_i) ∈ ℝ^512

Over the observation window [x_{t-1}, x_t]:
    cond = MLP([f_{t-1} ; f_t]) ∈ ℝ^obs_cond_dim

where [;] denotes concatenation and MLP is a two-layer ReLU network.

The cond vector is then used by ConditionalUnet1D exactly as the state-based
conditioning vector — it drives the FiLM parameters (γ, β) in every
ConditionalResidualBlock1D.

ImageNet normalisation
----------------------
When pretrained=True we apply the standard ImageNet normalisation:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
before the ResNet forward pass.  The raw zarr images are in [0, 255]; the
PushTImageDataset divides by 255 first, so inputs to this module are in [0, 1].
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ImageNet channel-wise statistics (used when pretrained=True)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class ResNetEncoder(nn.Module):
    """
    ResNet-18 visual encoder for visuomotor diffusion policy.

    Args:
        obs_horizon:      Number of consecutive RGB frames to encode (T_obs).
        obs_cond_dim:     Output dimension of the conditioning vector.  Must
                          match the ``cond_dim`` used in ConditionalUnet1D.
        pretrained:       If True, initialise the backbone with ImageNet weights
                          (requires internet on first call) and apply ImageNet
                          normalisation in ``forward``.
        freeze_backbone:  If True, fix backbone weights and only train the
                          projection MLP.  Useful for quick fine-tuning.

    Input shape:  (B, T_obs, 3, H, W)  — pixel values in [0, 1]
    Output shape: (B, obs_cond_dim)
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        obs_cond_dim: int = 256,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.obs_horizon   = obs_horizon
        self.obs_cond_dim  = obs_cond_dim
        self._pretrained   = pretrained

        # ------------------------------------------------------------------ #
        # Backbone: ResNet-18 without the classification head                 #
        # ------------------------------------------------------------------ #
        # Import here so the module is usable even without torchvision being
        # installed at import time (it's only needed if you instantiate this).
        try:
            import torchvision.models as tvm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torchvision is required for ResNetEncoder. "
                "Install it with: pip install torchvision"
            ) from exc

        if pretrained:
            from torchvision.models import ResNet18_Weights
            resnet = tvm.resnet18(weights=ResNet18_Weights.DEFAULT)
            logger.info("ResNetEncoder: loaded ImageNet-pretrained ResNet-18")
        else:
            resnet = tvm.resnet18(weights=None)
            logger.info("ResNetEncoder: initialised ResNet-18 from scratch")

        # Keep everything except avgpool and fc
        # For a 96×96 input the spatial size after each layer is:
        #   conv1+maxpool → 24×24   layer1 → 24×24   layer2 → 12×12
        #   layer3 → 6×6            layer4 → 3×3
        self.backbone = nn.Sequential(
            resnet.conv1,    # (B, 64, 48, 48)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B, 64, 24, 24)
            resnet.layer1,   # (B,  64, 24, 24)
            resnet.layer2,   # (B, 128, 12, 12)
            resnet.layer3,   # (B, 256,  6,  6)
            resnet.layer4,   # (B, 512,  3,  3)
        )

        # Adaptive global average pool: (B, 512, h, w) → (B, 512, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ------------------------------------------------------------------ #
        # Projection MLP                                                       #
        # ------------------------------------------------------------------ #
        backbone_feat_dim = 512 * obs_horizon   # concat features from all frames
        self.proj = nn.Sequential(
            nn.Linear(backbone_feat_dim, obs_cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obs_cond_dim, obs_cond_dim),
        )

        # ------------------------------------------------------------------ #
        # ImageNet normalisation buffers (used when pretrained=True)          #
        # ------------------------------------------------------------------ #
        # Register as buffers so they move with .to(device) automatically.
        mean = torch.tensor(_IMAGENET_MEAN).reshape(1, 1, 3, 1, 1)   # (1,1,3,1,1)
        std  = torch.tensor(_IMAGENET_STD ).reshape(1, 1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean)
        self.register_buffer("_imagenet_std",  std)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            logger.info("ResNetEncoder: backbone frozen")

        n_total     = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ResNetEncoder: obs_horizon=%d, obs_cond_dim=%d, "
            "params=%.2fM (trainable=%.2fM)",
            obs_horizon, obs_cond_dim,
            n_total / 1e6, n_trainable / 1e6,
        )

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode a stack of observation images into a conditioning vector.

        Args:
            obs: (B, T_obs, 3, H, W) float tensor with pixel values in [0, 1].

        Returns:
            cond: (B, obs_cond_dim) float tensor.
        """
        B, T, C, H, W = obs.shape
        assert T == self.obs_horizon, (
            f"ResNetEncoder: expected obs_horizon={self.obs_horizon} frames, got {T}"
        )
        assert C == 3, f"Expected 3-channel RGB input, got {C}"

        # Optional ImageNet normalisation
        x = obs
        if self._pretrained:
            x = (obs - self._imagenet_mean) / self._imagenet_std  # (B,T,3,H,W)

        # Merge batch and time dims for a single batched forward pass
        x = x.reshape(B * T, C, H, W)          # (B*T, 3, H, W)

        # ResNet body: (B*T, 512, h, w)
        features = self.backbone(x)

        # Global average pool + flatten: (B*T, 512)
        features = self.pool(features).flatten(1)

        # Restore (B, T, 512) then flatten to (B, T*512)
        features = features.reshape(B, T * 512)

        # Project: (B, obs_cond_dim)
        return self.proj(features)

    # ---------------------------------------------------------------------- #
    # Properties / helpers                                                     #
    # ---------------------------------------------------------------------- #

    @property
    def output_dim(self) -> int:
        """Dimension of the output conditioning vector."""
        return self.obs_cond_dim

    def __repr__(self) -> str:
        n_total     = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"ResNetEncoder("
            f"obs_horizon={self.obs_horizon}, "
            f"obs_cond_dim={self.obs_cond_dim}, "
            f"pretrained={self._pretrained}, "
            f"params={n_total/1e6:.2f}M, "
            f"trainable={n_trainable/1e6:.2f}M)"
        )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    enc = ResNetEncoder(obs_horizon=2, obs_cond_dim=256, pretrained=False)
    print(enc)

    x = torch.randn(4, 2, 3, 96, 96)       # (B=4, T_obs=2, C=3, H=96, W=96)
    cond = enc(x)
    assert cond.shape == (4, 256), f"Expected (4,256), got {cond.shape}"
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(cond.shape)}")
    print("ResNetEncoder sanity check passed!")
