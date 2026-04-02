"""
ema.py — Exponential Moving Average (EMA) for Model Weights
=============================================================

This module maintains a shadow copy of the model's parameters updated as
an exponential moving average of the training weights.

Why EMA?
--------
Each optimizer step pushes the weights toward minimizing the loss on the
current mini-batch, which is a noisy estimate of the true gradient.  The
training weights therefore oscillate around the optimum rather than settling
smoothly.

EMA smooths this out by tracking a weighted average of the last N ≈ 1/(1-decay)
checkpoints:

    θ_ema  ←  decay · θ_ema  +  (1 - decay) · θ_train

With ``decay = 0.995`` the effective window is about **200 steps**.  Recent
updates contribute more (decaying exponentially), old ones almost nothing.

Why does this matter for diffusion models specifically?
------------------------------------------------------
Diffusion inference is *iterative* — the reverse process runs K (or ddim_steps)
back-to-back model evaluations.  Any bias or high-frequency noise in the
weights gets **compounded** across those steps.  The EMA weights are smoother
and produce more consistent noise predictions, directly improving the quality
of the generated action trajectories.

The critical rule: **EMA weights are used only for evaluation / inference;
training always updates the original weights.**

Workflow::

    for batch in dataloader:
        loss = compute_loss(model, batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ema.update(model)          # update shadow weights after every step

    # ─── evaluation ──────────────────────────────────────────────────────────
    ema.apply(model)               # swap θ_ema → model params
    evaluate(model)                # inference uses smooth weights
    ema.restore(model)             # swap θ_train back for continued training

References
----------
- Polyak & Juditsky (1992) — "Acceleration of stochastic approximation by averaging"
- Ho et al., "DDPM" (NeurIPS 2020) — used EMA for diffusion model inference
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average of model weights.

    The EMA object is NOT an nn.Module — it lives outside the computational
    graph and is updated imperatively after each optimiser step.

    Args:
        model: The nn.Module whose parameters will be tracked.
        decay: EMA decay factor ∈ (0, 1).  Higher = smoother but slower to
               adapt.  Default 0.995 (≈ 200-step window).
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        # Deep-copy the initial model weights as the shadow parameters.
        # We store them on the same device as the model.
        self._shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        # Keep a backup of the training weights for the apply/restore cycle.
        self._backup: dict[str, torch.Tensor] = {}
        logger.debug(
            "EMA initialised: decay=%.4f, effective_window≈%d steps",
            decay, int(1 / (1 - decay)),
        )

    # ---------------------------------------------------------------------- #
    # Core update                                                             #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Perform one EMA update after an optimiser step.

        θ_ema  ←  decay · θ_ema  +  (1 - decay) · θ_train

        Args:
            model: The model that was just updated by the optimiser.
        """
        for name, param in model.named_parameters():
            if name not in self._shadow:
                # New parameter added after construction — initialise it.
                self._shadow[name] = param.data.clone()
                continue
            shadow = self._shadow[name]
            # In-place update to avoid allocating a new tensor each step
            shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    # ---------------------------------------------------------------------- #
    # Apply / restore (for evaluation)                                        #
    # ---------------------------------------------------------------------- #

    def apply(self, model: nn.Module) -> None:
        """
        Replace model parameters with EMA shadow weights (in-place).

        After calling this, the model can be used for inference.  Always
        call ``restore()`` afterwards to resume training.

        Args:
            model: The model to inject EMA weights into.
        """
        if self._backup:
            raise RuntimeError(
                "EMA.apply() called while backup exists. "
                "Did you forget to call EMA.restore() after the last apply()?"
            )
        for name, param in model.named_parameters():
            self._backup[name] = param.data.clone()          # save training weights
            param.data.copy_(self._shadow[name])              # inject EMA weights

    def restore(self, model: nn.Module) -> None:
        """
        Restore training weights after inference (undoes ``apply``).

        Args:
            model: The model that had EMA weights applied.
        """
        if not self._backup:
            raise RuntimeError(
                "EMA.restore() called with no backup. "
                "Call EMA.apply() first."
            )
        for name, param in model.named_parameters():
            param.data.copy_(self._backup[name])
        self._backup.clear()

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers                                                      #
    # ---------------------------------------------------------------------- #

    def state_dict(self) -> dict:
        """Return a serialisable dict of the shadow weights and decay."""
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from a saved checkpoint."""
        self.decay = state["decay"]
        self._shadow = {
            k: v.clone() for k, v in state["shadow"].items()
        }

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, "
            f"effective_window≈{int(1/(1-self.decay))} steps, "
            f"params={len(self._shadow)})"
        )


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Small dummy model
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.9)
    print(ema)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    initial_w = model.weight.data.clone()

    # Simulate a few training steps
    for step in range(5):
        x = torch.randn(2, 4)
        loss = model(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

    # Verify shadow weights differ from training weights (update happened)
    shadow_w = ema._shadow["weight"].clone()
    assert not torch.allclose(shadow_w, initial_w), "Shadow weights should have changed"
    assert not torch.allclose(shadow_w, model.weight.data), \
        "Shadow should differ from training weights (it lags behind)"

    # Apply / inference / restore cycle
    training_w = model.weight.data.clone()
    ema.apply(model)
    assert torch.allclose(model.weight.data, shadow_w), "EMA weights not applied"

    ema.restore(model)
    assert torch.allclose(model.weight.data, training_w), "Training weights not restored"

    # state_dict round-trip
    sd = ema.state_dict()
    ema2 = EMA(nn.Linear(4, 4), decay=0.5)
    ema2.load_state_dict(sd)
    assert ema2.decay == 0.9
    assert torch.allclose(ema2._shadow["weight"], shadow_w)

    print("All EMA sanity checks passed!")
