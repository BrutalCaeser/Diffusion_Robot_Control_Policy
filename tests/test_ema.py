"""
tests/test_ema.py — Unit tests for EMA.

Run with:  pytest tests/test_ema.py -v
"""

import pytest
import torch
import torch.nn as nn

from diffusion_policy.model.ema import EMA


@pytest.fixture
def model():
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    torch.manual_seed(0)
    return m


@pytest.fixture
def ema(model):
    return EMA(model, decay=0.9)


class TestEMAInit:
    def test_shadow_params_match_model(self, model, ema):
        for name, param in model.named_parameters():
            torch.testing.assert_close(ema._shadow[name], param.data)

    def test_invalid_decay_raises(self, model):
        with pytest.raises(ValueError):
            EMA(model, decay=0.0)
        with pytest.raises(ValueError):
            EMA(model, decay=1.0)
        with pytest.raises(ValueError):
            EMA(model, decay=-0.1)


class TestEMAUpdate:
    def test_shadow_changes_after_update(self, model, ema):
        initial = {k: v.clone() for k, v in ema._shadow.items()}
        # Modify model weights
        for p in model.parameters():
            p.data.fill_(99.0)
        ema.update(model)
        for name in initial:
            assert not torch.allclose(ema._shadow[name], initial[name])

    def test_shadow_lags_behind_model(self, model, ema):
        """EMA shadow should be between initial value and current model weights."""
        init_shadow = ema._shadow["0.weight"].clone()
        # Push model weights to a very different value
        for p in model.parameters():
            p.data.fill_(10.0)
        ema.update(model)
        shadow_after = ema._shadow["0.weight"]
        # shadow should be between initial (~0) and model (10) values
        assert (shadow_after < 10.0).all()
        assert (shadow_after > init_shadow).all()


class TestApplyRestore:
    def test_apply_injects_ema_weights(self, model, ema):
        # Do a few updates so shadow differs from model
        for p in model.parameters():
            p.data.fill_(5.0)
        ema.update(model)

        expected_shadow = {k: v.clone() for k, v in ema._shadow.items()}
        ema.apply(model)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.data, expected_shadow[name])

    def test_restore_returns_training_weights(self, model, ema):
        for p in model.parameters():
            p.data.fill_(5.0)
        ema.update(model)

        train_weights = {name: p.data.clone() for name, p in model.named_parameters()}
        ema.apply(model)
        ema.restore(model)

        for name, param in model.named_parameters():
            torch.testing.assert_close(param.data, train_weights[name])

    def test_double_apply_raises(self, model, ema):
        ema.apply(model)
        with pytest.raises(RuntimeError, match="backup exists"):
            ema.apply(model)
        ema.restore(model)

    def test_restore_without_apply_raises(self, model, ema):
        with pytest.raises(RuntimeError, match="no backup"):
            ema.restore(model)


class TestCheckpoint:
    def test_state_dict_round_trip(self, model, ema):
        for p in model.parameters():
            p.data.fill_(3.0)
        ema.update(model)

        sd = ema.state_dict()
        ema2 = EMA(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)), decay=0.5)
        ema2.load_state_dict(sd)

        assert ema2.decay == 0.9
        for name in ema._shadow:
            torch.testing.assert_close(ema2._shadow[name], ema._shadow[name])

    def test_repr(self, ema):
        r = repr(ema)
        assert "EMA" in r
        assert "decay" in r
