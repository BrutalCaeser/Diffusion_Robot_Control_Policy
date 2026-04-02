"""
pusht_env.py — PushT Environment Wrapper
==========================================

This module provides a clean wrapper around the PushT simulation environment,
ensuring a consistent observation and action interface for both data collection
and policy evaluation.

PushT environment description
------------------------------
PushT is a 2D planar manipulation task:
    - A circular AGENT (the "finger") can move in the plane
    - A T-shaped BLOCK sits on the table
    - The goal is to PUSH the T-block to match a target pose (position + angle)
    - The reward is based on the overlap between the current T-block pose and
      the target T-block pose

State space (obs_dim = 5):
    [agent_x, agent_y, block_x, block_y, block_angle]

Action space (action_dim = 2):
    [velocity_x, velocity_y] — velocity command for the circular agent

Why PushT?
----------
1. Lightweight: trains in 2-4 hours on a single GPU
2. Visually intuitive: easy to see if the policy is working
3. Multimodal: the agent can push from either side → natural test for
   diffusion's multimodality handling
4. Canonical benchmark: used in the original Diffusion Policy paper, so
   we have reference numbers to compare against

Physics engine: pymunk (2D rigid body physics)
Rendering: pygame

Installation
------------
    pip install gym-pusht

References:
    - Florence et al., "Implicit Behavioral Cloning" — original PushT env
    - Chi et al., "Diffusion Policy" — adopted as primary benchmark
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class PushTEnv:
    """
    Thin wrapper around the ``gym_pusht/PushT-v0`` Gymnasium environment.

    Provides a unified interface for:
    - Resetting the environment and obtaining the initial observation
    - Stepping with an action and receiving (obs, reward, done, info)
    - Rendering the current state to an RGB array
    - Maintaining an observation history deque for receding-horizon control

    The raw environment uses:
        observation_space: Box(5,) in [0, 512] (approx)
        action_space:      Box(2,) in [0, 512] (approx)

    We do NOT normalize inside the environment wrapper — normalization happens
    in the dataset and the evaluation loop so that the same MinMaxNormalizer
    statistics are used consistently.

    Args:
        render_size:       Side length (pixels) of the rendered square image.
        max_episode_steps: Episode length cap (default 300, matching training data).

    Raises:
        ImportError: if ``gym_pusht`` is not installed.
    """

    def __init__(
        self,
        render_size: int = 96,
        max_episode_steps: int = 300,
    ) -> None:
        try:
            import gymnasium as gym
            import gym_pusht  # noqa: F401 — registers the env
        except ImportError as e:
            raise ImportError(
                "gym_pusht is required for environment interaction.\n"
                "Install it with:  pip install gym-pusht\n"
                f"Original error: {e}"
            ) from e

        self._env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="state",
            render_mode="rgb_array",
            observation_width=render_size,
            observation_height=render_size,
            max_episode_steps=max_episode_steps,
        )
        self.obs_dim: int = 5
        self.action_dim: int = 2
        self.max_episode_steps: int = max_episode_steps
        self._step_count: int = 0
        self._last_obs: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------- #
    # Core interface                                                           #
    # ---------------------------------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        Args:
            seed: optional random seed for the reset (for reproducibility).

        Returns:
            obs: np.ndarray of shape (5,) — the initial state.
        """
        obs, _info = self._env.reset(seed=seed)
        self._step_count = 0
        self._last_obs = obs.astype(np.float32)
        return self._last_obs

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one action in the environment.

        Args:
            action: np.ndarray of shape (2,) — velocity command (unnormalized,
                    in the original action scale of the environment).

        Returns:
            obs:    np.ndarray (5,) — new state observation
            reward: float — coverage score ∈ [0, 1]
            done:   bool — True if episode has ended
            info:   dict — extra diagnostics (e.g. 'is_success')
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        done = bool(terminated or truncated)
        self._last_obs = obs.astype(np.float32)
        return self._last_obs, float(reward), done, info

    def render(self) -> np.ndarray:
        """
        Render the current environment state to an RGB image.

        Returns:
            image: np.ndarray of shape (render_size, render_size, 3), dtype uint8.
        """
        return self._env.render()

    def close(self) -> None:
        """Release environment resources."""
        self._env.close()

    # ---------------------------------------------------------------------- #
    # Observation history helper (for receding-horizon control)               #
    # ---------------------------------------------------------------------- #

    def make_obs_deque(self, obs_horizon: int) -> deque:
        """
        Create a fixed-length deque pre-filled with the current observation.

        During receding-horizon control we need a rolling buffer of the last
        T_obs observations. When the episode starts, there is only one
        observation, so we pad by repeating it obs_horizon times.

        Usage::

            env.reset()
            obs_deque = env.make_obs_deque(obs_horizon=2)
            # obs_deque contains [obs, obs] — same obs repeated twice

            obs, reward, done, info = env.step(action)
            obs_deque.append(obs)
            # obs_deque now contains [obs_t, obs_t+1]

        Args:
            obs_horizon: T_obs — size of the deque.

        Returns:
            deque of maxlen=obs_horizon, pre-filled with self._last_obs.
        """
        if self._last_obs is None:
            raise RuntimeError("Call reset() before make_obs_deque().")
        q: deque = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            q.append(self._last_obs.copy())
        return q

    @staticmethod
    def deque_to_array(obs_deque: deque) -> np.ndarray:
        """
        Convert an observation deque to a numpy array.

        Args:
            obs_deque: deque of np.ndarray, each shape (obs_dim,).

        Returns:
            np.ndarray of shape (T_obs, obs_dim).
        """
        return np.stack(list(obs_deque), axis=0).astype(np.float32)

    # ---------------------------------------------------------------------- #
    # Properties                                                               #
    # ---------------------------------------------------------------------- #

    @property
    def step_count(self) -> int:
        return self._step_count

    def __repr__(self) -> str:
        return (
            f"PushTEnv(obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"max_episode_steps={self.max_episode_steps})"
        )


# ==============================================================================
# Quick sanity check — run directly to verify env works
# ==============================================================================

if __name__ == "__main__":
    env = PushTEnv(render_size=96, max_episode_steps=10)
    print(env)

    obs = env.reset(seed=42)
    print(f"Initial obs: {obs} (shape: {obs.shape})")

    for step_i in range(5):
        action = np.array([256.0, 256.0], dtype=np.float32)  # move toward center
        obs, reward, done, info = env.step(action)
        print(f"  Step {step_i+1}: reward={reward:.4f}, done={done}")

    frame = env.render()
    print(f"Rendered frame shape: {frame.shape}")

    obs_deque = env.make_obs_deque(obs_horizon=2)
    print(f"Obs deque maxlen: {obs_deque.maxlen}")

    env.close()
    print("PushT env sanity check passed!")
