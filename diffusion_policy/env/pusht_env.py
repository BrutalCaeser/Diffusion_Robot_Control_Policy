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

Implementation plan (Phase 1):
    - PushTEnv class wrapping the raw environment with:
        - reset() → obs
        - step(action) → obs, reward, done, info
        - render() → image array (for visualization / GIF creation)
        - get_episode_data() → full trajectory for dataset creation
        - Consistent normalization interface with the dataset normalizer

References:
    - Florence et al., "Implicit Behavioral Cloning" — original PushT env
    - Chi et al., "Diffusion Policy" — adopted as primary benchmark
"""

# Implementation will be added in Phase 1.
