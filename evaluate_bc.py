"""
evaluate_bc.py — Evaluate the Behavioral Cloning Baseline
==========================================================

Runs the trained BC policy in the PushT simulation using the same
receding-horizon control loop as evaluate.py.

The BC policy replaces the diffusion denoising loop with a single
forward pass:  obs → model(obs) → action_chunk

Everything else is identical to diffusion evaluation:
  - Same environment
  - Same success threshold (coverage ≥ 0.9)
  - Same receding-horizon execution (T_action = 8 steps per plan)
  - Same 50-episode evaluation protocol

This identical setup makes the comparison fair: any performance gap
is due to the policy's ability to model multimodal distributions,
not due to evaluation differences.

Usage:
    python evaluate_bc.py --checkpoint checkpoints/bc/best.pt
    python evaluate_bc.py --checkpoint checkpoints/bc/best.pt --num_episodes 50
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig
from baselines.bc_policy import BCPolicy
from diffusion_policy.data.normalizer import MinMaxNormalizer

logger = logging.getLogger(__name__)

SUCCESS_THRESHOLD = 0.9   # Must match evaluate.py — coverage ≥ 90%


# ==============================================================================
# Checkpoint loading
# ==============================================================================

def load_bc_policy(
    checkpoint_path: str | Path,
    device: str,
) -> tuple[BCPolicy, MinMaxNormalizer, MinMaxNormalizer, TrainConfig]:
    """
    Load a trained BC policy from a checkpoint.

    Returns:
        (model, obs_normalizer, action_normalizer, cfg)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hp  = ckpt["bc_hparams"]
    cfg: TrainConfig = ckpt["config"]
    cfg.device = device

    model = BCPolicy(
        obs_horizon  = hp["obs_horizon"],
        obs_dim      = hp["obs_dim"],
        pred_horizon = hp["pred_horizon"],
        action_dim   = hp["action_dim"],
        hidden_dim   = hp["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_normalizer = MinMaxNormalizer()
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    action_normalizer = MinMaxNormalizer()
    action_normalizer.load_state_dict(ckpt["action_normalizer"])

    logger.info("Loaded BC policy from %s | %s", checkpoint_path, model)
    return model, obs_normalizer, action_normalizer, cfg


# ==============================================================================
# Single episode rollout
# ==============================================================================

@torch.no_grad()
def run_episode_bc(
    model: BCPolicy,
    obs_normalizer: MinMaxNormalizer,
    action_normalizer: MinMaxNormalizer,
    cfg: TrainConfig,
    save_frames: bool = False,
    seed: int | None = None,
) -> dict:
    """
    Run one evaluation episode with the BC policy and receding-horizon control.

    At each decision step:
      1. Stack the last T_obs observations → normalise
      2. BC model predicts a T_pred-step action chunk  (single forward pass)
      3. Execute the first T_action steps, collect new obs
      4. Repeat until done

    The key contrast with diffusion: step 2 is one matrix multiply,
    not 10–100 iterative denoising steps.  BC is fast but cannot
    recover from multimodal uncertainty.

    Returns:
        Dict: max_score, total_reward, ep_len, success, [frames]
    """
    from diffusion_policy.env.pusht_env import PushTEnv

    env     = PushTEnv(render_size=96, max_episode_steps=cfg.env.max_episode_steps)
    obs_np  = env.reset(seed=seed)

    obs_deque: deque = deque(maxlen=cfg.data.obs_horizon)
    for _ in range(cfg.data.obs_horizon):
        obs_deque.append(obs_np.copy())

    frames       = []
    total_reward = 0.0
    max_score    = 0.0
    ep_len       = 0
    done         = False

    while not done:
        # ── Build normalised observation tensor ──────────────────────────
        obs_seq   = np.stack(list(obs_deque), axis=0)              # (T_obs, obs_dim)
        obs_norm  = obs_normalizer.normalize(obs_seq)
        obs_t     = torch.from_numpy(obs_norm).unsqueeze(0).float().to(cfg.device)
        # shape: (1, T_obs, obs_dim)

        # ── BC prediction (single forward pass — no denoising loop) ─────
        action_norm = model(obs_t)   # (1, T_pred, action_dim)

        # ── Unnormalize and execute T_action steps ────────────────────────
        action_np   = action_norm.squeeze(0).cpu().numpy()          # (T_pred, action_dim)
        action_real = action_normalizer.unnormalize(action_np)

        for step_i in range(cfg.data.action_horizon):
            if done:
                break
            obs_np, reward, done, _ = env.step(action_real[step_i])
            obs_deque.append(obs_np.copy())
            total_reward += reward
            max_score     = max(max_score, reward)
            ep_len       += 1

            if save_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

    env.close()

    result = {
        "max_score":    max_score,
        "total_reward": total_reward,
        "ep_len":       ep_len,
        "success":      float(max_score >= SUCCESS_THRESHOLD),
    }
    if save_frames:
        result["frames"] = frames
    return result


# ==============================================================================
# Batch evaluation
# ==============================================================================

def run_evaluation_bc(
    model: BCPolicy,
    obs_normalizer: MinMaxNormalizer,
    action_normalizer: MinMaxNormalizer,
    cfg: TrainConfig,
    num_episodes: int = 50,
    save_gifs: bool = False,
    gif_dir: str | None = None,
) -> dict:
    """Evaluate BC policy over num_episodes rollouts and return aggregate metrics."""
    scores, successes, ep_lens = [], [], []
    t_total = 0.0

    if gif_dir is None:
        gif_dir = "plots/gifs/bc"

    for ep_i in range(num_episodes):
        t0 = time.time()
        result = run_episode_bc(
            model, obs_normalizer, action_normalizer, cfg,
            save_frames=save_gifs, seed=ep_i,
        )
        t_total += time.time() - t0

        scores.append(result["max_score"])
        successes.append(result["success"])
        ep_lens.append(result["ep_len"])

        if save_gifs and result.get("frames"):
            try:
                from visualize import save_rollout_gif
                Path(gif_dir).mkdir(parents=True, exist_ok=True)
                save_rollout_gif(result["frames"],
                                 f"{gif_dir}/episode_{ep_i:03d}.gif", fps=10)
            except Exception as e:
                logger.warning("Could not save GIF: %s", e)

        if (ep_i + 1) % max(1, num_episodes // 5) == 0:
            logger.info(
                "  Episode %3d/%d | max_score=%.3f | success_rate_so_far=%.3f",
                ep_i + 1, num_episodes, result["max_score"], np.mean(successes),
            )

    inf_time = t_total / max(1, sum(ep_lens))

    metrics = {
        "mean_score":              float(np.mean(scores)),
        "success_rate":            float(np.mean(successes)),
        "mean_ep_len":             float(np.mean(ep_lens)),
        "inference_time_per_step": inf_time,
        "policy":                  "bc",
        "num_episodes":            num_episodes,
    }
    logger.info(
        "BC eval done | mean_score=%.3f | success_rate=%.3f | inf_time/step=%.4fs",
        metrics["mean_score"], metrics["success_rate"], inf_time,
    )
    return metrics


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    p = argparse.ArgumentParser(description="Evaluate BC policy on PushT")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--save_gifs",    action="store_true")
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    model, obs_norm, action_norm, cfg = load_bc_policy(args.checkpoint, device)
    cfg.device = device

    metrics = run_evaluation_bc(
        model, obs_norm, action_norm, cfg,
        num_episodes=args.num_episodes,
        save_gifs=args.save_gifs,
    )

    print("\n" + "=" * 50)
    print("BC EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<30}: {v:.4f}" if isinstance(v, float) else f"  {k:<30}: {v}")
    print("=" * 50)
    print("\nNOTE: Compare this success_rate against Diffusion Policy results.")
    print("The gap demonstrates the multimodal distribution problem BC cannot solve.")
