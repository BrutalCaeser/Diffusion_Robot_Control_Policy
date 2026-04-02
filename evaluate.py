"""
evaluate.py — Policy Evaluation via Receding-Horizon Control
=============================================================

This script loads a trained checkpoint and evaluates the diffusion policy
in the PushT simulation environment using receding-horizon control.

Receding-horizon control (action chunking)
------------------------------------------
At each decision point the policy:

    1. Collects the last T_obs observation frames → obs history
    2. Runs DDPM or DDIM reverse diffusion to generate a full action chunk
       of T_pred steps
    3. Executes only the FIRST T_action of those steps in the environment
    4. Discards the remaining predictions and re-plans with fresh observations

This balances:
    - **Commitment** (T_action > 1): smooth, temporally coherent motion
    - **Reactivity** (T_action < T_pred): respond to unexpected state changes

Why use EMA weights?
--------------------
The EMA model is always used for evaluation (never the raw training weights).
EMA weights are a smoothed average of the last ~200 training iterations,
producing more stable and generalizable noise predictions.

Evaluation metric
-----------------
PushT reports a **coverage score** ∈ [0, 1] measuring the geometric overlap
between the current T-block pose and the target.  An episode is considered
**successful** if the maximum coverage reached in the episode exceeds 0.9
(90% overlap), matching the convention in the original paper.

Usage
-----
    # Fast evaluation (DDIM, 10 steps):
    python evaluate.py --checkpoint checkpoints/best.pt --sampler ddim

    # Full DDPM evaluation (100 steps, slower):
    python evaluate.py --checkpoint checkpoints/best.pt --sampler ddpm

    # Flow Matching evaluation:
    python evaluate.py --checkpoint checkpoints/best_fm.pt --sampler flow

    # Save rollout GIFs:
    python evaluate.py --checkpoint checkpoints/best.pt --save_gifs
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig
from diffusion_policy.data.normalizer import MinMaxNormalizer
from diffusion_policy.model.unet1d import ConditionalUnet1D
from diffusion_policy.model.ddpm import DDPMScheduler
from diffusion_policy.model.ddim import DDIMScheduler
from diffusion_policy.model.ema import EMA
from diffusion_policy.model.flow_matching import FlowMatchingScheduler

logger = logging.getLogger(__name__)

# An episode is "successful" if max coverage exceeds this threshold
SUCCESS_THRESHOLD = 0.9


# ==============================================================================
# Checkpoint loading
# ==============================================================================

def load_policy(
    checkpoint_path: str | Path,
    device: str,
) -> tuple[ConditionalUnet1D, MinMaxNormalizer, MinMaxNormalizer, TrainConfig]:
    """
    Load a trained policy from a checkpoint file.

    Restores:
      - Model architecture (from saved config)
      - EMA weights (used for inference)
      - Observation and action normalizers

    Args:
        checkpoint_path: Path to the ``*.pt`` checkpoint file.
        device:          Target device string.

    Returns:
        (model, obs_normalizer, action_normalizer, cfg)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["config"]
    cfg.device = device

    model = ConditionalUnet1D(
        action_dim               = cfg.env.action_dim,
        obs_horizon              = cfg.data.obs_horizon,
        obs_dim                  = cfg.env.obs_dim,
        diffusion_step_embed_dim = cfg.model.diffusion_step_embed_dim,
        down_dims                = cfg.model.down_dims,
        cond_dim                 = cfg.model.cond_dim,
        kernel_size              = cfg.model.kernel_size,
        n_groups                 = cfg.model.n_groups,
    ).to(device)

    # Load EMA weights (not raw training weights) for evaluation
    ema = EMA(model, decay=cfg.ema_decay)
    ema.load_state_dict(ckpt["ema_state_dict"])
    ema.apply(model)   # inject EMA weights; don't call restore() — inference only

    model.eval()

    obs_normalizer = MinMaxNormalizer()
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    action_normalizer = MinMaxNormalizer()
    action_normalizer.load_state_dict(ckpt["action_normalizer"])

    logger.info(
        "Loaded policy from %s | device=%s | method=%s",
        checkpoint_path, device, cfg.method,
    )
    return model, obs_normalizer, action_normalizer, cfg


# ==============================================================================
# Single episode rollout
# ==============================================================================

@torch.no_grad()
def run_episode(
    model: nn.Module,
    obs_normalizer: MinMaxNormalizer,
    action_normalizer: MinMaxNormalizer,
    cfg: TrainConfig,
    sampler: str,
    ddpm_scheduler: Optional[DDPMScheduler] = None,
    ddim_scheduler: Optional[DDIMScheduler] = None,
    fm_scheduler: Optional[FlowMatchingScheduler] = None,
    save_frames: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Run one evaluation episode with receding-horizon control.

    Args:
        model:             Trained ConditionalUnet1D (with EMA weights loaded).
        obs_normalizer:    Fitted normalizer for observations.
        action_normalizer: Fitted normalizer for actions.
        cfg:               Training configuration.
        sampler:           'ddpm', 'ddim', or 'flow'.
        save_frames:       Whether to collect rendered RGB frames.
        seed:              Optional environment reset seed.

    Returns:
        Dict with keys: 'max_score', 'total_reward', 'ep_len', 'success',
                        and optionally 'frames'.
    """
    from diffusion_policy.env.pusht_env import PushTEnv

    env = PushTEnv(
        render_size       = 96,
        max_episode_steps = cfg.env.max_episode_steps,
    )
    obs_np = env.reset(seed=seed)

    # Observation deque pre-filled with the initial observation
    obs_deque: deque = deque(maxlen=cfg.data.obs_horizon)
    for _ in range(cfg.data.obs_horizon):
        obs_deque.append(obs_np.copy())

    frames = []
    total_reward = 0.0
    max_score    = 0.0
    ep_len       = 0
    done         = False

    while not done:
        # ── Build normalised observation tensor ──────────────────────────
        obs_seq_np  = np.stack(list(obs_deque), axis=0)          # (T_obs, obs_dim)
        obs_norm    = obs_normalizer.normalize(obs_seq_np)        # (T_obs, obs_dim)
        obs_tensor  = torch.from_numpy(obs_norm).unsqueeze(0).to(cfg.device)
        # shape: (1, T_obs, obs_dim)

        # ── Run diffusion sampling ────────────────────────────────────────
        if sampler == "ddpm":
            action_norm = ddpm_scheduler.sample(
                model, obs_tensor,
                pred_horizon=cfg.data.pred_horizon,
                action_dim=cfg.env.action_dim,
            )
        elif sampler == "ddim":
            action_norm = ddim_scheduler.sample(
                model, obs_tensor,
                pred_horizon=cfg.data.pred_horizon,
                action_dim=cfg.env.action_dim,
            )
        else:  # flow
            action_norm = fm_scheduler.sample(
                model, obs_tensor,
                pred_horizon=cfg.data.pred_horizon,
                action_dim=cfg.env.action_dim,
            )

        # ── Unnormalize and execute T_action steps ────────────────────────
        action_norm_np = action_norm.squeeze(0).cpu().numpy()  # (T_pred, action_dim)
        action_real    = action_normalizer.unnormalize(action_norm_np)
        # Execute only the first T_action predictions (receding horizon)
        for step_i in range(cfg.data.action_horizon):
            if done:
                break
            act = action_real[step_i]                   # (action_dim,)
            obs_np, reward, done, info = env.step(act)
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

def run_evaluation(
    model: nn.Module,
    obs_normalizer: MinMaxNormalizer,
    action_normalizer: MinMaxNormalizer,
    cfg: TrainConfig,
    sampler: str = "ddim",
    num_episodes: int = 50,
    save_gifs: bool = False,
    gif_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate the policy over ``num_episodes`` rollouts and aggregate metrics.

    Args:
        model:             Trained ConditionalUnet1D.
        obs_normalizer:    Fitted observation normalizer.
        action_normalizer: Fitted action normalizer.
        cfg:               Training configuration.
        sampler:           Which sampler to use: 'ddpm', 'ddim', or 'flow'.
        num_episodes:      Number of evaluation rollouts.
        save_gifs:         Whether to save rendered GIFs for each episode.
        gif_dir:           Directory to save GIFs (defaults to logs/gifs).

    Returns:
        Dict with aggregate metrics: mean_score, success_rate, mean_ep_len,
        inference_time_per_step.
    """
    device = cfg.device

    # Build schedulers (shared across episodes)
    ddpm_sched = ddim_sched = fm_sched = None
    if sampler in ("ddpm", "ddim"):
        ddpm_sched = DDPMScheduler(
            num_diffusion_steps = cfg.diffusion.num_diffusion_steps,
            beta_start          = cfg.diffusion.beta_start,
            beta_end            = cfg.diffusion.beta_end,
        ).to(device)
        if sampler == "ddim":
            ddim_sched = DDIMScheduler(
                ddpm_scheduler = ddpm_sched,
                ddim_steps     = cfg.diffusion.ddim_steps,
                eta            = cfg.diffusion.ddim_eta,
            )
    else:
        fm_sched = FlowMatchingScheduler(
            num_inference_steps = cfg.flow_matching.num_inference_steps,
        )

    scores   = []
    successes= []
    ep_lens  = []
    t_total  = 0.0

    if gif_dir is None:
        gif_dir = str(Path(cfg.log_dir) / "gifs")

    logger.info(
        "Starting evaluation: sampler=%s, episodes=%d", sampler, num_episodes
    )

    for ep_i in range(num_episodes):
        t0 = time.time()
        result = run_episode(
            model             = model,
            obs_normalizer    = obs_normalizer,
            action_normalizer = action_normalizer,
            cfg               = cfg,
            sampler           = sampler,
            ddpm_scheduler    = ddpm_sched,
            ddim_scheduler    = ddim_sched,
            fm_scheduler      = fm_sched,
            save_frames       = save_gifs,
            seed              = ep_i,           # reproducible per-episode seeds
        )
        t_total += time.time() - t0

        scores.append(result["max_score"])
        successes.append(result["success"])
        ep_lens.append(result["ep_len"])

        if save_gifs and "frames" in result and result["frames"]:
            try:
                from visualize import save_rollout_gif
                Path(gif_dir).mkdir(parents=True, exist_ok=True)
                gif_path = Path(gif_dir) / f"episode_{ep_i:03d}.gif"
                save_rollout_gif(result["frames"], str(gif_path), fps=10)
                logger.info("Saved GIF: %s", gif_path)
            except Exception as exc:
                logger.warning("Could not save GIF: %s", exc)

        if (ep_i + 1) % max(1, num_episodes // 5) == 0:
            sr_so_far = np.mean(successes)
            logger.info(
                "  Episode %3d/%d | max_score=%.3f | success_rate_so_far=%.3f",
                ep_i + 1, num_episodes, result["max_score"], sr_so_far,
            )

    mean_score    = float(np.mean(scores))
    success_rate  = float(np.mean(successes))
    mean_ep_len   = float(np.mean(ep_lens))
    inf_time      = t_total / max(1, sum(ep_lens))  # seconds per environment step

    metrics = {
        "mean_score":             mean_score,
        "success_rate":           success_rate,
        "mean_ep_len":            mean_ep_len,
        "inference_time_per_step": inf_time,
        "sampler":                sampler,
        "num_episodes":           num_episodes,
    }

    logger.info(
        "Evaluation done | mean_score=%.3f | success_rate=%.3f | "
        "mean_ep_len=%.1f | inf_time/step=%.3fs",
        mean_score, success_rate, mean_ep_len, inf_time,
    )
    return metrics


# ==============================================================================
# CLI entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Diffusion Policy on PushT")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to checkpoint (.pt) file")
    p.add_argument("--sampler",      type=str, default="ddim",
                   choices=["ddpm", "ddim", "flow"],
                   help="Which sampler to use for inference")
    p.add_argument("--num_episodes", type=int, default=50,
                   help="Number of evaluation rollout episodes")
    p.add_argument("--device",       type=str, default=None,
                   help="Device override ('cpu', 'cuda', 'mps')")
    p.add_argument("--save_gifs",    action="store_true",
                   help="Save rendered rollout GIFs")
    p.add_argument("--gif_dir",      type=str, default=None,
                   help="Directory to save GIFs")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")

    model, obs_norm, action_norm, cfg = load_policy(args.checkpoint, device)
    cfg.device = device

    metrics = run_evaluation(
        model             = model,
        obs_normalizer    = obs_norm,
        action_normalizer = action_norm,
        cfg               = cfg,
        sampler           = args.sampler,
        num_episodes      = args.num_episodes,
        save_gifs         = args.save_gifs,
        gif_dir           = args.gif_dir,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        else:
            print(f"  {k:<30}: {v}")
    print("=" * 50)
