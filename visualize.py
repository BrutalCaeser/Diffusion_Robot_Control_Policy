"""
visualize.py — Visualization Utilities for Diffusion Policy
=============================================================

This module provides publication-quality plots and GIFs for:
    1. Dataset analysis (action/observation distributions, sample trajectories)
    2. Training diagnostics (loss curves, LR schedule, gradient norms)
    3. Evaluation results (success rate plots, score distributions)
    4. Diffusion process (forward noising + reverse denoising, step by step)
    5. Ablation comparisons (DDPM vs DDIM vs Flow Matching)

All plotting uses matplotlib with a consistent dark-ish scientific style.
Figures are saved to files — no interactive display needed.

Usage
-----
    # Dataset analysis
    from visualize import plot_dataset_summary
    plot_dataset_summary("data/pusht_cchi_v7_replay.zarr", save_dir="plots/")

    # After training, plot loss curve from CSV log
    from visualize import plot_training_curves
    plot_training_curves("logs/ddpm_20240401_120000_metrics.csv", "plots/")

    # Visualise forward diffusion process
    from visualize import visualize_diffusion_process
    visualize_diffusion_process(scheduler, sample_action, save_path="plots/diffusion.png")

    # Save GIF from frames collected during evaluation
    from visualize import save_rollout_gif
    save_rollout_gif(frames, "plots/rollout.gif", fps=10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# Use a non-interactive backend so this works headlessly
matplotlib.use("Agg")

# Consistent colour palette
_PALETTE = {
    "ddpm":  "#4C72B0",
    "ddim":  "#DD8452",
    "flow":  "#55A868",
    "train": "#4C72B0",
    "eval":  "#DD8452",
    "noise": "#C44E52",
    "clean": "#55A868",
}


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ==============================================================================
# 1. Dataset analysis
# ==============================================================================

def plot_dataset_summary(
    dataset_path: str,
    save_dir: str = "plots",
) -> None:
    """
    Load the PushT Zarr dataset and generate summary plots:
        - Action distribution (2D scatter + marginals)
        - Observation distribution for each dimension
        - Episode length histogram

    Args:
        dataset_path: Path to ``*.zarr`` dataset directory.
        save_dir:     Output directory for saved plots.
    """
    import zarr
    root = zarr.open(dataset_path, "r")
    states  = root["data"]["state"][:]        # (N, 5)
    actions = root["data"]["action"][:]       # (N, 2)
    ep_ends = root["meta"]["episode_ends"][:] # (num_eps,)

    # ── Action 2D scatter ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PushT Dataset Summary", fontsize=14, fontweight="bold")

    axes[0].scatter(
        actions[:, 0], actions[:, 1],
        alpha=0.05, s=1, c=_PALETTE["clean"],
    )
    axes[0].set_xlabel("action_x (velocity)")
    axes[0].set_ylabel("action_y (velocity)")
    axes[0].set_title("Action Distribution (2D scatter)")
    axes[0].set_aspect("equal")

    # Episode length histogram
    ep_lens = np.diff(np.concatenate([[0], ep_ends]))
    axes[1].hist(ep_lens, bins=30, color=_PALETTE["train"], edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Episode length (steps)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Episode Lengths (n={len(ep_lens)} episodes)")
    axes[1].axvline(ep_lens.mean(), color="red", ls="--", label=f"mean={ep_lens.mean():.0f}")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, Path(save_dir) / "dataset_summary.png")

    # ── Observation dimension distributions ──────────────────────────────
    obs_labels = ["agent_x", "agent_y", "block_x", "block_y", "block_angle"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3))
    fig.suptitle("Observation Dimension Distributions", fontsize=12, fontweight="bold")
    for i, (ax, label) in enumerate(zip(axes, obs_labels)):
        ax.hist(states[:, i], bins=50, color=_PALETTE["ddpm"], alpha=0.8, edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("value")
    plt.tight_layout()
    _save(fig, Path(save_dir) / "obs_distributions.png")


def plot_action_trajectory(
    actions: np.ndarray,
    save_path: str = "plots/action_trajectory.png",
    title: str = "Action Trajectory",
) -> None:
    """
    Plot a single action trajectory (T_pred, 2) as a line with a colour
    gradient showing temporal progression.

    Args:
        actions:   (T_pred, 2) array of (vx, vy) actions.
        save_path: Output path.
        title:     Plot title.
    """
    T = actions.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, T))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for t in range(T - 1):
        axes[0].plot(
            actions[t:t+2, 0], actions[t:t+2, 1],
            color=colors[t], linewidth=2,
        )
    axes[0].scatter(*actions[0], color="green", s=60, zorder=5, label="start")
    axes[0].scatter(*actions[-1], color="red",   s=60, zorder=5, label="end")
    axes[0].set_xlabel("vx"); axes[0].set_ylabel("vy")
    axes[0].set_title("2D Trajectory"); axes[0].legend()

    t_axis = np.arange(T)
    axes[1].plot(t_axis, actions[:, 0], label="vx", color=_PALETTE["ddpm"])
    axes[1].plot(t_axis, actions[:, 1], label="vy", color=_PALETTE["ddim"])
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Value")
    axes[1].set_title("Action components over time"); axes[1].legend()

    plt.tight_layout()
    _save(fig, save_path)


# ==============================================================================
# 2. Training curves
# ==============================================================================

def plot_training_curves(
    csv_path: str,
    save_dir: str = "plots",
) -> None:
    """
    Parse the CSV metric log written by train.py and generate:
        - Training loss curve (per step)
        - Evaluation success rate over epochs
        - Learning rate schedule

    Args:
        csv_path: Path to the ``*_metrics.csv`` file from training.
        save_dir: Directory to save plots.
    """
    import csv as csv_mod

    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append({k: _safe_float(v) for k, v in row.items()})

    if not rows:
        logger.warning("CSV log is empty: %s", csv_path)
        return

    steps = [r["step"] for r in rows if "step" in r and "loss" in r]
    losses= [r["loss"] for r in rows if "step" in r and "loss" in r]

    eval_rows = [r for r in rows if "success_rate" in r and r.get("success_rate") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Diagnostics", fontsize=14, fontweight="bold")

    if steps:
        axes[0].plot(steps, losses, color=_PALETTE["train"], lw=0.8, alpha=0.7)
        # Running mean (window=50)
        if len(losses) > 50:
            window = 50
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            axes[0].plot(
                steps[window-1:], smoothed,
                color="red", lw=1.5, label=f"moving avg (w={window})",
            )
            axes[0].legend()
        axes[0].set_xlabel("Training step"); axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Training Loss")
        axes[0].set_yscale("log")

    if eval_rows:
        eval_epochs = [r["epoch"] for r in eval_rows]
        success_rates = [r["success_rate"] for r in eval_rows]
        mean_scores   = [r.get("mean_score", 0) for r in eval_rows]

        axes[1].plot(eval_epochs, success_rates, "o-",
                     color=_PALETTE["eval"], label="success rate (≥0.9)")
        axes[1].plot(eval_epochs, mean_scores, "s--",
                     color=_PALETTE["ddim"], label="mean score")
        axes[1].axhline(0.9, ls=":", color="gray", label="threshold=0.9")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
        axes[1].set_title("Evaluation Performance"); axes[1].legend()

    plt.tight_layout()
    _save(fig, Path(save_dir) / "training_curves.png")


def _safe_float(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ==============================================================================
# 3. Diffusion process visualisation
# ==============================================================================

def visualize_diffusion_process(
    scheduler,
    clean_action: np.ndarray,
    save_path: str = "plots/diffusion_process.png",
    num_viz_steps: int = 10,
) -> None:
    """
    Visualize the DDPM forward and (approximate) reverse process on a single
    action sequence.

    Left panel:  Forward process — shows a_0, a_10, a_20, ..., a_K (noising)
    Right panel: Overlay of all noisy levels as scatter, showing noise growth

    Args:
        scheduler:       A fitted DDPMScheduler.
        clean_action:    (T_pred, action_dim) — one sample clean action.
        save_path:       Output path.
        num_viz_steps:   Number of noise levels to visualise.
    """
    import torch

    K = scheduler.K
    action_np = clean_action
    T, A = action_np.shape

    # Select evenly spaced timesteps to visualise
    viz_steps = np.linspace(0, K - 1, num_viz_steps, dtype=int)
    action_t  = torch.from_numpy(action_np).float().unsqueeze(0)  # (1, T, A)

    noisy_actions = []
    for k in viz_steps:
        noise = torch.randn_like(action_t)
        ks    = torch.tensor([k], dtype=torch.long)
        a_k   = scheduler.add_noise(action_t, noise, ks)
        noisy_actions.append(a_k.squeeze(0).numpy())

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    fig.suptitle("DDPM Forward Diffusion Process", fontsize=13, fontweight="bold")

    cmap = plt.cm.plasma
    for i, (k, a_k) in enumerate(zip(viz_steps, noisy_actions)):
        color = cmap(i / len(viz_steps))
        ax0.plot(np.arange(T), a_k[:, 0],
                 color=color, alpha=0.8, lw=1.2, label=f"k={k}" if i in (0, len(viz_steps)-1) else "")
    ax0.set_xlabel("Timestep in action sequence")
    ax0.set_ylabel("Action component 0 (vx)")
    ax0.set_title("Noising of action[0] across diffusion steps")
    ax0.legend()

    for i, (k, a_k) in enumerate(zip(viz_steps, noisy_actions)):
        color = cmap(i / len(viz_steps))
        ax1.scatter(a_k[:, 0], a_k[:, 1], color=color, alpha=0.4, s=20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, K-1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Diffusion step k")
    ax1.set_xlabel("vx (action dim 0)")
    ax1.set_ylabel("vy (action dim 1)")
    ax1.set_title("Action 2D distribution across noise levels")

    _save(fig, save_path)


# ==============================================================================
# 4. GIF saving
# ==============================================================================

def save_rollout_gif(
    frames: list[np.ndarray],
    path: str,
    fps: int = 10,
) -> None:
    """
    Save a list of RGB frames as an animated GIF.

    Args:
        frames: List of np.ndarray, each shape (H, W, 3) dtype uint8.
        path:   Output file path (e.g., "rollout.gif").
        fps:    Frames per second.
    """
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    logger.info("Saved GIF (%d frames, %.1f fps): %s", len(frames), fps, path)


# ==============================================================================
# 5. Ablation comparison
# ==============================================================================

def plot_eval_comparison(
    results: dict[str, dict],
    save_path: str = "plots/ablation_comparison.png",
    title: str = "Method Comparison on PushT",
) -> None:
    """
    Bar chart comparing evaluation metrics across different methods/settings.

    Args:
        results: Dict of {method_name: {metric_name: value}}.
                 e.g. {"DDPM (100 steps)": {"success_rate": 0.82, ...},
                        "DDIM (10 steps)":  {"success_rate": 0.79, ...}}
        save_path: Output path.
        title:     Figure title.
    """
    methods = list(results.keys())
    metrics = ["success_rate", "mean_score"]
    colors  = list(_PALETTE.values())[:len(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0.0) for m in methods]
        bars = ax.bar(methods, values, color=colors, edgecolor="white", alpha=0.85)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.axhline(0.9, ls=":", color="gray", lw=1, label="target (0.9)")
        ax.legend()

    plt.tight_layout()
    _save(fig, save_path)


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        # Plot action trajectory with random data
        actions = np.random.randn(16, 2).astype(np.float32)
        plot_action_trajectory(actions, save_path=os.path.join(tmp, "traj.png"))

        # Plot comparison
        results = {
            "DDPM (100)": {"success_rate": 0.82, "mean_score": 0.78},
            "DDIM (10)":  {"success_rate": 0.79, "mean_score": 0.75},
            "Flow Match": {"success_rate": 0.85, "mean_score": 0.81},
        }
        plot_eval_comparison(results, save_path=os.path.join(tmp, "comparison.png"))

        # GIF test (small frames)
        frames = [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(5)]
        save_rollout_gif(frames, os.path.join(tmp, "test.gif"), fps=5)

        print("All visualize sanity checks passed!")
