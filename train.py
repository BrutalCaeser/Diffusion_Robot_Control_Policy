"""
train.py — Main Training Script for Diffusion Policy
======================================================

This script orchestrates the full training pipeline:

    1. Parse CLI arguments and build config
    2. Set random seeds for reproducibility
    3. Build dataset and DataLoader
    4. Construct the ConditionalUnet1D model
    5. Build the DDPM or Flow Matching scheduler
    6. Initialize AdamW optimizer, cosine-warmup LR scheduler, EMA
    7. Training loop:
        a. Sample (obs, action) batch
        b. DDPM: sample timestep → add noise → predict noise → MSE loss
           FM:   sample t ~ U[0,1] → interpolate → predict velocity → MSE loss
        c. Backprop + gradient clip → optimizer step → EMA update → LR step
        d. Log loss to console and to CSV every N steps
    8. Periodic checkpointing (model, EMA, optimiser, normaliser stats)
    9. Periodic evaluation rollouts in PushT (via evaluate.py)

Usage
-----
    # Local CPU/MPS testing:
    python train.py --batch_size 4 --num_epochs 2

    # Full GPU training:
    python train.py

    # Flow Matching (Phase 5):
    python train.py --method flow_matching

    # Resume from checkpoint:
    python train.py --resume checkpoints/epoch_050.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from diffusion_policy.data.dataset       import PushTStateDataset
from diffusion_policy.data.image_dataset import PushTImageDataset
from diffusion_policy.model.unet1d       import ConditionalUnet1D
from diffusion_policy.model.vision_encoder import ResNetEncoder
from diffusion_policy.model.ddpm         import DDPMScheduler
from diffusion_policy.model.ddim         import DDIMScheduler
from diffusion_policy.model.ema          import EMA
from diffusion_policy.model.flow_matching import FlowMatchingScheduler


# ==============================================================================
# Logging setup
# ==============================================================================

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """
    Configure root logger to write to both stdout and a file.

    Args:
        log_dir:  Directory where log files are saved.
        run_name: Unique name for this run (used in filenames).

    Returns:
        Logger instance for this script.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{run_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    logger = logging.getLogger("train")
    logger.info("Logging to %s", log_path)
    return logger


# ==============================================================================
# Learning-rate schedule (cosine annealing with linear warmup)
# ==============================================================================

def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Learning-rate schedule: linear warmup then cosine decay.

    During warmup (step < warmup_steps):
        lr = learning_rate * (step / warmup_steps)

    During cosine decay (step >= warmup_steps):
        lr = learning_rate * 0.5 * (1 + cos(π · progress))
        where progress = (step - warmup) / (total - warmup)

    This prevents training instability from large gradients in the first
    few steps (when model weights are random) and allows fine-tuning with
    a very small LR near the end of training.

    Args:
        optimizer:    The AdamW optimizer.
        warmup_steps: Number of linear-warmup steps.
        total_steps:  Total number of training steps (epochs × batches/epoch).

    Returns:
        A LambdaLR scheduler that wraps the optimizer.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# Checkpoint helpers
# ==============================================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    obs_normalizer_state: dict,
    action_normalizer_state: dict,
    metrics: dict,
    cfg: TrainConfig,
) -> None:
    """Save a full training checkpoint to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "obs_normalizer": obs_normalizer_state,
            "action_normalizer": action_normalizer_state,
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> dict:
    """
    Restore model, EMA, optimizer, and LR scheduler from a checkpoint.

    Returns:
        The saved ``metrics`` dict (e.g. best success rate so far).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    return ckpt.get("metrics", {})


# ==============================================================================
# CSV metric logger
# ==============================================================================

class MetricLogger:
    """Simple CSV logger — writes one row per call to ``log()``."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._header_written = path.exists()  # don't overwrite existing file
        self._file = open(path, "a")

    def log(self, **kwargs: float | int | str) -> None:
        if not self._header_written:
            self._file.write(",".join(str(k) for k in kwargs) + "\n")
            self._header_written = True
        self._file.write(",".join(str(v) for v in kwargs.values()) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ==============================================================================
# Seed
# ==============================================================================

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Main training function
# ==============================================================================

def train(cfg: TrainConfig, resume_path: str | None = None) -> None:
    """
    Run the full training loop.

    Args:
        cfg:         Training configuration (from config.py).
        resume_path: Optional path to a checkpoint to resume from.
    """
    run_name = f"{cfg.method}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger   = setup_logging(cfg.log_dir, run_name)
    logger.info("Run: %s | device: %s | method: %s", run_name, cfg.device, cfg.method)

    seed_everything(cfg.seed)

    # ── Paths ────────────────────────────────────────────────────────────────
    ckpt_dir   = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metric_log = MetricLogger(Path(cfg.log_dir) / f"{run_name}_metrics.csv")

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    logger.info(
        "Loading dataset from: %s  [obs_type=%s]",
        cfg.data.dataset_path, cfg.data.obs_type,
    )
    if cfg.data.obs_type == "image":
        dataset = PushTImageDataset(
            dataset_path=cfg.data.dataset_path,
            obs_horizon=cfg.data.obs_horizon,
            pred_horizon=cfg.data.pred_horizon,
        )
        obs_normalizer    = None          # images are not min-max normalised
        action_normalizer = dataset.get_action_normalizer()
    else:
        dataset = PushTStateDataset(
            dataset_path=cfg.data.dataset_path,
            obs_horizon=cfg.data.obs_horizon,
            pred_horizon=cfg.data.pred_horizon,
        )
        obs_normalizer, action_normalizer = dataset.get_normalizers()
    logger.info("Dataset: %s", dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
        drop_last=True,
    )
    logger.info(
        "DataLoader: batch_size=%d, num_workers=%d, batches/epoch=%d",
        cfg.batch_size, cfg.num_workers, len(dataloader),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    # For image observations the UNet conditioning comes from the ResNetEncoder
    # (obs_cond_dim) instead of the flattened state vector (obs_horizon * obs_dim).
    # Both must equal cfg.model.cond_dim.
    vision_encoder: ResNetEncoder | None = None
    if cfg.data.obs_type == "image":
        vision_encoder = ResNetEncoder(
            obs_horizon    = cfg.data.obs_horizon,
            obs_cond_dim   = cfg.vision.obs_cond_dim,
            pretrained     = cfg.vision.pretrained,
            freeze_backbone= cfg.vision.freeze_backbone,
        ).to(cfg.device)
        logger.info("VisionEncoder: %s", vision_encoder)
        # The UNet's obs_dim is set to 0 when using a vision encoder so that
        # its internal obs embedding MLP is bypassed.  The cond vector is
        # injected directly by the training loop (see below).
        unet_obs_dim = 0
    else:
        unet_obs_dim = cfg.env.obs_dim

    model = ConditionalUnet1D(
        action_dim               = cfg.env.action_dim,
        obs_horizon              = cfg.data.obs_horizon,
        obs_dim                  = unet_obs_dim,
        diffusion_step_embed_dim = cfg.model.diffusion_step_embed_dim,
        down_dims                = cfg.model.down_dims,
        cond_dim                 = cfg.model.cond_dim,
        kernel_size              = cfg.model.kernel_size,
        n_groups                 = cfg.model.n_groups,
    ).to(cfg.device)
    logger.info("UNet parameters: %.2fM", model.num_parameters() / 1e6)

    # Combine UNet + VisionEncoder parameters for the optimizer
    all_params = list(model.parameters())
    if vision_encoder is not None:
        all_params += list(vision_encoder.parameters())
    total_params = sum(p.numel() for p in all_params) / 1e6
    logger.info("Total trainable parameters: %.2fM", total_params)

    # ── Scheduler ────────────────────────────────────────────────────────────
    if cfg.method == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_diffusion_steps = cfg.diffusion.num_diffusion_steps,
            beta_start          = cfg.diffusion.beta_start,
            beta_end            = cfg.diffusion.beta_end,
        ).to(cfg.device)
        ddim_scheduler = DDIMScheduler(
            ddpm_scheduler = noise_scheduler,
            ddim_steps     = cfg.diffusion.ddim_steps,
            eta            = cfg.diffusion.ddim_eta,
        )
    else:
        fm_scheduler = FlowMatchingScheduler(
            num_inference_steps = cfg.flow_matching.num_inference_steps,
        )

    # ── EMA, optimizer, LR schedule ──────────────────────────────────────────
    ema = EMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        all_params,
        lr           = cfg.learning_rate,
        betas        = cfg.betas,
        weight_decay = cfg.weight_decay,
    )

    total_steps = cfg.num_epochs * len(dataloader)
    lr_sched = get_cosine_warmup_scheduler(optimizer, cfg.lr_warmup_steps, total_steps)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_metrics = {"best_success_rate": 0.0}
    if resume_path:
        logger.info("Resuming from checkpoint: %s", resume_path)
        saved_metrics = load_checkpoint(
            Path(resume_path), model, ema, optimizer, lr_sched
        )
        best_metrics.update(saved_metrics)
        # Infer start epoch from checkpoint filename
        try:
            start_epoch = int(Path(resume_path).stem.split("_")[-1]) + 1
        except ValueError:
            pass
        logger.info("Resumed from epoch %d", start_epoch)

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        if vision_encoder is not None:
            vision_encoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:04d}/{cfg.num_epochs}", leave=False)
        for batch in pbar:
            obs_raw = batch["obs"].to(cfg.device, non_blocking=True)
            action  = batch["action"].to(cfg.device, non_blocking=True)  # (B, T_pred, 2)

            # ── Encode observations ────────────────────────────────────────
            if vision_encoder is not None:
                # obs_raw: (B, T_obs, 3, H, W) — encode to (B, cond_dim)
                # We pass it as obs to the UNet; since unet_obs_dim=0 the UNet
                # ignores the obs input and relies on the pre-computed cond
                # vector injected via the 'obs' argument being a 2-D tensor
                # (B, cond_dim) rather than (B, T_obs, obs_dim).
                obs = vision_encoder(obs_raw)   # (B, cond_dim) — used directly
            else:
                obs = obs_raw                   # (B, T_obs, obs_dim) — state

            # ── Compute loss ──────────────────────────────────────────────
            if cfg.method == "ddpm":
                timesteps    = noise_scheduler.sample_timesteps(obs.shape[0])
                noise        = torch.randn_like(action)
                noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
                noise_pred   = model(noisy_action, timesteps, obs)
                loss         = F.mse_loss(noise_pred, noise)
            else:  # flow_matching
                loss = fm_scheduler.get_loss(model, action, obs)

            # ── Backward pass ─────────────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            lr_sched.step()
            ema.update(model)

            epoch_loss  += loss.item()
            global_step += 1

            # ── Periodic logging ──────────────────────────────────────────
            if global_step % cfg.log_interval == 0:
                lr_now = lr_sched.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")
                metric_log.log(
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    lr=lr_now,
                )

        avg_loss = epoch_loss / len(dataloader)
        elapsed  = time.time() - t0
        logger.info(
            "Epoch %04d/%d | loss=%.4f | time=%.1fs | lr=%.2e",
            epoch + 1, cfg.num_epochs, avg_loss, elapsed,
            lr_sched.get_last_lr()[0],
        )
        metric_log.log(epoch=epoch, step=global_step, epoch_loss=avg_loss)

        # ── Checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            save_checkpoint(
                path                    = ckpt_path,
                epoch                   = epoch,
                model                   = model,
                ema                     = ema,
                optimizer               = optimizer,
                lr_scheduler            = lr_sched,
                obs_normalizer_state    = obs_normalizer.state_dict() if obs_normalizer else {},
                action_normalizer_state = action_normalizer.state_dict(),
                metrics                 = best_metrics,
                cfg                     = cfg,
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

        # ── Evaluation ────────────────────────────────────────────────────
        if (epoch + 1) % cfg.eval_interval == 0:
            try:
                from evaluate import run_evaluation
                ema.apply(model)
                eval_metrics = run_evaluation(
                    model            = model,
                    obs_normalizer   = obs_normalizer,
                    action_normalizer= action_normalizer,
                    cfg              = cfg,
                    sampler          = "ddim" if cfg.method == "ddpm" else "flow",
                    num_episodes     = cfg.num_eval_episodes,
                    save_gifs        = False,
                )
                ema.restore(model)

                success_rate = eval_metrics.get("success_rate", 0.0)
                logger.info(
                    "Epoch %04d eval | success_rate=%.3f | mean_score=%.3f",
                    epoch + 1, success_rate, eval_metrics.get("mean_score", 0.0),
                )
                metric_log.log(epoch=epoch, step=global_step, **eval_metrics)

                if success_rate > best_metrics["best_success_rate"]:
                    best_metrics["best_success_rate"] = success_rate
                    best_path = ckpt_dir / "best.pt"
                    save_checkpoint(
                        path                    = best_path,
                        epoch                   = epoch,
                        model                   = model,
                        ema                     = ema,
                        optimizer               = optimizer,
                        lr_scheduler            = lr_sched,
                        obs_normalizer_state    = obs_normalizer.state_dict() if obs_normalizer else {},
                        action_normalizer_state = action_normalizer.state_dict(),
                        metrics                 = best_metrics,
                        cfg                     = cfg,
                    )
                    logger.info(
                        "New best model! success_rate=%.3f → %s",
                        success_rate, best_path,
                    )
            except ImportError:
                logger.warning("evaluate.py not available; skipping eval.")
            except Exception as exc:
                logger.warning("Evaluation failed: %s", exc)

    metric_log.close()
    logger.info(
        "Training complete. Best success rate: %.3f",
        best_metrics["best_success_rate"],
    )


# ==============================================================================
# CLI entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Diffusion Policy on PushT")

    # Config overrides (subset of TrainConfig fields)
    p.add_argument("--method",          type=str,   default=None,
                   help="'ddpm' or 'flow_matching'")
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--num_epochs",      type=int,   default=None)
    p.add_argument("--learning_rate",   type=float, default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--dataset_path",    type=str,   default=None)
    p.add_argument("--checkpoint_dir",  type=str,   default=None)
    p.add_argument("--log_dir",         type=str,   default=None)
    p.add_argument("--device",          type=str,   default=None,
                   help="'cpu', 'cuda', 'mps', or None to auto-detect")
    p.add_argument("--obs_type",        type=str,   default=None,
                   help="'state' or 'image' — which observation modality to use")
    p.add_argument("--resume",          type=str,   default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--save_interval",   type=int,   default=None,
                   help="Save a checkpoint every N epochs (default: 50)")

    return p.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply non-None CLI arguments as overrides to the config."""
    if args.method         is not None: cfg.method             = args.method
    if args.batch_size     is not None: cfg.batch_size          = args.batch_size
    if args.num_epochs     is not None: cfg.num_epochs          = args.num_epochs
    if args.learning_rate  is not None: cfg.learning_rate       = args.learning_rate
    if args.seed           is not None: cfg.seed                = args.seed
    if args.dataset_path   is not None: cfg.data.dataset_path   = args.dataset_path
    if args.checkpoint_dir is not None: cfg.checkpoint_dir      = args.checkpoint_dir
    if args.log_dir        is not None: cfg.log_dir             = args.log_dir
    if args.device         is not None: cfg.device              = args.device
    if args.obs_type       is not None: cfg.data.obs_type       = args.obs_type
    if args.save_interval  is not None: cfg.save_interval       = args.save_interval
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg  = apply_overrides(TrainConfig(), args)
    train(cfg, resume_path=args.resume)
