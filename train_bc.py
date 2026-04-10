"""
train_bc.py — Train the Behavioral Cloning Baseline
=====================================================

This is intentionally the simplest possible training script.
Compare it against train.py to see what Diffusion Policy adds:

    train.py  (Diffusion):  ~560 lines — noise scheduler, EMA, K=100 denoising steps
    train_bc.py (BC):       ~200 lines — just MSE regression, nothing else

The simplicity is the point: BC is the natural baseline before introducing
diffusion. After training, run evaluate_bc.py to get success rates, then
compare against evaluate.py results for diffusion.

Usage:
    # Quick local test (CPU/MPS):
    python train_bc.py --num_epochs 5 --batch_size 16

    # Full training (GPU):
    python train_bc.py

    # Override dataset path (HPC):
    python train_bc.py --dataset_path /path/to/pusht.zarr --num_epochs 200
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from baselines.bc_policy import BCPolicy
from diffusion_policy.data.dataset import PushTStateDataset


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    return logging.getLogger("train_bc")


# ==============================================================================
# LR schedule (same as diffusion for a fair comparison)
# ==============================================================================

def cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    """Returns a LambdaLR lambda: linear warmup then cosine decay."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return fn


# ==============================================================================
# Main training function
# ==============================================================================

def train_bc(cfg: TrainConfig, num_epochs: int = 200) -> None:
    """
    Train the BC baseline with supervised MSE regression.

    No noise schedulers, no EMA, no denoising loops — just:
        loss = MSE( BCPolicy(obs), action_demo )

    The model learns to map observation history directly to actions.
    It will achieve lower success rates than Diffusion Policy because
    it cannot represent multimodal distributions (see bc_policy.py).
    """
    run_name = f"bc_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir  = "logs/bc"
    ckpt_dir = Path("checkpoints/bc")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, run_name)
    logger.info("Training BC baseline | device=%s | epochs=%d", cfg.device, num_epochs)

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Dataset (identical to diffusion training) ─────────────────────────────
    dataset = PushTStateDataset(
        dataset_path=cfg.data.dataset_path,
        obs_horizon=cfg.data.obs_horizon,
        pred_horizon=cfg.data.pred_horizon,
    )
    obs_normalizer, action_normalizer = dataset.get_normalizers()

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
        drop_last=True,
    )
    logger.info("Dataset: %d samples | %d batches/epoch", len(dataset), len(dataloader))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BCPolicy(
        obs_horizon  = cfg.data.obs_horizon,
        obs_dim      = cfg.env.obs_dim,
        pred_horizon = cfg.data.pred_horizon,
        action_dim   = cfg.env.action_dim,
        hidden_dim   = cfg.model.cond_dim,   # 256 — same as diffusion cond_dim
    ).to(cfg.device)
    logger.info("BC model: %s", model)

    # ── Optimizer (same hyperparameters as diffusion for fair comparison) ─────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        betas        = cfg.betas,
    )
    total_steps = num_epochs * len(dataloader)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        cosine_warmup_lambda(cfg.lr_warmup_steps, total_steps),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    # BC training is fast: one forward pass + MSE loss per batch.
    # No diffusion timestep sampling, no noise addition, no EMA.
    # Expect convergence in 50-100 epochs.
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{num_epochs}", leave=False)
        for batch in pbar:
            obs    = batch["obs"].to(cfg.device, non_blocking=True)    # (B, T_obs, obs_dim)
            action = batch["action"].to(cfg.device, non_blocking=True) # (B, T_pred, action_dim)

            pred   = model(obs)          # (B, T_pred, action_dim) — single forward pass
            loss   = F.mse_loss(pred, action)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            lr_sched.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        elapsed  = time.time() - t0
        logger.info(
            "Epoch %03d/%d | loss=%.4f | time=%.1fs",
            epoch + 1, num_epochs, avg_loss, elapsed,
        )

        # Save checkpoint every 50 epochs and at end
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save(
                {
                    "epoch":                    epoch,
                    "model_state_dict":         model.state_dict(),
                    "optimizer_state_dict":     optimizer.state_dict(),
                    "obs_normalizer":           obs_normalizer.state_dict(),
                    "action_normalizer":        action_normalizer.state_dict(),
                    "loss":                     avg_loss,
                    "config":                   cfg,
                    "bc_hparams": {
                        "obs_horizon":  cfg.data.obs_horizon,
                        "obs_dim":      cfg.env.obs_dim,
                        "pred_horizon": cfg.data.pred_horizon,
                        "action_dim":   cfg.env.action_dim,
                        "hidden_dim":   cfg.model.cond_dim,
                    },
                },
                ckpt_path,
            )
            logger.info("Saved: %s", ckpt_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(torch.load(ckpt_path, weights_only=False),
                           ckpt_dir / "best.pt")
                logger.info("  ↳ New best checkpoint (loss=%.4f)", best_loss)

    logger.info("BC training complete. Best loss: %.4f", best_loss)
    logger.info("Evaluate with: python evaluate_bc.py --checkpoint checkpoints/bc/best.pt")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Behavioral Cloning baseline on PushT")
    p.add_argument("--num_epochs",   type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--dataset_path", type=str,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = TrainConfig()

    if args.batch_size   is not None: cfg.batch_size         = args.batch_size
    if args.dataset_path is not None: cfg.data.dataset_path  = args.dataset_path
    if args.device       is not None: cfg.device             = args.device

    train_bc(cfg, num_epochs=args.num_epochs)
