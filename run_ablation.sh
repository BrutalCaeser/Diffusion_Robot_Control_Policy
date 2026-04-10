#!/usr/bin/env bash
# run_ablation.sh — Full Ablation Suite (local, sequential)
# ==========================================================
#
# Runs the complete set of experiments for the ML 6140 project:
#
#   Phase 1 — Train BC baseline              (~20 min on GPU)
#   Phase 2 — Train DDPM (100 epochs)        (~2 hrs on GPU, or use existing)
#   Phase 3 — Train Flow Matching (100 eps)  (~2 hrs on GPU, or use existing)
#   Phase 4 — DDIM inference steps ablation  (~15 min on GPU, no retraining)
#   Phase 5 — Evaluate all methods (50 eps each)
#   Phase 6 — Generate all comparison plots
#
# For GPU cluster training, use the SLURM scripts in hpc/ instead.
# This script is intended for quick local runs (reduce EPOCHS for speed).
#
# Usage:
#   ./run_ablation.sh [dataset_path] [epochs]
#   ./run_ablation.sh /path/to/pusht.zarr 100

set -euo pipefail

DATASET="${1:-/Volumes/Crucial_X9/Projects/ML_6140/pusht/pusht_cchi_v7_replay.zarr}"
EPOCHS="${2:-100}"
BATCH=256

echo "============================================================"
echo "  Diffusion Policy — Full Ablation Suite"
echo "  Dataset: $DATASET"
echo "  Epochs:  $EPOCHS  |  Batch: $BATCH"
echo "============================================================"

mkdir -p logs plots

# ── Phase 1: BC Baseline ──────────────────────────────────────────────────────
BC_CKPT="checkpoints/bc/best.pt"

if [ -f "$BC_CKPT" ]; then
    echo "[BC]   Checkpoint found at $BC_CKPT — skipping BC training."
else
    echo "[BC]   Training Behavioral Cloning baseline …"
    echo "       (This is the baseline that fails on multimodal data.)"
    python train_bc.py \
        --dataset_path "$DATASET" \
        --num_epochs   200 \
        --batch_size   "$BATCH"
    echo "[BC]   BC training complete."
fi

# ── Phase 2: DDPM ─────────────────────────────────────────────────────────────
DDPM_CKPT="checkpoints/run_ddpm/epoch_$(printf '%04d' $EPOCHS).pt"

if [ -f "$DDPM_CKPT" ]; then
    echo "[DDPM] Checkpoint found at $DDPM_CKPT — skipping DDPM training."
else
    echo "[DDPM] Training DDPM diffusion policy …"
    python train.py \
        --method         ddpm \
        --num_epochs     "$EPOCHS" \
        --batch_size     "$BATCH" \
        --dataset_path   "$DATASET" \
        --checkpoint_dir checkpoints/run_ddpm \
        --log_dir        logs/run_ddpm
    echo "[DDPM] Training complete."
fi
DDPM_BEST="checkpoints/run_ddpm/best.pt"
# Fall back to epoch checkpoint if no best.pt yet
[ -f "$DDPM_BEST" ] || DDPM_BEST="$DDPM_CKPT"

# ── Phase 3: Flow Matching ────────────────────────────────────────────────────
FM_CKPT="checkpoints/run_fm/epoch_$(printf '%04d' $EPOCHS).pt"

if [ -f "$FM_CKPT" ]; then
    echo "[FM]   Checkpoint found at $FM_CKPT — skipping FM training."
else
    echo "[FM]   Training Flow Matching policy …"
    python train.py \
        --method         flow_matching \
        --num_epochs     "$EPOCHS" \
        --batch_size     "$BATCH" \
        --dataset_path   "$DATASET" \
        --checkpoint_dir checkpoints/run_fm \
        --log_dir        logs/run_fm
    echo "[FM]   Training complete."
fi
FM_BEST="checkpoints/run_fm/best.pt"
[ -f "$FM_BEST" ] || FM_BEST="$FM_CKPT"

# ── Phase 4: DDIM Inference Steps Ablation ───────────────────────────────────
# This uses the trained DDPM checkpoint and evaluates at different step counts.
# No retraining needed — this is pure inference ablation.
echo ""
echo "[ABLATION] Running DDIM inference steps sweep …"
echo "           Step counts: 1 5 10 20 50 100 | 20 episodes each"
python ablation.py \
    --checkpoint   "$DDPM_BEST" \
    --steps        1 5 10 20 50 100 \
    --num_episodes 20 \
    --save_dir     logs/ablation \
    --plot_dir     plots \
    2>&1 | tee logs/ablation_steps.txt
echo "[ABLATION] Done. Plot: plots/steps_ablation.png"

# ── Phase 5: Evaluate all methods (50 episodes, reproducible seeds) ───────────
echo ""
echo "[EVAL] Evaluating BC baseline (50 episodes) …"
python evaluate_bc.py \
    --checkpoint   "$BC_CKPT" \
    --num_episodes 50 \
    2>&1 | tee logs/eval_bc.txt

echo ""
echo "[EVAL] Evaluating DDIM (10 steps, 50 episodes) …"
python evaluate.py \
    --checkpoint   "$DDPM_BEST" \
    --sampler      ddim \
    --num_episodes 50 \
    2>&1 | tee logs/eval_ddpm_ddim.txt

echo ""
echo "[EVAL] Evaluating DDPM (100 steps, 50 episodes) …"
python evaluate.py \
    --checkpoint   "$DDPM_BEST" \
    --sampler      ddpm \
    --num_episodes 50 \
    2>&1 | tee logs/eval_ddpm_ddpm.txt

echo ""
echo "[EVAL] Evaluating Flow Matching (50 episodes) …"
python evaluate.py \
    --checkpoint   "$FM_BEST" \
    --sampler      flow \
    --num_episodes 50 \
    2>&1 | tee logs/eval_fm.txt

# ── Phase 6: Generate all comparison plots ────────────────────────────────────
echo ""
echo "[PLOT] Generating comparison plots …"
python - <<'PYEOF'
import re, pathlib
from visualize import (
    plot_eval_comparison,
    plot_training_curves,
    plot_multimodal_trajectories,
)

# ── Parse success metrics from eval log files ──────────────────────────────
def parse_metric(log_path, key):
    try:
        txt = pathlib.Path(log_path).read_text()
        m   = re.search(rf'{key}\s*:\s*([0-9.]+)', txt)
        return float(m.group(1)) if m else 0.0
    except FileNotFoundError:
        return 0.0

logs = {
    "BC (baseline)":      "logs/eval_bc.txt",
    "DDPM (100 steps)":   "logs/eval_ddpm_ddpm.txt",
    "DDIM (10 steps)":    "logs/eval_ddpm_ddim.txt",
    "Flow Matching":      "logs/eval_fm.txt",
}

results = {
    name: {
        "success_rate": parse_metric(path, "success_rate"),
        "mean_score":   parse_metric(path, "mean_score"),
    }
    for name, path in logs.items()
}

print("\nMethod comparison:")
for name, m in results.items():
    print(f"  {name:<22} success={m['success_rate']:.3f}  mean_score={m['mean_score']:.3f}")

plot_eval_comparison(
    results,
    save_path="plots/final_comparison.png",
    title="BC Baseline vs Diffusion Policy — PushT (50 episodes)",
)
print("Saved: plots/final_comparison.png")

# ── Training curves ────────────────────────────────────────────────────────
for csv_path in pathlib.Path("logs").glob("**/*_metrics.csv"):
    try:
        plot_training_curves(str(csv_path), save_dir=f"plots/{csv_path.parent.name}")
    except Exception as e:
        print(f"  Could not plot {csv_path}: {e}")

# ── Multimodal trajectory plot (key motivation figure) ─────────────────────
import os
dataset_path = os.environ.get(
    "DATASET",
    "/Volumes/Crucial_X9/Projects/ML_6140/pusht/pusht_cchi_v7_replay.zarr",
)
try:
    plot_multimodal_trajectories(
        dataset_path,
        save_path="plots/multimodal_motivation.png",
        num_episodes=40,
    )
    print("Saved: plots/multimodal_motivation.png")
except Exception as e:
    print(f"  Could not plot multimodal trajectories: {e}")

print("\nAll plots generated in plots/")
PYEOF

echo ""
echo "============================================================"
echo "  Ablation study COMPLETE"
echo ""
echo "  Key outputs:"
echo "    plots/multimodal_motivation.png  — WHY diffusion is needed"
echo "    plots/final_comparison.png       — BC vs DDPM vs DDIM vs FM"
echo "    plots/steps_ablation.png         — DDIM step count ablation"
echo "    logs/ablation/steps_ablation.json — raw ablation numbers"
echo "============================================================"
