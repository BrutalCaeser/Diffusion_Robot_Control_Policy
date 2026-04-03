#!/usr/bin/env bash
# run_ablation.sh — Sequential ablation: DDPM (100ep) then Flow Matching (100ep)
#
# Usage:
#   ./run_ablation.sh [dataset_path]
#
# If the DDPM run is already done (checkpoints/run_100ep/epoch_0100.pt exists),
# only the FM run will execute.
#
# Logs and checkpoints are written to:
#   checkpoints/run_100ep/    (DDPM)
#   checkpoints/run_fm_100ep/ (Flow Matching)
#   logs/run_100ep/
#   logs/run_fm_100ep/

set -euo pipefail

DATASET="${1:-/Volumes/Crucial_X9/Projects/ML_6140/pusht/pusht_cchi_v7_replay.zarr}"
EPOCHS=100
BATCH=256

echo "============================================================"
echo "  Diffusion Policy Ablation Study"
echo "  Dataset: $DATASET"
echo "  Epochs:  $EPOCHS  |  Batch: $BATCH"
echo "============================================================"

# ── Phase 1: DDPM ────────────────────────────────────────────────────────────
DDPM_CKPT="checkpoints/run_100ep/epoch_$(printf '%04d' $EPOCHS).pt"

if [ -f "$DDPM_CKPT" ]; then
    echo "[DDPM] Checkpoint found at $DDPM_CKPT — skipping DDPM training."
else
    echo "[DDPM] Starting training …"
    python train.py \
        --dataset_path "$DATASET" \
        --method ddpm \
        --num_epochs $EPOCHS \
        --batch_size $BATCH \
        --checkpoint_dir checkpoints/run_100ep \
        --log_dir logs/run_100ep
    echo "[DDPM] Training complete."
fi

# ── Phase 2: Flow Matching ────────────────────────────────────────────────────
FM_CKPT="checkpoints/run_fm_100ep/epoch_$(printf '%04d' $EPOCHS).pt"

if [ -f "$FM_CKPT" ]; then
    echo "[FM]   Checkpoint found at $FM_CKPT — skipping FM training."
else
    echo "[FM]   Starting Flow Matching training …"
    python train.py \
        --dataset_path "$DATASET" \
        --method flow_matching \
        --num_epochs $EPOCHS \
        --batch_size $BATCH \
        --checkpoint_dir checkpoints/run_fm_100ep \
        --log_dir logs/run_fm_100ep
    echo "[FM]   Training complete."
fi

# ── Phase 3: Evaluate both checkpoints ───────────────────────────────────────
echo ""
echo "[EVAL] Evaluating DDPM checkpoint (DDIM sampler, 50 episodes) …"
python evaluate.py \
    --checkpoint "$DDPM_CKPT" \
    --sampler ddim \
    --num_episodes 50 \
    2>&1 | tee logs/eval_ddpm_ddim.txt

echo ""
echo "[EVAL] Evaluating DDPM checkpoint (DDPM sampler, 50 episodes) …"
python evaluate.py \
    --checkpoint "$DDPM_CKPT" \
    --sampler ddpm \
    --num_episodes 50 \
    2>&1 | tee logs/eval_ddpm_ddpm.txt

echo ""
echo "[EVAL] Evaluating Flow Matching checkpoint (50 episodes) …"
python evaluate.py \
    --checkpoint "$FM_CKPT" \
    --sampler flow \
    --num_episodes 50 \
    2>&1 | tee logs/eval_fm.txt

# ── Phase 4: Comparison plot ─────────────────────────────────────────────────
echo ""
echo "[PLOT] Generating ablation comparison plots …"
python -c "
from visualize import plot_eval_comparison, plot_training_curves
import re, pathlib

def parse_success(log_path):
    txt = pathlib.Path(log_path).read_text()
    m = re.search(r'success_rate\s*:\s*([0-9.]+)', txt)
    return float(m.group(1)) if m else 0.0

def parse_score(log_path):
    txt = pathlib.Path(log_path).read_text()
    m = re.search(r'mean_score\s*:\s*([0-9.]+)', txt)
    return float(m.group(1)) if m else 0.0

results = {
    'DDPM (100 steps)': {
        'success_rate': parse_success('logs/eval_ddpm_ddpm.txt'),
        'mean_score':   parse_score('logs/eval_ddpm_ddpm.txt'),
    },
    'DDIM (10 steps)': {
        'success_rate': parse_success('logs/eval_ddpm_ddim.txt'),
        'mean_score':   parse_score('logs/eval_ddpm_ddim.txt'),
    },
    'Flow Matching (10 steps)': {
        'success_rate': parse_success('logs/eval_fm.txt'),
        'mean_score':   parse_score('logs/eval_fm.txt'),
    },
}

plot_eval_comparison(results, save_path='plots/ablation_comparison.png')
print('Ablation comparison saved to plots/ablation_comparison.png')

# Also regenerate training curves for both runs
for csv in pathlib.Path('logs').glob('**/*_metrics.csv'):
    plot_training_curves(str(csv), save_dir=f'plots/{csv.parent.name}')
print('Training curves regenerated.')
"

echo ""
echo "============================================================"
echo "  Ablation study complete!"
echo "  Results in: logs/eval_*.txt"
echo "  Plots in:   plots/"
echo "============================================================"
