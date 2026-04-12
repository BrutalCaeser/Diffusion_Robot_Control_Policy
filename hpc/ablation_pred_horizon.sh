#!/usr/bin/env bash
# hpc/ablation_pred_horizon.sh — Sweep pred_horizon in [8, 16, 32]
# =================================================================
# Trains DDPM three times (100 epochs each) with different pred_horizon
# values and evaluates each with DDIM (10 steps, 30 episodes).
# All other hyperparameters are identical so pred_horizon is the only variable.
#
# action_horizon is always pred_horizon/2 — execute half, then re-plan.
#
# Results land in:
#   checkpoints/ablation_ph{8,16,32}/best.pt
#   logs/ablation_ph{8,16,32}/eval.txt
#
# Submit:  sbatch hpc/ablation_pred_horizon.sh
# Monitor: squeue -u $USER

#SBATCH --job-name=abl_pred_horizon
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sakhamuru.v@northeastern.edu
#SBATCH --output=/scratch/%u/diffusion_policy/logs/slurm/abl_pred_horizon_%j.out
#SBATCH --error=/scratch/%u/diffusion_policy/logs/slurm/abl_pred_horizon_%j.err

set -euo pipefail

module load cuda/12.1.1

PROJECT=/scratch/$USER/diffusion_policy
source $PROJECT/venv/bin/activate   # use project venv instead of conda
DATASET=$PROJECT/data/pusht_cchi_v7_replay.zarr

cd $PROJECT
mkdir -p logs/slurm

echo "Job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Sweeping pred_horizon: 8 16 32"

for PH in 8 16 32; do
    AH=$((PH / 2))   # action_horizon = half of pred_horizon
    echo ""
    echo "[$(date)] Training  pred_horizon=$PH  action_horizon=$AH"

    python train.py \
        --method         ddpm \
        --pred_horizon   $PH \
        --action_horizon $AH \
        --num_epochs     100 \
        --batch_size     256 \
        --dataset_path   "$DATASET" \
        --checkpoint_dir checkpoints/ablation_ph${PH} \
        --log_dir        logs/ablation_ph${PH} \
        --device         cuda

    echo "[$(date)] Evaluating pred_horizon=$PH"
    mkdir -p logs/ablation_ph${PH}
    python evaluate.py \
        --checkpoint   checkpoints/ablation_ph${PH}/best.pt \
        --sampler      ddim \
        --num_episodes 30 \
        --device       cuda \
        2>&1 | tee logs/ablation_ph${PH}/eval.txt
done

echo ""
echo "Done. Results in $PROJECT/logs/ablation_ph{8,16,32}/eval.txt"
