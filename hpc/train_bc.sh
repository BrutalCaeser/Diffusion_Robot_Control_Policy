#!/usr/bin/env bash
# hpc/train_bc.sh — Train Behavioral Cloning Baseline on Northeastern Explorer
# =============================================================================
# Cluster:  explorer.northeastern.edu  |  Login: $USER@explorer.northeastern.edu
# Requires: setup_env.sh to have run successfully first.
# Note:     BC is fast (~20 min). Submit this FIRST to get baseline results early.
#
# Submit:   sbatch hpc/train_bc.sh

#SBATCH --job-name=dp_bc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/$USER/diffusion_policy/logs/slurm/bc_%j.out
#SBATCH --error=/scratch/$USER/diffusion_policy/logs/slurm/bc_%j.err

set -euo pipefail

module load anaconda3/2024.06
module load cuda/12.1.1
CONDA_ENV=${CONDA_ENV:-diffusion}
source activate $CONDA_ENV

PROJECT=/scratch/$USER/diffusion_policy
DATASET=$PROJECT/data/pusht_cchi_v7_replay.zarr

echo "============================================"
echo " Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

cd $PROJECT
mkdir -p logs/slurm checkpoints/bc

echo "[$(date)] Training BC baseline (200 epochs) …"
python train_bc.py \
    --num_epochs     200 \
    --batch_size     256 \
    --dataset_path   "$DATASET" \
    --device         cuda

echo "[$(date)] BC training done. Evaluating (50 episodes) …"
python evaluate_bc.py \
    --checkpoint   checkpoints/bc/best.pt \
    --num_episodes 50 \
    --device       cuda \
    2>&1 | tee logs/eval_bc.txt

echo "[$(date)] BC job complete. Results: $PROJECT/logs/eval_bc.txt"
