#!/usr/bin/env bash
# hpc/train_ddpm.sh — Train DDPM Diffusion Policy on Northeastern Explorer
# =========================================================================
# Cluster:  explorer.northeastern.edu
# Requires: setup_env.sh to have run successfully first.
#
# Submit:   sbatch hpc/train_ddpm.sh
# Monitor:  squeue -u $USER
#           tail -f /scratch/$USER/diffusion_policy/logs/slurm/ddpm_<jobid>.out

#SBATCH --job-name=dp_ddpm
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/%u/diffusion_policy/logs/slurm/ddpm_%j.out
#SBATCH --error=/scratch/%u/diffusion_policy/logs/slurm/ddpm_%j.err

set -euo pipefail

module load anaconda3/2024.06
module load cuda/12.1.1
source activate diffpol

PROJECT=/scratch/$USER/diffusion_policy
DATASET=$PROJECT/data/pusht_cchi_v7_replay.zarr

echo "============================================"
echo " Job:     $SLURM_JOB_ID"
echo " Node:    $SLURMD_NODENAME"
echo " GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Python:  $(python --version)"
echo " Torch:   $(python -c 'import torch; print(torch.__version__, \"| CUDA:\", torch.cuda.is_available())')"
echo " Dataset: $DATASET"
echo "============================================"

cd $PROJECT
mkdir -p logs/slurm checkpoints/ddpm_300ep logs/ddpm_300ep

# ── Training (300 epochs, batch 256, A100 GPU) ─────────────────────────────
echo "[$(date)] Starting DDPM training …"
python train.py \
    --method         ddpm \
    --num_epochs     300 \
    --batch_size     256 \
    --dataset_path   "$DATASET" \
    --checkpoint_dir checkpoints/ddpm_300ep \
    --log_dir        logs/ddpm_300ep \
    --device         cuda
echo "[$(date)] Training complete."

# ── Evaluate with DDIM (fast, 10 steps) ───────────────────────────────────
echo "[$(date)] Evaluating DDIM (10 steps, 50 episodes) …"
python evaluate.py \
    --checkpoint   checkpoints/ddpm_300ep/best.pt \
    --sampler      ddim \
    --num_episodes 50 \
    --device       cuda \
    2>&1 | tee logs/eval_ddpm_ddim.txt

# ── Evaluate with full DDPM (100 steps) ───────────────────────────────────
echo "[$(date)] Evaluating DDPM (100 steps, 50 episodes) …"
python evaluate.py \
    --checkpoint   checkpoints/ddpm_300ep/best.pt \
    --sampler      ddpm \
    --num_episodes 50 \
    --device       cuda \
    2>&1 | tee logs/eval_ddpm_ddpm.txt

echo "[$(date)] DDPM job complete."
echo "  DDIM results: $PROJECT/logs/eval_ddpm_ddim.txt"
echo "  DDPM results: $PROJECT/logs/eval_ddpm_ddpm.txt"
