#!/usr/bin/env bash
# hpc/ablation_steps.sh — DDIM Inference Steps Ablation on Northeastern Explorer
# ================================================================================
# Cluster:  explorer.northeastern.edu  |  Login: gupta.yashv@explorer.northeastern.edu
# Requires: train_ddpm.sh to have completed (needs checkpoints/ddpm_300ep/best.pt).
#
# Submit AFTER DDPM training finishes:
#   sbatch hpc/ablation_steps.sh
#
# Override checkpoint:
#   CKPT=/scratch/gupta.yashv/diffusion_policy/checkpoints/ddpm_300ep/epoch_0150.pt \
#     sbatch hpc/ablation_steps.sh

#SBATCH --job-name=dp_ablation
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gupta.yashv/diffusion_policy/logs/slurm/ablation_%j.out
#SBATCH --error=/scratch/gupta.yashv/diffusion_policy/logs/slurm/ablation_%j.err

set -euo pipefail

module load anaconda3/2024.06
module load cuda/12.1.1
source activate diffpol

PROJECT=/scratch/gupta.yashv/diffusion_policy
CKPT="${CKPT:-$PROJECT/checkpoints/ddpm_300ep/best.pt}"

echo "============================================"
echo " Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo " Checkpoint: $CKPT"
echo "============================================"

cd $PROJECT
mkdir -p logs/slurm logs/ablation plots

echo "[$(date)] Running DDIM steps ablation [1, 5, 10, 20, 50, 100] …"
python ablation.py \
    --checkpoint   "$CKPT" \
    --steps        1 5 10 20 50 100 \
    --num_episodes 30 \
    --device       cuda \
    --save_dir     logs/ablation \
    --plot_dir     plots

echo "[$(date)] Ablation complete."
echo "  JSON:  $PROJECT/logs/ablation/steps_ablation.json"
echo "  Plot:  $PROJECT/plots/steps_ablation.png"
