#!/usr/bin/env bash
# hpc/train_fm.sh — Train Flow Matching Policy on Northeastern Explorer
# ======================================================================
# Cluster:  explorer.northeastern.edu  |  Login: $USER@explorer.northeastern.edu
# Requires: setup_env.sh to have run successfully first.
# Note:     Runs in parallel with train_ddpm.sh — same wall-clock, same GPU config.
#
# Submit:   sbatch hpc/train_fm.sh

#SBATCH --job-name=dp_fm
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/$USER/diffusion_policy/logs/slurm/fm_%j.out
#SBATCH --error=/scratch/$USER/diffusion_policy/logs/slurm/fm_%j.err

set -euo pipefail

module load anaconda3/2024.06
module load cuda/12.1.1
source activate diffpol

PROJECT=/scratch/$USER/diffusion_policy
DATASET=$PROJECT/data/pusht_cchi_v7_replay.zarr

echo "============================================"
echo " Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

cd $PROJECT
mkdir -p logs/slurm checkpoints/fm_300ep logs/fm_300ep

echo "[$(date)] Training Flow Matching (300 epochs) …"
python train.py \
    --method         flow_matching \
    --num_epochs     300 \
    --batch_size     256 \
    --dataset_path   "$DATASET" \
    --checkpoint_dir checkpoints/fm_300ep \
    --log_dir        logs/fm_300ep \
    --device         cuda

echo "[$(date)] FM training done. Evaluating (50 episodes) …"
python evaluate.py \
    --checkpoint   checkpoints/fm_300ep/best.pt \
    --sampler      flow \
    --num_episodes 50 \
    --device       cuda \
    2>&1 | tee logs/eval_fm.txt

echo "[$(date)] FM job complete. Results: $PROJECT/logs/eval_fm.txt"
