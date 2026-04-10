#!/usr/bin/env bash
# hpc/setup_env.sh — Create isolated 'diffpol' conda environment on Explorer
# ===========================================================================
#
# Run ONCE before submitting any training jobs.
# This MUST be a SLURM job (not a login-node command) because conda create
# requires ~4GB RAM that the login node will kill.
#
# Submit:
#   sbatch hpc/setup_env.sh
#
# When done (check: squeue -u gupta.yashv), proceed to submit training jobs.

#SBATCH --job-name=setup_diffpol_env
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/setup_%j.out
#SBATCH --error=logs/slurm/setup_%j.err

set -euo pipefail

echo "============================================"
echo " Setting up diffpol conda environment"
echo " Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "============================================"

module load anaconda3/2024.06
module load cuda/12.1.1

# Remove any broken previous attempt
conda env remove -n diffpol -y 2>/dev/null || true

# Create fresh isolated environment — Python 3.10 matches our local dev
echo "[1/4] Creating conda env diffpol (Python 3.10) …"
conda create -n diffpol python=3.10 -y

PYBIN=/home/gupta.yashv/.conda/envs/diffpol/bin
echo "Python: $($PYBIN/python --version)"

# Install PyTorch with CUDA 12.1 (matches cuda/12.1.1 module)
echo "[2/4] Installing PyTorch 2.3.0 + CUDA 12.1 …"
$PYBIN/pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

echo "  PyTorch: $($PYBIN/python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $($PYBIN/python -c 'import torch; print(torch.cuda.is_available())')"

# Install all project dependencies from requirements.txt
echo "[3/4] Installing project dependencies …"
$PYBIN/pip install \
    numpy>=1.24 \
    scipy>=1.10 \
    'zarr>=2.14,<4' \
    h5py>=3.8 \
    'gymnasium>=0.28' \
    'pygame>=2.3' \
    'pymunk>=6.4,<7' \
    'shapely>=2.0' \
    matplotlib>=3.7 \
    tqdm>=4.65 \
    imageio>=2.28 \
    imageio-ffmpeg>=0.4 \
    'einops>=0.6' \
    'diffusers>=0.25' \
    pyyaml>=6.0 \
    pytest>=7.0 \
    gym_pusht \
    --quiet

echo "[4/4] Verifying all imports …"
$PYBIN/python - <<'PYEOF'
import torch, numpy, scipy, zarr, gymnasium, pygame, pymunk, shapely
import matplotlib, tqdm, imageio, einops, diffusers, yaml
import gym_pusht
print("  All imports OK")
print(f"  torch:      {torch.__version__}")
print(f"  numpy:      {numpy.__version__}")
print(f"  zarr:       {zarr.__version__}")
print(f"  gymnasium:  {gymnasium.__version__}")
print(f"  gym_pusht:  {gym_pusht.__version__ if hasattr(gym_pusht, '__version__') else 'ok'}")
print(f"  CUDA:       {torch.cuda.is_available()}")
print(f"  GPU count:  {torch.cuda.device_count()}")
PYEOF

echo ""
echo "============================================"
echo " Environment setup COMPLETE"
echo " Activate with: source activate diffpol"
echo " Submit jobs now: sbatch hpc/train_bc.sh"
echo "============================================"
