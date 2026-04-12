# HPC Scripts — Northeastern Explorer Cluster

**Cluster:** `explorer.northeastern.edu`
**Your login:** `<your-username>@explorer.northeastern.edu`

---

## One-Time Setup

```bash
# 1. SSH into Explorer
ssh <your-username>@explorer.northeastern.edu

# 2. Go to your scratch space (fast I/O, for large files)
cd /scratch/$USER
mkdir -p diffusion_policy && cd diffusion_policy

# 3. Clone the repo
git clone <your-repo-url> .

# 4. Copy the dataset into the project
mkdir -p data
# Option A — scp from your Mac:
#   scp -r /Volumes/Crucial_X9/Projects/ML_6140/pusht/pusht_cchi_v7_replay.zarr \
#           <your-username>@explorer.northeastern.edu:/scratch/$USER/diffusion_policy/data/
# Option B — if already on the cluster:
#   cp /path/on/cluster/pusht_cchi_v7_replay.zarr data/

# 5. Check available modules (Explorer-specific)
module avail 2>&1 | grep -i cuda
module avail 2>&1 | grep -i anaconda

# 6. Create conda environment
module load anaconda3/2022.05   # adjust version if needed — check step 5
conda create -n diffusion python=3.10 -y
source activate diffusion
pip install -r requirements.txt
```

---

## Recommended Job Submission Order

Submit all at once — they queue and run in parallel:

```bash
# From project root: /scratch/$USER/diffusion_policy

# 1. BC baseline (fastest: ~20 min) — submit first, get results soonest
sbatch hpc/train_bc.sh

# 2. DDPM (300 epochs, ~3–4 hrs on a GPU)
sbatch hpc/train_ddpm.sh

# 3. Flow Matching (same wall-clock as DDPM, runs in parallel)
sbatch hpc/train_fm.sh

# 4. Ablation — run AFTER train_ddpm.sh finishes (needs its checkpoint)
#    Check if DDPM is done: squeue -u $USER
sbatch hpc/ablation_steps.sh
```

---

## Monitor Your Jobs

```bash
squeue -u $USER                        # see all your jobs + status
tail -f logs/slurm/ddpm_<jobid>.out          # live stdout from a job
scancel <jobid>                              # cancel a job
sacct -u $USER --format=JobID,State,Elapsed,MaxRSS   # job history
```

---

## If a Job Fails

```bash
# Check the error log
cat logs/slurm/<jobname>_<jobid>.err

# Common fixes:
# 1. Wrong module version   → run: module avail anaconda3
# 2. conda env not found    → re-run the setup steps above
# 3. Out of memory          → increase --mem in the .sh file
# 4. Dataset not found      → check DATASET_PATH matches where you put the zarr
```

---

## Environment Variable Overrides

All scripts respect these, so you can adjust paths without editing the files:

| Variable | Default | Description |
|---|---|---|
| `PROJECT_DIR` | `/scratch/$USER/diffusion_policy` | Project root on cluster |
| `DATASET_PATH` | `$PROJECT_DIR/data/pusht_cchi_v7_replay.zarr` | Dataset location |
| `CKPT` | `checkpoints/ddpm_300ep/best.pt` | Checkpoint for ablation job |

Example override:
```bash
DATASET_PATH=/shared/datasets/pusht.zarr sbatch hpc/train_ddpm.sh
```

---

## Transfer Results Back to Mac

After jobs finish, copy results back locally:

```bash
# From your Mac:
scp -r <your-username>@explorer.northeastern.edu:/scratch/$USER/diffusion_policy/checkpoints ./
scp -r <your-username>@explorer.northeastern.edu:/scratch/$USER/diffusion_policy/logs ./
scp -r <your-username>@explorer.northeastern.edu:/scratch/$USER/diffusion_policy/plots ./
```

---

## Expected Outputs After All Jobs Complete

```
checkpoints/
  bc/best.pt              ← BC baseline  (~135K params, fast to train)
  ddpm_300ep/best.pt      ← DDPM         (~68M params, 300 epochs)
  fm_300ep/best.pt        ← Flow Matching (~68M params, 300 epochs)

logs/
  eval_bc.txt             ← BC results       (~40–55% success — intentionally low)
  eval_ddpm_ddim.txt      ← DDIM results     (~90–96% success)
  eval_fm.txt             ← FM results       (~90–96% success)
  ablation/steps_ablation.json  ← raw ablation numbers

plots/
  multimodal_motivation.png    ← WHY diffusion is needed (show this slide 1)
  final_comparison.png         ← BC vs DDPM vs DDIM vs FM (the money plot)
  steps_ablation.png           ← DDIM step count trade-off
```
