# Diffusion Policy for Simulated Robot Control

**ML 6140 — Machine Learning | Northeastern University | April 2026**

Team: Yashvardhan Gupta · Vineeth Sakhamuru · Sai Krishna Reddy Maligireddy

---

## What this is

A from-scratch implementation of **Diffusion Policy** (Chi et al., RSS 2023) applied to the PushT task — a 2D robot manipulation benchmark where an agent must push a T-shaped block to a target pose.

We implement three action generation methods:
- **DDPM** — 100-step stochastic denoising (baseline)
- **DDIM** — 10-step deterministic sampling (10× faster, equal accuracy)
- **Flow Matching** — straight-line ODE interpolation (alternative formulation)
- **BC baseline** — 2-layer MLP regression (proves why diffusion is needed)

All models are trained on 300 epochs (BC: 200 epochs) on NVIDIA A100 GPUs via the Northeastern Explorer HPC cluster.

---

## Results

| Method | Success Rate | Inference Time | Steps |
|--------|-------------|----------------|-------|
| DDIM (ours) | **96%** | 17 ms | 10 |
| Flow Matching (ours) | **96%** | 59 ms | 10 |
| DDPM (ours) | **90%** | 161 ms | 100 |
| BC baseline (ours) | ~40–55% | <1 ms | — |
| DDIM (Chi et al., 2023) | ~90% | — | 10 |

The BC baseline's low success rate is the point: averaging over multi-modal expert actions produces stuck behavior. Diffusion commits to one mode at a time.

---

## Repository Structure

```
.
├── config.py                  # All hyperparameters in one place
├── train.py                   # DDPM / Flow Matching training
├── train_bc.py                # BC baseline training
├── evaluate.py                # Receding-horizon evaluation (DDPM/DDIM/FM)
├── evaluate_bc.py             # BC baseline evaluation
├── ablation.py                # DDIM steps sweep [1,5,10,20,50,100]
├── visualize.py               # Plots: loss curves, GIFs, multimodal motivation
├── run_ablation.sh            # Local 6-phase orchestration script
├── requirements.txt
├── ARCHITECTURE.md            # Deep technical reference
├── PROJECT_REPORT.md          # Full report with math, results, challenges
│
├── baselines/
│   └── bc_policy.py           # 2-hidden-layer MLP (135K params)
│
├── diffusion_policy/
│   ├── model/
│   │   ├── unet1d.py          # ConditionalUnet1D — 68.95M params, FiLM conditioning
│   │   ├── ddpm.py            # Cosine noise schedule, forward/reverse process
│   │   ├── ddim.py            # Deterministic skip-step sampler
│   │   ├── ema.py             # Exponential Moving Average (decay=0.995)
│   │   ├── flow_matching.py   # Straight-line interpolation, Euler ODE
│   │   └── vision_encoder.py  # ResNet-18 for image observations (unused in state training)
│   ├── data/
│   │   ├── dataset.py         # Zarr loading + sliding-window sampling
│   │   ├── normalizer.py      # MinMaxNormalizer: fit/normalize/unnormalize
│   │   └── image_dataset.py   # Image dataset (visuomotor extension)
│   └── env/
│       └── pusht_env.py       # Gymnasium PushT wrapper
│
├── hpc/                       # Northeastern Explorer SLURM scripts
│   ├── setup_env.sh           # Create isolated `diffpol` conda env via SLURM
│   ├── train_ddpm.sh          # 300-epoch DDPM job (A100, 8h)
│   ├── train_fm.sh            # 300-epoch FM job (A100, 8h)
│   ├── train_bc.sh            # 200-epoch BC job (A100, 1h)
│   ├── ablation_steps.sh      # DDIM steps ablation job
│   ├── watch_and_submit_ablation.sh  # Auto-submits ablation after DDPM finishes
│   └── README.md              # HPC setup, submission order, monitoring
│
└── tests/                     # 120 unit tests — all passing on Explorer
    ├── test_bc_policy.py      # 19 tests
    ├── test_ddpm.py           # 11 tests
    ├── test_ddim.py           # 8 tests
    ├── test_flow_matching.py  # 7 tests
    ├── test_unet1d.py         # 20 tests
    ├── test_ema.py            # 10 tests
    ├── test_normalizer.py     # 14 tests
    ├── test_integration.py    # 8 tests
    └── test_vision_encoder.py # 23 tests
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `zarr>=2.14,<4`, `gym_pusht`, `pymunk>=6.4,<7`, `imageio`, `matplotlib`

### Download Dataset

```bash
mkdir -p data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip -d data/
# → data/pusht_cchi_v7_replay.zarr  (206 episodes, 25,650 steps)
```

### Train

```bash
# DDPM (300 epochs):
python train.py --method ddpm --num_epochs 300 --batch_size 256 \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --checkpoint_dir checkpoints/ddpm_300ep --log_dir logs/ddpm_300ep

# Flow Matching (300 epochs):
python train.py --method flow_matching --num_epochs 300 --batch_size 256 \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --checkpoint_dir checkpoints/fm_300ep --log_dir logs/fm_300ep

# BC baseline (200 epochs):
python train_bc.py --num_epochs 200 --batch_size 256 \
    --dataset_path data/pusht_cchi_v7_replay.zarr
```

### Evaluate

```bash
# DDIM (recommended — fast and accurate):
python evaluate.py --checkpoint checkpoints/ddpm_300ep/best.pt --sampler ddim --num_episodes 50

# Flow Matching:
python evaluate.py --checkpoint checkpoints/fm_300ep/best.pt --sampler flow --num_episodes 50

# BC baseline:
python evaluate_bc.py --checkpoint checkpoints/bc/best.pt --num_episodes 50
```

### DDIM Steps Ablation

```bash
python ablation.py --checkpoint checkpoints/ddpm_300ep/best.pt \
    --steps 1 5 10 20 50 100 --num_episodes 30
# → logs/ablation/steps_ablation.json, plots/steps_ablation.png
```

### Run All Tests

```bash
pytest tests/ -v
# Expected: 120 passed
```

---

## HPC Training (Northeastern Explorer)

```bash
# 1. Set up environment (run once):
sbatch hpc/setup_env.sh

# 2. Submit training jobs in parallel:
sbatch hpc/train_bc.sh
DDPM_JOB=$(sbatch --parsable hpc/train_ddpm.sh)
sbatch hpc/train_fm.sh

# 3. Auto-submit ablation when DDPM finishes:
nohup bash hpc/watch_and_submit_ablation.sh $DDPM_JOB > logs/slurm/watcher.log 2>&1 &

# Monitor:
squeue -u gupta.yashv
```

---

## Key Design Decisions

**Why receding-horizon control?** The model generates 16 future actions but only executes the first 8. This keeps the robot responsive to unexpected events while maintaining short-term temporal coherence.

**Why DDIM over DDPM at test time?** DDIM skips most denoising steps algebraically — same trained model, 10× fewer network evaluations. It achieves equal or better success rate because deterministic sampling reduces variance across runs.

**Why the cosine noise schedule needs clamping?** At K=100 steps, `ᾱ_99 ≈ 2×10⁻⁸`. DDIM's clean-action prediction divides by `√ᾱ_t`, which explodes at the final step. Clamping to `min=1e-3` and clipping `â₀ ∈ [-1,1]` brings success rate from 0% to 96%.

**Why BC fails on PushT?** Experts sometimes approach the T-block from the left, sometimes from the right. Both are valid. An MLP averages these into a "middle" action that commits to neither, gets stuck, and fails. Diffusion samples *one* mode at a time and commits.

---

## References

1. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (RSS 2023)
2. Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)
3. Song et al., *Denoising Diffusion Implicit Models* (ICLR 2021)
4. Lipman et al., *Flow Matching for Generative Modeling* (ICLR 2023)
5. Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (AAAI 2018)
