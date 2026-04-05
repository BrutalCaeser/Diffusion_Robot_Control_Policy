# Diffusion Policy for Simulated Robot Control

**A Machine Learning Implementation of Action Diffusion for Robot Control in the PushT Environment**

## Project Overview

This project implements **Diffusion Policy** — a generative-model-based approach to robot control — as proposed by Chi et al. (RSS 2023). Rather than training a standard behavioral cloning policy that maps observations to a single deterministic action, we train a **Denoising Diffusion Probabilistic Model (DDPM)** that iteratively denoises action sequences conditioned on observations.

### Key Features

- **Core Architecture**: Conditional 1D Temporal U-Net with FiLM conditioning
- **Training Method**: DDPM with EMA weight averaging
- **Inference Methods**: 
  - DDPM sampling (100 steps, baseline)
  - DDIM sampling (10 steps, 10x faster)
  - Flow Matching (stretch goal, continuous-time ODE approach)
- **Environment**: PushT — a 2D pushing task where an agent pushes a T-block to match a target pose
- **Achieved Performance**: 96% success rate (DDIM, 50 episodes)

### Why Diffusion Policy?

Traditional behavioral cloning suffers from **multimodal action distribution** collapse — when multiple valid actions exist for the same observation, regression-based policies average over modes rather than committing to one. Diffusion models naturally represent multimodal distributions, enabling the policy to capture the full diversity of expert behavior without mode collapse.

---

## Team

| Name | Email |
|------|-------|
| Yashvardhan Gupta | gupta.yashv@northeastern.edu |
| Vineeth Sakhamuru | sakhamuru.v@northeastern.edu |
| Sai Krishna Reddy Maligireddy | maligireddy.s@northeastern.edu |

**Course**: ML 6140 — Machine Learning  
**Deadline**: April 15, 2026

---

## Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- macOS (CPU/MPS) or Linux (CUDA GPU)

### Installation

```bash
# Clone the repository
git clone <repo_url>
cd Diffusion_Robot_Control_Policy

# Install dependencies
pip install -r requirements.txt

# Verify setup
python config.py
```

This will print configuration parameters and verify the device is detected correctly (cpu, mps, or cuda).

### Local Development (MacBook)

```bash
# Test data pipeline (5 min on CPU)
python -c "from diffusion_policy.data.dataset import PushTStateDataset; print('Dataset import OK')"

# Single-batch overfit sanity check (< 1 min on CPU)
python train.py --batch_size 1 --num_epochs 1 --device cpu
```

### Remote Training (Google Colab / University GPU)

```bash
# In Colab notebook:
!git clone <repo_url>
%cd Diffusion_Robot_Control_Policy
!pip install -r requirements.txt

# Full GPU training (2-4 hours)
!python train.py --batch_size 256 --num_epochs 300
```

### Evaluation

```bash
# Evaluate trained model with DDIM (10 steps)
python evaluate.py --checkpoint checkpoints/best.pt --sampler ddim --num_episodes 100

# Generate rollout GIFs
python evaluate.py --checkpoint checkpoints/best.pt --sampler ddim --save_gifs
```

---

## Project Structure

```
diffusion_policy/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── unet1d.py              # Conditional 1D Temporal U-Net
│   ├── ddpm.py                # DDPM noise schedule & reverse sampling
│   ├── ddim.py                # DDIM accelerated sampling
│   ├── flow_matching.py        # Flow Matching (Phase 5 extension)
│   └── ema.py                 # Exponential Moving Average
├── data/
│   ├── __init__.py
│   ├── dataset.py             # PushT dataset with sliding window
│   └── normalizer.py          # Min-max normalization [-1, 1]
├── env/
│   ├── __init__.py
│   └── pusht_env.py           # PushT environment wrapper
├── train.py                   # Main training script
├── evaluate.py                # Evaluation & rollout script
├── visualize.py               # Visualization utilities
├── config.py                  # Centralized hyperparameter config
├── requirements.txt           # Python dependencies
├── PROJECT_REPORT.md          # Full project report with objectives
└── README.md                  # This file
```

---

## Technical Specification

### Model Architecture

**Conditional 1D Temporal U-Net**

The core model processes action sequences (T_pred, action_dim) along the temporal dimension using 1D convolutions.

| Component | Details |
|-----------|---------|
| Input channels | 2 (x, y velocity for PushT) |
| Channel progression | [256, 512, 1024] (3 U-Net levels) |
| Timestep embedding | 256-dim sinusoidal + 2-layer MLP |
| Observation conditioning | FiLM (Feature-wise Linear Modulation) |
| Residual blocks | Conv1d → GroupNorm → FiLM → Mish → Conv1d |

### Data Pipeline

1. **Sliding window extraction**: For each timestep in each episode:
   - Observation: last T_obs=2 frames → shape (2, 5)
   - Action: next T_pred=16 frames → shape (16, 2)
   
2. **Normalization**: Min-max to [-1, 1] using dataset statistics
   
3. **Padding**: Repeat first/last observation/action at episode boundaries

Expected data shapes:
```
obs:    (B, T_obs, obs_dim)   = (B, 2, 5)
action: (B, T_pred, action_dim) = (B, 16, 2)
```

### DDPM Training

Standard diffusion objective with linear noise schedule:

```
For each batch:
  1. Sample timestep k ~ Uniform{0, ..., K-1}
  2. Sample noise ε ~ N(0, I)
  3. Compute noisy actions: a_k = √(ᾱ_k) · a_0 + √(1 - ᾱ_k) · ε
  4. Predict noise: ε_θ = model(a_k, k, obs)
  5. Loss = MSE(ε_θ, ε)
```

**Key hyperparameters:**
- Diffusion steps: K = 100
- Noise schedule: cosine (K=100)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-6)
- LR schedule: Cosine annealing + 500-step warmup
- Batch size: 256
- EMA decay: 0.995
- Training epochs: 300-500

### DDPM & DDIM Inference

**DDPM Sampling** (100 steps):
```
1. Initialize: a_K ~ N(0, I)
2. For k = K-1, ..., 0:
     Predict noise: ε_θ = model(a_k, k, obs)
     Denoise: a_{k-1} = (1/√α_k)(a_k - (β_k/√(1-ᾱ_k)) · ε_θ) + σ_k · z
3. Execute first T_action=8 steps
4. Re-plan with updated observations
```

**DDIM Sampling** (10 steps, 10x faster):
```
1. Select timestep subsequence: [99, 89, 79, ..., 9, 0]
2. For each pair of timesteps:
     Compute predicted clean action: â_0 = (a_k - √(1-ᾱ_k) · ε_θ) / √(ᾱ_k)
     Deterministic step: a_{k'} = √(ᾱ_{k'}) · â_0 + √(1-ᾱ_{k'}) · ε_θ
3. Same receding-horizon execution
```

---

## Implementation Timeline

### Phase 1: Data Pipeline (Week 1)
- ✓ Set up environment and dependencies
- ✓ Implement data loading and normalization
- ✓ Verify data shapes and distributions

### Phase 2: Model Architecture (Week 2)
- Implement timestep embedding, FiLM conditioning
- Implement 1D residual blocks and full U-Net
- Local CPU sanity checks (gradient flow, shape verification)

### Phase 3: Training & Inference (Weeks 3-4)
- Implement DDPM forward/reverse process
- Implement EMA weight averaging
- Full training loop on GPU with loss logging
- DDPM and DDIM sampling
- Receding-horizon evaluation

### Phase 4: Ablations & Report (Weeks 5-9)
- **Prediction horizon**: {4, 8, 16, 32}
- **DDIM steps**: {5, 10, 20, 50} + timing
- **Observation horizon**: {1, 2, 4}
- **EMA vs. no EMA**
- **Diffusion Policy vs. MLP baseline**
- Write full technical report with figures

### Phase 5: Flow Matching (Post-deadline, optional)
- Implement continuous-time ODE formulation
- Velocity field prediction instead of noise
- Euler/RK4 ODE integration for inference
- Compare with DDPM: convergence, inference speed, success rate
- Bridge toward Schrödinger Bridge research

---

## Development Workflow

### Local (MacBook) — Code Development

```bash
# Develop with CPU/MPS
python config.py  # verify device is detected

# Test individual components
python -m pytest tests/  # (if test suite exists)

# Single-batch overfit sanity check
python train.py --batch_size 1 --num_epochs 1 --device cpu

# Visualize data
python -c "from diffusion_policy.data.dataset import PushTStateDataset; ..."
```

### Remote (Google Colab/University GPU) — Full Training

```bash
# 1. Push to GitHub
git add .
git commit -m "Implement DDPM training"
git push origin main

# 2. In Colab, clone and run
!git clone https://github.com/<your-repo>
!pip install -r requirements.txt
!python train.py  # runs on GPU

# 3. Download checkpoints back to MacBook
from google.colab import files
files.download('checkpoints/best.pt')
```

### Device-Agnostic Code

All code uses this pattern to work on any device:

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
model = model.to(device)
```

---

## Evaluation Metrics

- **Success Rate**: % of episodes where T-block reaches target pose (within distance/angle threshold)
- **Average Reward**: Continuous overlap score from PushT environment
- **Inference Speed**: Wall-clock time per action generation (DDPM vs. DDIM)
- **Rollout Quality**: Visual inspection of GIFs (agent pushes coherently, reaches target)

**Achieved Performance**: 96% success rate on PushT with DDIM sampling (10 steps); 96% with Flow Matching; 90% with DDPM (100 steps). All results meet or exceed Chi et al. (RSS 2023) reported numbers.

---

## Key Papers & References

1. **Diffusion Policy** — Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
   - Primary reference implementation target

2. **DDPM** — Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
   - Foundational noise schedule and training objective

3. **DDIM** — Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
   - Accelerated deterministic sampling

4. **Flow Matching** — Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
   - Continuous-time alternative to DDPM (Phase 5)

5. **Rectified Flows** — Liu et al., "Flow Straight and Fast" (ICLR 2023)
   - Straight-line interpolation for faster ODE solving

### Code References

- **Official Diffusion Policy**: https://github.com/real-stanford/diffusion_policy
- **LeRobot (HuggingFace)**: https://github.com/huggingface/lerobot
- **HuggingFace Diffusers**: https://github.com/huggingface/diffusers

---

## Repository Setup & Git Workflow

### Initial Setup

```bash
# Initialize git (if not already done)
git init
git add -A
git commit -m "Initial commit: project structure, config, and documentation"

# Add remote (replace with your actual GitHub URL)
git remote add origin https://github.com/<username>/Diffusion_Robot_Control_Policy.git
git branch -M main
git push -u origin main
```

### Regular Commits

```bash
# After completing a phase
git add diffusion_policy/ config.py requirements.txt
git commit -m "Phase 2: Implement conditional 1D U-Net architecture"
git push origin main

# Before pushing to Colab for training
git add -A
git commit -m "Ready for GPU training"
git push origin main
```

### .gitignore

Create `.gitignore` to exclude large files:

```
# Data
data/
*.zarr/

# Checkpoints & logs
checkpoints/
logs/
runs/

# Python
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
.env
```

---

## Configuration

All hyperparameters are centralized in [config.py](config.py). Modify these to run ablations:

```python
from config import TrainConfig

cfg = TrainConfig()
print(cfg.device)                    # Auto-detected device
print(cfg.data.pred_horizon)         # 16 (actions to predict)
print(cfg.data.action_horizon)       # 8 (actions to execute)
print(cfg.diffusion.ddim_steps)      # 10 (DDIM inference steps)
print(cfg.batch_size)                # 256
print(cfg.learning_rate)             # 1e-4
```

For ablations, override at runtime:

```bash
# Prediction horizon ablation
python train.py --pred_horizon 8 --num_epochs 100
python train.py --pred_horizon 16 --num_epochs 100
python train.py --pred_horizon 32 --num_epochs 100
```

---

## Troubleshooting

### Data Issues
- **Shapes don't match**: Check `config.py` values for `obs_horizon`, `pred_horizon`, `action_horizon`
- **Values not in [-1, 1]**: Ensure normalizer is fitted on full dataset before training
- **Missing dataset**: Download from [diffusion_policy repo](https://github.com/real-stanford/diffusion_policy) or use `pusht` package

### Training Issues
- **Loss doesn't decrease**: Check learning rate, gradient clipping, noise schedule (ᾱ_k values)
- **Out of memory on GPU**: Reduce `batch_size` in config.py
- **Model overfits on 1 batch**: Verify forward/reverse process with sanity checks in `ddpm.py`

### Inference Issues
- **Actions out of reasonable range**: Check unnormalization in `evaluate.py`
- **Low success rate**: Verify EMA weights are being used; check observation history construction
- **DDIM faster but worse quality**: Try more DDIM steps (e.g., 20 instead of 10)

---

## Contributing & Citation

If using this implementation in research:

```bibtex
@article{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and ...},
  journal={Robotics: Science and Systems},
  year={2023}
}
```

---

## Future Work

### Schrödinger Bridge Research
- Add diffusion coefficient σ(t) to FM ODE
- Estimate SB solutions using forward-backward SDE formulation
- Connect to optimal transport theory
- Application to multi-modal trajectory generation

---

## Contact & Questions

For questions or issues, refer to:
- **Project Report**: [PROJECT_REPORT.md](PROJECT_REPORT.md)
- **Timeline**: [TIMELINE.md](TIMELINE.md)
- **Code**: Individual module docstrings in `diffusion_policy/`

---

**Last Updated**: April 2026
**Status**: Complete — 101 tests passing, 100-epoch training runs finished for both DDPM and Flow Matching, full evaluation done.
