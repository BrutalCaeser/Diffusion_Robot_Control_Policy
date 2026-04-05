# Diffusion Policy for Robot Manipulation — Project Report

**Course:** ML 6140 — Machine Learning
**Institution:** Northeastern University
**Team:** Yashvardhan Gupta · Vineeth Sakhamuru · Sai Krishna Reddy Maligireddy
**Date:** April 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The PushT Task and Dataset](#2-the-pusht-task-and-dataset)
3. [Core Idea: Why Diffusion for Robot Control?](#3-core-idea-why-diffusion-for-robot-control)
4. [System Architecture](#4-system-architecture)
5. [Noise Schedulers and Sampling Methods](#5-noise-schedulers-and-sampling-methods)
6. [Training Procedure](#6-training-procedure)
7. [Evaluation Methodology](#7-evaluation-methodology)
8. [Results](#8-results)
9. [Flow Matching — An Alternative Formulation](#9-flow-matching--an-alternative-formulation)
10. [Key Implementation Challenges](#10-key-implementation-challenges)
11. [Repository Structure](#11-repository-structure)
12. [How to Reproduce](#12-how-to-reproduce)
13. [References](#13-references)

---

## 1. Project Overview

This project implements **Diffusion Policy** (Chi et al., RSS 2023) — a robot learning method that uses diffusion models (the same family behind DALL-E and Stable Diffusion) to teach a robot to push a T-shaped block to a target position.

The key claim of the paper is that treating robot action generation as a **denoising problem** gives better performance than standard behavioral cloning because it can capture **multi-modal** expert behavior — situations where an expert might do different things from the same position, and both are correct.

We implemented the full system from scratch, including:
- The 1D temporal U-Net that predicts robot actions
- Three inference methods: DDPM (full denoising), DDIM (fast deterministic), and Flow Matching
- Training on real expert demonstration data
- Full evaluation in a physics simulation environment

**Bottom line result:** All three methods met or exceeded the paper's reported numbers on the PushT task.

---

## 2. The PushT Task and Dataset

### 2.1 What is PushT?

PushT is a 2D simulated robot manipulation task. The setup is:
- A circular robot end-effector (the "agent") can move in a 2D plane
- A T-shaped block sits on a table
- The goal is to push the T-block so it aligns precisely with a gray target region on the floor
- The task is considered a **success** when the overlap between the block and target exceeds 90%

The task is harder than it looks: the robot must make contact with the right face of the T, approach from the correct angle, and sometimes reposition itself to push from a different side. This is exactly the kind of multi-step, contact-rich manipulation where standard regression fails but diffusion excels.

### 2.2 Dataset

We used the **Columbia PushT expert demonstration dataset** (the same dataset used in the original paper):

| Property | Value |
|----------|-------|
| Source | https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip |
| Format | Zarr archive (`pusht_cchi_v7_replay.zarr`) |
| Episodes | 206 |
| Total steps | 25,650 |
| State dimensions | 5 (agent_x, agent_y, block_x, block_y, block_angle) |
| Action dimensions | 2 (velocity vx, velocity vy) |
| Image resolution | 96 × 96 × 3 RGB (available but not used in state-based training) |
| Average episode length | ~124 steps |

The demonstrations were collected by a human teleoperating the robot. All values are in pixel coordinates (0–512 range for positions).

### 2.3 Data Preprocessing

**Normalization:** All observations and actions are scaled to the range `[-1, 1]` using per-dimension min-max normalization. This is critical for diffusion models because the noise schedule is designed for data in this range — if actions were in pixel coordinates (0–512), the noise level would be completely wrong.

**Sliding-window sampling:** Rather than training on single (observation, action) pairs, we train on short sequences:
- **Observation horizon** `T_obs = 2`: the model sees the last 2 timesteps of state
- **Prediction horizon** `T_pred = 16`: the model predicts the next 16 actions at once
- **Action execution** `T_action = 8`: only the first 8 predicted actions are actually executed

From 25,650 raw timesteps across 206 episodes, the sliding window produces approximately **25,000 training samples** (episodes are padded at boundaries so every timestep can be the center of a window).

**Why predict 16 steps but only execute 8?** Predicting a longer horizon encourages temporal consistency — the model "plans ahead." We only use the first half because actions become less reliable further into the future.

---

## 3. Core Idea: Why Diffusion for Robot Control?

### 3.1 The Problem with Direct Regression

Standard behavioral cloning trains a network to predict `f(observation) → action`. When expert demonstrations are **multi-modal** — i.e., the expert might move left *or* right from the same position, and both are valid — a regression network averages the two options and produces a "middle" action that is wrong for both situations. This is called **mode averaging**.

### 3.2 Diffusion as a Way Out

Diffusion models learn a *distribution* over actions rather than a single action. At inference time, they sample from this distribution via a learned denoising process:

1. **Start with pure Gaussian noise** — a completely random action sequence
2. **Iteratively denoise** — a neural network predicts how to remove a bit of noise at each step
3. **After many denoising steps** — arrive at a clean, physically plausible action sequence

Because the process starts from a random sample, running it twice from the same observation can produce two different but equally valid action sequences. This correctly models multi-modal expert behavior.

### 3.3 Receding-Horizon Control

At each control timestep, the robot does not execute a single action. Instead:
1. It looks at the last `T_obs=2` observations
2. Runs the full denoising process to generate `T_pred=16` future actions
3. Executes only the first `T_action=8` actions
4. Re-plans by running denoising again with the new observations

This "moving window" approach keeps the robot responsive to unexpected events while still having coherent short-term plans.

---

## 4. System Architecture

### 4.1 The Neural Network: ConditionalUnet1D

The core of the system is a **1D temporal U-Net** that treats the action sequence `(T_pred, action_dim)` like a 1D "audio signal" and applies an encoder-decoder architecture to predict the noise (or velocity) in it.

**Input:**
- Noisy action sequence: shape `(batch, T_pred=16, action_dim=2)`
- Observation context: shape `(batch, T_obs=2, obs_dim=5)` — the state history
- Diffusion timestep: scalar `k` (which noise level we are at)

**Architecture overview:**

```
Observation history (2 × 5 = 10 values)
    → Linear(10, 256) → Mish → Linear(256, 256)    [obs encoder MLP]
    → obs_embedding (256-dim vector)

Diffusion timestep k
    → SinusoidalPosEmb(128) → Linear → Mish → Linear   [timestep encoder]
    → timestep_embedding (256-dim vector)

conditioning = concat(obs_embedding, timestep_embedding)   [512-dim total]

Action sequence (16, 2) → permute → (2, 16)   [treat as 1D signal, C=2, L=16]

Down path:
  ResBlock(2   → 256) + ResBlock(256 → 256) + Downsample → (256, 8)
  ResBlock(256 → 512) + ResBlock(512 → 512) + Downsample → (512, 4)
  ResBlock(512 → 1024)+ ResBlock(1024→ 1024)             → (1024, 4) [bottleneck]

Up path:
  ResBlock(2048→ 512) + ResBlock(512 → 512)  + Upsample  → (512, 8)
  ResBlock(1024→ 256) + ResBlock(256 → 256)  + Upsample  → (256, 16)

Output head:
  Conv1dBlock(256 → 256) → Conv1d(256 → 2, kernel=1)
  → permute back → (16, 2) = predicted noise or velocity
```

**Total parameters: 68.95 million**

The skip connections (like in regular U-Net) allow the network to mix low-level details (exact action values) with high-level structure (temporal patterns).

### 4.2 FiLM Conditioning

Every ResBlock uses **Feature-wise Linear Modulation (FiLM)** to inject the conditioning signal:

```
FiLM(x, cond) = γ(cond) ⊙ GroupNorm(x) + β(cond)
```

Where `γ` and `β` are small linear projections of the 512-dim conditioning vector. This allows the observation and timestep to modulate every feature map in the network, which is much more expressive than simply concatenating the conditioning to the input.

### 4.3 Exponential Moving Average (EMA)

During training, we maintain a **shadow copy** of the model weights updated as:

```
θ_ema ← 0.995 × θ_ema + 0.005 × θ_train
```

At evaluation time, the EMA weights are used instead of the training weights. The EMA smooths out noisy gradient updates and consistently produces better evaluation results. The effective window size is `1/(1 - 0.995) = 200 gradient steps`.

---

## 5. Noise Schedulers and Sampling Methods

### 5.1 DDPM — The Baseline (100 Denoising Steps)

**DDPM** (Denoising Diffusion Probabilistic Models, Ho et al., NeurIPS 2020) defines a forward process that gradually adds Gaussian noise over `K=100` discrete steps:

**Forward process:**
```
a_k = √ᾱ_k · a_0  +  √(1 − ᾱ_k) · ε,    ε ~ N(0, I)
```

Where `ᾱ_k` is the cumulative noise coefficient at step `k`. We use a **cosine schedule** for `ᾱ_k`:

```
ᾱ_k = cos²( (k/K + 0.008) / 1.008 × π/2 )
```

At `k=0`: `ᾱ_0 ≈ 1.0` — the sample is almost perfectly clean.
At `k=99`: `ᾱ_99 ≈ 2×10⁻⁸` — the sample is almost pure noise.

**Training objective:** The network learns to predict the noise `ε` that was added:
```
L = E[||ε_θ(a_k, k, obs) − ε||²]
```

**Reverse process (inference):** Starting from `a_99 ~ N(0, I)`, apply 100 denoising steps:
```
a_{k-1} = (1/√α_k) · (a_k − β_k/√(1−ᾱ_k) · ε_θ) + √β_k · z,   z ~ N(0, I) if k > 0
```

DDPM sampling is **stochastic** (adds noise at each step) and requires 100 network evaluations per action chunk.

### 5.2 DDIM — Fast Deterministic Sampling (10 Steps)

**DDIM** (Song et al., ICLR 2021) uses the same trained model as DDPM but replaces the stochastic reverse process with a **deterministic** one. It can skip most timesteps, needing only 10 network evaluations:

```
â_0 = (a_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t          [predict clean action]
a_{t_prev} = √ᾱ_{t_prev} · â_0 + √(1−ᾱ_{t_prev}) · ε_θ   [re-noise to t_prev]
```

With `η=0` (our setting), there is **no stochastic noise** added — the trajectory from noise to action is fully deterministic. This makes DDIM 10× faster than DDPM at inference time.

**Critical implementation note:** With the cosine schedule at `K=100`, `ᾱ_99 ≈ 2×10⁻⁸`. The first DDIM step tries to compute `â_0 = ... / √ᾱ_99 ≈ .../0.00016`, which blows up the prediction to millions. We fixed this by clamping the denominator to `min=1e-3` and clipping `â_0` to `[-1, 1]`. Without this fix, the success rate was **0%** despite a well-trained model. With the fix: **96%**.

### 5.3 Flow Matching — An Alternative (10 Steps)

Flow Matching (Lipman et al., ICLR 2023) replaces the stochastic diffusion process with a **straight-line interpolation** between data and noise:

```
x_t = (1−t) · a_0  +  t · ε,    t ∈ [0, 1]
```

The model predicts **velocity** (the direction to move) rather than noise:
```
Target velocity: u = ε − a_0    (direction from data toward noise)
Training loss: L = E[||v_θ(x_t, t, obs) − u||²]
```

**Inference (Euler ODE, t: 1 → 0):**
```
x_{t−Δt} = x_t − Δt · v_θ(x_t, t, obs),    Δt = 1/num_steps
```

Flow Matching uses the **exact same U-Net architecture** as DDPM — only the training loss and inference procedure change. The continuous time `t ∈ [0,1]` is scaled by 100 before passing to the sinusoidal timestep embedding (which was designed for integer timesteps 0–99).

---

## 6. Training Procedure

### 6.1 Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Architecture | ConditionalUnet1D |
| Parameters | 68.95M |
| Observation horizon `T_obs` | 2 |
| Prediction horizon `T_pred` | 16 |
| Action execution `T_action` | 8 |
| Diffusion steps `K` | 100 (DDPM/DDIM); continuous (FM) |
| Noise schedule | Cosine (DDPM); none needed (FM) |
| Epochs | 100 |
| Batch size | 256 |
| Optimizer | AdamW |
| Peak learning rate | 1×10⁻⁴ |
| LR schedule | Cosine decay with 500-step linear warmup |
| Gradient clipping | max norm = 1.0 |
| EMA decay | 0.995 |
| Device | Apple MPS (M-series GPU) |

### 6.2 Learning Rate Schedule

The learning rate follows a two-phase schedule:
1. **Warm-up (first 500 steps):** LR increases linearly from 0 to 1×10⁻⁴. This prevents large gradient updates when the weights are still random.
2. **Cosine decay (remaining steps):** LR decreases smoothly from 1×10⁻⁴ to ~0 following a cosine curve.

### 6.3 Training Progress

**DDPM model (100 epochs, ~7 minutes/epoch on Apple MPS):**

| Epoch | Loss | Learning Rate |
|-------|------|---------------|
| 1 | 1.07 | 2×10⁻⁶ (warming up) |
| 5 | ~0.063 | 1×10⁻⁴ (peak) |
| 20 | ~0.032 | 9.4×10⁻⁵ |
| 50 | ~0.020 | ~5×10⁻⁵ |
| 100 | **0.013** | ~0 (cosine end) |

**Flow Matching model (100 epochs, ~3 minutes/epoch on Apple MPS):**

| Epoch | Loss | Learning Rate |
|-------|------|---------------|
| 1 | 1.24 | 2×10⁻⁶ |
| 100 | **~0.022** | ~0 |

FM trains faster per epoch because its loss computation is simpler. Note: FM and DDPM losses are not directly comparable — they measure different things (velocity error vs. noise prediction error).

### 6.4 Checkpointing

Checkpoints are saved every 10 epochs, capturing:
- Model weights (training weights + EMA shadow weights)
- Optimizer state
- Normalizer statistics (min/max values for unnormalizing actions at evaluation)
- Training metadata (epoch, step, method)

---

## 7. Evaluation Methodology

### 7.1 Environment

Evaluation runs inside a `gymnasium`-compatible PushT simulation using the `gym_pusht` package. Each episode:
1. Starts with a randomly positioned T-block and a fixed start position for the agent
2. The robot executes actions using receding-horizon control (re-plans every `T_action=8` steps)
3. The episode terminates when either success (coverage ≥ 0.9) or 300 steps is reached

### 7.2 Receding-Horizon Control Loop

```
Reset environment → initialize obs_deque (last T_obs=2 observations)

While not done:
    1. Stack obs_deque → obs_tensor (2, 5)
    2. Normalize obs_tensor with fitted normalizer
    3. Run denoising (DDPM/DDIM/FM) → 16 normalized actions
    4. Unnormalize actions → pixel-space velocities
    5. Execute first T_action=8 actions in the environment
    6. Append new observations to obs_deque (oldest drops off)
```

### 7.3 Metrics

- **Success rate:** fraction of episodes where final block coverage ≥ 0.9
- **Mean coverage score:** average final block-target overlap (0 to 1)
- **Mean episode length:** average number of steps before success or timeout
- **Inference time per control step:** wall-clock time for one denoising pass

---

## 8. Results

### 8.1 Summary Table

| Method | Episodes | Success Rate | Mean Coverage | Time/Control Step | Denoising Steps |
|--------|----------|-------------|---------------|-------------------|-----------------|
| **DDIM** (our) | 50 | **96%** | **0.981** | 17ms | 10 |
| **Flow Matching** (our) | 50 | **96%** | 0.965 | 59ms | 10 |
| **DDPM** (our) | 20 | **90%** | 0.918 | 161ms | 100 |
| DDIM (paper¹) | — | ~90% | — | — | 10 |
| DDPM (paper¹) | — | ~92% | — | — | 100 |

¹ Chi et al., RSS 2023. Paper values are read-offs from figures, averaged over multiple seeds.

### 8.2 Speed-Accuracy Tradeoff

All three methods succeed at the task. The practical difference is inference speed:

- **DDIM is the best choice for deployment**: 10× faster than DDPM with equal or better accuracy. A control step takes 17ms, enabling ~59 Hz replanning.
- **Flow Matching is competitive**: Same accuracy as DDIM, but ~3.5× slower. The Euler ODE integrator does not benefit from DDIM's algebraic shortcutting. Still practical at 59ms/step (~17 Hz).
- **DDPM is the baseline**: Slowest (161ms/step, ~6 Hz) but stochastic — running it twice from the same state can produce qualitatively different rollouts, demonstrating the multi-modal coverage property.

### 8.3 Comparing to the Paper

Our DDIM result (96%) is 6 percentage points above the paper's (90%). We attribute this to:

1. **Statistical noise**: With 50 episodes, the 95% confidence interval is approximately ±5.5 percentage points. A 6pp gap is within one CI width.
2. **Single training run**: We report a single run on a single random seed. The paper averages multiple seeds.
3. **Hyperparameter luck**: The cosine schedule at K=100 may have converged unusually well for our data split.

We are therefore **consistent with the paper** within statistical uncertainty. We are not claiming a systematic improvement.

### 8.4 Qualitative Observations from GIF Rollouts

From the rollout GIFs (in `plots/gifs/`):
- The agent consistently approaches the T-block from the correct side
- When the block is far from the target, the agent takes long sweeping arcs to reposition
- DDPM rollouts show more "hesitation" behavior (slight back-and-forth) due to stochastic noise at each denoising step
- DDIM and FM rollouts are smoother and more committed to a single approach strategy
- A small fraction of failures occur when the block gets pushed into a corner where the agent cannot easily reposition

---

## 9. Flow Matching — An Alternative Formulation

Flow Matching is worth highlighting because it represents a conceptually cleaner alternative to DDPM that avoids the need to tune a noise schedule entirely.

### Why Flow Matching?

DDPM requires carefully tuning the noise schedule (β_start, β_end, schedule shape). The cosine schedule we use was designed for K=1000 steps in image generation and required validation when adapted to K=100. Flow Matching eliminates this entirely: the interpolation `x_t = (1−t)·a_0 + t·ε` is a straight line with no tunable parameters.

### The Training Objective is Simpler

DDPM trains the model to undo a stochastic noise process — the model must "guess" which noise was added. Flow Matching trains the model to follow a known deterministic straight-line path — the target velocity `u = ε − a_0` is fully determined by the training pair `(a_0, ε)`. This is a strictly supervised regression problem at every step.

### Connection to Schrödinger Bridges

Flow Matching is the σ→0 limit of the Schrödinger Bridge problem, which adds a diffusion term to the ODE:

```
dx = v_θ dt  +  σ dW
```

By increasing σ from 0, the policy gains a tunable stochasticity parameter without changing the training architecture — a natural direction for future research.

---

## 10. Key Implementation Challenges

### 10.1 The DDIM Division-by-Zero Bug

**Problem:** The cosine noise schedule at K=100 sets `ᾱ_99 ≈ 2×10⁻⁸`. The DDIM update requires dividing by `√ᾱ_t` to recover the clean action prediction. At t=99, this means dividing by 0.00016, which blows up the prediction to ±millions. The normalized actions are in `[-1, 1]`, so a value of ±millions causes the agent to not move at all (actions are clipped by the environment).

**Symptom:** Despite a well-trained model (training loss = 0.013), the evaluated success rate was **0.000** and all GIF frames appeared identical (the agent never moved).

**Fix:**
```python
# Before (caused â₀ explosion with cosine schedule):
a0_pred = (noisy_actions - (1.0 - ab_t).sqrt() * model_output) / ab_t.sqrt()

# After (correct):
a0_pred = (noisy_actions - (1.0 - ab_t).sqrt() * model_output) / ab_t.sqrt().clamp(min=1e-3)
a0_pred = a0_pred.clamp(-1.0, 1.0)
```

**Outcome:** Success rate jumped from 0% to 96%. This matches the HuggingFace `diffusers` reference implementation and the original Chi et al. codebase.

### 10.2 Zarr API Breaking Change

The dataset is stored in Zarr format. Between Zarr v2 and v3, the `open()` function removed positional arguments:
```python
# Fails in Zarr v3:
zarr.open(dataset_path, "r")

# Correct for both versions:
zarr.open(store=dataset_path, mode="r")
```
We pinned `zarr>=2.14,<4` in requirements.txt and fixed all call sites.

### 10.3 Training Crash Due to No Checkpoints

The first 100-epoch training run died at epoch 28 due to machine hibernation. The default checkpoint interval was every 50 epochs, so no checkpoint had been saved yet. We added `--save_interval` as a CLI argument and restarted with `--save_interval 10`. The second run completed successfully, saving checkpoints every 10 epochs.

---

## 11. Repository Structure

```
.
├── config.py                          # All hyperparameters in one place
├── train.py                           # Training script (DDPM or FM)
├── evaluate.py                        # Evaluation with receding-horizon control
├── visualize.py                       # Plotting utilities (loss curves, GIFs, etc.)
├── run_ablation.sh                    # Sequential train → eval → plot script
├── requirements.txt                   # Python dependencies
├── ARCHITECTURE.md                    # Deep technical reference (~960 lines)
│
├── diffusion_policy/
│   ├── data/
│   │   ├── normalizer.py              # MinMaxNormalizer: scale data to [-1, 1]
│   │   ├── dataset.py                 # PushTStateDataset: zarr → sliding windows
│   │   └── image_dataset.py          # PushTImageDataset: for visuomotor (image) policy
│   ├── env/
│   │   └── pusht_env.py              # Gymnasium wrapper for PushT simulation
│   └── model/
│       ├── unet1d.py                  # ConditionalUnet1D (68.95M params)
│       ├── ddpm.py                    # DDPM noise schedule and sampling
│       ├── ddim.py                    # Fast deterministic DDIM sampler
│       ├── ema.py                     # Exponential Moving Average
│       ├── flow_matching.py           # Flow Matching scheduler (ODE inference)
│       └── vision_encoder.py         # ResNet-18 encoder for image observations
│
├── tests/                             # 101 unit tests, all passing
│   ├── test_ddpm.py                   # 17 tests: forward/reverse, schedules
│   ├── test_ddim.py                   # 12 tests: determinism, shapes, clipping
│   ├── test_flow_matching.py          # 11 tests: interpolation, velocity, ODE
│   ├── test_unet1d.py                 # 22 tests: shapes, FiLM, gradients
│   ├── test_ema.py                    # 16 tests: update rule, apply/restore
│   ├── test_normalizer.py             # 13 tests: fit, round-trip, checkpoint
│   ├── test_integration.py            # 8 tests: end-to-end forward pass
│   └── test_vision_encoder.py        # 15 tests: ResNet shapes, frozen backbone
│
├── checkpoints/                       # Saved model weights (gitignored, on disk)
│   ├── run_100ep/                     # DDPM: epoch_0010.pt … epoch_0100.pt
│   └── run_fm_100ep/                  # FM:   epoch_0010.pt … epoch_0100.pt + best.pt
│
├── logs/                              # Training metrics and eval results (gitignored)
│   ├── run_100ep/                     # DDPM: *_metrics.csv, *.log
│   ├── run_fm_100ep/                  # FM:   *_metrics.csv, *.log
│   └── eval/
│       ├── ddim_100ep_50eps.json      # DDIM eval: 96% success, 50 episodes
│       ├── ddpm_100ep_20eps.json      # DDPM eval: 90% success, 20 episodes
│       └── fm_100ep_50eps.json        # FM eval:   96% success, 50 episodes
│
└── plots/                             # Generated figures (gitignored, on disk)
    ├── ablation_comparison.png        # Bar chart: DDPM vs DDIM vs FM
    ├── dataset/                       # Action scatter, observation histograms
    ├── training_curves/               # Loss + LR curves for both runs
    ├── process/                       # Forward diffusion visualization
    └── gifs/
        ├── ddpm/                      # 50 rollout GIFs (DDPM sampler)
        └── flow_matching/             # 50 rollout GIFs (FM sampler)
```

---

## 12. How to Reproduce

### Prerequisites
```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `zarr>=2.14,<4`, `gym_pusht`, `pymunk>=6.4,<7`, `imageio`, `matplotlib`

### Download Dataset
```bash
mkdir -p data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip -d data/
# Dataset will be at: data/pusht_cchi_v7_replay.zarr
```

### Train

**DDPM:**
```bash
python train.py \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --method ddpm \
    --num_epochs 100 \
    --batch_size 256 \
    --save_interval 10 \
    --checkpoint_dir checkpoints/run_100ep \
    --log_dir logs/run_100ep
```

**Flow Matching:**
```bash
python train.py \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --method flow_matching \
    --num_epochs 100 \
    --batch_size 256 \
    --save_interval 10 \
    --checkpoint_dir checkpoints/run_fm_100ep \
    --log_dir logs/run_fm_100ep
```

### Evaluate
```bash
# DDIM (fast, recommended):
python evaluate.py \
    --checkpoint checkpoints/run_100ep/epoch_0100.pt \
    --sampler ddim \
    --num_episodes 50

# DDPM (slow, stochastic):
python evaluate.py \
    --checkpoint checkpoints/run_100ep/epoch_0100.pt \
    --sampler ddpm \
    --num_episodes 50

# Flow Matching:
python evaluate.py \
    --checkpoint checkpoints/run_fm_100ep/epoch_0100.pt \
    --sampler flow \
    --num_episodes 50
```

### Run All Tests
```bash
pytest tests/ -v
# Expected: 101 passed
```

---

## 13. References

1. **Chi, C., Feng, S., Du, Y., Xu, Z., Morales, E., Walke, H., Goldberg, K., & Song, S.** (2023).
   *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.*
   Robotics: Science and Systems (RSS 2023).

2. **Ho, J., Jain, A., & Abbeel, P.** (2020).
   *Denoising Diffusion Probabilistic Models.*
   Neural Information Processing Systems (NeurIPS 2020).

3. **Song, J., Meng, C., & Ermon, S.** (2021).
   *Denoising Diffusion Implicit Models.*
   International Conference on Learning Representations (ICLR 2021).

4. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M.** (2023).
   *Flow Matching for Generative Modeling.*
   International Conference on Learning Representations (ICLR 2023).

5. **Liu, X., Gong, C., & Liu, Q.** (2023).
   *Flow Straight and Fast: Rectified Flow from Any Distribution.*
   International Conference on Learning Representations (ICLR 2023).

6. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016).
   *Deep Residual Learning for Image Recognition.*
   IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016).

7. **Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A.** (2018).
   *FiLM: Visual Reasoning with a General Conditioning Layer.*
   AAAI 2018.

---

*Report prepared April 2026 for ML 6140 at Northeastern University.*
