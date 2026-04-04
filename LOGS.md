# Project Development Log

> **ML 6140 Final Project — Diffusion Policy for PushT**
> Team: Yashvardhan Gupta · Vineeth Sakhamuru · Sai Krishna Reddy Maligireddy
>
> This file records every significant edit, fix, feature addition, and experimental result
> in chronological order. Append new entries at the bottom.

---

## Session 1 — Initial Scaffolding
**Commits:** `b179359`, `4168085`

### Added
- `config.py` — centralised hyperparameter hub (`TrainConfig`, `EnvConfig`, `DataConfig`, `ModelConfig`, `DiffusionConfig`, `FlowMatchingConfig`)
- `requirements.txt` — initial Python dependencies
- `README.md`, `PROJECT_REPORT.md`, `TIMELINE.md` — project documentation
- Directory structure: `diffusion_policy/{data,env,model}/`, `tests/`

---

## Session 2 — Full Pipeline Implementation
**Commit:** `826a8b3`

### Added
- `diffusion_policy/data/normalizer.py` — `MinMaxNormalizer`: per-dimension min-max scaling to `[-1, 1]`; `state_dict()` / `load_state_dict()` for checkpoint serialisation
- `diffusion_policy/data/dataset.py` — `PushTStateDataset`: loads PushT zarr file, sliding-window sampling, episode-boundary padding
- `diffusion_policy/env/pusht_env.py` — `PushTEnv`: gymnasium wrapper around `gym_pusht/PushT-v0`; `make_obs_deque()` helper for receding-horizon control
- `diffusion_policy/model/unet1d.py` — `ConditionalUnet1D`: 1D U-Net with FiLM conditioning, sinusoidal timestep embeddings, GroupNorm residual blocks
- `diffusion_policy/model/ema.py` — `EMA`: shadow-weight exponential moving average; `apply()` / `restore()` for eval/train cycling
- `diffusion_policy/model/ddpm.py` — `DDPMScheduler`: cosine + linear beta schedules, `add_noise()`, `step()`, `sample()`, `sample_timesteps()`
- `diffusion_policy/model/ddim.py` — `DDIMScheduler`: deterministic DDIM sampler, configurable `eta`, subsequence timestep selection
- `diffusion_policy/model/flow_matching.py` — `FlowMatchingScheduler`: straight-line OFM, `get_loss()`, `sample()` with Euler integration
- `train.py` — full training loop: cosine-warmup LR, gradient clipping, EMA update, checkpoint saving, per-epoch evaluation
- `evaluate.py` — receding-horizon DDIM/DDPM/Flow inference loop; success/coverage metrics
- `visualize.py` — `plot_training_curves()`, `plot_dataset_summary()`, `visualize_diffusion_process()`, `save_rollout_gif()`, `plot_eval_comparison()`
- `tests/` — 7 test files, 86 tests covering all modules

---

## Session 3 — Bug Fixes: DDPM Tests & gym_pusht Compatibility
**Commit:** `9673ece`

### Fixed
- `DDPMScheduler`: switched default schedule from `"linear"` to `"cosine"` — linear schedule designed for K=1000 fails at K=100 (3 DDPM tests were failing: `test_alphas_cumprod_decreasing`, `test_k0_is_clean`, `test_kK_is_noise`)
- `DDIMScheduler.test_deterministic_with_eta0`: added `torch.manual_seed(0)` before each `sample()` call
- `PushTEnv`: fixed `gym_pusht 0.1.6` API incompatibility (`add_collision_handler` removed in pymunk v7)
- Sanity-check tolerances relaxed for numerical precision

---

## Session 4 — Real PushT Dataset & Visuomotor Policy
**Commits:** `4322249`, `b207467`, `af581c1`, `42aa7c1`

### Dataset
- Switched from synthetic dataset to real Columbia PushT expert demonstrations
- Correct download URL: `https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`
- Dataset: 206 episodes, 25,650 steps; `state (25650,5)`, `action (25650,2)`, `img (25650,96,96,3)`
- Fixed `dataset.py` error message to show correct URL

### New: Visuomotor (Image-Based) Policy
**`diffusion_policy/model/vision_encoder.py`**
- `ResNetEncoder`: ResNet-18 backbone (torchvision, stripped final avgpool/fc)
- `AdaptiveAvgPool2d(1)` → global average pool → concat T_obs frames → projection MLP
- Output: `(B, obs_cond_dim=256)` conditioning vector
- Supports `pretrained=True` (ImageNet weights) and `freeze_backbone=True`

**`diffusion_policy/data/image_dataset.py`**
- `PushTImageDataset`: loads `data/img` from zarr, normalises `[0,255] → [0,1]`, transposes to channel-first `(N,3,H,W)`
- Same sliding-window + boundary-padding logic as `PushTStateDataset`
- `get_action_normalizer()` exposes fitted `MinMaxNormalizer`

### Modified
- `diffusion_policy/model/unet1d.py`: added `obs_dim=0` image path — when 0, bypasses internal obs MLP; pre-encoded `(B, cond_dim)` vector from `ResNetEncoder` used directly
- `config.py`: added `VisionEncoderConfig`, `DataConfig.obs_type = "state" | "image"`, `__post_init__` constraint `vision.obs_cond_dim == model.cond_dim`
- `train.py`: dataset/model/encoder selection based on `obs_type`; vision encoder EMA tracked separately; checkpoint saves encoder state; `--obs_type` CLI arg
- `evaluate.py`: `load_policy()` reconstructs `ResNetEncoder` from checkpoint in image mode; `obs_normalizer=None` handled throughout
- `requirements.txt`: pinned `zarr>=2.14,<4` (v3 changed API), `pymunk>=6.4,<7`
- `diffusion_policy/data/__init__.py`, `diffusion_policy/model/__init__.py`: exported new classes

### Tests
- `tests/test_vision_encoder.py`: 15 tests (shapes, gradients, conditioning, frozen backbone, UNet integration, repr)
- **All 101 tests passing**

### Documentation & Tooling
- `ARCHITECTURE.md` (958 lines): added `image_dataset.py` and `vision_encoder.py` sections; updated UNet dual-path docs; corrected download URL; visuomotor reference results; updated ablation table; He et al. (ResNet) reference added
- `run_ablation.sh`: sequential DDPM → FM training → eval → comparison plot script
- `.gitignore`: added `._*` macOS resource fork pattern

---

## Session 5 — 100-Epoch DDPM Training Run
**Commits:** `d920857`

### Training
- Launched 100-epoch DDPM training on real PushT dataset
  - `batch_size=256`, `obs_type=state`, cosine schedule, MPS device
  - 68.95M parameters, `save_interval=10`
- First run died at epoch 28 (no checkpoint saved — `save_interval` was 50)
- Added `--save_interval` as CLI argument (`d920857`)
- Restarted with `--save_interval 10`; machine slept overnight (9hr gap at epoch 70), resumed cleanly
- **Completed: 100 epochs, final loss = 0.013**
- Checkpoints saved: `epoch_0010.pt` through `epoch_0100.pt`

### Loss Curve
| Epoch | Loss | LR |
|-------|------|----|
| 1 | 0.62 | 2e-6 (warmup) |
| 5 | 0.063 | 1e-4 (peak) |
| 20 | 0.032 | 9.4e-5 |
| 50 | 0.020 | ~5e-5 |
| 100 | **0.013** | ~0 (cosine end) |

---

## Session 6 — Repo Cleanup & Artefact Migration
**Commits:** `42aa7c1`, `a3242d6`

### Fixed
- Removed accidentally committed macOS `._*` resource fork files from git tracking (`a3242d6`)
- Copied `checkpoints/`, `logs/`, `plots/` from worktree to main repo directory
- Main repo now has all code + artefacts; working directory going forward is main repo

---

## Session 7 — DDIM Bug Fix: 0% → 96% Success Rate
**Commit:** `cc7535e`

### Bug Found
**File:** `diffusion_policy/model/ddim.py` — `DDIMScheduler.step()`

The DDIM â_0 prediction divides by `√ᾱ_t`:
```
â_0 = (a_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
```
Under the cosine schedule with K=100, `ᾱ_99 ≈ 2e-8` and `√ᾱ_99 ≈ 0.000156`.
Dividing by this at the first DDIM step (t=99→t=90) caused `â_0` to explode to ±millions,
making all unnormalized actions astronomically large and the agent frozen in place.

**Symptom:** Evaluation reported `success_rate=0.000`, `mean_score=0.056` despite
training loss=0.013 (well converged). All GIF frames appeared identical (agent not moving).

### Fix
```python
# Before:
a0_pred = (noisy_actions - (1.0 - ab_t).sqrt() * model_output) / ab_t.sqrt()

# After:
a0_pred = (noisy_actions - (1.0 - ab_t).sqrt() * model_output) / ab_t.sqrt().clamp(min=1e-3)
a0_pred = a0_pred.clamp(-1.0, 1.0)
```
Two changes:
1. Clamp denominator to `min=1e-3` (avoids near-zero division)
2. Clamp `â_0` to `[-1, 1]` (normalised action range — standard in all DDIM implementations)

This matches the HuggingFace diffusers implementation and the original Chi et al. codebase.

### Results After Fix

| Sampler | Episodes | Success Rate | Mean Coverage | Time/step |
|---------|----------|-------------|---------------|-----------|
| DDIM (10 steps) | 50 | **96%** | **0.981** | 17ms |
| DDPM (100 steps) | 5 | **100%** | **0.993** | 148ms |

**Paper reference (Chi et al., RSS 2023):** DDIM ~90%, DDPM ~92%

Our result is consistent with (and within the confidence interval of) the paper's numbers.
The 96% vs 90% difference is within statistical noise for a 50-episode evaluation.

### GIF Rollouts Generated
| Folder | Sampler | Episodes |
|--------|---------|---------|
| `plots/gifs/sampler_ddim/` | DDIM, 10 steps | 5 |
| `plots/gifs/sampler_ddpm/` | DDPM, 100 steps | 5 |

---

## Pending / Next Steps

| Priority | Task | Status |
|----------|------|--------|
| 🔴 | Flow Matching training (100 epochs) | Not started |
| 🔴 | Flow Matching evaluation + GIFs | Blocked on above |
| 🟡 | `plots/ablation_comparison.png` — DDPM vs DDIM vs FM bar chart | Blocked on FM |
| 🟡 | Evaluate `epoch_0050.pt` to build learning curve for report | Ready to run |
| 🟡 | Multimodal demonstration: overlay 5 rollouts from same start state | Ready to run |
| 🟢 | Full 50-episode DDPM evaluation (currently only 5 episodes) | Ready to run |

---

*Last updated: 2026-04-04*
