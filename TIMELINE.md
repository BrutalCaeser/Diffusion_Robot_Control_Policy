# Project Timeline — Diffusion Policy for PushT
## Feb 7, 2026 → April 15, 2026 (9.5 weeks)

> **Philosophy:** Learn every line. No vibe coding. Understand the math, then implement it.

---

## Week 1 (Feb 7–13): Environment Setup & Data Pipeline
**Goal:** Be able to load, visualize, and iterate over training batches.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Install dependencies, verify PushT env runs, download Zarr dataset | Working env + raw data on disk |
| Mon-Tue | Implement `normalizer.py` (MinMaxNormalizer: fit, normalize, unnormalize) | Tested normalizer with unit tests |
| Wed-Thu | Implement `dataset.py` (Zarr loading, sliding window, padding, normalization) | `DataLoader` yields correct `(obs, action)` shapes |
| Fri | Implement dataset visualizations in `visualize.py` (trajectories, distributions) | Plots of expert demos, histograms of normalized data |

**Sanity checks (all on CPU):**
- [ ] `obs.shape == (B, 2, 5)` and `action.shape == (B, 16, 2)`
- [ ] Normalized values are in `[-1, 1]`
- [ ] Unnormalize(normalize(x)) ≈ x (round-trip test)
- [ ] Visualized trajectories look like an agent pushing a T-block

---

## Week 2 (Feb 14–20): U-Net Architecture
**Goal:** Forward pass compiles and produces correct output shapes on CPU.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Implement `SinusoidalPosEmb` + timestep MLP. Study: why sinusoidal? | Tested embedding module |
| Mon | Implement observation encoder MLP | Tested: `(B, T_obs*obs_dim) → (B, cond_dim)` |
| Tue-Wed | Implement `FiLMConditionedResBlock` (Conv1d → GroupNorm → FiLM → Mish → Conv1d) | Tested residual block |
| Thu-Fri | Assemble full `ConditionalUnet1D` (encoder, bottleneck, decoder, skip connections) | `model(noisy_actions, timestep, obs)` → correct shape |

**Sanity checks (all on CPU):**
- [ ] Output shape == input shape `(B, T_pred, action_dim)`
- [ ] Model accepts both integer timesteps (DDPM) and float timesteps (FM)
- [ ] Gradient flows: `loss.backward()` completes without error
- [ ] Parameter count is reasonable (~10-50M for PushT)

---

## Week 3 (Feb 21–27): DDPM Training Pipeline
**Goal:** Loss decreases on full dataset. Model overfits on 1 batch.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Implement `ddpm.py`: noise schedule (β, α, ᾱ), `add_noise()`, `sample_timesteps()` | Tested forward process |
| Mon | Implement `ema.py`: EMA update, apply, restore, save/load | Tested EMA on a toy model |
| Tue-Wed | Implement `train.py`: full training loop with optimizer, LR schedule, logging, checkpointing | Working training script |
| Thu | **Local sanity check**: overfit on 1 batch on CPU (loss → ~0 in <100 steps) | Confirmed model can memorize |
| Fri | Push to GitHub, clone into Colab, start full GPU training run | Training running on Colab, loss curve decreasing |

**Sanity checks:**
- [ ] `add_noise(clean, noise, k=0)` ≈ clean (almost no noise at k=0)
- [ ] `add_noise(clean, noise, k=99)` ≈ pure noise (signal destroyed at k=K-1)
- [ ] ᾱ values monotonically decrease from ~1 to ~0
- [ ] Single-batch overfit: loss < 0.01 within 100 steps
- [ ] Full training: loss curve shows consistent decrease over first few epochs

---

## Week 4 (Feb 28–Mar 6): DDPM & DDIM Inference
**Goal:** Trained model generates coherent action trajectories. First evaluation rollouts.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Implement DDPM reverse sampling loop in `ddpm.py` | Can generate action sequences from noise |
| Mon | Implement receding-horizon control loop in `evaluate.py` | Agent runs in PushT env using predicted actions |
| Tue-Wed | Implement `ddim.py`: timestep subsequence, deterministic step, full sampling loop | DDIM sampling works as drop-in replacement |
| Thu | First evaluation: 50 rollouts with DDPM and DDIM, compute success rate | Quantitative metrics |
| Fri | Create rollout GIF visualizations, debug if performance is poor | Visual confirmation of policy behavior |

**Sanity checks:**
- [ ] DDPM 100-step sampling produces actions in [-1, 1] range
- [ ] DDIM 10-step sampling produces similar quality to DDPM 100-step
- [ ] Unnormalized actions are in physically reasonable range
- [ ] GIFs show the agent actively pushing (not standing still or diverging)

---

## Week 5 (Mar 7–13): Performance Tuning & Flow Matching Start
**Goal:** Achieve ≥70% success rate. Begin Flow Matching implementation.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Debug & iterate on DDPM if success rate < 70%: check normalization, EMA, horizons | Improved performance |
| Mon-Tue | Implement `flow_matching.py`: interpolation, velocity target, Euler solver | FM training loss compiles |
| Wed-Thu | Implement FM training mode in `train.py` (--method flow_matching flag) | FM training runs on CPU (sanity check) |
| Fri | Push FM code, start FM training on Colab alongside continued DDPM tuning | FM training running on GPU |

---

## Week 6 (Mar 14–20): Flow Matching Evaluation & Ablation Setup
**Goal:** FM produces rollouts. Begin systematic ablation experiments.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Implement FM inference in `evaluate.py`, run first FM rollouts | FM success rate measured |
| Mon-Tue | Set up ablation experiment scripts: prediction horizon {4, 8, 16, 32} | Automated ablation launcher |
| Wed-Thu | Run ablation: DDIM steps {5, 10, 20, 50} + wall-clock timing | Results table |
| Fri | Run ablation: observation horizon {1, 2, 4} | Results table |

---

## Week 7 (Mar 21–27): Remaining Ablations & MLP Baseline
**Goal:** All ablation data collected.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Implement MLP Behavioral Cloning baseline (same data, same horizons) | Trained MLP baseline |
| Mon-Tue | Compare MLP vs. Diffusion: success rate + action distribution visualization | Multimodality comparison plots |
| Wed | Run ablation: EMA vs. no EMA | Results table |
| Thu-Fri | Run ablation: DDPM vs. Flow Matching (success rate, convergence, inference speed) | Comprehensive comparison |

---

## Week 8 (Mar 28–Apr 3): Analysis & Report Writing
**Goal:** All figures generated. Report draft complete.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Generate all figures: loss curves, ablation bar charts, rollout GIF grids | Figures directory |
| Mon-Tue | Write report: Introduction, Background, Method sections | Draft sections 1-3 |
| Wed-Thu | Write report: Experiments, Discussion sections | Draft sections 4-5 |
| Fri | Write report: Future Work (FM → Schrödinger Bridges), Conclusion | Full draft |

---

## Week 9 (Apr 4–10): Report Polishing & Final Experiments
**Goal:** Publication-quality report. Fill any experimental gaps.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Re-run any failed/incomplete ablations, generate missing figures | Complete experimental results |
| Mon-Tue | Polish report: proofread, fix figures, ensure math notation is consistent | Revised draft |
| Wed-Thu | Add appendix: implementation details, full hyperparameter tables, code snippets | Complete appendix |
| Fri | Final review pass, check all citations and references | Near-final report |

---

## Week 10 (Apr 11–15): Final Submission
**Goal:** Everything submitted.

| Day | Task | Deliverable |
|-----|------|-------------|
| Sat-Sun | Final report polish, prepare code repository for submission | Clean repo |
| Mon-Tue | Create README.md with setup instructions, run final eval to confirm numbers | Reproducible results |
| **Wed Apr 15** | **SUBMIT** | **Report + code + checkpoints** |

---

## Post-Deadline (Apr 16+): Thesis Bridge Work
- Polish Flow Matching implementation
- Experiment with Optimal Transport conditioning for FM
- Begin exploring Schrödinger Bridge formulation (add diffusion coefficient σ)
- This becomes the foundation for Fall 2026 thesis work

---

## Risk Mitigation
| Risk | Mitigation |
|------|-----------|
| PushT env setup issues | Have backup: use pre-recorded dataset only, skip live env initially |
| Poor DDPM performance | Checklist in spec: normalization, EMA, noise schedule, action execution |
| Colab GPU timeouts | Save checkpoints every 50 epochs, resume from last checkpoint |
| Flow Matching doesn't converge | It's a stretch goal — report still has DDPM/DDIM as core contribution |
| Time crunch on report | Start writing Method section in Week 5 (don't wait until Week 8) |
