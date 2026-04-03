# Diffusion Policy — Architecture & Technical Reference

> **ML 6140 Final Project — Northeastern University**
> Team: Yashvardhan Gupta · Vineeth Sakhamuru · Sai Krishna Reddy Maligireddy

---

## Quick-Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install gym-pusht           # PushT simulation environment

# 2. Download the expert demonstration dataset (Columbia, ~100 MB)
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip -d data/
# → data/pusht_cchi_v7_replay.zarr  (206 episodes, 25,650 steps)

# 3a. Train the policy — state observations (fast, ~14 hrs on MPS / ~2 hrs on GPU)
python train.py \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --num_epochs 100 --batch_size 256

# 3b. Train with image observations (visuomotor, ResNet-18 encoder)
python train.py \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --obs_type image --num_epochs 100 --batch_size 64

# 3c. Train with Flow Matching instead of DDPM
python train.py \
    --dataset_path data/pusht_cchi_v7_replay.zarr \
    --method flow_matching --num_epochs 100 --batch_size 256

# 4. Evaluate the trained policy
python evaluate.py --checkpoint checkpoints/best.pt --sampler ddim --num_episodes 50

# 5. Run unit tests
pytest tests/ -v
```

---

## The Big Picture — Why Diffusion Policy?

### The Problem with Standard Behavioural Cloning

A standard neural policy learns `f(obs) → action` by minimising MSE on expert demonstrations.  The fundamental flaw is that many real tasks are **multimodal** — there are *multiple equally valid strategies* for the same situation.

> **Example (PushT):** The agent can push the T-block from the left or from the right.  Both reach the goal.  An MSE-trained policy averages over these two modes, predicting a path straight *through* the block — which is physically impossible.

### The Diffusion Solution

Instead of predicting a single action, the policy learns a **probability distribution** over actions.  Sampling from this distribution naturally produces diverse, valid behaviours.

Concretely, the policy learns to **iteratively denoise** a random action sequence conditioned on the current observation:

```
obs + pure noise → [100 denoising steps] → coherent action sequence
```

Each denoising step makes the action sequence slightly less noisy and more like what an expert would do, conditioned on what the robot currently sees.  The key insight from Chi et al. (RSS 2023):

> **"Represent the visuomotor policy as a conditional score function of the action distribution."**

---

## Project File Map

```
Diffusion_Robot_Control_Policy/
├── config.py                          ← All hyperparameters (one place)
├── train.py                           ← Training orchestration (state + image)
├── evaluate.py                        ← Evaluation / rollout script
├── visualize.py                       ← Plotting and GIF utilities
├── ARCHITECTURE.md                    ← This document
├── requirements.txt                   ← Python dependencies
│
├── diffusion_policy/
│   ├── data/
│   │   ├── normalizer.py              ← MinMaxNormalizer: scale to [-1,1]
│   │   ├── dataset.py                 ← PushTStateDataset: Zarr + sliding window
│   │   └── image_dataset.py           ← PushTImageDataset: RGB frames variant
│   ├── env/
│   │   └── pusht_env.py               ← PushTEnv: gymnasium wrapper
│   └── model/
│       ├── unet1d.py                  ← ConditionalUnet1D: core network (state+image)
│       ├── vision_encoder.py          ← ResNetEncoder: ResNet-18 visual backbone
│       ├── ema.py                     ← EMA: weight averaging for stable inference
│       ├── ddpm.py                    ← DDPMScheduler: noise schedule + sampling
│       ├── ddim.py                    ← DDIMScheduler: fast 10-step inference
│       └── flow_matching.py           ← FlowMatchingScheduler: FM extension
│
└── tests/
    ├── test_normalizer.py
    ├── test_unet1d.py
    ├── test_ddpm.py
    ├── test_ddim.py
    ├── test_ema.py
    ├── test_flow_matching.py
    ├── test_vision_encoder.py         ← 15 tests for ResNetEncoder
    └── test_integration.py            ← End-to-end pipeline tests
```

---

## File-by-File Reference

---

### `config.py` — Centralised Hyperparameter Hub

#### Intuitive Explanation
Every tunable number in the project lives here — no magic constants scattered across files.  This makes ablation studies trivial: change `pred_horizon` from 16 to 8, re-run training, and compare.

The config is implemented as nested Python dataclasses.  The IDE provides auto-complete and type checking; `__post_init__` validates logical constraints at instantiation time.

#### Code Structure
```
TrainConfig
├── env:            EnvConfig            # obs_dim=5, action_dim=2
├── data:           DataConfig           # horizons, dataset path, obs_type, norm_range
│   └── obs_type: str                   #   "state" (default) or "image"
├── model:          ModelConfig          # down_dims, cond_dim, kernel_size
├── vision:         VisionEncoderConfig  # obs_cond_dim=256, pretrained, freeze_backbone
├── diffusion:      DiffusionConfig      # num_diffusion_steps, beta_start/end, DDIM params
├── flow_matching:  FlowMatchingConfig   # num_inference_steps
├── method:         str                  # "ddpm" or "flow_matching"
├── learning_rate:  float                # 1e-4
├── batch_size:     int                  # 256 (state) / 64 (image)
├── ema_decay:      float                # 0.995
└── device:         str                  # auto-detected: "cuda"/"mps"/"cpu"
```

**`obs_type` selects the full observation pipeline:**
- `"state"` → `PushTStateDataset` → 5-dim state vector → UNet obs MLP
- `"image"` → `PushTImageDataset` → 96×96 RGB frames → `ResNetEncoder` → UNet image path

**Constraint enforced by `__post_init__`:**
```python
if data.obs_type == "image":
    assert vision.obs_cond_dim == model.cond_dim  # both must be 256
```

#### Key Insight — The Three Horizons
```
│  obs_{t-1}  obs_t  │  act_t  act_{t+1}  ...  act_{t+15}  │
│ ◄── T_obs=2 ──►    │ ◄────── T_pred=16 ─────────────────► │
│                     │ ◄── T_action=8 ──►                   │
│                     │   (actually executed)                 │
```
- **T_obs = 2** — the model sees two consecutive frames; implicit velocity information.
- **T_pred = 16** — the model predicts 16 future steps; gives temporally coherent motion.
- **T_action = 8** — only the first 8 are executed before re-planning; balance commitment/reactivity.

---

### `diffusion_policy/data/normalizer.py` — MinMaxNormalizer

#### Intuitive Explanation
Diffusion models learn to map Gaussian noise back to data.  If the data lives in `[0, 512]` but the noise is `N(0, 1)`, there's a catastrophic scale mismatch — the model would need to amplify the noise by ×512 just to reach the right range.  Normalization to `[-1, 1]` removes this mismatch.

Think of it like unit conversion: before comparing apples (positions in pixels) to oranges (angles in radians), you convert everything to the same unit.

#### Code Walkthrough
```python
normalizer = MinMaxNormalizer()
normalizer.fit(all_actions)           # compute min/max per dimension over full dataset
norm   = normalizer.normalize(x)      # maps x ∈ [x_min, x_max] → [-1, 1]
unnorm = normalizer.unnormalize(y)    # inverse map: [-1, 1] → [x_min, x_max]
```

The normalizer accepts both `np.ndarray` and `torch.Tensor` and returns the same type, preserving device placement.

Statistics are serialized via `state_dict()` / `load_state_dict()` and saved alongside model checkpoints so that inference uses *exactly* the same scale as training.

#### Mathematical Foundation
Per-dimension min-max normalization:

```
x_norm = 2 · (x - x_min) / (x_max - x_min) - 1

Inverse: x = (x_norm + 1) / 2 · (x_max - x_min) + x_min
```

For a zero-variance dimension (constant feature), the range is set to 1.0 to avoid division by zero:

```
range = max(x_max - x_min, ε)    where ε = 1e-8
```

---

### `diffusion_policy/data/dataset.py` — PushTStateDataset

#### Intuitive Explanation
The raw dataset has ~200 episodes of expert demonstrations, each ~300 steps long.  That's only 200 data points if you use each episode as one sample — nowhere near enough for a diffusion model.

The **sliding window** trick extracts one `(observation history, future action chunk)` pair from *every timestep* of *every episode*, giving ~200 × 300 = **~60,000 training samples** from the same data.

Think of it like reading a book: instead of reading one page per chapter, you read every possible consecutive pair of pages.

#### Code Walkthrough
```python
dataset = PushTStateDataset(
    dataset_path="data/pusht_cchi_v7_replay.zarr",
    obs_horizon=2,      # T_obs
    pred_horizon=16,    # T_pred
)

sample = dataset[0]
# sample["obs"]    → shape (2, 5):  last 2 normalised observations
# sample["action"] → shape (16, 2): next 16 normalised actions
```

**Zarr structure:**
```
root.zarr/
├── data/state     (N_total, 5)    — all observations concatenated
├── data/action    (N_total, 2)    — all actions concatenated
└── meta/episode_ends (num_eps,)   — index where each episode ends
```

**Boundary handling:**
- *Episode start:* pad the observation window by repeating the first frame.
- *Episode end:* pad the action window by repeating the last action.
- This ensures a valid sample exists for every timestep without crossing episode boundaries.

#### Mathematical Foundation
For timestep `t` in episode `[ep_start, ep_end)`:

```
obs_window:    stack(obs[max(ep_start, t-T_obs+1) : t+1])   padded to shape (T_obs, D_obs)
action_window: stack(act[t : min(ep_end, t+T_pred)])          padded to shape (T_pred, D_act)
```

---

### `diffusion_policy/env/pusht_env.py` — PushTEnv

#### Intuitive Explanation
PushT is a 2D physics simulation: a circular "finger" agent pushes a T-shaped block to match a target pose.  The environment is built on `pymunk` (2D rigid body physics) and rendered with `pygame`.

The wrapper provides a clean, consistent interface identical to standard gym environments, plus a helper for the **observation deque** needed in receding-horizon control.

#### Code Walkthrough
```python
env = PushTEnv(render_size=96, max_episode_steps=300)

obs = env.reset(seed=42)             # (5,) — raw unnormalized state
obs, reward, done, info = env.step(action)   # action is (2,) unnormalized velocity

# For receding-horizon control:
obs_deque = env.make_obs_deque(obs_horizon=2)   # deque(maxlen=2) pre-filled
obs_arr   = PushTEnv.deque_to_array(obs_deque)  # (2, 5) ready for normalizer
```

**State space (5D):**
```
[agent_x, agent_y, block_x, block_y, block_angle]
```

**Action space (2D):**
```
[velocity_x, velocity_y]   — velocity command in pixel/step units
```

**Reward:** geometric overlap (coverage score) ∈ [0, 1] between current and target T-block pose.  An episode is considered successful at coverage ≥ 0.9.

---

### `diffusion_policy/data/image_dataset.py` — PushTImageDataset

#### Intuitive Explanation
The image-observation variant of the dataset loads the raw RGB video frames stored in the PushT zarr file (`data/img`) alongside actions.  Instead of returning a 5-dim state vector per timestep, it returns a `(T_obs, 3, H, W)` tensor of normalised RGB frames.

Using images as observations moves the policy much closer to what a real robot would see from a camera — no privileged state information, just pixels.  This is the **visuomotor** setting of Chi et al. (RSS 2023), where success rates of ~95% coverage were reported with a pretrained ResNet encoder.

#### Code Walkthrough
```python
dataset = PushTImageDataset(
    dataset_path="data/pusht_cchi_v7_replay.zarr",
    obs_horizon=2,      # T_obs  — number of frames per sample
    pred_horizon=16,    # T_pred — future action chunk length
)

sample = dataset[0]
# sample["obs"]    → (2, 3, 96, 96)  float32, range [0, 1]
# sample["action"] → (16, 2)         float32, range [-1, 1]
```

**Zarr structure used:**
```
root.zarr/
├── data/img      (N_total, 96, 96, 3)  uint8/float32 [0, 255]
├── data/action   (N_total, 2)
└── meta/episode_ends (num_eps,)
```

**Preprocessing pipeline:**
1. Load `data/img` as float32, divide by 255 → `[0, 1]`
2. Transpose `(N, H, W, C) → (N, C, H, W)` (PyTorch channel-first convention)
3. Same sliding-window + boundary padding logic as `PushTStateDataset`
4. Actions are min-max normalised to `[-1, 1]` exactly as in the state variant

The `get_action_normalizer()` method returns the fitted `MinMaxNormalizer` so `train.py` and `evaluate.py` can unnormalize predictions at inference time.

#### Why No Image Normalizer?
Image pixels in `[0, 1]` are already in a safe range for neural networks.  When `pretrained=True`, the `ResNetEncoder` applies ImageNet mean/std normalisation internally.  Storing a separate `MinMaxNormalizer` for images would add complexity with no benefit.

---

### `diffusion_policy/model/vision_encoder.py` — ResNetEncoder

#### Intuitive Explanation
The `ResNetEncoder` is the "eyes" of the visuomotor policy.  It takes T_obs consecutive RGB frames, extracts a rich visual feature from each, concatenates them, and projects everything down to a single conditioning vector of size `obs_cond_dim=256`.

This conditioning vector plays exactly the same role as the normalised state vector in the state-based policy: it tells the UNet "what the robot currently sees" so the UNet can generate appropriate actions.

Think of it like this: the ResNet-18 backbone is a pretrained image-understanding engine (trained on ImageNet to recognise 1000 categories).  We strip its final classification head and repurpose its 512-dimensional feature as a rich spatial representation of each frame.  Two frames concatenated → 1024 dimensions.  A small projection MLP then compresses this to 256, matching the UNet's conditioning dimension.

#### Code Walkthrough — Architecture
```python
encoder = ResNetEncoder(
    obs_horizon=2,        # T_obs — how many frames to process
    obs_cond_dim=256,     # output conditioning vector size
    pretrained=False,     # True → ImageNet weights (better for real images)
    freeze_backbone=False # True → only train the projection MLP
)

obs_imgs = torch.randn(B, 2, 3, 96, 96)   # (batch, T_obs, C, H, W)
cond     = encoder(obs_imgs)               # (B, 256) — conditioning vector
```

**Internal data flow:**
```
obs_imgs: (B, T_obs, 3, 96, 96)
    │
    ▼  reshape to process all frames at once
(B·T_obs, 3, 96, 96)
    │
    ▼  ResNet-18 backbone (conv1→bn1→relu→maxpool→layer1→layer2→layer3→layer4)
(B·T_obs, 512, 3, 3)     ← spatial feature map for each frame
    │
    ▼  AdaptiveAvgPool2d(1)  — global average pooling
(B·T_obs, 512, 1, 1)
    │
    ▼  flatten
(B·T_obs, 512)
    │
    ▼  reshape to concatenate time dimension
(B, T_obs × 512) = (B, 1024)    ← all frames' features concatenated
    │
    ▼  projection MLP: Linear(1024→256) → ReLU → Linear(256→256)
(B, 256)    ← final conditioning vector
```

**Key design choices:**
- **No final avgpool/fc from torchvision** — ResNet-18's default head is stripped; the `AdaptiveAvgPool2d(1)` is added explicitly.  This allows any input resolution (96×96, 128×128, etc.) without resizing.
- **Concatenation over time** — frames are concatenated in feature space rather than averaged, preserving temporal information (frame 0 ≠ frame 1 in the conditioning vector).
- **Shared backbone across frames** — the same ResNet weights process each frame; implicit weight sharing encourages consistent feature extraction regardless of temporal position.

#### Mathematical Foundation

**ResNet-18 residual block:**
```
y = F(x, {W_i}) + x

where F(x) = BN(W₂ · ReLU(BN(W₁ · x)))
```

The `+x` skip connection ensures gradients flow directly back to early layers, enabling training of very deep networks.

**Global average pooling:**
```
For feature map x ∈ ℝ^{C × H × W}:
    GAP(x)_c = (1/H·W) · Σ_{h,w} x[c, h, w]
```

This collapses spatial information to a single vector while being invariant to input resolution.

**Projection MLP:**
```
cond = W₂ · ReLU(W₁ · concat(GAP(frame₁), ..., GAP(frame_{T_obs})))
```

**ImageNet normalisation (when `pretrained=True`):**
```
x_norm = (x − μ_ImageNet) / σ_ImageNet
μ_ImageNet = [0.485, 0.456, 0.406]
σ_ImageNet = [0.229, 0.224, 0.225]
```

---

### `diffusion_policy/model/unet1d.py` — ConditionalUnet1D

#### Intuitive Explanation
The U-Net is the "brain" of the diffusion policy.  Its job: given a noisy action sequence, tell the model *what noise was added* so it can be removed (DDPM), or tell it the *velocity of the transport path* (Flow Matching).

Think of the 16-step action trajectory as a short 1D audio clip with 2 channels (vx, vy).  The U-Net processes it like a spectrogram: first encoding it into a compact latent representation (encoder), then reconstructing the full-resolution prediction (decoder), while injecting the timestep and observation conditioning at every layer.

**FiLM conditioning** is the clever trick that makes this work: instead of simply appending the conditioning information to the input (which the first layer would then have to "decode"), FiLM injects it at **every residual block** by multiplicatively modulating the features.

#### Code Walkthrough — Dual Observation Paths

The UNet supports two observation modes, selected by `obs_dim`:
- **State mode** (`obs_dim=5`): raw state vector → internal MLP → conditioning
- **Image mode** (`obs_dim=0`): pre-encoded vector from `ResNetEncoder` passed directly

```python
# State mode (obs_dim > 0):
model = ConditionalUnet1D(action_dim=2, obs_dim=5, obs_horizon=2, cond_dim=256)
obs_state = torch.randn(B, 2, 5)     # (B, T_obs, obs_dim)
out = model(noisy_actions, timesteps, obs_state)

# Image mode (obs_dim = 0):
model = ConditionalUnet1D(action_dim=2, obs_dim=0, obs_horizon=2, cond_dim=256)
obs_cond  = encoder(obs_imgs)         # (B, 256) — from ResNetEncoder
out = model(noisy_actions, timesteps, obs_cond)
```

When `obs_dim=0`, the `obs_encoder` MLP is bypassed — the external conditioning vector is used directly.  This avoids a pointless `Linear(256 → 256)` bottleneck and cleanly separates concerns between the visual backbone and the temporal denoiser.

#### Code Walkthrough — Data Flow

```
Input: noisy_actions (B, 16, 2)
         │
         ▼ permute to (B, 2, 16) for Conv1d

Condition encoding (STATE mode, obs_dim=5):
  obs (B, 2, 5) → flatten → (B, 10) → MLP → obs_emb (B, 256)
  timestep (B,) → SinusoidalPosEmb → MLP → t_emb (B, 256)
  cond = cat([t_emb, obs_emb]) → (B, 512)

Condition encoding (IMAGE mode, obs_dim=0):
  obs_cond already (B, 256) from ResNetEncoder
  timestep (B,) → SinusoidalPosEmb → MLP → t_emb (B, 256)
  cond = cat([t_emb, obs_cond]) → (B, 512)

Encoder (skip connections saved before downsampling):
  Level 0: ResBlock(2→256) + ResBlock(256→256)     T=16 → skip0
           Downsample2x                             T=8
  Level 1: ResBlock(256→512) + ResBlock(512→512)   T=8  → skip1
           Downsample2x                             T=4
  Level 2: ResBlock(512→1024) + ResBlock(1024→1024) T=4  → skip2

Bottleneck:
  ResBlock(1024→1024)  ResBlock(1024→1024)          T=4

Decoder (cat skip before each ResBlock):
  cat(x, skip2) → (B,2048,4)
  ResBlock(2048→512) + ResBlock(512→512)             T=4 → Upsample→T=8
  cat(x, skip1) → (B,1024,8)
  ResBlock(1024→256) + ResBlock(256→256)             T=8 → Upsample→T=16

Output:
  Conv1dBlock(256→256) + Conv1d(256→2)              T=16
  permute → (B, 16, 2)
```

**FiLM block detail:**

```python
out = conv_block1(x)          # (B, C_out, T)
gamma, beta = film_proj(cond).chunk(2, dim=-1)   # each (B, C_out)
out = gamma.unsqueeze(-1) * out + beta.unsqueeze(-1)   # scale & shift
out = conv_block2(out) + residual_proj(x)
```

#### Mathematical Foundation

**Sinusoidal positional embedding** (for timestep `k`):

```
emb[2i]   = sin(k / 10000^(2i/d))
emb[2i+1] = cos(k / 10000^(2i/d))

for i = 0, 1, ..., d/2 - 1
```

**FiLM (Feature-wise Linear Modulation):**

```
h_out = γ(cond) ⊙ GroupNorm(h)  +  β(cond)

where γ, β : ℝ^{cond_dim} → ℝ^{C_out}  are learned linear maps
```

**GroupNorm (within a residual block):**

```
For a feature map x ∈ ℝ^{B × C × T}, divide channels into G groups.
For each sample b and group g:
  μ_{b,g} = mean(x[b, g·(C/G):(g+1)·(C/G), :])
  σ_{b,g} = std(x[b, g·(C/G):(g+1)·(C/G), :])
  x_norm  = (x - μ) / σ     (within the group)
```

---

### `diffusion_policy/model/ema.py` — EMA

#### Intuitive Explanation
Training is noisy: each gradient step pushes the weights toward minimising the loss on one random mini-batch.  The weights oscillate around the true optimum.

EMA is like a low-pass filter on the weight sequence.  It keeps a "shadow" copy of the weights that is a running average of the last ~200 training iterations:

```
θ_ema  ←  0.995 · θ_ema  +  0.005 · θ_train
```

This is especially critical for diffusion models because inference is **iterative** — the model runs 10–100 forward passes in sequence.  Any high-frequency noise in the weights compounds across those steps.

#### Code Walkthrough

```python
ema = EMA(model, decay=0.995)

# After every optimizer step:
ema.update(model)       # θ_ema ← 0.995·θ_ema + 0.005·θ_train

# For evaluation:
ema.apply(model)        # swap θ_train with θ_ema
result = evaluate(model)
ema.restore(model)      # swap back → training can continue
```

#### Mathematical Foundation

**EMA update rule:**

```
θ_ema^{(t+1)} = decay · θ_ema^{(t)}  +  (1 - decay) · θ_train^{(t)}
```

**Effective window size:**

```
Window ≈ 1 / (1 - decay)

With decay = 0.995:  window ≈ 200 steps
```

The weight at step `s` in the past contributes approximately `(1 - decay) · decay^s` to the current EMA.

---

### `diffusion_policy/model/ddpm.py` — DDPMScheduler

#### Intuitive Explanation
DDPM defines a two-phase process:

1. **Forward (noising)** — used during training.  Take a clean action sequence and gradually corrupt it with Gaussian noise over `K=100` steps until it looks like pure noise.

2. **Reverse (denoising)** — used at inference.  Start from pure Gaussian noise and iteratively remove predicted noise for K steps until a clean action emerges.

Think of the forward process like slowly dissolving ink in water.  The reverse process is the model's job: given a partially-dissolved image, predict what the original ink looked like.

#### Code Walkthrough — Training

```python
scheduler = DDPMScheduler(K=100)

# Per training batch:
timesteps    = scheduler.sample_timesteps(B)          # random k ~ Uniform{0,...,99}
noise        = torch.randn_like(clean_actions)         # ε ~ N(0,I)
noisy_action = scheduler.add_noise(clean_actions, noise, timesteps)
noise_pred   = model(noisy_action, timesteps, obs)     # ε_θ
loss         = F.mse_loss(noise_pred, noise)           # train to predict noise
```

#### Code Walkthrough — Inference

```python
# Start from pure noise, iteratively denoise
a_k = torch.randn(B, T_pred, action_dim)
for k in reversed(range(100)):
    eps_pred = model(a_k, k, obs)
    a_k = scheduler.step(eps_pred, k, a_k)
# a_k is now a_0: a clean action sequence
```

#### Mathematical Foundation

**Linear noise schedule:**

```
β_k = β_start + (β_end - β_start) · k/(K-1)

β_start = 1e-4,  β_end = 0.02,  K = 100
```

**Derived quantities:**

```
α_k  = 1 - β_k                       (signal retention at step k)
ᾱ_k  = ∏_{i=1}^{k} α_i              (cumulative signal retention)
```

**Forward process (reparameterization trick):**

```
q(a_k | a_0) = N(a_k;  √ᾱ_k · a_0,  (1 - ᾱ_k) · I)

⟹  a_k = √ᾱ_k · a_0  +  √(1-ᾱ_k) · ε,    ε ~ N(0, I)
```

This lets us jump directly to any noise level in one step — O(1) instead of iterating k times.

**Reverse process:**

```
p_θ(a_{k-1} | a_k) = N(a_{k-1};  μ_θ(a_k, k),  β_k · I)

μ_θ = (1/√α_k) · [a_k  −  (β_k / √(1-ᾱ_k)) · ε_θ(a_k, k, obs)]
```

**Training objective (simplified ELBO):**

```
L = E_{k ~ U[0,K-1], a_0 ~ data, ε ~ N(0,I)} [
    ||ε  −  ε_θ(√ᾱ_k · a_0 + √(1-ᾱ_k) · ε,  k,  obs)||²
]
```

---

### `diffusion_policy/model/ddim.py` — DDIMScheduler

#### Intuitive Explanation
DDPM takes 100 steps at inference — slow when running a robot in real time.  DDIM asks: *can we skip most of those steps?*

DDPM is stochastic: each step adds fresh Gaussian noise.  This means every step must be taken in order.  DDIM reformulates the reverse process as **deterministic** (when η=0), which means it only depends on the *cumulative* noise level ᾱ_k, not the path taken to get there.  We can jump from k=99 to k=89 to k=79... in 10 steps instead of 100.

**Result: 10× speedup with negligible quality loss.**

The η parameter controls stochasticity:
- η=0 → fully deterministic, reproducible evaluation
- η=1 → equivalent to DDPM
- 0<η<1 → partial stochasticity

#### Code Walkthrough

```python
ddim = DDIMScheduler(ddpm_scheduler, ddim_steps=10, eta=0.0)
# timestep_seq = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

actions = ddim.sample(model, obs, pred_horizon=16, action_dim=2)
# 10 network evaluations instead of 100
```

#### Mathematical Foundation

**DDIM update rule (per step, from t to t_prev < t):**

```
â_0 = (a_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t          ← predict clean action

σ_t = η · √((1−ᾱ_{t-1})/(1−ᾱ_t)) · √(1 − ᾱ_t/ᾱ_{t-1})

a_{t-1} = √ᾱ_{t-1} · â_0
           + √(1 − ᾱ_{t-1} − σ_t²) · ε_θ
           + σ_t · z,     z ~ N(0,I)
```

**Why skipping works:** the update from t to t_prev only uses ᾱ_t and ᾱ_{t-1}, not any intermediate values.  So we can choose *any* subsequence of {0,...,K-1} as long as it's decreasing.

---

### `diffusion_policy/model/flow_matching.py` — FlowMatchingScheduler

#### Intuitive Explanation
Flow Matching is an elegant alternative to DDPM that replaces the complex noise schedule with straight-line paths.

Imagine every training data point as a location in action-space, and every noise sample as another location.  Flow Matching learns a **vector field** (arrows) pointing from the noise distribution to the data distribution, along straight paths:

```
noise (t=1)  ──────────────────────────►  data (t=0)
                  straight-line path
```

At training time: sample t ~ Uniform[0,1], interpolate between data and noise, train the model to predict the velocity along the straight path.

At inference time: start at t=1 (noise) and follow the predicted arrows for a few Euler steps until t=0 (data).  Because the paths are straight, you need far fewer steps than DDPM.

**Connection to Yashvardhan's thesis:** Flow Matching is the σ→0 limit of Schrödinger Bridges.  Adding a diffusion term `σ dW` recovers the SB formulation.  This project's FM implementation is thus a direct code-level bridge to SB research.

#### Code Walkthrough — Training

```python
fm = FlowMatchingScheduler(num_inference_steps=10)

# Per training batch:
loss = fm.get_loss(model, clean_actions, obs)
# Internally: sample t~U[0,1], interpolate, compute MSE(v_θ, ε - a_0)
```

#### Code Walkthrough — Inference

```python
actions = fm.sample(model, obs, pred_horizon=16, action_dim=2)
# Euler ODE: x_{t-Δt} = x_t - Δt · v_θ(x_t, t, obs)
# 10 steps from t=1 (noise) to t=0 (data)
```

#### Mathematical Foundation

**Interpolation (straight-line transport):**

```
x_t = (1 − t) · a_0  +  t · ε,    t ∈ [0, 1],  ε ~ N(0, I)
```

**Conditional velocity target:**

```
u_t(x_t | a_0, ε) = dx_t/dt = ε − a_0
```

**Training objective:**

```
L_FM = E_{t ~ U[0,1], a_0 ~ data, ε ~ N(0,I)} [
    ||v_θ(x_t, t · τ, obs)  −  (ε − a_0)||²
]
```

where τ=100 scales t into the U-Net's sinusoidal embedding range.

**Euler ODE integration (inference, step i from t_i to t_{i-1} = t_i − Δt):**

```
x_{t-Δt} = x_t  −  Δt · v_θ(x_t, t, obs)

where Δt = 1 / num_steps  and  t goes from 1.0 down to 0.0
```

**Connection to Schrödinger Bridges:**

FM solves the deterministic ODE:  `dx/dt = v_θ(x_t, t)`

The Schrödinger Bridge (SB) with diffusion coefficient σ solves the SDE:
`dx = v_θ(x_t, t) dt  +  σ dW`

As σ → 0, the SB solution converges to the FM solution.  Increasing σ adds entropy to the transport, making the policy more exploratory.

---

### `train.py` — Training Orchestration

#### Intuitive Explanation
The training script is the conductor: it builds all the components (dataset, model, scheduler, EMA, optimizer) and runs the training loop, logging everything, saving checkpoints, and periodically evaluating the policy.

**Key design choices:**
- `set_to_none=True` in `zero_grad()` → faster than zeroing (avoids memory writes)
- Gradient clipping to 1.0 → prevents catastrophic divergence
- Cosine-warmup LR schedule → stable start + fine-grained end
- EMA updated after every step → smooth inference weights throughout training

#### Code Walkthrough — Dataset and Model Selection

`train.py` selects the full pipeline based on `cfg.data.obs_type`:

```python
if cfg.data.obs_type == "image":
    dataset           = PushTImageDataset(dataset_path, obs_horizon, pred_horizon)
    obs_normalizer    = None                          # images need no state normalizer
    action_normalizer = dataset.get_action_normalizer()
    vision_encoder    = ResNetEncoder(obs_horizon, obs_cond_dim, pretrained, freeze_backbone)
    unet_obs_dim      = 0                             # UNet bypasses internal MLP
else:
    dataset           = PushTStateDataset(dataset_path, obs_horizon, pred_horizon)
    obs_normalizer, action_normalizer = dataset.get_normalizers()
    vision_encoder    = None
    unet_obs_dim      = cfg.env.obs_dim               # UNet uses internal MLP
```

When `obs_type == "image"`, the vision encoder is updated by EMA separately from the UNet, and both are saved in the checkpoint.

#### Code Walkthrough — Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        obs_raw, action = batch["obs"], batch["action"]

        # ── Encode observations ──────────────────────────────────────────
        if vision_encoder is not None:
            obs = vision_encoder(obs_raw)     # (B, T_obs, C, H, W) → (B, cond_dim)
        else:
            obs = obs_raw                     # (B, T_obs, obs_dim) — state

        # ── DDPM path ──
        k          = scheduler.sample_timesteps(B)        # random k
        noise      = torch.randn_like(action)              # ε ~ N(0,I)
        noisy      = scheduler.add_noise(action, noise, k) # a_k
        noise_pred = model(noisy, k, obs)                  # ε_θ
        loss       = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        ema.update(model)
        if vision_encoder is not None:
            ema_encoder.update(vision_encoder)
```

#### Mathematical Foundation — LR Schedule

**Linear warmup** (steps 0 → `warmup_steps`):

```
lr(t) = lr_max · t / warmup_steps
```

**Cosine decay** (steps `warmup_steps` → `total_steps`):

```
progress = (t − warmup) / (total − warmup)
lr(t)    = lr_max · 0.5 · (1 + cos(π · progress))
```

This schedule is motivated by empirical evidence that warmup prevents large gradient updates early (when weights are random), and cosine decay allows fine-tuning with a very small LR at the end without a sharp cutoff.

---

### `evaluate.py` — Receding-Horizon Control Evaluation

#### Intuitive Explanation
The evaluation loop runs the trained policy in the PushT environment using **receding-horizon control** (also called MPC-style action chunking):

1. **Plan:** run diffusion sampling to predict T_pred=16 future actions.
2. **Execute:** only perform the first T_action=8 of those actions.
3. **Replan:** collect fresh observations and repeat.

Why execute only 8 of 16? Pure commitment (execute all 16) leads to rigid behaviour that can't react to unexpected state changes.  Pure reactivity (execute 1 step) is noisy and incoherent.  The midpoint (execute 8) gives smooth, committed motion that can still adapt.

#### Code Walkthrough — Inference Loop

```python
obs_deque = deque(maxlen=T_obs)  # rolling observation buffer

while not done:
    obs_arr   = stack(obs_deque)                    # (T_obs, obs_dim)
    obs_norm  = obs_normalizer.normalize(obs_arr)   # scale to [-1, 1]
    obs_t     = tensor(obs_norm).unsqueeze(0)       # (1, T_obs, obs_dim)

    # ── Run diffusion (DDPM or DDIM) ──────────────────────────────────
    act_norm = ddim.sample(model, obs_t, T_pred, action_dim)    # (1, 16, 2)
    act_real = action_normalizer.unnormalize(act_norm.numpy())  # back to pixels

    # ── Execute first T_action steps ──────────────────────────────────
    for i in range(T_action):
        obs_new, reward, done, _ = env.step(act_real[i])
        obs_deque.append(obs_new)
```

#### Success Metric

An episode is **successful** if the T-block reaches ≥ 90% coverage of the target pose at any point in the episode:

```
success = max_coverage_in_episode ≥ 0.9
```

The reported **success rate** is the fraction of successful episodes over 50 evaluation rollouts.

**Image mode:** when `cfg.data.obs_type == "image"`, `load_policy()` recreates the `ResNetEncoder` from the checkpoint, loads its EMA weights, and uses it to encode each frame in the evaluation loop:

```python
obs_deque: deque  # holds (T_obs, 3, 96, 96) frames
frames_tensor = torch.stack(list(obs_deque)).unsqueeze(0)  # (1, T_obs, 3, H, W)
obs_cond = vision_encoder(frames_tensor)                    # (1, 256)
# obs_cond passed directly to model (obs_dim=0 path)
```

**Reference numbers (from Chi et al., RSS 2023):**

| Method | Observations | Success Rate | Inference Steps |
|--------|-------------|-------------|-----------------|
| Implicit BC (IBC) | State | ~0.66 | — |
| Explicit BC (LSTM-GMM) | State | ~0.61 | — |
| **Diffusion Policy (DDPM)** | **State** | **~0.92** | 100 |
| **Diffusion Policy (DDIM)** | **State** | **~0.90** | 10 |
| **Diffusion Policy (DDPM)** | **Image (ResNet-18)** | **~0.96** | 100 |
| **Diffusion Policy (DDIM)** | **Image (ResNet-18)** | **~0.95** | 10 |

---

### `visualize.py` — Visualization Utilities

#### Intuitive Explanation
Good visualizations are essential for debugging, understanding, and presenting ML results.  This module provides five categories of plots, all saved to files (no interactive display needed).

| Function | What it shows | When to use |
|----------|---------------|-------------|
| `plot_dataset_summary()` | Action scatter + episode lengths | After loading data, to verify dataset |
| `plot_training_curves()` | Loss curve + eval success rate | After training, to diagnose convergence |
| `visualize_diffusion_process()` | Forward noising of an action | To intuitively show what DDPM "does" |
| `save_rollout_gif()` | Animated policy rollout | For presentation / report |
| `plot_eval_comparison()` | Bar chart: DDPM vs DDIM vs FM | For ablation study results |

---

## Ablation Study Guide

### What to Vary

| Variable | Values to try | Expected effect |
|----------|--------------|-----------------|
| `obs_type` | `"state"` vs `"image"` | Image → higher ceiling, slower training |
| `pred_horizon` (T_pred) | 8, 16, 32 | Longer → smoother but harder to train |
| `action_horizon` (T_action) | 1, 4, 8, 16 | Longer → more committed, less reactive |
| `obs_horizon` (T_obs) | 1, 2, 4 | More → implicit velocity signal |
| `ddim_steps` | 5, 10, 20 | More → slower but higher quality |
| EMA decay | 0.99, 0.995, 0.999 | Higher → smoother but slower to adapt |
| Down dims | (256,512) vs (256,512,1024) | Larger → more capacity, slower |
| Method | ddpm vs flow_matching | FM tends to need fewer inference steps |
| `freeze_backbone` | True vs False | Frozen → faster training, less overfit risk |

### How to Read Results

- **Success rate** is the primary metric; target ≥ 0.90.
- **Mean score** (mean coverage) gives a softer gradient — useful when success rate is 0 (early in training).
- **Inference time per step** reveals the DDPM vs DDIM speed trade-off.
- Plot `success_rate` vs `epoch` to detect whether 300 epochs is enough.

---

## Key Papers

1. Chi et al., **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"**, RSS 2023.
   — The primary reference; introduces Diffusion Policy for robot control.

2. Ho et al., **"Denoising Diffusion Probabilistic Models"**, NeurIPS 2020.
   — Introduces DDPM; our training objective and noise schedule.

3. Song et al., **"Denoising Diffusion Implicit Models"**, ICLR 2021.
   — Introduces DDIM; our fast inference method.

4. Lipman et al., **"Flow Matching for Generative Modeling"**, ICLR 2023.
   — Flow Matching; our Phase 5 extension.

5. Perez et al., **"FiLM: Visual Reasoning with a General Conditioning Layer"**, AAAI 2018.
   — FiLM conditioning; our observation conditioning mechanism.

6. Florence et al., **"Implicit Behavioral Cloning"**, CoRL 2021.
   — Original PushT environment; provides baseline comparisons.

7. He et al., **"Deep Residual Learning for Image Recognition"**, CVPR 2016.
   — ResNet architecture; our visual backbone (`ResNet-18`).
