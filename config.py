"""
config.py — Centralized Hyperparameter Configuration
=====================================================

Every tunable number in this project lives here. Nothing is hardcoded in the
model, training loop, or data pipeline. This makes ablation experiments trivial:
change one value here and re-run, rather than hunting through scattered files.

Design decision: we use a plain Python dataclass rather than YAML/JSON so that
the IDE gives us autocomplete and type checking, and so that derived quantities
(like alpha_bar) can be computed as properties.

Usage:
    from config import TrainConfig
    cfg = TrainConfig()
    print(cfg.pred_horizon)   # 16
    print(cfg.device)         # "cpu" or "cuda" or "mps" depending on hardware
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import torch


def _auto_device() -> str:
    """
    Detect the best available compute device.

    Priority order:
        1. CUDA  — NVIDIA GPUs (Colab, university cluster)
        2. MPS   — Apple Silicon GPU (MacBook M-series)
        3. CPU   — fallback for any machine

    This function is called once at config instantiation time, so every module
    that reads `cfg.device` gets a consistent device string without any
    hardcoding.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ==============================================================================
#  Environment & Task Configuration
# ==============================================================================

@dataclass
class EnvConfig:
    """
    Parameters that describe the PushT task itself.
    These are fixed by the environment and should NOT be changed.
    """

    # --- Observation space ---
    # PushT state-based observation: (agent_x, agent_y, block_x, block_y, block_angle)
    obs_dim: int = 5

    # --- Action space ---
    # PushT actions: (velocity_x, velocity_y) of the circular agent
    action_dim: int = 2

    # --- Episode properties ---
    max_episode_steps: int = 300  # PushT default episode length


# ==============================================================================
#  Data Pipeline Configuration
# ==============================================================================

@dataclass
class DataConfig:
    """
    Parameters controlling how raw demonstration episodes are processed into
    training samples via the sliding-window approach.

    Key concept — the three horizons:
    ┌─────────────────────────────────────────────────────────────┐
    │  ... obs_t-1  obs_t │ act_t  act_t+1 ... act_t+15          │
    │  ◄─ T_obs=2 ──►    │ ◄──────── T_pred=16 ────────►        │
    │                     │ ◄─ T_action=8 ──►                    │
    └─────────────────────────────────────────────────────────────┘

    - T_obs (obs_horizon): How many past observations the model sees.
      More history gives the model velocity/acceleration information implicitly.
      Default 2: current + previous frame → the model can infer velocity.

    - T_pred (pred_horizon): How many future actions the model predicts.
      Longer horizons give more temporally coherent trajectories but make the
      denoising problem harder (higher-dimensional output space).
      Default 16: empirically best in the Diffusion Policy paper.

    - T_action (action_horizon): How many of the predicted actions are actually
      EXECUTED before re-planning. Always T_action ≤ T_pred.
      Default 8: execute first half, then re-plan with fresh observations.
      This receding-horizon scheme balances commitment (smooth motion) with
      reactivity (responding to new observations).
    """

    obs_horizon: int = 2       # T_obs: number of observation frames to condition on
    pred_horizon: int = 16     # T_pred: number of future actions to predict
    action_horizon: int = 8    # T_action: number of predicted actions to execute

    # --- Dataset source ---
    dataset_path: str = "data/pusht_cchi_v7_replay.zarr"  # path to Zarr dataset

    # --- Normalization ---
    # Diffusion models work best when inputs/outputs are in a bounded range.
    # We normalize all observations and actions to [-1, 1] using dataset-wide
    # min/max statistics. This is crucial: without it, the Gaussian noise
    # assumption of DDPM doesn't match the data scale.
    norm_range: Tuple[float, float] = (-1.0, 1.0)


# ==============================================================================
#  Model Architecture Configuration
# ==============================================================================

@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for the Conditional 1D Temporal U-Net.

    The U-Net processes action sequences along the TEMPORAL dimension using 1D
    convolutions. Think of the action trajectory (T_pred, action_dim) as a
    1D "signal" with `action_dim` channels and `T_pred` spatial positions.
    The Conv1d operations slide along the T_pred axis.

    Convention: we reshape (B, T_pred, action_dim) → (B, action_dim, T_pred)
    before feeding into the U-Net, because PyTorch Conv1d expects (B, C, L).
    """

    # --- Channel progression through the U-Net ---
    # Each entry defines the number of channels at that resolution level.
    # Level 0: 256 channels at full temporal resolution (T_pred=16)
    # Level 1: 512 channels at half resolution (8)
    # Level 2: 1024 channels at quarter resolution (4)
    #
    # Why these sizes? The action_dim (2) is tiny, so we need to project up to
    # a high-dimensional latent space where the model has enough capacity to
    # learn the conditional denoising function. The paper uses [256, 512, 1024].
    down_dims: Tuple[int, ...] = (256, 512, 1024)

    # --- Diffusion timestep embedding ---
    # The integer diffusion step k (or continuous time t for Flow Matching) is
    # embedded via sinusoidal positional encoding into this dimensionality,
    # then passed through a 2-layer MLP.
    #
    # Why sinusoidal? Same idea as Transformer positional encoding — it gives
    # the model a smooth, information-rich representation of "how noisy is this
    # input?" that generalizes across timesteps it hasn't seen exact matches of.
    diffusion_step_embed_dim: int = 256

    # --- Observation conditioning ---
    # The observation encoder MLP maps the flattened observation history
    # (T_obs * obs_dim) to this dimensionality. This vector is then used
    # for FiLM conditioning in every residual block.
    #
    # FiLM (Feature-wise Linear Modulation) works by applying an affine
    # transform after GroupNorm: γ(cond) * x + β(cond), where γ and β are
    # learned linear projections of the conditioning vector. This is more
    # expressive than simple concatenation because it modulates features
    # multiplicatively — it can scale, shift, or suppress entire channels.
    cond_dim: int = 256

    # --- Residual block settings ---
    n_groups: int = 8  # number of groups for GroupNorm (must divide channel count)

    # --- Kernel size for Conv1d layers ---
    # Kernel size 5 means each convolution looks at 5 consecutive timesteps.
    # This gives a receptive field that grows with depth, allowing deeper layers
    # to capture longer-range temporal dependencies in the action sequence.
    kernel_size: int = 5


# ==============================================================================
#  Diffusion Process Configuration (DDPM / DDIM)
# ==============================================================================

@dataclass
class DiffusionConfig:
    """
    Parameters defining the forward (noising) and reverse (denoising) diffusion
    processes.

    The forward process adds Gaussian noise to clean actions over K steps:
        a_k = √(ᾱ_k) · a_0 + √(1 - ᾱ_k) · ε,   ε ~ N(0, I)

    where ᾱ_k = ∏_{i=1}^{k} (1 - β_i) is the cumulative noise schedule.

    At k=0:  ᾱ_0 ≈ 1    →  a_0 ≈ clean action (almost no noise)
    At k=K:  ᾱ_K ≈ 0    →  a_K ≈ pure Gaussian noise

    The model is trained to predict the noise ε given (a_k, k, observation),
    then the reverse process iteratively removes the predicted noise.
    """

    # --- Number of diffusion steps ---
    # More steps = finer-grained noise levels = potentially better quality,
    # but slower training (marginal) and slower DDPM inference (linear).
    # 100 is standard for DDPM. DDIM can use a subset (e.g., 10 steps).
    num_diffusion_steps: int = 100  # K in the equations

    # --- Linear noise schedule ---
    # β_k increases linearly from beta_start to beta_end.
    # β controls how much noise is added at each step.
    #
    # Why these values? From the original DDPM paper (Ho et al., 2020).
    # - beta_start=1e-4: very little noise at the beginning, so the model
    #   learns fine details first (easy denoising near k=0).
    # - beta_end=0.02: substantial noise near k=K, so the model learns the
    #   global structure (hard denoising near k=K).
    #
    # The cumulative product ᾱ_K ≈ 0.0 at step K, meaning the signal is
    # essentially destroyed — which is what we want (pure noise at the end).
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # --- DDIM-specific ---
    # Number of steps to use during DDIM inference. Fewer steps = faster,
    # with minimal quality loss. 10 steps gives ~10x speedup over DDPM.
    ddim_steps: int = 10

    # DDIM stochasticity parameter η:
    #   η = 0: fully deterministic (recommended default for evaluation)
    #   η = 1: equivalent to DDPM (fully stochastic)
    # In between: partially stochastic. We default to 0 for reproducible evals.
    ddim_eta: float = 0.0


# ==============================================================================
#  Flow Matching Configuration (Phase 5 — Stretch Goal)
# ==============================================================================

@dataclass
class FlowMatchingConfig:
    """
    Parameters for the Flow Matching (FM) alternative to DDPM.

    Key difference from DDPM:
    - DDPM: discrete timesteps k ∈ {0, ..., K-1}, predicts noise ε
    - FM: continuous time t ∈ [0, 1], predicts velocity v

    The same U-Net architecture is used. Only the training loss and inference
    loop change. FM training interpolates linearly between data (t=0) and
    noise (t=1):
        x_t = (1 - t) · a_0 + t · ε

    and the model learns the conditional velocity field v_θ(x_t, t) = ε - a_0.

    Inference is an ODE solve from t=1 (noise) to t=0 (data).
    """

    # Number of Euler steps for ODE integration during inference.
    # FM typically needs fewer steps than DDPM because the transport paths
    # are straighter (especially with optimal transport conditioning).
    num_inference_steps: int = 10

    # Whether to use mini-batch Optimal Transport (OT) pairing.
    # Standard FM pairs each data sample with an independent noise sample.
    # OT-FM pairs them to minimize transport cost (Wasserstein distance),
    # giving straighter interpolation paths → better performance with fewer steps.
    use_ot: bool = False

    # ODE solver: "euler" (simple, fast) or "rk4" (higher-order, more accurate)
    ode_solver: str = "euler"


# ==============================================================================
#  Training Configuration
# ==============================================================================

@dataclass
class TrainConfig:
    """
    All hyperparameters for the training loop.

    Composed from the sub-configs above so you can do:
        cfg = TrainConfig()
        model = ConditionalUnet1D(cfg.model)
        dataset = PushTDataset(cfg.data)
    """

    # --- Sub-configurations ---
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)

    # --- Training method ---
    # "ddpm" for standard denoising diffusion, "flow_matching" for FM
    method: str = "ddpm"

    # --- Optimizer ---
    # AdamW = Adam with decoupled weight decay. Weight decay prevents the
    # model weights from growing unboundedly (L2 regularization), which
    # improves generalization. AdamW decouples the weight decay from the
    # gradient update, which is theoretically more correct than Adam's
    # built-in L2 regularization.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam momentum parameters

    # --- Learning rate schedule ---
    # Cosine annealing: LR smoothly decays from `learning_rate` to ~0 following
    # a cosine curve. This avoids the sharp drops of step-based schedules and
    # allows the model to fine-tune with a very low LR at the end of training.
    #
    # Linear warmup: for the first `lr_warmup_steps` steps, the LR linearly
    # ramps from 0 to `learning_rate`. This prevents early training instability
    # when the model weights are randomly initialized and gradients are noisy.
    lr_warmup_steps: int = 500

    # --- Batch size ---
    # 256 is the default from the Diffusion Policy paper. Larger batch sizes
    # give more stable gradient estimates but use more memory.
    # For local CPU testing, override this to something small (e.g., 4).
    batch_size: int = 256

    # --- Training duration ---
    num_epochs: int = 300

    # --- EMA (Exponential Moving Average) ---
    # EMA maintains a shadow copy of the model weights that is a running
    # exponential average: θ_ema ← decay * θ_ema + (1 - decay) * θ_train
    #
    # Why? The training weights oscillate as the optimizer takes noisy gradient
    # steps. The EMA weights smooth out these oscillations, giving a more stable
    # model for inference. The EMA model is ALWAYS used for evaluation.
    #
    # decay = 0.995 means the EMA weights are a weighted average of roughly
    # the last 1/(1-0.995) = 200 training steps.
    ema_decay: float = 0.995

    # --- Gradient clipping ---
    # Clips the total gradient norm to prevent exploding gradients.
    # If ||∇L|| > max_grad_norm, scale all gradients down proportionally.
    # This is a safety net — with a well-tuned LR, gradients rarely explode,
    # but it prevents catastrophic training failures if they do.
    max_grad_norm: float = 1.0

    # --- Logging & checkpointing ---
    log_interval: int = 10       # print loss every N steps
    save_interval: int = 50      # save checkpoint every N epochs
    eval_interval: int = 50      # run evaluation rollouts every N epochs
    num_eval_episodes: int = 50  # number of rollout episodes per evaluation

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Reproducibility ---
    seed: int = 42

    # --- Device (auto-detected) ---
    device: str = field(default_factory=_auto_device)

    # --- Number of DataLoader workers ---
    # For CPU: 0 (single-process, easier to debug)
    # For GPU: 4-8 (parallel data loading to saturate GPU)
    num_workers: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.data.action_horizon <= self.data.pred_horizon, (
            f"action_horizon ({self.data.action_horizon}) must be <= "
            f"pred_horizon ({self.data.pred_horizon}). You can't execute more "
            f"actions than you predict."
        )
        assert self.method in ("ddpm", "flow_matching"), (
            f"method must be 'ddpm' or 'flow_matching', got '{self.method}'"
        )
        assert self.diffusion.ddim_eta >= 0.0 and self.diffusion.ddim_eta <= 1.0, (
            f"ddim_eta must be in [0, 1], got {self.diffusion.ddim_eta}"
        )


# ==============================================================================
#  Quick sanity check — run this file directly to verify config loads correctly
# ==============================================================================

if __name__ == "__main__":
    cfg = TrainConfig()
    print("=" * 60)
    print("Configuration loaded successfully!")
    print("=" * 60)
    print(f"  Device:           {cfg.device}")
    print(f"  Method:           {cfg.method}")
    print(f"  Obs dim:          {cfg.env.obs_dim}")
    print(f"  Action dim:       {cfg.env.action_dim}")
    print(f"  Obs horizon:      {cfg.data.obs_horizon}")
    print(f"  Pred horizon:     {cfg.data.pred_horizon}")
    print(f"  Action horizon:   {cfg.data.action_horizon}")
    print(f"  U-Net channels:   {cfg.model.down_dims}")
    print(f"  Diffusion steps:  {cfg.diffusion.num_diffusion_steps}")
    print(f"  DDIM steps:       {cfg.diffusion.ddim_steps}")
    print(f"  Batch size:       {cfg.batch_size}")
    print(f"  Learning rate:    {cfg.learning_rate}")
    print(f"  EMA decay:        {cfg.ema_decay}")
    print(f"  Seed:             {cfg.seed}")
    print("=" * 60)
