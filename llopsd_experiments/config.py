"""
LLOPSD (Latent-Loop On-Policy Self-Distillation) Training Configuration.

Based on RLTT config but with LLOPSD-specific configurations for
on-policy self-distillation across latent thought loops of looped
language models. The teacher (running R loops) distills knowledge
into a student (running R~ < R loops) using loop-aligned KL divergence.

Key differences from RLTT:
- Teacher/student loop asymmetry (teacher_loops R vs student_loops R~)
- Loop mapping strategies (shift, linear, fixed) to align teacher and student loops
- Per-loop distillation weight schedules (uniform, late_heavy, terminal_only)
- Divergence choice (forward_kl, reverse_kl, jsd)
- Teacher mode (frozen, ema, same) with optional EMA decay
- OPSD-style teacher context (append ground-truth answer)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

# Resolve paths relative to this file's location
_THIS_DIR = Path(__file__).parent.resolve()
_LLOPSD_ROOT = _THIS_DIR.parent


def get_default_sft_checkpoint() -> str:
    """Get default SFT checkpoint path relative to LLOPSD root."""
    return str(_LLOPSD_ROOT / "sft_experiments" / "sft_ouro_math_output")


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"

    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    use_cache: bool = False  # Disable for training

    # Fixed recurrent loops for Ouro model
    total_ut_steps: int = 4

    # Initialize from SFT checkpoint (LoRA adapter)
    sft_checkpoint_path: Optional[str] = None
    # Default: relative to LLOPSD root (../sft_experiments/sft_ouro_math_output from llopsd_experiments/)
    default_sft_checkpoint: str = ""  # Computed at runtime via get_default_sft_checkpoint()


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl"
    math500_test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
    preprocessing_num_workers: int = 4

    # Data format for verl
    prompt_key: str = "prompt"
    answer_key: str = "answer"


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # Use "all-linear" to apply LoRA to all linear layers for maximum expressiveness
    lora_target_modules: str = "all-linear"
    lora_bias: str = "none"


@dataclass
class VLLMRolloutConfig:
    """vLLM rollout configuration for accelerated generation."""
    # Use vLLM for generation (recommended)
    name: str = "vllm"  # "vllm" or "hf"

    # vLLM specific settings
    tensor_model_parallel_size: int = 1  # Number of GPUs for TP during inference
    gpu_memory_utilization: float = 0.7  # GPU memory fraction for vLLM
    enforce_eager: bool = False  # False = enable CUDA graphs
    free_cache_engine: bool = False  # Keep cache between batches
    enable_prefix_caching: bool = True  # Enable prefix caching

    # Generation parameters
    n: int = 8  # Number of samples per prompt (group size)
    temperature: float = 0.9
    top_p: float = 1.0
    max_response_length: int = 3072


@dataclass
class LLOPSDConfig:
    """LLOPSD-specific configuration for loop-aligned on-policy self-distillation.

    Controls the teacher/student loop asymmetry, how teacher and student loops
    are aligned, per-loop distillation weights, divergence measure, and teacher
    update strategy.
    """
    # === Loop configuration ===
    # Number of loops the teacher runs (R in the paper)
    teacher_loops: int = 4

    # Number of loops the student runs at inference (R~ in the paper, R~ < R)
    student_loops: int = 2

    # === Loop mapping strategy ===
    # How to align student loops to teacher loops for distillation:
    # - "shift": align student loop r~ to teacher loop r~ + (R - R~),
    #            i.e., student loops map to the last R~ teacher loops
    # - "linear": linearly interpolate student loops onto the teacher loop range,
    #             e.g., student loop r~ maps to teacher loop round(r~ * (R-1) / (R~-1))
    # - "fixed": each student loop r~ distills from the same fixed teacher loop (the last one, R)
    loop_mapping: str = "shift"

    # === Per-loop distillation weight schedule ===
    # How to weight the KL loss across aligned loop pairs:
    # - "uniform": equal weight 1/R~ on every student loop
    # - "late_heavy": weight proportional to r~^gamma / sum(s~^gamma), emphasising later loops
    # - "terminal_only": all weight on the final student loop (r~ = R~), zero elsewhere
    weight_schedule: str = "uniform"

    # Exponent for the "late_heavy" schedule (only used when weight_schedule="late_heavy")
    # gamma = 0 recovers uniform; gamma = 1 gives linear; gamma = 2 gives quadratic
    weight_gamma: float = 1.0

    # === Divergence measure ===
    # KL variant used for the distillation loss:
    # - "forward_kl": KL(teacher || student) -- mode-covering
    # - "reverse_kl": KL(student || teacher) -- mode-seeking
    # - "jsd": Jensen-Shannon divergence (symmetric, bounded)
    divergence: str = "forward_kl"

    # === Teacher context (OPSD-style) ===
    # How the teacher prompt is constructed:
    # - "opsd": append the ground-truth answer to the teacher's prompt context
    teacher_context: str = "opsd"

    # === Teacher update mode ===
    # How the teacher model is maintained relative to the student:
    # - "frozen": teacher weights are frozen at initialisation (never updated)
    # - "ema": teacher is an exponential moving average of the student
    # - "same": teacher and student share weights (self-distillation)
    teacher_mode: str = "same"

    # EMA decay coefficient (only used when teacher_mode="ema")
    # theta_teacher <- ema_decay * theta_teacher + (1 - ema_decay) * theta_student
    ema_decay: float = 0.999


@dataclass
class ActorConfig:
    """Actor (policy) model training configuration."""
    # LLOPSD-specific (replaces GRPO's kl_loss)
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"

    # PPO mini-batch size (for gradient updates)
    ppo_mini_batch_size: int = 64
    ppo_micro_batch_size_per_gpu: int = 8

    # Dynamic batch size optimization
    use_dynamic_bsz: bool = True
    ppo_max_token_len_per_gpu: int = 48000

    # Optimizer
    optim_lr: float = 1e-6
    optim_weight_decay: float = 0.1
    optim_betas: tuple = (0.9, 0.99)
    max_grad_norm: float = 0.1

    # Schedule
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # FSDP settings
    fsdp_param_offload: bool = False
    fsdp_optimizer_offload: bool = False

    # Gradient checkpointing
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True


@dataclass
class RefConfig:
    """Reference model configuration."""
    log_prob_max_token_len_per_gpu: int = 64000
    fsdp_param_offload: bool = True  # Offload ref model to save memory


@dataclass
class LLOPSDTrainingConfig:
    """LLOPSD Training hyperparameters."""
    output_dir: str = "./llopsd_output"

    # === LLOPSD-specific hyperparameters ===
    beta: float = 0.001  # KL coefficient
    num_generations: int = 8  # Samples per prompt (same as rollout.n)
    num_prompts_per_batch: int = 128  # Unique prompts per batch

    # Total batch = num_prompts_per_batch * num_generations
    # e.g., 128 * 8 = 1024 samples per batch

    # === Sequence lengths ===
    max_prompt_length: int = 1024
    max_completion_length: int = 3072

    # === Training duration ===
    total_epochs: int = 20

    # === Precision ===
    bf16: bool = True
    fp16: bool = False

    # === Saving and Validation ===
    save_freq: int = 7  # Save every N epochs
    test_freq: int = 7  # Validate every N epochs
    val_before_train: bool = True  # Run validation before training

    # === Random seed ===
    seed: int = 42

    # === Data filtering ===
    filter_overlong_prompts: bool = True
    truncation: str = "error"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    # Multi-GPU / Multi-node settings
    n_gpus_per_node: int = 8
    nnodes: int = 1

    # Resource pool configuration
    # Default: all GPUs in single pool for actor+rollout+ref
    use_separate_pools: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = True
    wandb_project: str = "llopsd-ouro-math"
    wandb_run_name: Optional[str] = None
    wandb_offline: bool = True

    experiment_name: str = "LLOPSD-Ouro-MATH"

    # Rollout logging
    log_rollouts: bool = True
    rollout_log_freq: int = 1  # Log rollouts every N steps


@dataclass
class ValidationConfig:
    """Validation configuration."""
    # Validation frequency (in epochs, aligned with verl)
    eval_epochs: int = 7

    # Generation settings for validation
    max_prompt_length: int = 1024
    max_new_tokens: int = 3072
    temperature: float = 0.0  # Greedy decoding for evaluation
    do_sample: bool = False
    use_few_shot: bool = False

    # MATH-500 test file
    math500_test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
    math500_samples: Optional[int] = None  # None = all 500

    # Batch size for evaluation
    eval_batch_size: int = 64


@dataclass
class LLOPSDFullConfig:
    """Combined LLOPSD configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rollout: VLLMRolloutConfig = field(default_factory=VLLMRolloutConfig)
    llopsd: LLOPSDConfig = field(default_factory=LLOPSDConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    training: LLOPSDTrainingConfig = field(default_factory=LLOPSDTrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
