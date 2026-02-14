#!/bin/bash
#SBATCH --job-name=llopsd_train
#SBATCH --output=logs/llopsd_train_%j.out
#SBATCH --error=logs/llopsd_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=60:00:00

# Adjust partition/account as needed
#SBATCH --partition=ailab

# ============================================================================
# LLOPSD HYPERPARAMETERS - TUNE THESE VIA SLURM SUBMISSION
# ============================================================================
# Usage examples:
#   sbatch run_llopsd.slurm                                                      # Default (shift mapping, R=4, R~=2)
#   sbatch --export=ALL,STUDENT_LOOPS=3 run_llopsd.slurm                        # Less compression
#   sbatch --export=ALL,LOOP_MAPPING=fixed run_llopsd.slurm                     # Fixed mapping
#   sbatch --export=ALL,DIVERGENCE=reverse_kl run_llopsd.slurm                  # Reverse KL
#   sbatch --export=ALL,WEIGHT_SCHEDULE=late_heavy,WEIGHT_GAMMA=2 run_llopsd.slurm  # Late-heavy weights
#
# Training Modes:
#   USE_LORA=true  -> LoRA fine-tuning (default, ~2-3 GB per GPU)
#   USE_LORA=false -> Full parameter fine-tuning (~40-60 GB per GPU for 2.6B model)
#
# Full Fine-tuning Workflow:
#   1. Loads SFT checkpoint (LoRA adapter)
#   2. Merges LoRA into base model
#   3. Trains ALL parameters with multi-GPU FSDP
#
# Multi-GPU Training:
#   This script uses verl with a custom LLOPSD worker for multi-GPU training.
#   The full LLOPSD objective (loop-to-loop distillation) is preserved
#   across all GPUs via FSDP sharding and the custom actor implementation.
#   Single-GPU training (N_GPUS=1) uses a simplified trainer instead.

# === LLOPSD-specific hyperparameters ===
TEACHER_LOOPS=${TEACHER_LOOPS:-4}
STUDENT_LOOPS=${STUDENT_LOOPS:-2}
LOOP_MAPPING=${LOOP_MAPPING:-"shift"}          # shift, linear, fixed
WEIGHT_SCHEDULE=${WEIGHT_SCHEDULE:-"uniform"}   # uniform, late_heavy, terminal_only
WEIGHT_GAMMA=${WEIGHT_GAMMA:-1.0}
DIVERGENCE=${DIVERGENCE:-"forward_kl"}         # forward_kl, reverse_kl, jsd
TEACHER_CONTEXT=${TEACHER_CONTEXT:-"opsd"}
TEACHER_MODE=${TEACHER_MODE:-"same"}           # frozen, ema, same
EMA_DECAY=${EMA_DECAY:-0.999}

# === KL penalty and RL hyperparameters ===
BETA=${BETA:-0.001}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
NUM_PROMPTS_PER_BATCH=${NUM_PROMPTS_PER_BATCH:-32}
TEMPERATURE=${TEMPERATURE:-1}

# === Optimizer hyperparameters ===
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.1}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}  # cosine or constant (both include warmup)

# === Batch sizes ===
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
# Max tokens per micro-batch during actor update (with dynamic batching)
# Controls memory usage during backprop. Lower = less memory but slower.
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192} #change back to 16384 for 8 GPU
# Max tokens per micro-batch during log-prob computation (rollout/ref)
# Controls memory usage during forward pass for log-prob calculation.
# Lower = less memory but slower.
LOG_PROB_MAX_TOKEN_LEN=${LOG_PROB_MAX_TOKEN_LEN:-8192} #change back to 16384 for 8 GPU
# Number of dataloader workers (reduce to save CPU RAM if OOM)
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

# === Sequence lengths ===
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}

# === Training duration ===
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:--1}

# === Training Mode: LoRA vs Full Fine-tuning ===
# USE_LORA=true  -> LoRA fine-tuning (default, memory efficient)
# USE_LORA=false -> Full parameter fine-tuning (requires more GPU memory)
#
# Full Fine-tuning Workflow:
#   1. Load SFT checkpoint (which contains LoRA adapter)
#   2. Merge LoRA weights into base model
#   3. Train ALL model parameters with LLOPSD
#
# Example usage:
#   sbatch --export=ALL,USE_LORA=false run_llopsd.slurm  # Full fine-tuning
#   sbatch --export=ALL,USE_LORA=true run_llopsd.slurm   # LoRA fine-tuning (default)
#
USE_LORA=${USE_LORA:-false}

# LoRA configuration (only used when USE_LORA=true)
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
# LORA_TARGET_MODULES: "all-linear" applies to all linear layers,
#                      "attention" applies only to q_proj, k_proj, v_proj, o_proj
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-attention}

# === Ouro model configuration ===
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}

# === Validation configuration ===
EVAL_STEPS=${EVAL_STEPS:-5}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_MAX_NEW_TOKENS=${VAL_MAX_NEW_TOKENS:-3072}
NO_VALIDATION=${NO_VALIDATION:-true}
SKIP_INITIAL_VALIDATION=${SKIP_INITIAL_VALIDATION:-true}
MATH500_TEST_FILE=${MATH500_TEST_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"}

# === SFT checkpoint initialization ===
# Set USE_SFT_CHECKPOINT=false to start from fresh MODEL_PATH weights (no SFT initialization)
# Example: sbatch --export=ALL,USE_SFT_CHECKPOINT=false run_llopsd.slurm
USE_SFT_CHECKPOINT=${USE_SFT_CHECKPOINT:-false}
SFT_CHECKPOINT_PATH=${SFT_CHECKPOINT_PATH:-""}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-""}

# === Logging ===
LOGGING_STEPS=${LOGGING_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-10}
WANDB_PROJECT=${WANDB_PROJECT:-llopsd-ouro-math}

# === vLLM configuration ===
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.45}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-1}
USE_VLLM=${USE_VLLM:-true}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}

# === Multi-GPU settings ===
if [ -n "$N_GPUS" ]; then
    N_GPUS=$N_GPUS
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    N_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    N_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    N_GPUS=2
fi
NNODES=${NNODES:-${SLURM_NNODES:-1}}

# === Random seed ===
SEED=${SEED:-42}

# ============================================================================
# PATHS - Adjust if needed
# ============================================================================
MODEL_PATH=${MODEL_PATH:-"/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"}
TRAIN_FILE=${TRAIN_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./llopsd_output/${SLURM_JOB_ID}"}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "LLOPSD Training - Loop-to-Loop On-Policy Self-Distillation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "=========================================="
if [ "$USE_LORA" = "true" ]; then
    echo "*** TRAINING MODE: LoRA FINE-TUNING ***"
else
    echo "*** TRAINING MODE: FULL PARAMETER FINE-TUNING ***"
fi
echo "=========================================="
echo ""

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set wandb to offline mode
export WANDB_MODE=offline

# Set cache directories
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Ray settings for verl
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/distillation/LLOPSD/llopsd_experiments}"
LLOPSD_ROOT="$(dirname "$WORK_DIR")"
cd $WORK_DIR

# Set default SFT paths relative to LLOPSD root if not specified
if [ -z "$SFT_OUTPUT_DIR" ]; then
    SFT_OUTPUT_DIR="$LLOPSD_ROOT/sft_experiments/sft_ouro_math_output"
fi

# Create logs directory
mkdir -p logs

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# Load CUDA toolkit matching PyTorch's CUDA version (12.8)
module load cudatoolkit/12.8

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Check for vLLM
if python -c "import vllm" 2>/dev/null; then
    echo "vLLM: available"
    VLLM_AVAILABLE=true
else
    echo "vLLM: not available (will use HF generation)"
    VLLM_AVAILABLE=false
fi

# Check for verl
if python -c "import verl" 2>/dev/null; then
    echo "verl: available"
    VERL_AVAILABLE=true
else
    echo "verl: not available (will use simple trainer)"
    VERL_AVAILABLE=false
fi
echo ""

# Print configuration
echo "=========================================="
echo "LLOPSD Configuration"
echo "=========================================="
echo ""
echo "LLOPSD-Specific Settings:"
echo "  Teacher loops (R): $TEACHER_LOOPS"
echo "  Student loops (R~): $STUDENT_LOOPS"
echo "  Loop mapping: $LOOP_MAPPING"
echo "  Weight schedule: $WEIGHT_SCHEDULE"
echo "  Weight gamma: $WEIGHT_GAMMA"
echo "  Divergence: $DIVERGENCE"
echo "  Teacher context: $TEACHER_CONTEXT"
echo "  Teacher mode: $TEACHER_MODE"
if [ "$TEACHER_MODE" = "ema" ]; then
    echo "  EMA decay: $EMA_DECAY"
fi
echo "  Total UT steps: $TOTAL_UT_STEPS"
echo ""
echo "Model Settings:"
echo "  Model path: $MODEL_PATH"
echo ""
echo "Data Settings:"
echo "  Train file: $TRAIN_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "RL Hyperparameters:"
echo "  Beta (KL coef): $BETA"
echo "  Num generations per prompt: $NUM_GENERATIONS"
echo "  Num prompts per batch: $NUM_PROMPTS_PER_BATCH"
echo "  Temperature: $TEMPERATURE"
echo ""
COMPUTED_BATCH_SIZE=$((NUM_PROMPTS_PER_BATCH * NUM_GENERATIONS))
echo "Training Hyperparameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  LR scheduler: $LR_SCHEDULER_TYPE"
echo "  Warmup ratio: $WARMUP_RATIO"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Max grad norm: $MAX_GRAD_NORM"
echo "  Epochs: $NUM_EPOCHS"
echo "  Total batch size: $COMPUTED_BATCH_SIZE ($NUM_PROMPTS_PER_BATCH prompts x $NUM_GENERATIONS gens)"
echo "  Dynamic batch sizing: enabled (ppo_max_token_len=$PPO_MAX_TOKEN_LEN)"
echo "  Log-prob max token len: $LOG_PROB_MAX_TOKEN_LEN"
echo ""
echo "Sequence Lengths:"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Max completion length: $MAX_COMPLETION_LENGTH"
echo ""
echo "=========================================="
echo "Training Mode Configuration:"
echo "=========================================="
if [ "$USE_LORA" = "true" ]; then
    echo "  *** TRAINING MODE: LoRA FINE-TUNING ***"
    echo "  Only LoRA adapter parameters will be updated"
    echo "  LoRA r: $LORA_R"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  LoRA target modules: $LORA_TARGET_MODULES"
else
    echo "  *** TRAINING MODE: FULL PARAMETER FINE-TUNING ***"
    echo "  All model parameters will be updated"
    echo "  Note: Requires more GPU memory than LoRA"
    if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
        echo "  Workflow: SFT LoRA -> Merge into base -> Train all params"
    fi
fi
echo ""
echo "Multi-GPU Configuration:"
echo "  Number of GPUs: $N_GPUS"
echo "  Number of nodes: $NNODES"
echo "  vLLM TP size: $VLLM_TP_SIZE"
echo ""
echo "vLLM Configuration:"
echo "  Use vLLM: $USE_VLLM"
echo "  GPU memory utilization: $VLLM_GPU_MEM"
echo ""
echo "SFT Checkpoint:"
echo "  Use SFT checkpoint: $USE_SFT_CHECKPOINT"
if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
    echo "  SFT checkpoint path: $SFT_CHECKPOINT_PATH"
else
    echo "  Starting from fresh model weights (no SFT initialization)"
fi
echo ""
echo "=========================================="
echo ""

# Explain LLOPSD loop mapping and weight schedule
echo "LLOPSD Loop Mapping & Weight Schedule:"
echo "======================================="
echo ""
echo "Loop Mapping ($LOOP_MAPPING):"
if [ "$LOOP_MAPPING" = "shift" ]; then
    echo "  Using SHIFT mapping: student loop s maps to teacher loop s+1"
    echo "  Student R~=$STUDENT_LOOPS loops learn from teacher R=$TEACHER_LOOPS loops"
    echo "  Each student loop distills from one loop ahead in the teacher"
elif [ "$LOOP_MAPPING" = "linear" ]; then
    echo "  Using LINEAR mapping: student loops are linearly spaced across teacher loops"
    echo "  Student R~=$STUDENT_LOOPS loops map proportionally to teacher R=$TEACHER_LOOPS loops"
elif [ "$LOOP_MAPPING" = "fixed" ]; then
    echo "  Using FIXED mapping: all student loops distill from the final teacher loop"
    echo "  Student R~=$STUDENT_LOOPS loops all target teacher loop R=$TEACHER_LOOPS"
fi
echo ""
echo "Weight Schedule ($WEIGHT_SCHEDULE):"
if [ "$WEIGHT_SCHEDULE" = "uniform" ]; then
    echo "  Using UNIFORM weights: all loop pairs contribute equally"
elif [ "$WEIGHT_SCHEDULE" = "late_heavy" ]; then
    echo "  Using LATE-HEAVY weights: later loop pairs receive more weight (gamma=$WEIGHT_GAMMA)"
elif [ "$WEIGHT_SCHEDULE" = "terminal_only" ]; then
    echo "  Using TERMINAL-ONLY weights: only the final loop pair contributes to the loss"
fi
echo ""
echo "Divergence ($DIVERGENCE):"
if [ "$DIVERGENCE" = "forward_kl" ]; then
    echo "  Using Forward KL: KL(teacher || student) - mean-seeking"
elif [ "$DIVERGENCE" = "reverse_kl" ]; then
    echo "  Using Reverse KL: KL(student || teacher) - mode-seeking"
elif [ "$DIVERGENCE" = "jsd" ]; then
    echo "  Using Jensen-Shannon Divergence: symmetric combination"
fi
echo ""

# Calculate steps based on dataset size
DATASET_SIZE=7500
STEPS_PER_EPOCH=$((DATASET_SIZE / NUM_PROMPTS_PER_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))
echo "Training Progress Guide:"
echo "========================"
echo "  Dataset size: $DATASET_SIZE prompts"
echo "  Prompts per step: $NUM_PROMPTS_PER_BATCH"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Total steps: $TOTAL_STEPS (over $NUM_EPOCHS epoch(s))"
echo ""
echo "Starting LLOPSD training..."
echo ""

# Build command line arguments
LORA_ARGS="--lora_target_modules $LORA_TARGET_MODULES"
if [ "$USE_LORA" = "false" ]; then
    LORA_ARGS="$LORA_ARGS --no_lora"
fi

# Build SFT checkpoint arguments
SFT_ARGS=""
if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
    if [ -n "$SFT_CHECKPOINT_PATH" ] && [ -d "$SFT_CHECKPOINT_PATH" ]; then
        SFT_ARGS="--sft_checkpoint $SFT_CHECKPOINT_PATH"
    else
        SFT_ARGS="--use_latest_sft --sft_output_dir $SFT_OUTPUT_DIR"
    fi
else
    # Explicitly skip SFT checkpoint - start from fresh MODEL_PATH weights
    SFT_ARGS="--no_sft_checkpoint"
fi

# Build validation arguments
VAL_ARGS="--eval_steps $EVAL_STEPS --val_batch_size $VAL_BATCH_SIZE --val_max_new_tokens $VAL_MAX_NEW_TOKENS --math500_test_file $MATH500_TEST_FILE"
if [ "$NO_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --no_validation"
fi
if [ "$SKIP_INITIAL_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --skip_initial_validation"
fi

# Build vLLM arguments
VLLM_ARGS=""
if [ "$USE_VLLM" = "false" ]; then
    VLLM_ARGS="--no_vllm"
else
    VLLM_ARGS="--vllm_gpu_memory_utilization $VLLM_GPU_MEM --vllm_tensor_parallel_size $VLLM_TP_SIZE"
    if [ "$VLLM_ENFORCE_EAGER" = "true" ]; then
        VLLM_ARGS="$VLLM_ARGS --vllm_enforce_eager"
    fi
fi

# Run training
python llopsd_train.py \
    --model_path "$MODEL_PATH" \
    --total_ut_steps "$TOTAL_UT_STEPS" \
    --teacher_loops "$TEACHER_LOOPS" \
    --student_loops "$STUDENT_LOOPS" \
    --loop_mapping "$LOOP_MAPPING" \
    --weight_schedule "$WEIGHT_SCHEDULE" \
    --weight_gamma "$WEIGHT_GAMMA" \
    --divergence "$DIVERGENCE" \
    --teacher_context "$TEACHER_CONTEXT" \
    --teacher_mode "$TEACHER_MODE" \
    --ema_decay "$EMA_DECAY" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --beta "$BETA" \
    --num_generations "$NUM_GENERATIONS" \
    --num_prompts_per_batch "$NUM_PROMPTS_PER_BATCH" \
    --temperature "$TEMPERATURE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --ppo_max_token_len "$PPO_MAX_TOKEN_LEN" \
    --log_prob_max_token_len "$LOG_PROB_MAX_TOKEN_LEN" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --wandb_project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --n_gpus "$N_GPUS" \
    --nnodes "$NNODES" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    $LORA_ARGS \
    $SFT_ARGS \
    $VAL_ARGS \
    $VLLM_ARGS

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  Model checkpoint: $OUTPUT_DIR/"
echo "  Rollout logs: $OUTPUT_DIR/rollout_logs/"
echo "  wandb logs: $OUTPUT_DIR/wandb/"
echo ""
echo "To sync wandb (when internet available):"
echo "  wandb sync $OUTPUT_DIR/wandb/offline-run-*"
echo ""
echo "=========================================="
