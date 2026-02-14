#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --output=logs/sft_eval_%j.out
#SBATCH --error=logs/sft_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

# Adjust partition as needed
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# SFT Unified Evaluation Script
# ============================================================================
# This script provides unified evaluation across multiple benchmarks for SFT.
# Supports both direct model paths and experiment checkpoints, with optional
# compressed student loop evaluation.
#
# Usage examples:
#   # Evaluate a model directly:
#   sbatch --export=ALL,BENCHMARK=math500,MODEL_PATH=/path/to/model run_eval.slurm
#
#   # Evaluate from experiment checkpoint:
#   sbatch --export=ALL,BENCHMARK=aime24,EXPERIMENT_DIR=/path/to/exp,CHECKPOINT=step_140 run_eval.slurm
#
#   # Evaluate with compressed student (2 loops):
#   sbatch --export=ALL,BENCHMARK=math500,MODEL_PATH=/path/to/model,TOTAL_UT_STEPS=2 run_eval.slurm
#
# Required parameters:
#   BENCHMARK - One of: math500, gsm8k, aime24, beyondaime
#
# Model loading (one of the following):
#   MODEL_PATH - Direct path to model directory
#   EXPERIMENT_DIR + CHECKPOINT - Path to experiment output and checkpoint name
#
# Optional parameters:
#   TOTAL_UT_STEPS - Number of compressed student loops (default: use model default)
#   NUM_RUNS - Number of evaluation runs (default: 10)
#   TEMPERATURE - Sampling temperature (default: 0.2)
#   OUTPUT_DIR - Output directory (default: ../eval_outputs)
#   METHOD - Method name for results (default: sft)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"math500"}

# Model loading: either MODEL_PATH or EXPERIMENT_DIR+CHECKPOINT
MODEL_PATH=${MODEL_PATH:-""}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-""}
CHECKPOINT=${CHECKPOINT:-""}

# SFT-specific: evaluate with compressed student loops
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-""}  # empty = use model default

NUM_RUNS=${NUM_RUNS:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/distillation/LLOPSD/sft_experiments/eval_outputs"}
METHOD=${METHOD:-"sft"}

# Benchmark-specific settings
case $BENCHMARK in
    math500)
        SCRIPT="eval_math500.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    gsm8k)
        SCRIPT="eval_gsm8k.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
        ;;
    aime24)
        SCRIPT="eval_aime24.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    beyondaime)
        SCRIPT="eval_beyondaime.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    *)
        echo "ERROR: Unknown benchmark: $BENCHMARK"
        echo "Valid options: math500, gsm8k, aime24, beyondaime"
        exit 1
        ;;
esac

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "SFT Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Benchmark: $BENCHMARK"
if [ -n "$MODEL_PATH" ]; then
    echo "Model path: $MODEL_PATH"
else
    echo "Experiment dir: $EXPERIMENT_DIR"
    echo "Checkpoint: $CHECKPOINT"
fi
echo "Number of runs: $NUM_RUNS"
echo "Temperature: $TEMPERATURE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Max prompt length: $MAX_PROMPT_LENGTH"
if [ -n "$TOTAL_UT_STEPS" ]; then
    echo "Total UT steps (compressed): $TOTAL_UT_STEPS"
fi
echo "Output dir: $OUTPUT_DIR"
echo "Method: $METHOD"
echo ""

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set cache directories
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/distillation/LLOPSD/sft_experiments}"
cd $WORK_DIR

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# Load CUDA toolkit
module load cudatoolkit/12.8

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# ============================================================================
# RUN EVALUATION
# ============================================================================
echo "=========================================="
echo "Running evaluation..."
echo "=========================================="

# Build command dynamically based on provided parameters
CMD="python $SCRIPT"

# Add model loading arguments
if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
else
    if [ -z "$EXPERIMENT_DIR" ] || [ -z "$CHECKPOINT" ]; then
        echo "ERROR: Must provide either MODEL_PATH or both EXPERIMENT_DIR and CHECKPOINT"
        exit 1
    fi
    CMD="$CMD --experiment_dir $EXPERIMENT_DIR --checkpoint $CHECKPOINT"
fi

# Add standard evaluation parameters
CMD="$CMD --num_runs $NUM_RUNS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --max_prompt_length $MAX_PROMPT_LENGTH"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --method $METHOD"

# Add LLOPSD-specific parameter if provided
if [ -n "$TOTAL_UT_STEPS" ]; then
    CMD="$CMD --total_ut_steps $TOTAL_UT_STEPS"
fi

# Print and execute command
echo "Executing: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"

exit $EXIT_CODE
