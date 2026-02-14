#!/bin/bash
#SBATCH --job-name=eval_loops_cmp
#SBATCH --output=logs/eval_loops_cmp_%j.out
#SBATCH --error=logs/eval_loops_cmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:58:00
#SBATCH --partition=gpu80

echo "=========================================="
echo "Loop-count comparison: Ouro-2.6B-Thinking"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# ── Environment ──────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TOKENIZERS_PARALLELISM=false

# Fix for MIG-enabled nodes
if [[ "$CUDA_VISIBLE_DEVICES" == *"MIG"* ]]; then
    echo "Detected MIG GPU, setting CUDA_VISIBLE_DEVICES=0"
    export CUDA_VISIBLE_DEVICES=0
fi

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/distillation/LLOPSD/sandbox}"
cd "$WORK_DIR"
mkdir -p logs

# Conda
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# ── Configurable parameters (override via environment) ───────────
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking}"
TEST_FILE="${TEST_FILE:-/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl}"
NUM_QUESTIONS="${NUM_QUESTIONS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
LOOPS="${LOOPS:-2 4}"
SEED="${SEED:-42}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/rollouts_${SLURM_JOB_ID}}"

# Print env info
echo "Python: $(which python)"
echo "vLLM:   $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo ""

echo "Configuration:"
echo "  Model:            $MODEL_PATH"
echo "  Test file:        $TEST_FILE"
echo "  Num questions:    $NUM_QUESTIONS"
echo "  Max new tokens:   $MAX_NEW_TOKENS"
echo "  Max prompt len:   $MAX_PROMPT_LENGTH"
echo "  Loops:            $LOOPS"
echo "  Seed:             $SEED"
echo "  GPU mem util:     $GPU_MEM_UTIL"
echo "  Output dir:       $OUTPUT_DIR"
echo ""

# ── Run ──────────────────────────────────────────────────────────
python eval_loops_comparison.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --num_questions "$NUM_QUESTIONS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --loops $LOOPS \
    --seed "$SEED" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Done! End time: $(date)"
echo "=========================================="
