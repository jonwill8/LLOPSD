#!/usr/bin/env python3
"""
Evaluation Script for MATH-500 Benchmark (LLOPSD)

This script evaluates a model checkpoint N times under different random seeds
to enable statistical significance testing.

Supports two modes:
  1. Direct model path via --model_path
  2. FSDP checkpoint via --experiment_dir + --checkpoint (with automatic merge)

Optionally sets total_ut_steps in the model config for compressed student inference.

Usage:
    # Direct model path:
    python eval_math500.py \
        --model_path /path/to/merged_model \
        --num_runs 10 \
        --total_ut_steps 4

    # FSDP checkpoint:
    python eval_math500.py \
        --experiment_dir /path/to/llopsd_output/12345 \
        --checkpoint step_100 \
        --num_runs 10
"""
import os
import sys
import json
import argparse
import logging
import random
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_utils import (
    rl_extract_boxed_answer as extract_boxed_answer,
    rl_check_math_answer as check_math_answer,
    rl_get_gold_answer as get_gold_answer,
    format_math_prompt,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load MATH-500 test set from JSONL."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(data)} problems from {filepath}")
    return data


def find_merged_model(experiment_dir: str, checkpoint: str) -> Optional[str]:
    """Find the merged model path for a given checkpoint.

    Searches in eval directories for the merged model, or returns the path
    where it would be created.
    """
    # Check in existing eval directories
    for dirname in os.listdir(experiment_dir):
        if dirname.startswith("eval_"):
            eval_dir = os.path.join(experiment_dir, dirname)
            merged_path = os.path.join(eval_dir, checkpoint, "merged_model")
            if os.path.isdir(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
                logger.info(f"Found existing merged model at: {merged_path}")
                return merged_path

    return None


def merge_checkpoint(experiment_dir: str, checkpoint: str, output_dir: str) -> str:
    """Merge FSDP checkpoint to HuggingFace format."""
    import subprocess

    step_num = checkpoint.replace("step_", "")
    fsdp_checkpoint_dir = os.path.join(experiment_dir, f"global_step_{step_num}", "actor")
    merged_model_dir = os.path.join(output_dir, "merged_model")

    if not os.path.isdir(fsdp_checkpoint_dir):
        raise ValueError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")

    # Check if already merged
    if os.path.isdir(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
        logger.info(f"Using existing merged model at: {merged_model_dir}")
        return merged_model_dir

    logger.info(f"Merging FSDP checkpoint from: {fsdp_checkpoint_dir}")
    logger.info(f"Target directory: {merged_model_dir}")

    # Fix checkpoint files first (same as in run_eval_checkpoints.slurm)
    ckpt_hf_dir = os.path.join(fsdp_checkpoint_dir, "huggingface")
    base_model_dir = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT"

    # Fix config.json
    config_path = os.path.join(ckpt_hf_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_content = f.read()
        config_content = config_content.replace(
            '"AutoModelForCausalLM": "peft_model.PeftModelForCausalLM"',
            '"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"'
        )
        with open(config_path, "w") as f:
            f.write(config_content)

    # Copy modeling_ouro.py if missing
    modeling_path = os.path.join(ckpt_hf_dir, "modeling_ouro.py")
    if not os.path.exists(modeling_path):
        import shutil
        shutil.copy(os.path.join(base_model_dir, "modeling_ouro.py"), modeling_path)

    # Run merger
    result = subprocess.run([
        "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--trust-remote-code",
        "--local_dir", fsdp_checkpoint_dir,
        "--target_dir", merged_model_dir,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Merge failed: {result.stderr}")
        raise RuntimeError(f"Failed to merge checkpoint: {result.stderr}")

    logger.info("Checkpoint merged successfully")
    return merged_model_dir


def evaluate_single_run(
    model_path: str,
    test_data: List[Dict[str, Any]],
    tokenizer,
    llm,
    seed: int,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
) -> Dict[str, Any]:
    """Run a single evaluation with the given seed."""
    set_seed(seed)

    # Prepare prompts
    prompts = []
    gold_answers = []
    levels = []

    for example in test_data:
        problem = example.get("problem", "")
        prompts.append(format_math_prompt(problem, tokenizer, use_few_shot=False))
        gold_answers.append(get_gold_answer(example))
        levels.append(example.get("level", "unknown"))

    # Setup sampling with temperature for variance
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    correct_count = 0
    level_stats = {}

    for i, (output, gold_answer, level) in enumerate(zip(outputs, gold_answers, levels)):
        response = output.outputs[0].text
        pred_answer = extract_boxed_answer(response)
        is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)

        if is_correct:
            correct_count += 1

        # Track by level
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1

    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0

    # Compute level accuracies
    level_accuracies = {}
    for level, stats in level_stats.items():
        level_accuracies[level] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "level_accuracies": level_accuracies,
        "level_stats": level_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation on MATH-500 for LLOPSD with statistical significance testing")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Direct path to a merged/HF model directory. If provided, --experiment_dir and --checkpoint are ignored.")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to experiment output directory (e.g., llopsd_output/12345)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to evaluate (e.g., step_100)")
    parser.add_argument("--total_ut_steps", type=int, default=None,
                        help="If provided, sets total_ut_steps in the model config before loading. "
                             "Allows evaluating the compressed student at inference with fewer loops.")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of evaluation runs with different seeds")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generating run seeds")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (>0 for variance)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--test_file", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
                        help="Path to MATH-500 test file")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/distillation/LLOPSD/eval_outputs",
                        help="Output directory for results")
    parser.add_argument("--method", type=str, default="llopsd",
                        help="Method name for output file naming")
    args = parser.parse_args()

    # Validate inputs: need either model_path or (experiment_dir + checkpoint)
    if args.model_path is None and (args.experiment_dir is None or args.checkpoint is None):
        parser.error("Either --model_path or both --experiment_dir and --checkpoint must be provided.")

    if args.model_path is not None and not os.path.isdir(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")

    if args.experiment_dir is not None and not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    if not os.path.isfile(args.test_file):
        raise ValueError(f"Test file not found: {args.test_file}")

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required for evaluation")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model path
    if args.model_path is not None:
        model_path = args.model_path
        exp_id = os.path.basename(model_path)
        ckpt_label = "direct"
    else:
        # Find or create merged model from FSDP checkpoint
        exp_id = os.path.basename(args.experiment_dir)
        ckpt_label = args.checkpoint

        model_path = find_merged_model(args.experiment_dir, args.checkpoint)
        if model_path is None:
            temp_dir = os.path.join(args.output_dir, f"temp_{args.method}_{exp_id}_{args.checkpoint}")
            os.makedirs(temp_dir, exist_ok=True)
            model_path = merge_checkpoint(args.experiment_dir, args.checkpoint, temp_dir)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ut_suffix = f"_ut{args.total_ut_steps}" if args.total_ut_steps is not None else ""
    output_csv = os.path.join(
        args.output_dir,
        f"{args.method}_math500_{exp_id}_{ckpt_label}_n{args.num_runs}{ut_suffix}_{timestamp}.csv"
    )

    logger.info("=" * 60)
    logger.info("MATH-500 Evaluation (LLOPSD)")
    logger.info("=" * 60)
    logger.info(f"Model path: {model_path}")
    if args.experiment_dir:
        logger.info(f"Experiment directory: {args.experiment_dir}")
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Total UT steps: {args.total_ut_steps}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info("")

    # Optionally set total_ut_steps in the model config
    if args.total_ut_steps is not None:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        config["total_ut_steps"] = args.total_ut_steps
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Set total_ut_steps={args.total_ut_steps} in {config_path}")

    logger.info(f"Using model at: {model_path}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Initialize vLLM
    max_model_len = args.max_prompt_length + args.max_new_tokens
    logger.info(f"Initializing vLLM engine (max_model_len={max_model_len})...")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    logger.info("vLLM engine initialized")

    # Load test data
    test_data = load_dataset(args.test_file)

    # Generate seeds for each run
    set_seed(args.base_seed)
    run_seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_runs)]

    # Run evaluations
    results = []

    for run_idx, seed in enumerate(run_seeds):
        logger.info("")
        logger.info(f"[Run {run_idx + 1}/{args.num_runs}] Seed: {seed}")

        run_result = evaluate_single_run(
            model_path=model_path,
            test_data=test_data,
            tokenizer=tokenizer,
            llm=llm,
            seed=seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        results.append(run_result)
        logger.info(f"  Accuracy: {run_result['accuracy']*100:.2f}% ({run_result['correct']}/{run_result['total']})")

    # Write results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["run", "seed", "accuracy", "correct", "total"]
        # Add level columns
        all_levels = sorted(set(
            level for r in results for level in r["level_accuracies"].keys()
        ))
        for level in all_levels:
            header.append(f"level_{level}_accuracy")
        writer.writerow(header)

        # Data rows
        for run_idx, r in enumerate(results):
            row = [run_idx + 1, r["seed"], r["accuracy"], r["correct"], r["total"]]
            for level in all_levels:
                row.append(r["level_accuracies"].get(level, 0))
            writer.writerow(row)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)

    # Compute summary statistics
    accuracies = [r["accuracy"] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample std

    logger.info(f"Mean accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    logger.info(f"Min: {min(accuracies)*100:.2f}%, Max: {max(accuracies)*100:.2f}%")
    logger.info(f"Results saved to: {output_csv}")

    # Save summary JSON
    summary = {
        "method": args.method,
        "benchmark": "math500",
        "model_path": model_path,
        "experiment_dir": args.experiment_dir,
        "checkpoint": args.checkpoint,
        "total_ut_steps": args.total_ut_steps,
        "num_runs": args.num_runs,
        "temperature": args.temperature,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "accuracies": accuracies,
        "run_seeds": run_seeds,
        "output_csv": output_csv,
    }

    summary_json = output_csv.replace(".csv", "_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
