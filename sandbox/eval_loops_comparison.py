#!/usr/bin/env python3
"""
Compare Ouro-2.6B-Thinking accuracy on random MATH questions
under different loop counts, each with configurable max decode tokens.

Usage:
    python eval_loops_comparison.py
    python eval_loops_comparison.py --num_questions 20 --loops 1 2 4 --max_new_tokens 4096
"""
import os
import sys
import json
import time
import random
import shutil
import argparse
import tempfile
from datetime import datetime

# Offline mode for cluster
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Fix for MIG-partitioned GPUs: vLLM cannot parse MIG UUIDs in
# CUDA_VISIBLE_DEVICES.  Override with "0" (matches run_eval_*.slurm pattern).
if "MIG" in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent directory for math_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_utils import (
    extract_boxed_answer,
    check_math_answer,
    get_gold_answer,
    format_math_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_random_questions(path: str, n: int, seed: int):
    """Load n random questions from a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    rng = random.Random(seed)
    return rng.sample(data, min(n, len(data)))


def make_model_copy(src: str, total_ut_steps: int) -> str:
    """Create a temp copy of the model dir with a modified total_ut_steps.

    Only copies metadata files (config, tokenizer, modeling code) and
    symlinks the large weight files to avoid duplicating ~5 GB.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"ouro_ut{total_ut_steps}_")

    for entry in os.listdir(src):
        src_path = os.path.join(src, entry)
        dst_path = os.path.join(tmp_dir, entry)
        if entry.endswith((".safetensors", ".bin", ".pt")):
            os.symlink(src_path, dst_path)
        else:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    # Patch config
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["total_ut_steps"] = total_ut_steps
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return tmp_dir


def run_eval(model_dir: str, questions: list, args):
    """Load model from model_dir and evaluate on questions.

    Returns (accuracy, correct_count, total, elapsed_sec, details).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
    )

    max_model_len = args.max_prompt_length + args.max_new_tokens
    llm = LLM(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )

    # Build prompts
    prompts = []
    gold_answers = []
    for q in questions:
        prompts.append(format_math_prompt(q["problem"], tokenizer, use_few_shot=True))
        gold_answers.append(get_gold_answer(q))

    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        stop_token_ids=stop_ids or None,
    )

    # Generate & time
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    # Score
    correct = 0
    details = []
    for i, (out, gold) in enumerate(zip(outputs, gold_answers)):
        response = out.outputs[0].text
        pred = extract_boxed_answer(response)
        is_correct = pred is not None and check_math_answer(pred, gold)
        if is_correct:
            correct += 1
        details.append({
            "index": i,
            "problem": questions[i]["problem"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "correct": is_correct,
            "prompt": prompts[i],
            "response": response,
        })

    total = len(questions)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, elapsed, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare Ouro-2.6B-Thinking accuracy across loop counts"
    )
    parser.add_argument("--model_path", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking",
                        help="Path to base model weights")
    parser.add_argument("--test_file", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
                        help="Path to MATH test JSONL file")
    parser.add_argument("--num_questions", type=int, default=10,
                        help="Number of random questions to sample")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens for generation")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--loops", type=int, nargs="+", default=[2, 4],
                        help="Loop counts to compare (e.g. --loops 2 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for question sampling and generation")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization fraction")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graph capture (saves memory)")
    parser.add_argument("--output_dir", type=str, default="./logs",
                        help="Directory for rollout JSON output files")
    args = parser.parse_args()

    print("=" * 65)
    print("  Ouro-2.6B-Thinking  |  Loop-count comparison on MATH")
    print("=" * 65)
    print(f"  Model:          {args.model_path}")
    print(f"  Test file:      {args.test_file}")
    print(f"  Questions:      {args.num_questions}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Loop configs:   {args.loops}")
    print(f"  Seed:           {args.seed}")
    print(f"  GPU mem util:   {args.gpu_memory_utilization}")
    print(f"  Enforce eager:  {args.enforce_eager}")

    # Sample questions once so all configs answer the same set
    questions = load_random_questions(args.test_file, args.num_questions, args.seed)
    print(f"\nSampled {len(questions)} random MATH-500 questions (seed={args.seed})")
    for i, q in enumerate(questions):
        print(f"  Q{i+1}: {q['problem'][:80]}...")

    results = {}

    for num_loops in args.loops:
        print(f"\n{'─' * 65}")
        print(f"  Running with {num_loops} loops, max_new_tokens={args.max_new_tokens}")
        print(f"{'─' * 65}")

        # Create a temp model dir with the desired loop count
        tmp_dir = make_model_copy(args.model_path, num_loops)
        try:
            acc, correct, total, elapsed, details = run_eval(
                tmp_dir, questions, args,
            )
            results[num_loops] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "elapsed_sec": elapsed,
                "details": details,
            }

            print(f"\n  Loops={num_loops}  |  Accuracy: {acc*100:.1f}% ({correct}/{total})  |  Time: {elapsed:.2f}s")
            for d in details:
                status = "✓" if d["correct"] else "✗"
                print(f"    {status}  gold={d['gold_answer']}  pred={d['predicted_answer']}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Loops':<8} {'Accuracy':<18} {'Time (s)':<12}")
    print(f"  {'─'*8} {'─'*18} {'─'*12}")
    for num_loops in args.loops:
        r = results[num_loops]
        print(f"  {num_loops:<8} {r['accuracy']*100:.1f}% ({r['correct']}/{r['total']}){'':<5} {r['elapsed_sec']:.2f}")
    print(f"{'=' * 65}")

    # -----------------------------------------------------------------------
    # Save rollouts to JSON
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    for num_loops in args.loops:
        r = results[num_loops]
        rollout_data = {
            "config": {
                "model_path": args.model_path,
                "test_file": args.test_file,
                "num_questions": args.num_questions,
                "max_new_tokens": args.max_new_tokens,
                "max_prompt_length": args.max_prompt_length,
                "total_ut_steps": num_loops,
                "seed": args.seed,
                "job_id": job_id,
                "timestamp": timestamp,
            },
            "metrics": {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
                "elapsed_sec": r["elapsed_sec"],
            },
            "rollouts": r["details"],
        }
        fname = f"rollouts_loops{num_loops}_{job_id}_{timestamp}.json"
        fpath = os.path.join(args.output_dir, fname)
        with open(fpath, "w") as f:
            json.dump(rollout_data, f, indent=2)
        print(f"  Saved rollouts ({num_loops} loops): {fpath}")

    # Combined summary JSON
    summary = {
        "timestamp": timestamp,
        "job_id": job_id,
        "config": {
            "model_path": args.model_path,
            "test_file": args.test_file,
            "num_questions": args.num_questions,
            "max_new_tokens": args.max_new_tokens,
            "loops": args.loops,
            "seed": args.seed,
        },
        "results": {
            str(nl): {
                "accuracy": results[nl]["accuracy"],
                "correct": results[nl]["correct"],
                "total": results[nl]["total"],
                "elapsed_sec": results[nl]["elapsed_sec"],
            }
            for nl in args.loops
        },
    }
    summary_path = os.path.join(args.output_dir, f"summary_{job_id}_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
