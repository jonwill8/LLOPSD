"""
Math reward function with rollout logging for RL training.

This module provides a reward function compatible with verl's reward manager
interface, with additional logging capabilities for debugging and analysis.

Uses centralized answer_parsing for answer extraction and checking.

Deduplication: Only logs unique (question, response, ground_truth) combinations.
Rollouts are written immediately during training (streaming).
"""
import os
import json
import logging
import threading
import atexit
from datetime import datetime
from typing import Optional, Dict, Any, Set, List, Callable
from collections import defaultdict

from .answer_parsing import extract_boxed_answer, extract_first_boxed_answer, find_first_boxed_end, check_math_answer

logger = logging.getLogger(__name__)

# Thread-safe file writing and deduplication
_file_lock = threading.Lock()
_rollout_file: Optional[str] = None
_header_written = False

# Deduplication: track seen (question, response, ground_truth) tuples
_seen_rollouts: Set[tuple] = set()

# Track prompts by (question, ground_truth) -> prompt_id and rollout_count
_prompt_data = defaultdict(lambda: {"prompt_id": None, "rollout_count": 0})
_next_prompt_id = 0

# Write buffer for efficient batched I/O
# Buffer size of 512 matches typical batch size (64 prompts × 8 generations)
# This results in ~1 file write per training step instead of 512
# Memory overhead is minimal (~1-2MB for buffered JSON strings)
_write_buffer: List[str] = []
_BUFFER_SIZE = 512


def _flush_buffer(rollout_file: str) -> None:
    """Flush the write buffer to disk (must be called with _file_lock held)."""
    global _write_buffer
    if not _write_buffer:
        return
    try:
        with open(rollout_file, "a") as f:
            f.write("\n".join(_write_buffer) + "\n")
        _write_buffer = []
    except Exception as e:
        logger.warning(f"Failed to flush rollout buffer: {e}")


def flush_rollout_buffer() -> None:
    """Flush any remaining buffered rollouts to disk. Call at end of training."""
    global _write_buffer
    rollout_file = _get_rollout_file()
    if rollout_file is None:
        return
    with _file_lock:
        _flush_buffer(rollout_file)


# Register flush on exit to ensure no data is lost
atexit.register(flush_rollout_buffer)


def _get_rollout_file() -> Optional[str]:
    """Get the rollout log file path from environment and initialize header.

    Checks for output directory in order: ROLLOUT_OUTPUT_DIR, RLTT_OUTPUT_DIR, GRPO_OUTPUT_DIR.
    """
    global _rollout_file, _header_written

    if _rollout_file is None:
        # Check multiple env vars for compatibility with both RLTT and GRPO
        output_dir = (
            os.environ.get("ROLLOUT_OUTPUT_DIR")
            or os.environ.get("RLTT_OUTPUT_DIR")
            or os.environ.get("GRPO_OUTPUT_DIR")
            or "./rl_output"
        )
        os.makedirs(output_dir, exist_ok=True)
        _rollout_file = os.path.join(output_dir, "rollouts.jsonl")

        # Determine experiment type for header
        experiment_type = "RL"
        if os.environ.get("RLTT_OUTPUT_DIR"):
            experiment_type = "RLTT"
        elif os.environ.get("GRPO_OUTPUT_DIR"):
            experiment_type = "GRPO"

        # Initialize file with header
        if not _header_written:
            try:
                with open(_rollout_file, "w") as f:
                    f.write(json.dumps({
                        "type": "header",
                        "timestamp": datetime.now().isoformat(),
                        "description": f"{experiment_type} training rollouts (deduplicated)",
                    }) + "\n")
                _header_written = True
            except Exception as e:
                logger.warning(f"Could not initialize rollout log file: {e}")

    return _rollout_file


def _log_rollout(
    question: str,
    response: str,
    ground_truth: str,
    reward: float,
    pred: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
):
    """Log a rollout to JSONL file (with deduplication).

    Only logs unique (question, response, ground_truth) combinations
    to avoid duplicates from multiple workers or batches.
    """
    global _seen_rollouts, _prompt_data, _next_prompt_id

    rollout_file = _get_rollout_file()
    if rollout_file is None:
        return

    # Create deduplication key: (question, response, ground_truth)
    dedup_key = (question, response, ground_truth)

    with _file_lock:
        # Skip if we've already seen this exact rollout
        if dedup_key in _seen_rollouts:
            return
        _seen_rollouts.add(dedup_key)

        # Track prompt by (question, ground_truth)
        prompt_key = (question, ground_truth)
        prompt_info = _prompt_data[prompt_key]

        # Assign prompt_id if this is a new prompt
        if prompt_info["prompt_id"] is None:
            _next_prompt_id += 1
            prompt_info["prompt_id"] = _next_prompt_id

        # Increment rollout count for this prompt
        prompt_info["rollout_count"] += 1

        prompt_id = prompt_info["prompt_id"]
        rollout_num = prompt_info["rollout_count"]

        # Build log entry
        entry = {
            "type": "rollout",
            "prompt_id": prompt_id,
            "rollout_num": rollout_num,
            "timestamp": datetime.now().isoformat(),
            "question": question if question else "",
            "response": response if response else "",
            "ground_truth": ground_truth,
            "reward": reward,
            "correct": reward > 0.5,
            "extracted_pred": pred,
        }

        if extra_info:
            entry["extra_info"] = {
                k: v for k, v in extra_info.items()
                if isinstance(v, (str, int, float, bool))
            }

        # Add to buffer and flush if full (batched I/O for efficiency)
        _write_buffer.append(json.dumps(entry))
        if len(_write_buffer) >= _BUFFER_SIZE:
            _flush_buffer(rollout_file)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute reward for a math problem solution with logging.

    This function is compatible with verl's reward manager interface.
    Uses centralized answer_parsing for consistent answer checking.

    Args:
        data_source: Source of the data (e.g., "math")
        solution_str: Model's generated solution
        ground_truth: Gold answer string
        extra_info: Optional additional information

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    # Extract predicted answer from the FIRST \boxed{} (not last)
    pred_answer = extract_first_boxed_answer(solution_str)

    if pred_answer is None:
        reward = 0.0
    elif check_math_answer(pred_answer, ground_truth):
        reward = 1.0
    else:
        reward = 0.0

    # Log the rollout
    question = ""
    if extra_info and "problem" in extra_info:
        question = extra_info["problem"]
    elif extra_info and "question" in extra_info:
        question = extra_info["question"]

    _log_rollout(
        question=question,
        response=solution_str,
        ground_truth=ground_truth,
        reward=reward,
        pred=pred_answer,
        extra_info=extra_info,
    )

    return reward


# Alias for compatibility with different verl versions
compute_reward = compute_score


def compute_math_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute reward for a math problem solution (without logging).

    This function is compatible with verl's reward manager interface.
    Uses centralized answer_parsing for consistent answer checking.

    Args:
        data_source: Source of the data (e.g., "math")
        solution_str: Model's generated solution
        ground_truth: Gold answer string
        extra_info: Optional additional information (unused)

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    # Extract predicted answer from the FIRST \boxed{} (not last)
    pred_answer = extract_first_boxed_answer(solution_str)

    if pred_answer is None:
        return 0.0

    # Check if correct using centralized math_utils
    if check_math_answer(pred_answer, ground_truth):
        return 1.0

    return 0.0


def create_math_reward_func(log_rollouts: bool = False) -> Callable:
    """Create a reward function for MATH problems.

    Returns a function compatible with RL trainer interface.

    Args:
        log_rollouts: If True, use compute_score (with logging).
                      If False, use compute_math_reward (no logging).

    Returns:
        Reward function that takes (prompts, completions, answer) and returns List[float].
    """
    reward_fn = compute_score if log_rollouts else compute_math_reward

    def reward_func(prompts: List, completions: List, answer: List, **kwargs) -> List[float]:
        """Compute rewards based on answer correctness.

        Args:
            prompts: List of prompts (not used)
            completions: List of model completions
            answer: List of gold answers

        Returns:
            List of float rewards (1.0 for correct, 0.0 for incorrect)
        """
        # Handle different completion formats
        if isinstance(completions[0], list) and isinstance(completions[0][0], dict):
            responses = [completion[0]["content"] for completion in completions]
        elif isinstance(completions[0], str):
            responses = completions
        else:
            responses = [str(c) for c in completions]

        rewards = []
        for resp, gold in zip(responses, answer):
            reward = reward_fn(
                data_source="math",
                solution_str=resp,
                ground_truth=gold or "",
            )
            rewards.append(reward)

        return rewards

    return reward_func


def truncate_responses_at_first_boxed(batch, tokenizer) -> Dict[str, float]:
    """Truncate response_mask and reward placement to end at first \\boxed{...}.

    For each sample in the batch, decodes the response tokens, finds where the
    first \\boxed{...} ends, and zeros out the response_mask for all tokens after
    that position. Also moves the reward (token_level_scores) from the old last
    valid token to the new truncated last valid token.

    This ensures that loss functions only operate on the reasoning trace up to
    and including the first boxed answer.

    Args:
        batch: DataProto object with batch keys:
            - 'responses': (batch_size, response_length) token IDs
            - 'prompts': (batch_size, prompt_length) token IDs
            - 'attention_mask': (batch_size, total_length) binary mask
            - 'response_mask': (batch_size, response_length) binary mask
            - 'token_level_scores': (batch_size, response_length) reward tensor
        tokenizer: HuggingFace tokenizer for decoding/encoding.

    Returns:
        Dict with truncation metrics:
            - 'truncation/mean_tokens_dropped': average tokens removed per truncated sample
            - 'truncation/pct_truncated': percentage of samples that were truncated
            - 'truncation/mean_new_response_length': average response length after truncation
    """
    import torch

    response_ids = batch.batch["responses"]         # (batch_size, response_length)
    prompt_ids = batch.batch["prompts"]              # (batch_size, prompt_length)
    attention_mask = batch.batch["attention_mask"]    # (batch_size, total_length)
    response_mask = batch.batch["response_mask"]     # (batch_size, response_length)
    prompt_length = prompt_ids.shape[-1]

    has_scores = "token_level_scores" in batch.batch.keys()
    if has_scores:
        scores = batch.batch["token_level_scores"]   # (batch_size, response_length)

    batch_size = response_ids.shape[0]
    num_truncated = 0
    total_tokens_dropped = 0
    new_lengths = []

    for i in range(batch_size):
        # Get valid response length from attention_mask
        valid_response_length = attention_mask[i, prompt_length:].sum().item()
        valid_response_length = int(valid_response_length)

        if valid_response_length == 0:
            new_lengths.append(0)
            continue

        # Decode the valid response tokens
        valid_ids = response_ids[i, :valid_response_length]
        response_str = tokenizer.decode(valid_ids, skip_special_tokens=True)

        # Find end of first \boxed{...}
        boxed_end_char = find_first_boxed_end(response_str)

        if boxed_end_char is None:
            # No \boxed{} found - keep full response
            new_lengths.append(valid_response_length)
            continue

        # Truncate the text at the boxed end position
        truncated_text = response_str[:boxed_end_char]

        # Re-encode to find the token boundary
        # We encode the truncated text and count how many tokens it produces
        truncated_ids = tokenizer.encode(truncated_text, add_special_tokens=False)
        new_valid_length = len(truncated_ids)

        # Clamp to not exceed original valid length
        new_valid_length = min(new_valid_length, valid_response_length)

        if new_valid_length >= valid_response_length:
            # No truncation needed (boxed end is at/beyond valid response)
            new_lengths.append(valid_response_length)
            continue

        # Truncation needed
        num_truncated += 1
        tokens_dropped = valid_response_length - new_valid_length
        total_tokens_dropped += tokens_dropped
        new_lengths.append(new_valid_length)

        # Zero out response_mask after the truncated position
        response_mask[i, new_valid_length:] = 0

        # Move reward from old last valid position to new last valid position
        if has_scores and new_valid_length > 0:
            old_reward_pos = valid_response_length - 1
            new_reward_pos = new_valid_length - 1
            if old_reward_pos != new_reward_pos:
                reward_val = scores[i, old_reward_pos].item()
                scores[i, old_reward_pos] = 0.0
                scores[i, new_reward_pos] = reward_val

    # Compute metrics
    metrics = {}
    if num_truncated > 0:
        metrics["truncation/mean_tokens_dropped"] = total_tokens_dropped / num_truncated
    else:
        metrics["truncation/mean_tokens_dropped"] = 0.0
    metrics["truncation/pct_truncated"] = num_truncated / batch_size * 100.0
    metrics["truncation/mean_new_response_length"] = sum(new_lengths) / max(len(new_lengths), 1)

    if num_truncated > 0:
        logger.info(
            f"Truncated {num_truncated}/{batch_size} responses at first \\boxed{{}}. "
            f"Mean tokens dropped: {metrics['truncation/mean_tokens_dropped']:.1f}"
        )

    return metrics
