"""
Core LLOPSD (Latent-Loop On-Policy Self-Distillation) algorithm functions.

Implements the cross-loop distillation objective for LoopLMs (Equation 1 from paper):

    L_loop-distill(x,y) = (1/|y|) sum_{t=1}^{|y|} sum_{r=1}^{R~} alpha_r
        * D(p_S^(r)(.|x, y<t) || stopgrad(p_T^(phi(r))(.|x, c, y<t)))

Where:
    R~ = number of student loops, R = number of teacher loops
    phi: {1,...,R~} -> {1,...,R} is a loop mapping function
    alpha_r are per-loop weights (summing to 1)
    D(. || .) is a divergence measure (forward KL, reverse KL, or JSD)
    stopgrad blocks gradients through the teacher
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Literal, Dict, Tuple


def compute_loop_mapping(student_loops: int, teacher_loops: int, strategy: str = "shift") -> List[int]:
    """Compute loop mapping phi: {1,...,R~} -> {1,...,R}.

    Maps each student loop index to a corresponding teacher loop index, enabling
    cross-loop distillation between models running different numbers of recurrent loops.

    Args:
        student_loops: R~ (number of student loops). Must be >= 1.
        teacher_loops: R (number of teacher loops). Must be >= student_loops.
        strategy: One of "shift", "linear", or "fixed".

    Returns:
        List of length R~ where element r maps student loop r to teacher loop phi(r).
        Uses 0-indexed loops internally (i.e., values are in [0, R-1]).

    Strategies:
        shift:  phi(r) = r + (R - R~).
                Maps student loops to the last R~ teacher loops.
                E.g., (R, R~) = (4, 2): [0->2, 1->3]
        linear: phi(r) = round(r * (R-1) / (R~-1)).
                Linearly interpolates student loops across the full teacher range.
                E.g., (R, R~) = (4, 2): [0->0, 1->3]
                Special case: if R~ == 1, maps to the last teacher loop.
        fixed:  phi(r) = R-1 for all r.
                Every student loop distills from the final teacher loop.
                E.g., (R, R~) = (4, 2): [0->3, 1->3]

    Raises:
        ValueError: If student_loops < 1, teacher_loops < 1, or
                    teacher_loops < student_loops, or strategy is unknown.
    """
    if student_loops < 1:
        raise ValueError(f"student_loops must be >= 1, got {student_loops}")
    if teacher_loops < 1:
        raise ValueError(f"teacher_loops must be >= 1, got {teacher_loops}")
    if teacher_loops < student_loops:
        raise ValueError(
            f"teacher_loops ({teacher_loops}) must be >= student_loops ({student_loops})"
        )

    R = teacher_loops
    R_tilde = student_loops

    if strategy == "shift":
        # Student loop r maps to teacher loop r + (R - R~)
        offset = R - R_tilde
        mapping = [r + offset for r in range(R_tilde)]

    elif strategy == "linear":
        # Linearly interpolate student loops onto teacher loop range
        if R_tilde == 1:
            # Single student loop maps to the final teacher loop
            mapping = [R - 1]
        else:
            mapping = [round(r * (R - 1) / (R_tilde - 1)) for r in range(R_tilde)]

    elif strategy == "fixed":
        # Every student loop maps to the final teacher loop
        mapping = [R - 1] * R_tilde

    else:
        raise ValueError(
            f"Unknown loop mapping strategy: '{strategy}'. "
            f"Supported strategies: 'shift', 'linear', 'fixed'."
        )

    return mapping


def compute_student_weights(
    student_loops: int,
    schedule: str = "uniform",
    gamma: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute alpha_r weights over student loops.

    Determines how much each student loop contributes to the total distillation loss.
    Weights are normalized to sum to 1.

    Args:
        student_loops: R~ (number of student loops). Must be >= 1.
        schedule: One of "uniform", "late_heavy", or "terminal_only".
        gamma: Exponent for the "late_heavy" schedule (alpha_r proportional to r^gamma).
               Only used when schedule="late_heavy". Must be >= 0.
        device: Torch device for the output tensor. Defaults to CPU.

    Returns:
        Tensor of shape (R~,) with non-negative weights summing to 1.

    Schedules:
        uniform:       alpha_r = 1 / R~ for all r.
        late_heavy:    alpha_r = r^gamma / sum_{s=1}^{R~} s^gamma
                       where r is 1-indexed (r in {1, ..., R~}).
                       gamma=0 recovers uniform; gamma=1 gives linear weighting;
                       gamma=2 gives quadratic weighting toward later loops.
        terminal_only: alpha_{R~} = 1, all others = 0.
                       Equivalent to distilling only from the final student loop.

    Raises:
        ValueError: If student_loops < 1, gamma < 0, or schedule is unknown.
    """
    if student_loops < 1:
        raise ValueError(f"student_loops must be >= 1, got {student_loops}")

    R_tilde = student_loops

    if schedule == "uniform":
        weights = torch.ones(R_tilde, device=device) / R_tilde

    elif schedule == "late_heavy":
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0 for late_heavy schedule, got {gamma}")
        # 1-indexed loop indices: {1, 2, ..., R~}
        indices = torch.arange(1, R_tilde + 1, dtype=torch.float64, device=device)
        raw_weights = indices.pow(gamma)
        weights = (raw_weights / raw_weights.sum()).to(torch.float32)

    elif schedule == "terminal_only":
        weights = torch.zeros(R_tilde, device=device)
        weights[-1] = 1.0

    else:
        raise ValueError(
            f"Unknown weight schedule: '{schedule}'. "
            f"Supported schedules: 'uniform', 'late_heavy', 'terminal_only'."
        )

    return weights


def compute_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    divergence_type: str = "forward_kl",
) -> torch.Tensor:
    """Compute D(p_S || stopgrad(p_T)) per token position.

    Computes a divergence measure between student and teacher distributions
    at a single token position. All computation is done in log-domain for
    numerical stability.

    Args:
        student_logits: Shape (batch_size, vocab_size). Raw student logits for one
            token position.
        teacher_logits: Shape (batch_size, vocab_size). Raw teacher logits for one
            token position. Should already be detached (stopgrad applied externally),
            but this function also calls .detach() defensively.
        divergence_type: One of "forward_kl", "reverse_kl", or "jsd".

    Returns:
        Per-sample divergence of shape (batch_size,). Values are non-negative
        (up to floating-point precision).

    Divergences (all computed in log-domain):
        forward_kl: KL(p_T || p_S) = sum_v p_T(v) * (log p_T(v) - log p_S(v))
            Teacher is the "target" distribution, student is the "approximation".
            This is the standard distillation direction (mode-covering).
        reverse_kl: KL(p_S || p_T) = sum_v p_S(v) * (log p_S(v) - log p_T(v))
            Student is the "target", teacher is the "approximation" (mode-seeking).
        jsd: JSD(p_S, p_T) = 0.5 * KL(p_S || M) + 0.5 * KL(p_T || M)
            where M = 0.5 * (p_S + p_T). Symmetric and bounded by log(2).

    Raises:
        ValueError: If divergence_type is unknown.
    """
    # Defensive detach on teacher
    teacher_logits = teacher_logits.detach()

    # Compute log-probabilities for numerical stability
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)

    if divergence_type == "forward_kl":
        # KL(p_T || p_S) = sum_v p_T(v) * (log p_T(v) - log p_S(v))
        p_t = log_p_t.exp()
        # Using the identity: KL(P||Q) = sum P * (log P - log Q)
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
        return kl

    elif divergence_type == "reverse_kl":
        # KL(p_S || p_T) = sum_v p_S(v) * (log p_S(v) - log p_T(v))
        p_s = log_p_s.exp()
        kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
        return kl

    elif divergence_type == "jsd":
        # JSD(p_S, p_T) = 0.5 * KL(p_S || M) + 0.5 * KL(p_T || M)
        # where M = 0.5 * (p_S + p_T)
        # Compute log M in log-domain: log(0.5 * (exp(log_p_s) + exp(log_p_t)))
        #   = logsumexp(log_p_s, log_p_t) + log(0.5)
        #   = logsumexp(log_p_s, log_p_t) - log(2)
        log_m = torch.logsumexp(torch.stack([log_p_s, log_p_t], dim=0), dim=0) - torch.log(
            torch.tensor(2.0, dtype=student_logits.dtype, device=student_logits.device)
        )

        p_s = log_p_s.exp()
        p_t = log_p_t.exp()

        # KL(p_S || M) = sum_v p_S(v) * (log p_S(v) - log M(v))
        kl_s_m = (p_s * (log_p_s - log_m)).sum(dim=-1)
        # KL(p_T || M) = sum_v p_T(v) * (log p_T(v) - log M(v))
        kl_t_m = (p_t * (log_p_t - log_m)).sum(dim=-1)

        jsd = 0.5 * kl_s_m + 0.5 * kl_t_m
        return jsd

    else:
        raise ValueError(
            f"Unknown divergence type: '{divergence_type}'. "
            f"Supported types: 'forward_kl', 'reverse_kl', 'jsd'."
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Compute masked mean, with fallback if verl utility is unavailable.

    Args:
        values: Tensor of values to average.
        mask: Binary mask of same shape as values. 1 = include, 0 = exclude.
        dim: Dimension along which to compute the mean. None = all elements.

    Returns:
        Masked mean, avoiding division by zero with an epsilon term.
    """
    try:
        from verl.utils.torch_functional import masked_mean
        return masked_mean(values, mask, axis=dim)
    except ImportError:
        masked_values = values * mask
        return masked_values.sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


def compute_llopsd_loss(
    student_per_loop_logits: List[torch.Tensor],
    teacher_per_loop_logits: List[torch.Tensor],
    loop_mapping: List[int],
    student_weights: torch.Tensor,
    divergence_type: str,
    response_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute full LLOPSD loss (Equation 1 from the paper).

    L = (1/|y|) sum_{t=1}^{|y|} sum_{r=1}^{R~} alpha_r
        * D(p_S^(r)(.|x, y<t) || stopgrad(p_T^(phi(r))(.|x, c, y<t)))

    The outer sum over tokens t is masked by response_mask and averaged over
    the number of valid response tokens per sample (then averaged over the batch).

    Args:
        student_per_loop_logits: List of R~ tensors, each of shape
            (batch_size, seq_len, vocab_size). Student logits at each loop iteration.
        teacher_per_loop_logits: List of R tensors, each of shape
            (batch_size, seq_len, vocab_size). Teacher logits at each loop iteration.
            Should already be detached (stopgrad); detach is also applied defensively.
        loop_mapping: List of R~ ints mapping student loop index to teacher loop index.
            E.g., [2, 3] means student loop 0 maps to teacher loop 2, student loop 1
            maps to teacher loop 3.
        student_weights: alpha_r weights of shape (R~,) summing to 1.
        divergence_type: "forward_kl", "reverse_kl", or "jsd".
        response_mask: Binary mask of shape (batch_size, seq_len). 1 for valid
            response tokens, 0 for padding / prompt tokens.

    Returns:
        loss: Scalar loss value (mean over batch).
        metrics: Dictionary with diagnostic information including:
            - "llopsd/loss": the scalar loss value
            - "llopsd/div_loop_{r}": per-loop mean divergence for each student loop r
            - "llopsd/response_length_mean": mean number of response tokens
            - "llopsd/weight_sum": sum of student weights (should be ~1.0)

    Raises:
        ValueError: If lengths of student_per_loop_logits and loop_mapping disagree,
            or if any teacher loop index in loop_mapping is out of range.
    """
    R_tilde = len(student_per_loop_logits)
    R = len(teacher_per_loop_logits)

    if len(loop_mapping) != R_tilde:
        raise ValueError(
            f"loop_mapping length ({len(loop_mapping)}) must match number of student loops "
            f"({R_tilde})"
        )
    for r, phi_r in enumerate(loop_mapping):
        if phi_r < 0 or phi_r >= R:
            raise ValueError(
                f"loop_mapping[{r}] = {phi_r} is out of range for {R} teacher loops "
                f"(expected 0 <= phi(r) < {R})"
            )

    if student_weights.shape[0] != R_tilde:
        raise ValueError(
            f"student_weights length ({student_weights.shape[0]}) must match number of "
            f"student loops ({R_tilde})"
        )

    batch_size, seq_len = response_mask.shape
    device = response_mask.device
    dtype = student_per_loop_logits[0].dtype

    # Number of valid response tokens per sample, shape (batch_size,)
    response_lengths = response_mask.sum(dim=-1)  # (batch_size,)

    metrics: Dict[str, float] = {}
    metrics["llopsd/response_length_mean"] = response_lengths.float().mean().item()
    metrics["llopsd/weight_sum"] = student_weights.sum().item()

    # Accumulate weighted loss across loops
    # We compute: for each sample, (1/|y|) * sum_r alpha_r * sum_t mask_t * D(...)
    # Then average over the batch.

    # total_weighted_div: (batch_size,) accumulates sum_r alpha_r * sum_t mask_t * D(...)
    total_weighted_div = torch.zeros(batch_size, device=device, dtype=torch.float32)

    for r in range(R_tilde):
        phi_r = loop_mapping[r]
        alpha_r = student_weights[r]

        # Skip if weight is zero (e.g., terminal_only schedule)
        if alpha_r.item() == 0.0:
            metrics[f"llopsd/div_loop_{r}"] = 0.0
            continue

        student_logits_r = student_per_loop_logits[r]  # (batch_size, seq_len, vocab_size)
        teacher_logits_r = teacher_per_loop_logits[phi_r].detach()  # (batch_size, seq_len, vocab_size)

        # Compute per-token divergence for this loop pair
        # We iterate over the sequence dimension for memory efficiency with large vocabs.
        # However, for practical seq_lens, we can vectorize by reshaping.
        # Reshape to (batch_size * seq_len, vocab_size) for compute_divergence
        s_flat = student_logits_r.reshape(-1, student_logits_r.shape[-1])  # (B*T, V)
        t_flat = teacher_logits_r.reshape(-1, teacher_logits_r.shape[-1])  # (B*T, V)

        div_flat = compute_divergence(s_flat, t_flat, divergence_type)  # (B*T,)
        div_per_token = div_flat.reshape(batch_size, seq_len)  # (batch_size, seq_len)

        # Mask out non-response tokens
        div_masked = div_per_token * response_mask  # (batch_size, seq_len)

        # Per-sample sum of divergence over response tokens
        div_sum_per_sample = div_masked.sum(dim=-1)  # (batch_size,)

        # Accumulate alpha_r * div_sum for this loop
        total_weighted_div = total_weighted_div + alpha_r * div_sum_per_sample

        # Log per-loop mean divergence (averaged over valid tokens and batch)
        loop_mean_div = _masked_mean(div_per_token, response_mask)
        metrics[f"llopsd/div_loop_{r}"] = loop_mean_div.item()

    # Normalize by response length per sample: (1/|y|) * total_weighted_div
    # Avoid division by zero for samples with no response tokens
    per_sample_loss = total_weighted_div / response_lengths.clamp(min=1.0)  # (batch_size,)

    # Average over batch
    loss = per_sample_loss.mean()

    metrics["llopsd/loss"] = loss.item()

    return loss, metrics


def construct_teacher_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ground_truths: list,
    tokenizer,
    teacher_context: str = "opsd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct teacher input with privileged context (OPSD-style).

    For OPSD-style teacher context, the ground-truth answer is appended to the
    prompt so that the teacher has access to privileged information when generating
    its logits. The teacher sees:

        [prompt tokens] + [privileged context tokens] + [response tokens]

    Since verl typically stores prompt and response separately, and inserting tokens
    into the middle of a padded batch is complex, this initial implementation takes
    a simpler approach: it decodes each sequence, inserts the ground-truth text,
    re-tokenizes, and pads back to a uniform length.

    NOTE: This function assumes left-padded input_ids (as is standard in verl for
    generation). The attention_mask indicates which positions are real tokens (1)
    versus padding (0).

    Args:
        input_ids: (batch_size, seq_len) - Original input token IDs
            (prompt + response, left-padded).
        attention_mask: (batch_size, seq_len) - Binary mask where 1 = real token,
            0 = padding.
        ground_truths: List of str of length batch_size. Ground-truth answers to
            inject as privileged context for the teacher.
        tokenizer: HuggingFace tokenizer for encoding/decoding.
        teacher_context: Context injection strategy. Currently only "opsd" is
            supported.

    Returns:
        teacher_input_ids: (batch_size, new_seq_len) - Teacher input with
            privileged context injected.
        teacher_attention_mask: (batch_size, new_seq_len) - Corresponding
            attention mask.

    Raises:
        ValueError: If teacher_context is not a supported strategy.

    TODO:
        - This re-tokenization approach is expensive. A more efficient implementation
          would directly manipulate token IDs: find the prompt/response boundary from
          metadata, encode only the ground-truth snippet, and insert/concatenate the
          token ID tensors with proper padding adjustment.
        - Support additional teacher_context modes beyond "opsd" (e.g., "cot" for
          chain-of-thought hints, "rationale" for step-by-step hints).
        - Handle edge cases where injecting context causes the sequence to exceed
          the model's maximum context length (truncation strategy).
    """
    if teacher_context != "opsd":
        raise ValueError(
            f"Unsupported teacher_context: '{teacher_context}'. "
            f"Currently only 'opsd' is supported."
        )

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if len(ground_truths) != batch_size:
        raise ValueError(
            f"Number of ground_truths ({len(ground_truths)}) must match "
            f"batch_size ({batch_size})"
        )

    # Decode each sequence to text, stripping padding
    teacher_texts = []
    for i in range(batch_size):
        # Extract non-padding tokens
        valid_mask = attention_mask[i].bool()
        valid_ids = input_ids[i][valid_mask]

        # Decode to text
        text = tokenizer.decode(valid_ids, skip_special_tokens=False)

        # For OPSD: we prepend the ground-truth answer as privileged context
        # Format: "[Ground Truth: {answer}]\n{original_text}"
        # This simple approach puts context at the beginning of the sequence.
        # A more sophisticated approach would insert it between prompt and response.
        gt_context = f"[Ground Truth: {ground_truths[i]}]\n"
        teacher_text = gt_context + text
        teacher_texts.append(teacher_text)

    # Re-tokenize with padding
    teacher_encoded = tokenizer(
        teacher_texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    teacher_input_ids = teacher_encoded["input_ids"].to(device)
    teacher_attention_mask = teacher_encoded["attention_mask"].to(device)

    return teacher_input_ids, teacher_attention_mask
