"""
Custom LLOPSD Actor for verl that implements cross-loop distillation.

This actor extends DataParallelPPOActor to support the LLOPSD (Latent-Loop
On-Policy Self-Distillation) objective, which distills knowledge from a teacher
model running R loops to a student model running R~ < R loops using per-loop
KL divergence between logit distributions.

Key difference from RLTT actor:
  - RLTT: computes per-loop log-probs, aggregates with weights, uses REINFORCE
    loss with advantages.
  - LLOPSD: computes per-loop LOGITS for both student AND teacher, then computes
    cross-loop KL divergence (or reverse KL / JSD) between mapped loop pairs.
"""

import copy
import inspect
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .llopsd_algos import (
    compute_divergence,
    compute_llopsd_loss,
    compute_loop_mapping,
    compute_student_weights,
    construct_teacher_input,
)

__all__ = ["LLOPSDDataParallelPPOActor"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class LLOPSDDataParallelPPOActor(DataParallelPPOActor):
    """LLOPSD-aware Actor that computes cross-loop distillation loss.

    This actor extends the standard DataParallelPPOActor to:
    1. Run the student model with R~ loops and collect per-loop logits.
    2. Run the teacher model with R loops (under torch.no_grad) and collect
       per-loop logits at the mapped teacher loop indices.
    3. Compute the weighted cross-loop divergence loss between student and
       teacher loop pairs using a configurable divergence measure.
    4. Optionally maintain an EMA copy of the student as the teacher.

    The teacher can operate in three modes:
      - "same": teacher IS the student model, but runs with more loops (R > R~).
        This is the simplest cross-loop self-distillation setup.
      - "frozen": a separate frozen copy of the model (no gradient updates).
      - "ema": an exponential moving average of the student parameters,
        updated after each optimizer step.
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        teacher_loops: int = 4,
        student_loops: int = 2,
        loop_mapping_strategy: str = "shift",
        weight_schedule: str = "uniform",
        weight_gamma: float = 1.0,
        divergence_type: str = "forward_kl",
        teacher_mode: str = "same",
        teacher_module: Optional[nn.Module] = None,
        ema_decay: float = 0.999,
        teacher_context: str = "opsd",
    ):
        """Initialize LLOPSD actor.

        Args:
            config: Actor configuration from verl (ActorConfig or dict-like).
            actor_module: The student model (looped/recurrent model).
            actor_optimizer: Optimizer for training the student.
            teacher_loops: R -- number of recurrent loops for the teacher.
            student_loops: R~ -- number of recurrent loops for the student.
                Must satisfy 1 <= R~ <= R.
            loop_mapping_strategy: How to map student loops to teacher loops.
                One of "shift", "linear", "fixed". See llopsd_algos.compute_loop_mapping.
            weight_schedule: How to weight each student loop's contribution to
                the loss. One of "uniform", "late_heavy", "terminal_only".
                See llopsd_algos.compute_student_weights.
            weight_gamma: Exponent for "late_heavy" schedule (alpha_r ~ r^gamma).
            divergence_type: Divergence measure between student and teacher
                distributions. One of "forward_kl", "reverse_kl", "jsd".
            teacher_mode: How the teacher model is managed:
                "same"   -- teacher IS the student, but with more loops.
                "frozen" -- a separate frozen copy passed via teacher_module.
                "ema"    -- an EMA copy of the student, updated after each step.
            teacher_module: Pre-built teacher module for "frozen" or "ema" modes.
                If teacher_mode is "ema" and teacher_module is None, a deep copy
                of actor_module will be created automatically.
            ema_decay: EMA decay factor tau for the "ema" teacher mode.
                theta' <- tau * theta' + (1 - tau) * theta.
            teacher_context: How to provide context to the teacher model.
                One of "opsd" (privileged context from ground truth) or "same"
                (same input as student).
        """
        super().__init__(config, actor_module, actor_optimizer)

        # -------------------------------------------------------------------
        # Store LLOPSD configuration
        # -------------------------------------------------------------------
        self.teacher_loops = teacher_loops
        self.student_loops = student_loops
        self.loop_mapping_strategy = loop_mapping_strategy
        self.weight_schedule = weight_schedule
        self.weight_gamma = weight_gamma
        self.divergence_type = divergence_type
        self.teacher_mode = teacher_mode
        self.ema_decay = ema_decay
        self.teacher_context = teacher_context

        # -------------------------------------------------------------------
        # Validate teacher_mode
        # -------------------------------------------------------------------
        valid_teacher_modes = ("same", "frozen", "ema")
        if self.teacher_mode not in valid_teacher_modes:
            raise ValueError(
                f"Invalid teacher_mode '{self.teacher_mode}'. "
                f"Must be one of {valid_teacher_modes}."
            )

        # -------------------------------------------------------------------
        # Compute loop mapping and student weights (static, computed once)
        # -------------------------------------------------------------------
        self.loop_mapping = compute_loop_mapping(
            student_loops=self.student_loops,
            teacher_loops=self.teacher_loops,
            strategy=self.loop_mapping_strategy,
        )
        self.student_weights = compute_student_weights(
            student_loops=self.student_loops,
            schedule=self.weight_schedule,
            gamma=self.weight_gamma,
            device=torch.cuda.current_device(),
        )

        # -------------------------------------------------------------------
        # Check if the model supports native per-loop logit computation
        # -------------------------------------------------------------------
        self.model_supports_per_loop = self._check_model_rltt_support()

        # -------------------------------------------------------------------
        # Teacher module setup
        # -------------------------------------------------------------------
        if self.teacher_mode == "same":
            # Teacher IS the student; no separate module needed.
            self._teacher_module = None
        elif self.teacher_mode == "frozen":
            if teacher_module is None:
                raise ValueError(
                    "teacher_mode='frozen' requires a pre-built teacher_module."
                )
            self._teacher_module = teacher_module
            # Freeze all parameters
            for param in self._teacher_module.parameters():
                param.requires_grad = False
            self._teacher_module.eval()
        elif self.teacher_mode == "ema":
            if teacher_module is not None:
                self._teacher_module = teacher_module
            else:
                # Create a deep copy for EMA
                print(
                    "LLOPSD Actor: Creating EMA teacher via deep copy of actor_module. "
                    "This may use significant memory.",
                    flush=True,
                )
                self._teacher_module = copy.deepcopy(actor_module)
            # Freeze EMA parameters (gradients are never needed)
            for param in self._teacher_module.parameters():
                param.requires_grad = False
            self._teacher_module.eval()

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(
                f"LLOPSD Actor initialized:\n"
                f"  teacher_loops (R)       = {self.teacher_loops}\n"
                f"  student_loops (R~)      = {self.student_loops}\n"
                f"  loop_mapping_strategy   = {self.loop_mapping_strategy}\n"
                f"  loop_mapping            = {self.loop_mapping}\n"
                f"  weight_schedule         = {self.weight_schedule}\n"
                f"  weight_gamma            = {self.weight_gamma}\n"
                f"  student_weights         = {self.student_weights.tolist()}\n"
                f"  divergence_type         = {self.divergence_type}\n"
                f"  teacher_mode            = {self.teacher_mode}\n"
                f"  teacher_context         = {self.teacher_context}\n"
                f"  ema_decay               = {self.ema_decay}\n"
                f"  model_supports_per_loop = {self.model_supports_per_loop}",
                flush=True,
            )

    # ======================================================================
    # Model introspection
    # ======================================================================

    def _check_model_rltt_support(self) -> bool:
        """Check if the model supports native per-loop logit computation.

        Unwraps FSDP / PEFT wrappers to inspect the forward() signature of the
        underlying model.  Returns True if the model accepts both
        ``return_per_loop_logits`` and ``return_exit_pdf`` keyword arguments.
        """
        model = self.actor_module

        # Iteratively unwrap FSDP, torch.compile, PEFT, etc.
        max_unwrap = 10
        for _ in range(max_unwrap):
            unwrapped = False
            if hasattr(model, "_fsdp_wrapped_module"):
                model = model._fsdp_wrapped_module
                unwrapped = True
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
                unwrapped = True
            if hasattr(model, "module") and not isinstance(
                getattr(model, "module", None), type(None)
            ):
                inner = model.module
                if inner is not model and not callable(inner):
                    model = inner
                    unwrapped = True
            if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                model = model.base_model.model
                unwrapped = True
            elif hasattr(model, "get_base_model"):
                try:
                    base = model.get_base_model()
                    if base is not model:
                        model = base
                        unwrapped = True
                except Exception:
                    pass
            if not unwrapped:
                break

        try:
            sig = inspect.signature(model.forward)
            params = sig.parameters
            has_logits = "return_per_loop_logits" in params
            has_exit_pdf = "return_exit_pdf" in params
            result = has_logits and has_exit_pdf
            if torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True:
                print(
                    f"[LLOPSD] _check_model_rltt_support: "
                    f"model_type={type(model).__name__}, "
                    f"return_per_loop_logits={has_logits}, "
                    f"return_exit_pdf={has_exit_pdf} -> {result}",
                    flush=True,
                )
            return result
        except Exception as e:
            print(f"[LLOPSD] Exception checking forward signature: {e}", flush=True)
            return False

    # ======================================================================
    # Teacher access
    # ======================================================================

    def _get_teacher_module(self) -> nn.Module:
        """Return the teacher model based on the configured teacher_mode.

        - "same":   returns self.actor_module (the student model itself).
        - "frozen": returns the frozen self._teacher_module.
        - "ema":    returns the EMA self._teacher_module.
        """
        if self.teacher_mode == "same":
            return self.actor_module
        else:
            if self._teacher_module is None:
                raise RuntimeError(
                    f"teacher_mode='{self.teacher_mode}' but _teacher_module is None."
                )
            return self._teacher_module

    # ======================================================================
    # EMA update
    # ======================================================================

    def update_ema_teacher(self) -> None:
        """Update the EMA teacher parameters after an optimizer step.

        theta' <- tau * theta' + (1 - tau) * theta

        where tau = self.ema_decay, theta' = teacher params, theta = student params.
        This should only be called when teacher_mode == "ema".
        """
        if self.teacher_mode != "ema":
            return
        if self._teacher_module is None:
            warnings.warn(
                "update_ema_teacher called but _teacher_module is None. Skipping."
            )
            return

        tau = self.ema_decay
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self._teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data
                if teacher_param.device != student_data.device:
                    student_data = student_data.to(device=teacher_param.device)
                teacher_param.data.mul_(tau).add_(student_data, alpha=(1.0 - tau))

    # ======================================================================
    # Forward passes: student and teacher
    # ======================================================================

    def _set_model_ut_steps(self, model: nn.Module, num_steps: int) -> Optional[int]:
        """Temporarily set total_ut_steps on the model config.

        Returns the original value so it can be restored, or None if the
        attribute does not exist.
        """
        # Try common config paths
        config_obj = getattr(model, "config", None)
        if config_obj is not None and hasattr(config_obj, "total_ut_steps"):
            original = config_obj.total_ut_steps
            config_obj.total_ut_steps = num_steps
            return original
        return None

    def _restore_model_ut_steps(
        self, model: nn.Module, original_value: Optional[int]
    ) -> None:
        """Restore total_ut_steps on the model config."""
        if original_value is not None:
            config_obj = getattr(model, "config", None)
            if config_obj is not None:
                config_obj.total_ut_steps = original_value

    def _forward_student_micro_batch(
        self,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> List[torch.Tensor]:
        """Run the student forward pass with R~ loops and collect per-loop logits.

        Args:
            micro_batch: Dict with keys 'input_ids', 'attention_mask',
                'position_ids', 'responses', and optionally 'multi_modal_inputs'.
            temperature: Temperature for logit scaling.

        Returns:
            List of R~ tensors, each of shape (batch_size, response_length, vocab_size).
            These are the student logits at each loop iteration, sliced to the
            response portion and divided by temperature.
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]],
                    dim=0,
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.model_supports_per_loop:
                # ---------------------------------------------------------
                # Native per-loop logits path
                # ---------------------------------------------------------
                # Temporarily set the model to use student_loops
                original_ut = self._set_model_ut_steps(
                    self.actor_module, self.student_loops
                )

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_per_loop_logits=True,
                    return_exit_pdf=False,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                self._restore_model_ut_steps(self.actor_module, original_ut)

                per_loop_logits_raw = output.per_loop_logits  # List of R~ tensors
                del output

                # Slice each to response portion and apply temperature
                per_loop_logits = []
                for i, logits in enumerate(per_loop_logits_raw):
                    logits = logits / temperature
                    logits = logits[
                        :, -response_length - 1 : -1, :
                    ]  # (bs, response_length, vocab)
                    per_loop_logits.append(logits)
                    per_loop_logits_raw[i] = None  # Allow GC

                del per_loop_logits_raw

            else:
                # ---------------------------------------------------------
                # Fallback: run model R~ times with different total_ut_steps
                # ---------------------------------------------------------
                per_loop_logits = []
                for t in range(1, self.student_loops + 1):
                    original_ut = self._set_model_ut_steps(self.actor_module, t)

                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                    )

                    self._restore_model_ut_steps(self.actor_module, original_ut)

                    logits = output.logits / temperature
                    logits = logits[:, -response_length - 1 : -1, :]
                    per_loop_logits.append(logits)
                    del output

        return per_loop_logits

    def _forward_teacher_micro_batch(
        self,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> List[torch.Tensor]:
        """Run the teacher forward pass with R loops and collect per-loop logits.

        All computation is performed under torch.no_grad().  Only the logits at
        teacher loop indices that appear in self.loop_mapping are retained (to
        save memory); the rest are discarded.

        Args:
            micro_batch: Dict with keys 'input_ids', 'attention_mask',
                'position_ids', 'responses', and optionally 'multi_modal_inputs'.
                The input_ids may include privileged context for the teacher.
            temperature: Temperature for logit scaling.

        Returns:
            List of R tensors (some may be None for unmapped loops), each of
            shape (batch_size, response_length, vocab_size).  Indexed by teacher
            loop index.  Only the entries referenced by self.loop_mapping are
            guaranteed to be non-None.
        """
        teacher_module = self._get_teacher_module()
        response_length = micro_batch["responses"].size(-1)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]],
                    dim=0,
                )

        # Determine which teacher loop indices we actually need
        needed_teacher_indices = set(self.loop_mapping)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.model_supports_per_loop:
                # ---------------------------------------------------------
                # Native per-loop logits path
                # ---------------------------------------------------------
                original_ut = self._set_model_ut_steps(
                    teacher_module, self.teacher_loops
                )

                output = teacher_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_per_loop_logits=True,
                    return_exit_pdf=False,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                self._restore_model_ut_steps(teacher_module, original_ut)

                per_loop_logits_raw = output.per_loop_logits
                del output

                # Only keep the teacher loops we need; discard the rest
                per_loop_logits = []
                for i in range(len(per_loop_logits_raw)):
                    if i in needed_teacher_indices:
                        logits = per_loop_logits_raw[i] / temperature
                        logits = logits[:, -response_length - 1 : -1, :]
                        per_loop_logits.append(logits.detach())
                    else:
                        per_loop_logits.append(None)
                    per_loop_logits_raw[i] = None  # Allow GC

                del per_loop_logits_raw

            else:
                # ---------------------------------------------------------
                # Fallback: run model once with full R loops and get final,
                # or run R times. Since we need specific loop indices, we
                # run each needed loop count separately.
                # ---------------------------------------------------------
                per_loop_logits = [None] * self.teacher_loops

                for t_idx in sorted(needed_teacher_indices):
                    # t_idx is 0-indexed; model expects total_ut_steps = t_idx + 1
                    original_ut = self._set_model_ut_steps(
                        teacher_module, t_idx + 1
                    )

                    output = teacher_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                    )

                    self._restore_model_ut_steps(teacher_module, original_ut)

                    logits = output.logits / temperature
                    logits = logits[:, -response_length - 1 : -1, :]
                    per_loop_logits[t_idx] = logits.detach()
                    del output

        return per_loop_logits

    # ======================================================================
    # Main training entry point
    # ======================================================================

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        """Update the student policy using the LLOPSD cross-loop distillation loss.

        For each mini-batch / micro-batch:
        1. Run the student model with R~ loops to get per-loop logits (with gradients).
        2. Run the teacher model with R loops to get per-loop logits (no gradients).
        3. Compute the LLOPSD loss: weighted sum of divergences between mapped
           student-teacher loop pairs.
        4. Backpropagate and step the optimizer.
        5. Optionally update the EMA teacher.

        Args:
            data: DataProto from the verl trainer containing at minimum:
                batch keys: 'input_ids', 'attention_mask', 'position_ids',
                    'responses', 'advantages' (may be unused but present).
                non_tensor_batch keys: 'reward_model' (dict with 'ground_truth').
                meta_info: 'temperature'.

        Returns:
            Dict of training metrics.
        """
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        # ------------------------------------------------------------------
        # Select keys from data
        # ------------------------------------------------------------------
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        # Include advantages and old_log_probs if present (may be needed for
        # combined RL+distillation objectives in future, or for logging)
        if "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if "old_log_probs" in data.batch.keys():
            select_keys.append("old_log_probs")
        if "response_mask" in data.batch.keys():
            select_keys.append("response_mask")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        # Keep reward_model for ground_truth access (used for teacher context)
        if "reward_model" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("reward_model")

        # ------------------------------------------------------------------
        # Build dataloader (mini-batches -> micro-batches)
        # ------------------------------------------------------------------
        if has_multi_modal_inputs:
            num_mini_batches = (
                data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            )
            dataloader = data.select(
                select_keys, non_tensor_select_keys
            ).chunk(num_mini_batches)
        else:
            selected_data = data.select(
                batch_keys=select_keys,
                non_tensor_batch_keys=non_tensor_select_keys,
            )
            dataloader = selected_data.split(self.config.ppo_mini_batch_size)

        metrics: Dict[str, Any] = {}

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data in enumerate(dataloader):
                # Split into micro-batches
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch_data.batch.batch_size[0]
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch_data.select(
                        select_keys, non_tensor_select_keys
                    ).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    # Convert DataProto to dict for rearrange_micro_batches
                    mini_batch_dict = {
                        **mini_batch_data.batch,
                        **mini_batch_data.non_tensor_batch,
                    }
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch_dict, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch_data.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    # -------------------------------------------------------
                    # Prepare micro-batch data
                    # -------------------------------------------------------
                    if isinstance(micro_data, DataProto):
                        micro_dict = {
                            **micro_data.batch.to(torch.cuda.current_device()),
                            **micro_data.non_tensor_batch,
                        }
                    else:
                        micro_dict = micro_data.to(torch.cuda.current_device())

                    responses = micro_dict["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_dict["attention_mask"]

                    # Derive response_mask: prefer explicit key, else use
                    # the tail of the attention_mask
                    if "response_mask" in micro_dict:
                        response_mask = micro_dict["response_mask"]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    micro_batch_size = responses.size(0)

                    # -------------------------------------------------------
                    # Extract ground truths (for potential teacher context)
                    # -------------------------------------------------------
                    ground_truths = None
                    if "reward_model" in micro_dict:
                        rm_info = micro_dict["reward_model"]
                        if isinstance(rm_info, dict) and "ground_truth" in rm_info:
                            ground_truths = rm_info["ground_truth"]
                        elif isinstance(rm_info, (list, tuple)):
                            # Each element is a dict for one sample
                            try:
                                ground_truths = [
                                    item.get("ground_truth", "")
                                    if isinstance(item, dict)
                                    else ""
                                    for item in rm_info
                                ]
                            except Exception:
                                ground_truths = None

                    # -------------------------------------------------------
                    # Student forward pass (R~ loops) -- WITH gradients
                    # -------------------------------------------------------
                    student_per_loop_logits = self._forward_student_micro_batch(
                        micro_dict, temperature
                    )

                    # -------------------------------------------------------
                    # Teacher forward pass (R loops) -- NO gradients
                    # -------------------------------------------------------
                    # For v1, the teacher uses the same input as the student.
                    # The "privileged context" comes from the teacher running
                    # more loops (R > R~), giving it a deeper representation.
                    #
                    # Full OPSD-style context injection (ground truth in input)
                    # can be enabled later by using construct_teacher_input.
                    teacher_micro_dict = micro_dict  # Same input for now

                    if ground_truths is not None and self.teacher_mode != "same":
                        # Log a note that full OPSD context is not yet active
                        # (only on first micro-batch to avoid log spam)
                        if batch_idx == 0 and epoch == 0:
                            rank = (
                                torch.distributed.get_rank()
                                if torch.distributed.is_initialized()
                                else 0
                            )
                            if rank == 0:
                                logger.info(
                                    "LLOPSD: ground_truths available but OPSD-style "
                                    "teacher context injection is not yet enabled. "
                                    "Teacher uses same input as student. "
                                    "Set teacher_context='opsd' in future versions "
                                    "to enable privileged teacher context."
                                )

                    teacher_per_loop_logits = self._forward_teacher_micro_batch(
                        teacher_micro_dict, temperature
                    )

                    # -------------------------------------------------------
                    # Compute LLOPSD loss
                    # -------------------------------------------------------
                    loss, llopsd_metrics = compute_llopsd_loss(
                        student_per_loop_logits=student_per_loop_logits,
                        teacher_per_loop_logits=teacher_per_loop_logits,
                        loop_mapping=self.loop_mapping,
                        student_weights=self.student_weights,
                        divergence_type=self.divergence_type,
                        response_mask=response_mask,
                    )

                    # -------------------------------------------------------
                    # Scale loss for gradient accumulation
                    # -------------------------------------------------------
                    if self.config.use_dynamic_bsz:
                        scaled_loss = loss * (
                            micro_batch_size / self.config.ppo_mini_batch_size
                        )
                    else:
                        scaled_loss = loss / self.gradient_accumulation

                    scaled_loss.backward()

                    # -------------------------------------------------------
                    # Free logits to reduce memory pressure
                    # -------------------------------------------------------
                    del student_per_loop_logits, teacher_per_loop_logits

                    # -------------------------------------------------------
                    # Collect metrics
                    # -------------------------------------------------------
                    step_metrics = {
                        "llopsd/total_loss": loss.detach().item(),
                    }
                    step_metrics.update(llopsd_metrics)
                    append_to_dict(metrics, step_metrics)

                # -----------------------------------------------------------
                # Optimizer step
                # -----------------------------------------------------------
                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {"actor/grad_norm": grad_norm.detach().item()},
                )

                # -----------------------------------------------------------
                # EMA teacher update (after each optimizer step)
                # -----------------------------------------------------------
                if self.teacher_mode == "ema":
                    self.update_ema_teacher()

        # Final zero_grad to clean up
        self.actor_optimizer.zero_grad()
        return metrics
