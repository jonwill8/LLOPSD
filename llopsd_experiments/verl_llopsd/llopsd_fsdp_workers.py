"""
LLOPSD-aware FSDP Workers for verl.

This module provides custom worker classes that integrate the LLOPSD actor
into verl's distributed training framework, supporting loop-aware
teacher-student self-distillation.
"""

import logging
import os
import sys

import ray
import torch
from omegaconf import DictConfig

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch
import verl.utils.py_functional as verl_py_functional

from .llopsd_actor import LLOPSDDataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


# Monkey-patch convert_to_regular_types to handle comma-separated target_modules
# We need to patch both the module AND the imported reference in fsdp_workers
import verl.workers.fsdp_workers as verl_fsdp_workers

_original_convert_to_regular_types = verl_py_functional.convert_to_regular_types


def _patched_convert_to_regular_types(obj):
    """Patched version that splits comma-separated strings for target_modules.

    This is needed because:
    1. verl's HFModelConfig expects target_modules as a string (for OmegaConf validation)
    2. PEFT expects target_modules as a list for specific module names
    3. Only "all-linear" works as a string in PEFT

    This patch converts comma-separated strings like "q_proj,k_proj,v_proj,o_proj"
    into lists ["q_proj", "k_proj", "v_proj", "o_proj"] when convert_to_regular_types is called.
    """
    result = _original_convert_to_regular_types(obj)

    # If result is a comma-separated string (not "all-linear"), split it into a list
    if isinstance(result, str) and ',' in result and result != "all-linear":
        return [m.strip() for m in result.split(',')]

    return result


# Apply the monkey-patch to both locations
verl_py_functional.convert_to_regular_types = _patched_convert_to_regular_types
verl_fsdp_workers.convert_to_regular_types = _patched_convert_to_regular_types

__all__ = ['LLOPSDActorRolloutRefWorker']


@ray.remote
class LLOPSDActorRolloutRefWorker(ActorRolloutRefWorker):
    """LLOPSD-aware version of ActorRolloutRefWorker.

    This worker uses LLOPSDDataParallelPPOActor instead of the standard
    DataParallelPPOActor, enabling loop-aware teacher-student distillation
    for the full LLOPSD objective.

    Teacher modes:
    - "same": Self-distillation. The actor model is used as its own teacher
      (with torch.no_grad). This is the default and fully supported mode.
    - "frozen": Intended for a frozen copy of the initial model as teacher.
      In v1, falls back to "same" mode with a warning due to FSDP constraints
      on maintaining separate model copies.
    - "ema": Intended for an exponential moving average teacher. In v1, falls
      back to "same" mode with a warning due to FSDP constraints.
    """

    def __init__(self, config: DictConfig, role: str):
        """Initialize LLOPSD worker.

        Args:
            config: Worker configuration (should include LLOPSD-specific options)
            role: Worker role ('actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref')
        """
        super().__init__(config, role)

        # Extract LLOPSD-specific configuration
        self.teacher_loops = config.get('teacher_loops', 2)
        self.student_loops = config.get('student_loops', 1)
        self.loop_mapping_strategy = config.get('loop_mapping_strategy', 'shift')
        self.weight_schedule = config.get('weight_schedule', 'uniform')
        self.weight_gamma = config.get('weight_gamma', 1.0)
        self.divergence_type = config.get('divergence_type', 'forward_kl')
        self.teacher_context = config.get('teacher_context', 'response_only')
        self.teacher_mode = config.get('teacher_mode', 'same')
        self.ema_decay = config.get('ema_decay', 0.999)

        if self.rank == 0:
            print(f"LLOPSD Worker initialized:")
            print(f"  teacher_loops: {self.teacher_loops}")
            print(f"  student_loops: {self.student_loops}")
            print(f"  loop_mapping_strategy: {self.loop_mapping_strategy}")
            print(f"  weight_schedule: {self.weight_schedule}")
            print(f"  weight_gamma: {self.weight_gamma}")
            print(f"  divergence_type: {self.divergence_type}")
            print(f"  teacher_context: {self.teacher_context}")
            print(f"  teacher_mode: {self.teacher_mode}")
            if self.teacher_mode == 'ema':
                print(f"  ema_decay: {self.ema_decay}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with LLOPSD actor instead of standard actor.

        This calls the parent init_model() and then replaces the standard
        DataParallelPPOActor with LLOPSDDataParallelPPOActor, configured
        with the appropriate teacher module based on teacher_mode.
        """
        print(f"[LLOPSD WORKER DEBUG] init_model() called on rank {self.rank}", flush=True)
        sys.stdout.flush()

        # Call parent implementation - this handles all the model loading,
        # optimizer creation, rollout/ref setup, etc. with proper trust_remote_code
        # Note: target_modules conversion is handled by the monkey-patched
        # convert_to_regular_types function at module level
        super().init_model()

        # Replace standard actor with LLOPSD actor
        if self._is_actor:
            print(f"[LLOPSD WORKER DEBUG] Replacing actor with LLOPSDDataParallelPPOActor on rank {self.rank}", flush=True)
            sys.stdout.flush()

            # Determine teacher_module based on teacher_mode
            teacher_module = None

            if self.teacher_mode == "same":
                # Self-distillation: actor uses its own weights as teacher (with no_grad)
                # teacher_module=None signals the actor to reuse self.actor_module
                teacher_module = None
            elif self.teacher_mode == "frozen":
                # Frozen teacher: ideally a separate copy of the initial weights.
                # With FSDP, maintaining a separate full model copy is non-trivial
                # (deepcopy of FSDP modules is problematic). For v1, we fall back
                # to self-distillation and log a warning.
                teacher_module = None
                if self.rank == 0:
                    print(
                        "[LLOPSD WARNING] teacher_mode='frozen' requested, but full frozen "
                        "teacher requires a separate model copy which is not supported with "
                        "FSDP in v1. Falling back to 'same' mode (self-distillation with "
                        "torch.no_grad). The actor model will serve as its own teacher.",
                        flush=True,
                    )
            elif self.teacher_mode == "ema":
                # EMA teacher: ideally an exponential moving average of the actor weights.
                # With FSDP, maintaining a separate EMA model copy is non-trivial.
                # For v1, we fall back to self-distillation and log a warning.
                teacher_module = None
                if self.rank == 0:
                    print(
                        f"[LLOPSD WARNING] teacher_mode='ema' (decay={self.ema_decay}) requested, "
                        "but EMA teacher requires a separate model copy which is not supported "
                        "with FSDP in v1. Falling back to 'same' mode (self-distillation with "
                        "torch.no_grad). The actor model will serve as its own teacher.",
                        flush=True,
                    )
            else:
                if self.rank == 0:
                    print(
                        f"[LLOPSD WARNING] Unknown teacher_mode='{self.teacher_mode}'. "
                        "Falling back to 'same' mode (self-distillation).",
                        flush=True,
                    )
                teacher_module = None

            # Create LLOPSD actor with all configuration parameters
            self.actor = LLOPSDDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                teacher_module=teacher_module,
                teacher_loops=self.teacher_loops,
                student_loops=self.student_loops,
                loop_mapping_strategy=self.loop_mapping_strategy,
                weight_schedule=self.weight_schedule,
                weight_gamma=self.weight_gamma,
                divergence_type=self.divergence_type,
                teacher_context=self.teacher_context,
                teacher_mode=self.teacher_mode,
                ema_decay=self.ema_decay,
            )

            if self.rank == 0:
                print(f"Created LLOPSDDataParallelPPOActor with:")
                print(f"  teacher_module: {'provided' if teacher_module is not None else 'None (self-distillation)'}")
                print(f"  teacher_loops: {self.teacher_loops}")
                print(f"  student_loops: {self.student_loops}")
                print(f"  loop_mapping_strategy: {self.loop_mapping_strategy}")
                print(f"  weight_schedule: {self.weight_schedule}")
                print(f"  weight_gamma: {self.weight_gamma}")
                print(f"  divergence_type: {self.divergence_type}")
                print(f"  teacher_context: {self.teacher_context}")
                print(f"  teacher_mode: {self.teacher_mode}")
                if self.teacher_mode == 'ema':
                    print(f"  ema_decay: {self.ema_decay}")

            # Log trainable vs frozen parameters (all ranks participate in all_reduce)
            self._log_trainable_parameters()

        print(f"[LLOPSD WORKER DEBUG] init_model() completed on rank {self.rank}", flush=True)
        sys.stdout.flush()

    def _log_trainable_parameters(self):
        """Log trainable vs frozen parameter counts.

        Computes the TOTAL parameter count across all FSDP shards by summing
        local counts from each rank. This gives the true model size before sharding.

        All ranks must call this method (for all_reduce), but only rank 0 prints.
        """
        # Count local (sharded) parameters
        local_total = sum(p.numel() for p in self.actor_module_fsdp.parameters())
        local_trainable = sum(p.numel() for p in self.actor_module_fsdp.parameters() if p.requires_grad)

        # Sum across all ranks to get global totals (FSDP shards parameters across ranks)
        world_size = torch.distributed.get_world_size()

        # Create tensors for all_reduce
        total_tensor = torch.tensor([local_total], dtype=torch.long, device='cuda')
        trainable_tensor = torch.tensor([local_trainable], dtype=torch.long, device='cuda')

        # Sum across all ranks - ALL ranks must participate
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(trainable_tensor, op=torch.distributed.ReduceOp.SUM)

        # Only rank 0 prints the result
        if self.rank == 0:
            global_total = total_tensor.item()
            global_trainable = trainable_tensor.item()
            global_frozen = global_total - global_trainable

            if global_total > 0:
                trainable_pct = 100 * global_trainable / global_total
                frozen_pct = 100 * global_frozen / global_total
            else:
                trainable_pct = frozen_pct = 0.0

            print(f"Parameter summary (summed across {world_size} FSDP shards):")
            print(f"  Total params:     {global_total:,}")
            print(f"  Trainable params: {global_trainable:,} ({trainable_pct:.2f}%)")
            print(f"  Frozen params:    {global_frozen:,} ({frozen_pct:.2f}%)")
