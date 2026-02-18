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
from omegaconf import DictConfig, OmegaConf

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fs import copy_to_local
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
    - "frozen": A separate FSDP-wrapped copy of the initial model weights,
      loaded from disk with CPU offloading. Never updated during training.
    - "ema": An exponential moving average of the student parameters, built
      as a separate FSDP-wrapped model with CPU offloading. Updated after
      each optimizer step via tau*teacher + (1-tau)*student.
    """

    def __init__(self, config: DictConfig, role: str):
        """Initialize LLOPSD worker.

        Args:
            config: Worker configuration (should include LLOPSD-specific options)
            role: Worker role ('actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref')
        """
        super().__init__(config, role)

        # Extract LLOPSD-specific configuration
        self.total_ut_steps = config.get('total_ut_steps', None)
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

        # Override total_ut_steps on the loaded model config (safety belt in case
        # override_config didn't apply, e.g. when loading from SFT checkpoint)
        if self.total_ut_steps is not None and self._is_actor:
            if hasattr(self.actor_module_fsdp, 'config') and hasattr(self.actor_module_fsdp.config, 'total_ut_steps'):
                self.actor_module_fsdp.config.total_ut_steps = self.total_ut_steps
                if self.rank == 0:
                    print(f"Set actor model.config.total_ut_steps = {self.total_ut_steps}")

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
            elif self.teacher_mode in ("frozen", "ema"):
                # Build a separate FSDP-wrapped model from disk for the teacher.
                # Uses verl's ref model mechanism: FSDP with CPU offloading to
                # minimize GPU memory overhead.
                teacher_module = self._build_teacher_module()
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

    def _build_teacher_module(self):
        """Build a separate FSDP-wrapped teacher model for frozen/EMA modes.

        Uses verl's _build_model_optimizer with role="ref" to create an
        FSDP-sharded copy with CPU offloading. This keeps GPU memory
        overhead near zero (params are offloaded to CPU and brought to
        GPU on-demand during forward passes).

        Returns:
            The FSDP-wrapped teacher module with all parameters frozen.
        """
        model_path = self.config.model.path
        use_shm = self.config.model.get("use_shm", False)
        local_path = copy_to_local(model_path, use_shm=use_shm)

        # Reconstruct override_model_config (same as parent uses)
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )

        if self.rank == 0:
            print(
                f"[LLOPSD] Building {self.teacher_mode} teacher model from: {model_path}",
                flush=True,
            )

        teacher_fsdp = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
            optim_config=None,
            override_model_config=override_model_config,
            use_remove_padding=self.config.model.get("use_remove_padding", False),
            use_fused_kernels=self.config.model.get("use_fused_kernels", False),
            trust_remote_code=self.config.model.get("trust_remote_code", False),
            use_liger=self.config.model.get("use_liger", False),
            role="ref",
        )[0]

        # Freeze all parameters
        for param in teacher_fsdp.parameters():
            param.requires_grad = False
        teacher_fsdp.eval()

        # Set total_ut_steps on teacher config
        if self.total_ut_steps is not None:
            if hasattr(teacher_fsdp, 'config') and hasattr(teacher_fsdp.config, 'total_ut_steps'):
                teacher_fsdp.config.total_ut_steps = self.total_ut_steps
                if self.rank == 0:
                    print(f"[LLOPSD] Set teacher model.config.total_ut_steps = {self.total_ut_steps}")

        if self.rank == 0:
            print(
                f"[LLOPSD] {self.teacher_mode.upper()} teacher model built successfully "
                f"(FSDP with CPU offload)",
                flush=True,
            )

        return teacher_fsdp
