"""
Custom verl components for LLOPSD (Looped Language On-Policy Self-Distillation).

This module provides custom actor workers and algorithms that implement
the LLOPSD objective with loop-aware teacher-student distillation,
enabling multi-GPU training with verl while preserving the complete
LLOPSD loss function.

Usage:
    from verl_llopsd import LLOPSDActorRolloutRefWorker

    # Use LLOPSDActorRolloutRefWorker instead of ActorRolloutRefWorker
    # in your verl training configuration
"""

from .llopsd_actor import LLOPSDDataParallelPPOActor
from .llopsd_algos import (
    compute_llopsd_loss,
    compute_loop_mapping,
    compute_student_weights,
    compute_divergence,
    construct_teacher_input,
)
from .llopsd_fsdp_workers import LLOPSDActorRolloutRefWorker

__all__ = [
    # Actor
    'LLOPSDDataParallelPPOActor',
    # Algorithms
    'compute_llopsd_loss',
    'compute_loop_mapping',
    'compute_student_weights',
    'compute_divergence',
    'construct_teacher_input',
    # Workers
    'LLOPSDActorRolloutRefWorker',
]
