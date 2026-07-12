"""
Custom layers and utilities for HCNN models.

This module provides specialized layers including:
- Custom linear layers with configurable initialization
- Sparse linear layers for large-scale models
- Custom dropout implementations for partial teacher forcing
"""

from .linear import CustomLinear, DiagonalMatrix
from .sparse import CustomSparseLinear
from .dropout import (
    PartialTeacherForcingDropout,
    create_ptf_dropout,
    AdaptiveDropout,
    LinearScheduleDropout,
    ExponentialScheduleDropout,
    CosineAnnealingDropout,
    StepScheduleDropout
)

__all__ = [
    "CustomLinear",
    "DiagonalMatrix",
    "CustomSparseLinear",
    "PartialTeacherForcingDropout",
    "AdaptiveDropout",
    "LinearScheduleDropout",
    "ExponentialScheduleDropout",
    "CosineAnnealingDropout",
    "StepScheduleDropout",
    # Backward compatibility (deprecated)
    "create_ptf_dropout",
]
