"""
Training module.

- ``BaseTrainer``: family-agnostic epoch loop / selection / checkpointing.
- ``HCNNTrainer``: single HCNN model trainer.
- ``EnsembleTrainer``: trains each ensemble member with its own single-model trainer.
"""

from .base_trainer import BaseTrainer, EnsembleTrainer
from .hcnn_trainer import HCNNTrainer

__all__ = ["BaseTrainer", "HCNNTrainer", "EnsembleTrainer"]
