"""
Vanilla HCNN. See ``README.md`` in this folder.
"""

from ...core.registry import ArchitectureSpec, register_architecture
from .cell import VanillaHCNNCell
from .model import Vanilla_Model
from .ensemble import VanillaHCNNEnsemble

SPEC = register_architecture(ArchitectureSpec(
    name="vanilla",
    model_cls=Vanilla_Model,
    cell_cls=VanillaHCNNCell,
    ensemble_cls=VanillaHCNNEnsemble,
    summary="s_{t+1} = A tanh(s_t - C^T (y_hat_t - y_t))",
    maintainers=("rock-feller",),
))

__all__ = ["VanillaHCNNCell", "Vanilla_Model", "VanillaHCNNEnsemble", "SPEC"]
