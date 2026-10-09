"""
LSTM-formulation (LForm) HCNN. See ``README.md`` in this folder.
"""

from ...core.registry import ArchitectureSpec, register_architecture
from .cell import LFormHCNNCell
from .model import LForm_Model
from .ensemble import LFormHCNNEnsemble

SPEC = register_architecture(ArchitectureSpec(
    name="lform",
    model_cls=LForm_Model,
    cell_cls=LFormHCNNCell,
    ensemble_cls=LFormHCNNEnsemble,
    summary="s_{t+1} = r_t + D (A tanh(r_t) - r_t), D diagonal in (0, 1)",
    maintainers=("rock-feller",),
))

__all__ = ["LFormHCNNCell", "LForm_Model", "LFormHCNNEnsemble", "SPEC"]
