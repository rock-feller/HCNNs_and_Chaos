"""
Partial Teacher Forcing (PTF) HCNN. See ``README.md`` in this folder.
"""

from ...core.registry import ArchitectureSpec, register_architecture
from .cell import PTFHCNNCell
from .model import PTF_Model
from .ensemble import PTFHCNNEnsemble

SPEC = register_architecture(ArchitectureSpec(
    name="ptf",
    model_cls=PTF_Model,
    cell_cls=PTFHCNNCell,
    ensemble_cls=PTFHCNNEnsemble,
    summary="vanilla with scheduled dropout on the correction delta_t",
    maintainers=("rock-feller",),
))

__all__ = ["PTFHCNNCell", "PTF_Model", "PTFHCNNEnsemble", "SPEC"]
