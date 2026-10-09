"""
Large-Sparse (LSpa) HCNN. See ``README.md`` in this folder.
"""

from ...core.registry import ArchitectureSpec, register_architecture
from .cell import LSpaHCNNCell
from .model import LSpa_Model
from .ensemble import LSpaHCNNEnsemble

SPEC = register_architecture(ArchitectureSpec(
    name="lspa",
    model_cls=LSpa_Model,
    cell_cls=LSpaHCNNCell,
    ensemble_cls=LSpaHCNNEnsemble,
    summary="s_{t+1} = A_sparse tanh(r_t), masked transition matrix",
    maintainers=("rock-feller",),
))

__all__ = ["LSpaHCNNCell", "LSpa_Model", "LSpaHCNNEnsemble", "SPEC"]
