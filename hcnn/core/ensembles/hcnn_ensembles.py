"""
Legacy module: the HCNN ensembles moved to ``hcnn/architectures/<name>/ensemble.py``
and the generic base to :mod:`hcnn.core.ensembles.base`.

Kept so ``from hcnn.core.ensembles.hcnn_ensembles import VanillaHCNNEnsemble, ...``
(used by the notebooks) keeps working.
"""

from .base import HCNNEnsemble, _HCNNEnsembleBase  # noqa: F401
from ...architectures.vanilla import VanillaHCNNEnsemble  # noqa: F401
from ...architectures.ptf import PTFHCNNEnsemble  # noqa: F401
from ...architectures.lform import LFormHCNNEnsemble  # noqa: F401
from ...architectures.lspa import LSpaHCNNEnsemble  # noqa: F401

__all__ = [
    "HCNNEnsemble",
    "VanillaHCNNEnsemble",
    "PTFHCNNEnsemble",
    "LFormHCNNEnsemble",
    "LSpaHCNNEnsemble",
]
