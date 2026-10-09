"""
Ensembles.

:class:`HCNNEnsemble` (in :mod:`.base`) is the generic ensemble every architecture
subclasses. The concrete ensembles live with their architecture
(``hcnn/architectures/<name>/ensemble.py``) and the RNN/LSTM ones in
:mod:`hcnn.baselines`; they are re-exported here lazily for backward compatibility.
"""

import importlib

from .base import HCNNEnsemble

_MOVED = {
    "VanillaHCNNEnsemble": "hcnn.architectures.vanilla",
    "PTFHCNNEnsemble": "hcnn.architectures.ptf",
    "LFormHCNNEnsemble": "hcnn.architectures.lform",
    "LSpaHCNNEnsemble": "hcnn.architectures.lspa",
    "RNNEnsemble": "hcnn.baselines",
    "LSTMEnsemble": "hcnn.baselines",
}

__all__ = ["HCNNEnsemble", *_MOVED]


def __getattr__(name):
    if name in _MOVED:
        return getattr(importlib.import_module(_MOVED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
