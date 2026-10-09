"""
Legacy import path for the model wrappers.

HCNN models now live in ``hcnn/architectures/<name>/model.py`` and the RNN/LSTM
baselines in :mod:`hcnn.baselines`. Names are resolved lazily (see
:mod:`hcnn.core.cells` for why).
"""

import importlib

_MOVED = {
    "Vanilla_Model": "hcnn.architectures.vanilla",
    "PTF_Model": "hcnn.architectures.ptf",
    "LForm_Model": "hcnn.architectures.lform",
    "LSpa_Model": "hcnn.architectures.lspa",
    "RNNModel": "hcnn.baselines",
    "LSTMModel": "hcnn.baselines",
}

__all__ = list(_MOVED)


def __getattr__(name):
    if name in _MOVED:
        return getattr(importlib.import_module(_MOVED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
