"""
Legacy import path for the HCNN cells.

Cells now live with their architecture in ``hcnn/architectures/<name>/cell.py``.
This module keeps ``from hcnn.core.cells import VanillaHCNNCell`` working; names
are resolved lazily so importing :mod:`hcnn.core` never imports the architectures
(which themselves depend on :mod:`hcnn.core`).
"""

import importlib

_MOVED = {
    "VanillaHCNNCell": "hcnn.architectures.vanilla",
    "PTFHCNNCell": "hcnn.architectures.ptf",
    "LFormHCNNCell": "hcnn.architectures.lform",
    "LSpaHCNNCell": "hcnn.architectures.lspa",
}

__all__ = list(_MOVED)


def __getattr__(name):
    if name in _MOVED:
        return getattr(importlib.import_module(_MOVED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
