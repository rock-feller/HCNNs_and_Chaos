"""
Shared HCNN framework - the part every architecture builds on.

- :mod:`.base`       - ``BaseHCNNCell``, ``BaseHCNNModel`` (the one shared rollout),
                       ``CellOutput`` and the output named tuples
- :mod:`.initial_state` - how each window's first state is built (``SharedInitialState``,
                       ``DelayEmbeddingInitialState``)
- :mod:`.registry`   - architecture registry (``build_model``, ``list_architectures``)
- :mod:`.layers`     - reusable layers (``CustomLinear``, ``DiagonalMatrix``,
                       ``CustomSparseLinear``, PTF dropout schedules)
- :mod:`.ensembles`  - generic ``HCNNEnsemble``
- :mod:`.training`   - ``BaseTrainer``, ``HCNNTrainer``, ``EnsembleTrainer``

The architectures themselves live in :mod:`hcnn.architectures`. ``cells`` and
``models`` remain here only as legacy import paths.
"""

from . import base
from . import initial_state
from . import registry
from . import layers
from . import ensembles
from . import cells
from . import models

__all__ = [
    "base",
    "initial_state",
    "registry",
    "layers",
    "ensembles",
    "cells",
    "models",
]
