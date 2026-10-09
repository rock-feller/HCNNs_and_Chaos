"""
HCNN architectures - one self-contained package per variant.

Each sub-package (``vanilla/``, ``ptf/``, ``lform/``, ``lspa/``, ...) holds the
variant's cell, model wrapper, ensemble and README, and registers itself with
:mod:`hcnn.core.registry` on import. Sub-packages are discovered automatically,
so adding an architecture never requires editing a shared file: copy
``_template/`` to a new folder and fill it in (see ``README.md`` here).

Packages whose name starts with an underscore (e.g. ``_template``) are skipped.
"""

import importlib
import pkgutil

from ..core.registry import (
    ArchitectureSpec,
    build_ensemble,
    build_model,
    describe_architectures,
    get_architecture,
    list_architectures,
    register_architecture,
)

for _info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    if _info.ispkg and not _info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_info.name}")

# Built-in architectures, re-exported for convenience.
from .vanilla import VanillaHCNNCell, Vanilla_Model, VanillaHCNNEnsemble  # noqa: E402
from .ptf import PTFHCNNCell, PTF_Model, PTFHCNNEnsemble  # noqa: E402
from .lform import LFormHCNNCell, LForm_Model, LFormHCNNEnsemble  # noqa: E402
from .lspa import LSpaHCNNCell, LSpa_Model, LSpaHCNNEnsemble  # noqa: E402

__all__ = [
    "ArchitectureSpec",
    "register_architecture",
    "get_architecture",
    "list_architectures",
    "describe_architectures",
    "build_model",
    "build_ensemble",
    "VanillaHCNNCell", "Vanilla_Model", "VanillaHCNNEnsemble",
    "PTFHCNNCell", "PTF_Model", "PTFHCNNEnsemble",
    "LFormHCNNCell", "LForm_Model", "LFormHCNNEnsemble",
    "LSpaHCNNCell", "LSpa_Model", "LSpaHCNNEnsemble",
]
