"""
Legacy module: the HCNN models moved to ``hcnn/architectures/<name>/model.py``.

Kept so ``from hcnn.core.models.hcnn_models import Vanilla_Model, ...`` (used by
the notebooks) keeps working. Prefer ``from hcnn import Vanilla_Model`` or
``hcnn.build_model("vanilla", ...)`` in new code.
"""

from ...architectures.vanilla import Vanilla_Model  # noqa: F401
from ...architectures.ptf import PTF_Model  # noqa: F401
from ...architectures.lform import LForm_Model  # noqa: F401
from ...architectures.lspa import LSpa_Model  # noqa: F401

__all__ = ["Vanilla_Model", "PTF_Model", "LForm_Model", "LSpa_Model"]
