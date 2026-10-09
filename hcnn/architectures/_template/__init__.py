"""
TEMPLATE for a new HCNN architecture - never imported (leading underscore).

    cp -r hcnn/architectures/_template hcnn/architectures/<your_name>

then follow ``hcnn/architectures/README.md``. Once the folder name has no leading
underscore it is discovered and registered automatically on ``import hcnn``.
"""

from ...core.registry import ArchitectureSpec, register_architecture
from .cell import TemplateHCNNCell
from .model import Template_Model
from .ensemble import TemplateHCNNEnsemble

SPEC = register_architecture(ArchitectureSpec(
    name="template",  # TODO: must equal the folder name
    model_cls=Template_Model,
    cell_cls=TemplateHCNNCell,
    ensemble_cls=TemplateHCNNEnsemble,
    summary="TODO: one-line transition, e.g. s_{t+1} = f(r_t)",
    maintainers=("your-github-handle",),  # TODO
))

__all__ = ["TemplateHCNNCell", "Template_Model", "TemplateHCNNEnsemble", "SPEC"]
