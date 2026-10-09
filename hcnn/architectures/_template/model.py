"""
<Name> HCNN model - TEMPLATE.

The rollout is inherited from :class:`hcnn.core.base.BaseHCNNModel` - do NOT
write a ``forward`` here. Only build the cell (and, if your cell returns
``extras``, override ``_pack_output`` as PTF does).
"""

from typing import Optional, Tuple, Literal

from ...core.base import BaseHCNNModel
from .cell import TemplateHCNNCell


class Template_Model(BaseHCNNModel):  # TODO: rename, e.g. MyIdea_Model
    """TODO: one-line description."""

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal["zeros_", "random_"] = "random_",
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        n_ext_vars: Optional[int] = None,
        # TODO: forward your cell's hyper-parameters (keep defaults).
    ):
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
        )
        self.cell = TemplateHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
        )
        self.name = "Template_Model"  # TODO

    @property
    def model_type(self) -> str:
        return "template_hcnn"  # TODO
