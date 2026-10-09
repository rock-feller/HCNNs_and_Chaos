"""
LSTM-formulation (LForm) HCNN model.

The two-phase rollout (teacher-forced calibration, then autonomous forecast) is
inherited from :class:`hcnn.core.base.BaseHCNNModel`; this wrapper only builds
the cell. See ``README.md`` in this folder for the math.
"""

import torch
from typing import Optional, Tuple, Literal

from ...core.base import BaseHCNNModel
from .cell import LFormHCNNCell


class LForm_Model(BaseHCNNModel):
    """
    LSTM-formulation HCNN: a diagonal-gated residual update

        ``s_{t+1} = r_t + D * (A * tanh(r_t) - r_t)``

    where ``D`` is a learnable diagonal gate in (0, 1). ``D -> 0`` preserves the
    (corrected) state (pure memory); ``D -> 1`` recovers vanilla dynamics.
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_diag: float = 1.0,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None,
    ):
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
        )
        self.init_diag = init_diag
        self.cell = LFormHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            init_diag=init_diag,
            n_ext_vars=n_ext_vars,
        )
        self.name = "LForm_Model"

    @property
    def model_type(self) -> str:
        return "lform_hcnn"

    def get_diagonal_values(self) -> torch.Tensor:
        """Current values of the diagonal memory gate D."""
        return self.cell.get_diagonal_values()

    def get_diagonal_matrix(self) -> torch.Tensor:
        """Full weight matrix of the diagonal memory gate D."""
        return self.cell.get_diagonal_matrix()
