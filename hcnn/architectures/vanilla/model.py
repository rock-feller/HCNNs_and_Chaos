"""
Vanilla HCNN model: the baseline teacher-forced HCNN.

The two-phase rollout (teacher-forced calibration, then autonomous forecast) is
inherited from :class:`hcnn.core.base.BaseHCNNModel`; this wrapper only builds
the cell. See ``README.md`` in this folder for the math.
"""

import torch
from typing import Optional, Tuple, Literal

from ...core.base import BaseHCNNModel
from .cell import VanillaHCNNCell


class Vanilla_Model(BaseHCNNModel):
    """
    Vanilla HCNN: standard teacher forcing with a nonlinear tanh state transition.

    Transition:  ``s_{t+1} = A * tanh(r_t)`` with ``r_t = s_t - C^T (C s_t - y_t)`` during
    teacher forcing (the observed part of ``r_t`` is the data ``y_t``), and ``r_t = s_t``
    in the autonomous rollout.
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
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
        self.cell = VanillaHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
        )
        self.name = "Vanilla_Model"

    @property
    def model_type(self) -> str:
        return "vanilla_hcnn"
