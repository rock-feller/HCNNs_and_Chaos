"""
Partial Teacher Forcing (PTF) HCNN model.

The two-phase rollout (teacher-forced calibration, then autonomous forecast) is
inherited from :class:`hcnn.core.base.BaseHCNNModel`; this wrapper only builds
the cell and packs the PTF-specific output. See ``README.md`` in this folder for the math.
"""

import torch
from typing import Optional, Tuple, Literal

from ...core.base import BaseHCNNModel, PTFHCNNOutput
from .cell import PTFHCNNCell


class PTF_Model(BaseHCNNModel):
    """
    HCNN with Partial Teacher Forcing: dropout is applied to the correction term
    ``delta_t`` before it feeds back into the state, letting training interpolate
    between fully teacher-forced and autonomous behaviour. Six dropout schedules
    are supported at the cell level (see :mod:`hcnn.core.layers.dropout`).

    In addition to the standard fields, the output carries ``partial_delta_terms``
    (the dropout-masked delta), assembled here via :meth:`_pack_output`.
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        target_prob: float = 0.5,
        drop_output: bool = False,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        n_ext_vars: Optional[int] = None,
        dropout_strategy: str = 'adaptive',
        dropout_params: Optional[dict] = None,
    ):
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
        )
        self.target_prob = target_prob
        self.drop_output = drop_output
        self.dropout_strategy = dropout_strategy
        self.dropout_params = dropout_params or {}
        self.cell = PTFHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
            dropout_strategy=dropout_strategy,
            dropout_params=dropout_params,
        )
        self.name = "PTF_Model"

    @property
    def model_type(self) -> str:
        return "ptf_hcnn"

    def update_dropout_epoch(self, epoch: int):
        """Advance the cell's dropout schedule to ``epoch`` (1-based)."""
        self.cell.update_dropout_epoch(epoch)

    def _pack_output(self, expectations, states, delta_terms, forecasts, future_states, extras):
        return PTFHCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            partial_delta_terms=extras.get("partial_delta_terms"),
            forecasts=forecasts,
            future_states=future_states,
        )
