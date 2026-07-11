"""
HCNN model wrappers (Vanilla, PTF, LForm, LSpa).

Every variant shares the exact same two-phase rollout (teacher-forced calibration
over the observed window, then autonomous forecasting). That rollout lives once in
:class:`hcnns_chaos.core.base.BaseHCNNModel.forward`; each model here only:

- constructs its variant-specific cell, and
- (if it has extra outputs, like PTF) overrides ``_pack_output``.

This keeps the four models to a few lines each and guarantees they cannot drift
apart in their sequence handling - the historical bug where the identical
``forward`` was copy-pasted (and separately edited) four times.
"""

import torch
from typing import Optional, Tuple, Literal

from ..base import BaseHCNNModel, HCNNOutput, PTFHCNNOutput
from ..cells import VanillaHCNNCell, PTFHCNNCell, LFormHCNNCell, LSpaHCNNCell


class Vanilla_Model(BaseHCNNModel):
    """
    Vanilla HCNN: standard teacher forcing with a nonlinear tanh state transition.

    Transition:  ``s_{t+1} = A * tanh(s_t - C^T delta_t)``  (delta_t = y_t - C s_t
    during teacher forcing, else the correction term vanishes and the rollout is
    autonomous).
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


class PTF_Model(BaseHCNNModel):
    """
    HCNN with Partial Teacher Forcing: dropout is applied to the correction term
    ``delta_t`` before it feeds back into the state, letting training interpolate
    between fully teacher-forced and autonomous behaviour. Six dropout schedules
    are supported at the cell level (see :mod:`hcnns_chaos.core.layers.dropout`).

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


class LSpa_Model(BaseHCNNModel):
    """
    Large-Sparse HCNN: vanilla dynamics with a structured-sparse transition matrix
    ``A_sparse`` for scaling to high-dimensional state spaces. Two mask types are
    supported: ``random_block`` (uniform random) and ``non_obs_block`` (sparsify
    only the hidden-related blocks).
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        sparsity_ratio: float = 0.5,
        mask_type: Literal['non_obs_block', 'random_block'] = 'random_block',
        bias: bool = False,
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
        self.sparsity_ratio = sparsity_ratio
        self.mask_type = mask_type
        self.bias = bias
        self.cell = LSpaHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            bias=bias,
            mask_type=mask_type,
            sparsity_ratio=sparsity_ratio,
            n_ext_vars=n_ext_vars,
        )
        self.name = "LSpa_Model"

    @property
    def model_type(self) -> str:
        return "lspa_hcnn"

    def get_sparsity_info(self) -> dict:
        """Statistics about the current sparsity pattern."""
        return self.cell.get_sparsity_info()

    def visualize_sparsity_pattern(self) -> torch.Tensor:
        """Binary (non-zero) pattern of the sparse transition matrix."""
        return self.cell.visualize_sparsity_pattern()

    def get_sparse_transition_matrix(self) -> torch.Tensor:
        """Weights of the sparse transition matrix ``A_sparse``."""
        return self.cell.get_sparse_transition_matrix()
