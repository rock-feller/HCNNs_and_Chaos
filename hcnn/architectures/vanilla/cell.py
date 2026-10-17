"""
Vanilla HCNN cell implementation.


This module contains the basic Historical Consistent Neural Network (HCNN) cell
that performs state-to-state mapping with teacher forcing mechanism support.
"""

import torch
from typing import Optional, Tuple
from ...core.base import BaseHCNNCell
from ...core.defaults import DEFAULT_INIT_RANGE
from ...core.layers import CustomLinear


class VanillaHCNNCell(BaseHCNNCell):
    """
    Vanilla HCNN Cell implementation.

    This class implements the basic version of the Historical Consistent Neural Network (HCNN) Cell.
    It performs a state-to-state mapping and produces outputs extracted from hidden states.
    It is designed to support teacher forcing during training.

    The step itself (read-out, teacher-forcing correction, external inputs) is
    :meth:`BaseHCNNCell.forward`; this cell supplies the transition:

    - State transition: s_{t+1} = A * tanh(r_t) + B * u_t (if external inputs)
    - Observation: y_t = C * s_t (where C is the observation matrix)
    - Teacher forcing: r_t = s_t - C^T * (y_pred - y_true) when teacher forcing is enabled,
    otherwise the cell operates in autonomous prediction mode: r_t = s_t.

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables (i.e., the dimensionality of the observed state variables)
    n_hid_vars : int
        Number of hidden variables (i.e., the dimensionality of the hidden state variables)
    init_range : Tuple[float, float], default=DEFAULT_INIT_RANGE
        Tuple specifying the range for uniform weight initialization in the linear modules
    n_ext_vars : Optional[int], default=None
        Number of external variables (i.e., the dimensionality of the external variables)

    Attributes
    ----------
    A : CustomLinear
        State transition matrix (n_state_vars x n_state_vars)
    B : CustomLinear, optional
        External input matrix (n_ext_vars x n_state_vars), only if n_ext_vars is provided
    ConMat : torch.Tensor
        Observation matrix that maps hidden states to observed outputs
        Shape: (n_obs_vars, n_state_vars)
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = DEFAULT_INIT_RANGE,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_obs_vars, n_hid_vars, init_range, n_ext_vars)

        # State transition matrix A
        self.A = CustomLinear(
            in_features=self.n_state_vars,
            out_features=self.n_state_vars,
            bias=False,
            init_range=self.init_range
        )
        # External input matrix B (optional), created after A
        self._init_external_matrix()

    @property
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""
        return "vanilla_hcnn_cell"

    def transition(self, r: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = A tanh(r_t) (before external inputs)."""
        return self.A(torch.tanh(r))

    def get_state_transition_matrix(self) -> torch.Tensor:
        """Get the current state transition matrix weights."""
        return self.A.weight

    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        self.A.reset_parameters()
        if self.B is not None:
            self.B.reset_parameters()
