"""
LSTM Formulation (LForm) HCNN cell implementation.


This module contains the LSTM-inspired HCNN cell that introduces memory-preserving
behavior using a learnable diagonal matrix for modulating long-term dependencies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...core.base import BaseHCNNCell
from ...core.layers import CustomLinear, DiagonalMatrix


class LFormHCNNCell(BaseHCNNCell):
    """
    LSTM Formulation of the HCNN Cell (HCNN-LForm Cell).

    This module implements an LSTM-inspired variant of the Historical Consistent Neural Network (HCNN) cell.
    It performs a non-linear residual update using a learnable diagonal matrix to regulate the embedding of 
    residuals in the hidden state space. This structure is designed to support both autonomous dynamics and 
    teacher-forced based training.

    The key innovation is the LSTM-like dynamics combining:
    1. Nonlinear transformations of the residual state (A)
    2. Diagonal modulation (D) for memory conservation
    3. Residual connections for improved gradient flow

    The dynamics are:
    - r_t = s_t - C^T * (y_pred - y_true) (if teacher forcing; observed part of r_t = y_true)
    - lstm_block = A(tanh(r_t)) - r_t
    - s_{t+1} = r_t + D(lstm_block)

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables (i.e., the output dimensionality of the system)
    n_hid_vars : int
        Number of hidden variables (i.e., the internal state dimensionality)
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Range for uniform initialization of the linear transformation weights
    init_diag : float, default=1.0
        Initial value for diagonal elements of the diagonal matrix D
    n_ext_vars : Optional[int], default=None
        Number of external variables

    Attributes
    ----------
    A : CustomLinear
        State transition matrix for nonlinear transformation
    D : DiagonalMatrix
        Learnable diagonal matrix for memory modulation
    B : CustomLinear, optional
        External input matrix, only if n_ext_vars is provided
    ConMat : torch.Tensor
        Observation matrix that maps hidden states to observed outputs
    init_diag : float
        Initial diagonal value for the D matrix
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        init_diag: float = 1.0,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_obs_vars, n_hid_vars, init_range, n_ext_vars)
        
        self.init_diag = init_diag

        # State transition matrix A
        self.A = CustomLinear(
            in_features=self.n_state_vars,
            out_features=self.n_state_vars,
            bias=False,
            init_range=self.init_range
        )

        # Diagonal matrix D for memory modulation
        self.D = DiagonalMatrix(
            n_features=self.n_state_vars,
            bias=False,
            init_diag=self.init_diag
        )

        # External input matrix B (optional), created after the transition parameters
        self._init_external_matrix()

    @property
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""
        return "lform"

    def transition(self, r: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = r_t + D (A tanh(r_t) - r_t) (before external inputs): a gated residual."""
        lstm_block = self.A(torch.tanh(r)) - r
        return r + self.D(lstm_block)

    def get_state_transition_matrix(self) -> torch.Tensor:
        """Get the current state transition matrix weights."""
        return self.A.weight

    def get_diagonal_matrix(self) -> torch.Tensor:
        """Get the current diagonal matrix weights."""
        return self.D.weight

    def get_diagonal_values(self) -> torch.Tensor:
        """Get current diagonal values."""
        return self.D.get_diagonal_values()

    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        self.A.reset_parameters()
        self.D.reset_parameters()
        if self.B is not None:
            self.B.reset_parameters()

    def extra_repr(self) -> str:
        """Return extra representation string for the cell."""
        return (
            f'n_obs_vars={self.n_obs_vars}, '
            f'n_hid_vars={self.n_hid_vars}, '
            f'n_ext_vars={self.n_ext_vars}, '
            f'init_range={self.init_range}, '
            f'init_diag={self.init_diag}'
        )
