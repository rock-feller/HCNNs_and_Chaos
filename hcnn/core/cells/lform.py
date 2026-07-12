"""
LSTM Formulation (LForm) HCNN cell implementation.


This module contains the LSTM-inspired HCNN cell that introduces memory-preserving
behavior using a learnable diagonal matrix for modulating long-term dependencies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..base import BaseHCNNCell, CellOutput
from ..layers import CustomLinear, DiagonalMatrix


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
    - r_t = s_t - C^T * (y_true - y_pred) (if teacher forcing)
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
    Ide : torch.Tensor
        Identity matrix for state operations
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

        # External input matrix B (optional)
        if n_ext_vars is not None:
            self.B = CustomLinear(
                in_features=n_ext_vars,
                out_features=self.n_state_vars,
                bias=False,
                init_range=self.init_range
            )
        else:
            self.B = None

        # Observation matrix (maps state to observations)
        self.register_buffer(
            'ConMat',
            torch.eye(self.n_obs_vars, self.n_state_vars),
            persistent=False
        )

        # Identity matrix for state operations
        self.register_buffer(
            'Ide',
            torch.eye(self.n_state_vars),
            persistent=False
        )

    @property
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""
        return "lform"

    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None
    ) -> CellOutput:
        """
        Forward pass through the LSTM Formulation HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor
            Current state tensor of shape (batch_size, n_state_vars)
        teacher_forcing : bool, default=False
            Whether to apply teacher forcing correction
        observation : Optional[torch.Tensor], default=None
            Ground truth observation for teacher forcing of shape (batch_size, n_obs_vars)
            Required when teacher_forcing=True
        externals : Optional[torch.Tensor], default=None
            External variables of shape (batch_size, n_ext_vars)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            - expectation: Predicted observation (batch_size, n_obs_vars)
            - next_state: Next state (batch_size, n_state_vars)
            - delta_term: Observation error (y_true - y_pred) if teacher_forcing, else None

        Raises
        ------
        ValueError
            If teacher_forcing=True but observation is not provided
            If externals is provided but cell doesn't support external variables
            If externals is not provided but cell expects external variables
        """
        # Compute expectation (predicted observation)
        expectation = torch.matmul(state, torch.as_tensor(self.ConMat, device=state.device).T)

        # Handle external variables
        external_contribution = torch.zeros_like(state)
        if self.n_ext_vars is not None:
            if externals is None:
                raise ValueError(
                    "External variables expected but not provided. "
                    f"Cell was initialized with n_ext_vars={self.n_ext_vars}"
                )
            if externals.shape[-1] != self.n_ext_vars:
                raise ValueError(
                    f"Expected {self.n_ext_vars} external variables, "
                    f"got {externals.shape[-1]}"
                )
            if self.B is not None:
                external_contribution = self.B(externals)
        elif externals is not None:
            raise ValueError(
                "External variables provided but cell doesn't support them. "
                "Initialize cell with n_ext_vars parameter."
            )

        # Apply teacher forcing if requested
        if teacher_forcing:
            if observation is None:
                raise ValueError(
                    "`observation` must be provided when `teacher_forcing` is True."
                )
            if observation.shape[-1] != self.n_obs_vars:
                raise ValueError(
                    f"Expected {self.n_obs_vars} observation variables, "
                    f"got {observation.shape[-1]}"
                )

            # Compute delta term (observation error)
            delta_term = observation - expectation

            # Apply teacher forcing correction
            teach_forc = torch.matmul(delta_term, torch.as_tensor(self.ConMat, device=delta_term.device))
            r_state = state - teach_forc

            # LSTM-like computation
            # lstm_block = A(tanh(r_state)) - r_state (residual connection)
            lstm_block = self.A(torch.tanh(r_state)) - r_state

            # Apply diagonal modulation and add to corrected state
            next_state = r_state + self.D(lstm_block) + external_contribution

            return CellOutput(expectation, next_state, delta_term, {})

        else:
            # No teacher forcing - standard forward pass
            r_state = torch.matmul(state, torch.as_tensor(self.Ide, device=state.device))

            # LSTM-like computation without teacher forcing
            lstm_block = self.A(torch.tanh(r_state)) - r_state
            next_state = r_state + self.D(lstm_block) + external_contribution

            return CellOutput(expectation, next_state, None, {})

    def get_observation_matrix(self) -> torch.Tensor:
        """Get the observation matrix."""
        return torch.as_tensor(self.ConMat, device=next(self.parameters()).device)

    def get_state_transition_matrix(self) -> torch.Tensor:
        """Get the current state transition matrix weights."""
        return self.A.weight

    def get_diagonal_matrix(self) -> torch.Tensor:
        """Get the current diagonal matrix weights."""
        return self.D.weight

    def get_diagonal_values(self) -> torch.Tensor:
        """Get current diagonal values."""
        return self.D.get_diagonal_values()

    def get_external_input_matrix(self) -> Optional[torch.Tensor]:
        """Get the external input matrix weights if available."""
        return self.B.weight if self.B is not None else None

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
