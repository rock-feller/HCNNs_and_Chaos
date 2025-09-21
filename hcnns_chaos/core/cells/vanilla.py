"""
Vanilla HCNN cell implementation.


This module contains the basic Historical Consistent Neural Network (HCNN) cell
that performs state-to-state mapping with teacher forcing mechanism support.
"""

import torch
from typing import Optional, Tuple
from ..base import BaseHCNNCell
from ..layers import CustomLinear


class VanillaHCNNCell(BaseHCNNCell):
    """
    Vanilla HCNN Cell implementation.

    This class implements the basic version of the Historical Consistent Neural Network (HCNN) Cell.
    It performs a state-to-state mapping and produces outputs extracted from hidden states.
    It is designed to support teacher forcing during training.

    The cell implements the following dynamics:
    - State transition: s_{t+1} = A * tanh(s_t) + B * u_t (if external inputs)
    - Observation: y_t = C * s_t (where C is the observation matrix)
    - Teacher forcing: r_t = s_t - C^T * (y_pred - y_true) when teacherr forcing is enabled,
    otherwise the cell operates in autonomous prediction mode: r_t = s_t.

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables (i.e., the dimensionality of the observed state variables)
    n_hid_vars : int
        Number of hidden variables (i.e., the dimensionality of the hidden state variables)
    init_range : Tuple[float, float], default=(-0.75, 0.75)
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
    Ide : torch.Tensor
        Identity matrix for state operations
        Shape: (n_state_vars, n_state_vars)
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = (-0.75, 0.75),
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
        # ConMat = [I_obs, 0] where I_obs is identity matrix of size n_obs_vars
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
        return "vanilla_hcnn_cell"

    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Vanilla HCNN Cell.

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
            Only used if the cell was initialized with n_ext_vars

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
            # self.B is guaranteed to be not None when n_ext_vars is not None
            assert self.B is not None, "B should not be None when n_ext_vars is provided"
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
            # Correction is applied by subtracting C^T * delta from state
            teach_forc = torch.matmul(delta_term, torch.as_tensor(self.ConMat, device=delta_term.device))
            corrected_state = state - teach_forc

            # Compute next state with correction
            next_state = self.A(torch.tanh(corrected_state)) + external_contribution

            return expectation, next_state, delta_term

        else:
            # No teacher forcing - standard forward pass
            # Apply identity transformation to state (for consistency)
            r_state = torch.matmul(state, torch.as_tensor(self.Ide, device=state.device))

            # Compute next state
            next_state = self.A(torch.tanh(r_state)) + external_contribution

            return expectation, next_state, None

    def get_observation_matrix(self) -> torch.Tensor:
        """Get the observation matrix."""
        return torch.as_tensor(self.ConMat, device=next(self.parameters()).device)

    def get_state_transition_matrix(self) -> torch.Tensor:
        """Get the current state transition matrix weights."""
        return self.A.weight

    def get_external_input_matrix(self) -> Optional[torch.Tensor]:
        """Get the external input matrix weights if available."""
        return self.B.weight if self.B is not None else None

    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        self.A.reset_parameters()
        if self.B is not None:
            self.B.reset_parameters()

    def extra_repr(self) -> str:
        """Return extra representation string for the cell."""
        return (
            f'n_obs_vars={self.n_obs_vars}, '
            f'n_hid_vars={self.n_hid_vars}, '
            f'n_ext_vars={self.n_ext_vars}, '
            f'init_range={self.init_range}'
        )
