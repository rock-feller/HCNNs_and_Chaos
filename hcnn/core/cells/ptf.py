"""
Partial Teacher Forcing (PTF) HCNN cell implementation.


This module contains the PTF HCNN cell that incorporates partial teacher forcing
via dropout scaling during training.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..base import BaseHCNNCell, CellOutput
from ..layers.linear import CustomLinear
from ..layers.dropout import (
    
    PartialTeacherForcingDropout,
    AdaptiveDropout,
    LinearScheduleDropout,
    ExponentialScheduleDropout,
    CosineAnnealingDropout,
    StepScheduleDropout
)


class PTFHCNNCell(BaseHCNNCell):
    """
    Partial Teacher Forcing HCNN Cell (HCNN-pTF).

    This class implements a HCNN Cell endowed with a partial teacher forcing mechanism during training.
    It performs a state-to-state mapping and produces outputs based on hidden states.
    It allows a controlled adjustment of the teacher forcing behavior using dropout probabilities.

    The key innovation is the application of dropout to the delta term (y_true - y_pred) before
    applying the teacher forcing correction, allowing for partial guidance during training.

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
    dropout_strategy : str, default='adaptive'
        Dropout strategy to use for partial teacher forcing. Options:
        - 'adaptive': AdaptiveDropout (0 for first half, incremental for second half)
        - 'linear': LinearScheduleDropout (linear interpolation from start_p to end_p)
        - 'exponential': ExponentialScheduleDropout (exponential change)
        - 'cosine': CosineAnnealingDropout (cosine annealing schedule)
        - 'step': StepScheduleDropout (step-wise changes at specific epochs)
        - 'constant': PartialTeacherForcingDropout (constant probability)
    dropout_params : dict, default=None
        Parameters for the dropout strategy. If None, uses default parameters

    Attributes
    ----------
    A : CustomLinear
        State transition matrix (n_state_vars x n_state_vars)
    B : CustomLinear, optional
        External input matrix (n_ext_vars x n_state_vars), only if n_ext_vars is provided
    ConMat : torch.Tensor
        Observation matrix that maps hidden states to observed outputs
    Ide : torch.Tensor
        Identity matrix for state operations
    dropout_strategy : str
        String indicating the dropout strategy type
    dropout_module : nn.Module
        Dropout module implementing the chosen strategy for partial teacher forcing
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None,
        dropout_strategy: str = 'adaptive',
        dropout_params: Optional[dict] = None
    ):
        super().__init__(n_obs_vars, n_hid_vars, init_range, n_ext_vars)

        # Store dropout strategy type and parameters
        self.dropout_strategy = dropout_strategy
        self.dropout_params = dropout_params or {}

        # Initialize dropout module with proper type annotation
        self.dropout_module: nn.Module

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

        # Initialize dropout module
        self.dropout_module = self._create_dropout_strategy(dropout_strategy, dropout_params)
        if not isinstance(self.dropout_module, nn.Module):
            raise TypeError("Dropout strategy must be an nn.Module, got {}".format(type(self.dropout_module)))

    def _create_dropout_strategy(self, dropout_strategy: str, params: Optional[dict] = None) -> nn.Module:
        """
        Create dropout strategy based on the specified type.

        Parameters
        ----------
        dropout_strategy : str
            Dropout strategy type
        params : Optional[dict]
            Parameters for the dropout strategy

        Returns
        -------
        nn.Module
            Configured dropout module
        """
        if params is None:
            params = {}

        # Default parameters for each strategy
        default_params = {
            'adaptive': {'target_p': 0.3, 'total_epochs': 100},
            'linear': {'start_p': 0.0, 'end_p': 0.3, 'total_epochs': 100},
            'exponential': {'start_p': 0.0, 'end_p': 0.3, 'total_epochs': 100, 'decay_rate': 0.1},
            'cosine': {'start_p': 0.0, 'end_p': 0.3, 'total_epochs': 100},
            'step': {'schedule': [(50, 0.1), (75, 0.2), (90, 0.3)]},
            'constant': {'p': 0.1}
        }

        # Merge default params with user params
        final_params = default_params.get(dropout_strategy, {}).copy()
        final_params.update(params)

        if dropout_strategy == 'adaptive':
            return AdaptiveDropout(**final_params)
        elif dropout_strategy == 'linear':
            return LinearScheduleDropout(**final_params)
        elif dropout_strategy == 'exponential':
            return ExponentialScheduleDropout(**final_params)
        elif dropout_strategy == 'cosine':
            return CosineAnnealingDropout(**final_params)
        elif dropout_strategy == 'step':
            return StepScheduleDropout(**final_params)
        elif dropout_strategy == 'constant':
            return PartialTeacherForcingDropout(**final_params)
        else:
            raise ValueError(f"Unknown dropout strategy: {dropout_strategy}")

    @property
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""
        return "ptf"

    def update_dropout_epoch(self, epoch: int):
        """
        Update the dropout strategy for the current epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch (1-based)
        """
        if hasattr(self.dropout_module, 'update_epoch'):
            self.dropout_module.update_epoch(epoch)  # type: ignore

    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None
    ) -> CellOutput:
        """
        Forward pass through the Partial Teacher Forcing HCNN Cell.

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
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
            - expectation: Predicted observation (batch_size, n_obs_vars)
            - next_state: Next state (batch_size, n_state_vars)
            - delta_term: Full observation error (y_true - y_pred) if teacher_forcing, else None
            - partial_delta_term: Dropout-masked delta term if teacher_forcing, else None

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

        # Apply partial teacher forcing if requested
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

            # Compute full delta term (observation error)
            delta_term = observation - expectation

            # Apply partial teacher forcing dropout to delta term using the configured strategy
            if not callable(self.dropout_module):
                raise TypeError("Dropout module is not callable. Ensure it is an nn.Module implementing __call__.")
            partial_delta_term = self.dropout_module(delta_term)

            # Apply teacher forcing correction using partial delta
            teach_forc = torch.matmul(partial_delta_term, torch.as_tensor(self.ConMat, device=partial_delta_term.device))
            corrected_state = state - teach_forc

            # Compute next state with correction
            next_state = self.A(torch.tanh(corrected_state)) + external_contribution

            return CellOutput(
                expectation, next_state, delta_term,
                {"partial_delta_terms": partial_delta_term},
            )

        else:
            # No teacher forcing - standard forward pass
            r_state = torch.matmul(state, torch.as_tensor(self.Ide, device=state.device))
            next_state = self.A(torch.tanh(r_state)) + external_contribution

            return CellOutput(expectation, next_state, None, {})

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
