"""
Large Sparse (LSpa) HCNN cell implementation.

This module contains the Large Sparse HCNN cell that supports structured sparsity
for efficient computation in high-dimensional dynamical systems.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from ..base import BaseHCNNCell
from ..layers.sparse import  CustomSparseLinear
from ..layers.linear import CustomLinear



class LSpaHCNNCell(BaseHCNNCell):
    """
    Large Sparse HCNN Cell.

    Implements a scalable variant of the Historical Consistent Neural Network (HCNN) cell
    designed for high-dimensional dynamical systems using sparsity-aware nonlinear transformation.
    It performs a state-to-state mapping using sparsity-aware nonlinear transformation
    and produces outputs based on hidden states.

    This cell supports structured sparsity for efficient computation and also supports
    teacher forcing during training. The key innovation is the use of a sparse transformation
    matrix that can handle large state spaces efficiently.

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables (i.e., the dimensionality of the observed state variables)
    n_hid_vars : int
        Number of hidden variables (i.e., the dimensionality of the hidden state variables)
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Range for uniform initialization of weights
    bias : bool, default=False
        Whether to include bias terms in linear transformations
    mask_type : Literal['non_obs_block', 'random_block'], default='random_block'
        Type of sparsity mask to apply:
        - 'random_block': Applies uniform random sparsity over the entire weight matrix
        - 'non_obs_block': Applies sparsity only to the non-observable block of the matrix
    sparsity_ratio : float, default=0.0
        Proportion of weights to set to zero. Must be in the range [0.0, 1.0]
    n_ext_vars : Optional[int], default=None
        Number of external variables

    Attributes
    ----------
    Sparse_A : CustomSparseLinear
        Sparse state transition matrix with structured sparsity
    B : CustomLinear, optional
        External input matrix, only if n_ext_vars is provided
    ConMat : torch.Tensor
        Observation matrix that maps hidden states to observed outputs
    Ide : torch.Tensor
        Identity matrix for state operations
    bias : bool
        Whether bias terms are included
    mask_type : str
        Type of sparsity mask applied
    sparsity_ratio : float
        Sparsity ratio applied to the transformation matrix
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        bias: bool = False,
        mask_type: Literal['non_obs_block', 'random_block'] = 'random_block',
        sparsity_ratio: float = 0.0,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_obs_vars, n_hid_vars, init_range, n_ext_vars)
        
        self.bias = bias
        self.mask_type = mask_type
        self.sparsity_ratio = sparsity_ratio

        # Sparse state transition matrix
        self.Sparse_A = CustomSparseLinear(
            n_obs_vars=self.n_obs_vars,
            n_hid_vars=self.n_hid_vars,
            bias=self.bias,
            init_range=self.init_range,
            sparsity_ratio=self.sparsity_ratio,
            mask_type=self.mask_type
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
        return "lspa"

    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Large Sparse HCNN Cell.

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
            corrected_state = state - teach_forc

            # Compute next state using sparse transformation
            next_state = self.Sparse_A(torch.tanh(corrected_state)) + external_contribution

            return expectation, next_state, delta_term

        else:
            # No teacher forcing - standard forward pass
            r_state = torch.matmul(state, torch.as_tensor(self.Ide, device=state.device))
            next_state = self.Sparse_A(torch.tanh(r_state)) + external_contribution

            return expectation, next_state, None

    def get_observation_matrix(self) -> torch.Tensor:
        """Get the observation matrix."""
        return torch.as_tensor(self.ConMat, device=next(self.parameters()).device)

    def get_sparse_transition_matrix(self) -> torch.Tensor:
        """Get the current sparse state transition matrix weights."""
        return self.Sparse_A.weight

    def get_sparsity_info(self) -> dict:
        """Get information about the current sparsity pattern."""
        return self.Sparse_A.get_sparsity_info()

    def visualize_sparsity_pattern(self) -> torch.Tensor:
        """Get the current sparsity pattern for visualization."""
        return self.Sparse_A.visualize_sparsity_pattern()

    def get_external_input_matrix(self) -> Optional[torch.Tensor]:
        """Get the external input matrix weights if available."""
        return self.B.weight if self.B is not None else None

    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        self.Sparse_A.reset_parameters()
        if self.B is not None:
            self.B.reset_parameters()

    def extra_repr(self) -> str:
        """Return extra representation string for the cell."""
        return (
            f'n_obs_vars={self.n_obs_vars}, '
            f'n_hid_vars={self.n_hid_vars}, '
            f'n_ext_vars={self.n_ext_vars}, '
            f'init_range={self.init_range}, '
            f'bias={self.bias}, '
            f'mask_type={self.mask_type}, '
            f'sparsity_ratio={self.sparsity_ratio}'
        )
