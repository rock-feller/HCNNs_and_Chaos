"""
Large Sparse (LSpa) HCNN cell implementation.


This module contains the Large Sparse HCNN cell that supports structured sparsity
for efficient computation in high-dimensional dynamical systems.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from ...core.base import BaseHCNNCell
from ...core.defaults import DEFAULT_INIT_RANGE
from ...core.layers.sparse import  CustomSparseLinear
from ...core.layers.linear import CustomLinear



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
    init_range : Tuple[float, float], default=DEFAULT_INIT_RANGE
        Range for uniform initialization of weights
    bias : bool, default=False
        Whether to include bias terms in linear transformations
    mask_type : Literal['non_obs_block', 'random_block'], default='random_block'
        Type of sparsity mask to apply:
        - 'random_block': Applies uniform random sparsity over the entire weight matrix
        - 'non_obs_block': Applies sparsity only to the columns that read the hidden
          variables (the hid_to_obs and hid_to_hid blocks)
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
        init_range: Tuple[float, float] = DEFAULT_INIT_RANGE,
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

        # External input matrix B (optional), created after the transition parameters
        self._init_external_matrix()

    @property
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""
        return "lspa"

    def transition(self, r: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = (M * A) tanh(r_t) (before external inputs)."""
        return self.Sparse_A(torch.tanh(r))

    def get_sparse_transition_matrix(self) -> torch.Tensor:
        """Get the current sparse state transition matrix weights."""
        return self.Sparse_A.weight

    def get_sparsity_info(self) -> dict:
        """Get information about the current sparsity pattern."""
        return self.Sparse_A.get_sparsity_info()

    def visualize_sparsity_pattern(self) -> torch.Tensor:
        """Get the current sparsity pattern for visualization."""
        return self.Sparse_A.visualize_sparsity_pattern()

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
