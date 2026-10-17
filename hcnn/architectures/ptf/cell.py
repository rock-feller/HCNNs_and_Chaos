"""
Partial Teacher Forcing (PTF) HCNN cell implementation.


This module contains the PTF HCNN cell that incorporates partial teacher forcing
via dropout scaling during training.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ...core.base import BaseHCNNCell
from ...core.defaults import DEFAULT_INIT_RANGE
from ...core.layers.linear import CustomLinear
from ...core.layers.dropout import (
    
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
    init_range : Tuple[float, float], default=DEFAULT_INIT_RANGE
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
    dropout_strategy : str
        String indicating the dropout strategy type
    dropout_module : nn.Module
        Dropout module implementing the chosen strategy for partial teacher forcing
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = DEFAULT_INIT_RANGE,
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

        # External input matrix B (optional), created after A
        self._init_external_matrix()

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

    def correction(self, delta: torch.Tensor) -> torch.Tensor:
        """g(delta_t) = m_t * delta_t: the PTF dropout mask on the teacher-forcing error.
        Observed coordinates whose correction is kept are replaced by y_t; dropped ones
        (m_t = 0) keep the model's own prediction. In eval() the mask keeps everything."""
        return self.dropout_module(delta)

    def step_extras(self, delta: torch.Tensor, correction: torch.Tensor):
        """Return the masked error as ``partial_delta_terms``."""
        return {"partial_delta_terms": correction}

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
