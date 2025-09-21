"""
Corrected HCNN Models Implementation

This module contains the corrected implementations of Historical Consistent Neural Networks (HCNNs)
following the original src/ folder implementation patterns exactly.

Key fixes for NaN losses:
- Cell-based architecture with proper activation functions
- Teacher forcing mechanism during training
- Connection matrix (ConMat) for expectations
- torch.tanh activation functions
- Complex state evolution with r_state computations
- Proper weight initialization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
import os
import glob

from ..base import BaseHCNNModel, HCNNOutput, PTFHCNNOutput
from ..cells import VanillaHCNNCell, PTFHCNNCell, LFormHCNNCell, LSpaHCNNCell



class Vanilla_Model(BaseHCNNModel):
    """
    Vanilla HCNN: Historical Consistent Neural Network with standard teacher forcing.

    This model implements the foundational HCNN architecture that maintains temporal
    consistency through teacher forcing during training. The model processes sequential
    observations and learns to predict future states while correcting for prediction
    errors using ground truth observations.

    The Vanilla HCNN uses a nonlinear state transition function with tanh activation
    and applies full teacher forcing correction at each time step during training.
    During inference, it operates autonomously without teacher forcing.

    Key Features:
    - Standard teacher forcing mechanism
    - Nonlinear state transitions with tanh activation
    - Support for external variables
    - Autonomous forecasting capabilities
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        n_ext_vars: Optional[int] = None
    ):
        # Call parent constructor first
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars
        )

        # Create the VanillaHCNNCell
        self.cell = VanillaHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            n_ext_vars=n_ext_vars
        )

        self.name = "Vanilla_Model"

    @property
    def model_type(self) -> str:
        """Return the type of HCNN model."""
        return "vanilla_hcnn"

    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        **kwargs
    ) -> HCNNOutput:
        """
        Forward pass through the Vanilla HCNN.

        Processes a sequence of observations through the HCNN with teacher forcing
        during training and optionally generates autonomous forecasts.

        Args:
            data_window (torch.Tensor): Input observation sequence of shape
                [batch_size, seq_len, n_obs_vars]
            forecast_horizon (Optional[int]): Number of future time steps to forecast.
                If None or 0, no forecasting is performed.
            externals (Optional[torch.Tensor]): External variables during training of shape
                [batch_size, seq_len, n_ext_vars]. Required if model was initialized with n_ext_vars.
            future_externals (Optional[torch.Tensor]): External variables for forecasting of shape
                [batch_size, forecast_horizon, n_ext_vars]. Required for forecasting if model uses externals.

        Returns:
            HCNNOutput: Named tuple containing:
                - expectations (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_obs_vars]
                - states (torch.Tensor): Internal states of shape [batch_size, seq_len, n_state_vars]
                - delta_terms (torch.Tensor): Prediction errors of shape [batch_size, seq_len, n_obs_vars]
                - forecasts (Optional[torch.Tensor]): Future predictions of shape
                  [batch_size, forecast_horizon, n_obs_vars] if forecast_horizon > 0, else None
                - future_states (Optional[torch.Tensor]): Future states of shape
                  [batch_size, forecast_horizon, n_state_vars] if forecast_horizon > 0, else None
        """
        batch_size, seq_len, _ = data_window.shape
        device = data_window.device

        # Initialize lists to collect results (avoid in-place operations)
        states_list = []
        expectations_list = []
        delta_terms_list = []

        # Set initial state for all sequences in batch
        current_state = self.s0.to(device).expand(batch_size, -1).clone()
        states_list.append(current_state)

        # Process observed data with teacher forcing
        for t in range(seq_len - 1):
            ext_t = externals[:, t, :] if externals is not None else None
            expectation, next_state, delta_term = self.cell(
                state=current_state,
                teacher_forcing=True,
                observation=data_window[:, t, :],
                externals=ext_t
            )

            expectations_list.append(expectation)
            delta_terms_list.append(delta_term)
            current_state = next_state
            states_list.append(current_state)

        # Final time step
        ext_final = externals[:, seq_len - 1, :] if externals is not None else None
        final_expectation, _, final_delta = self.cell(
            state=current_state,
            teacher_forcing=True,
            observation=data_window[:, seq_len - 1, :],
            externals=ext_final
        )
        expectations_list.append(final_expectation)
        delta_terms_list.append(final_delta)

        # Stack results into tensors
        states = torch.stack(states_list, dim=1)
        expectations = torch.stack(expectations_list, dim=1)
        delta_terms = torch.stack(delta_terms_list, dim=1)
        
        # Forecasting
        forecasts = None
        future_states = None
        
        if forecast_horizon and forecast_horizon > 0:
            forecasts_list = []
            future_states_list = []

            # Initialize forecasting from last state
            current_state = states[:, seq_len - 1, :]

            for t in range(forecast_horizon):
                # Get external variables for forecasting if available
                ext_forecast = future_externals[:, t, :] if future_externals is not None else None

                expectation, next_state, _ = self.cell(
                    state=current_state,
                    teacher_forcing=False,
                    externals=ext_forecast
                )

                forecasts_list.append(expectation)
                future_states_list.append(next_state)
                current_state = next_state

            # Stack results into tensors
            forecasts = torch.stack(forecasts_list, dim=1)
            future_states = torch.stack(future_states_list, dim=1)
        
        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )




class PTF_Model(BaseHCNNModel):
    """
    HCNN with partial teacher forcing mechanism.

    This model implements the Partial Teacher Forcing variant of HCNN that applies
    dropout to the teacher forcing signal during training. Instead of using the full
    prediction error for state correction, PTF-HCNN randomly masks portions of the
    error term, allowing the model to gradually transition from teacher-forced to
    autonomous behavior.

    The partial teacher forcing mechanism helps prevent over-reliance on ground truth
    during training and improves long-term prediction stability. The model supports
    multiple dropout scheduling strategies that control how the dropout probability
    evolves during training.

    Key Features:
    - Partial teacher forcing with configurable dropout strategies
    - Multiple scheduling algorithms (adaptive, linear, exponential, cosine, step, constant)
    - Improved long-term prediction stability
    - Support for external variables
    - Epoch-aware dropout probability adjustment
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
        dropout_params: Optional[dict] = None
    ):
        # Call parent constructor first
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars
        )

        # Store PTF-specific attributes
        self.target_prob = target_prob
        self.drop_output = drop_output
        self.dropout_strategy = dropout_strategy
        self.dropout_params = dropout_params or {}
        self.name = "PTF_Model"

        # Create the PTFHCNNCell with dropout strategy
        self.cell = PTFHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
            dropout_strategy=dropout_strategy,
            dropout_params=dropout_params
        )

    @property
    def model_type(self) -> str:
        """Return the type of HCNN model."""
        return "ptf_hcnn"

    def update_dropout_epoch(self, epoch: int):
        """Update the dropout strategy for the current epoch."""
        self.cell.update_dropout_epoch(epoch)

    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        prob: Optional[float] = None,
        **kwargs
    ) -> PTFHCNNOutput:
        """
        Forward pass through the HCNN with partial teacher forcing mechanism.

        Processes a sequence of observations with partial teacher forcing, where dropout
        is applied to the prediction error before state correction. This allows gradual
        transition from teacher-forced to autonomous behavior during training.

        Args:
            data_window (torch.Tensor): Input observation sequence of shape
                [batch_size, seq_len, n_obs_vars]
            forecast_horizon (Optional[int]): Number of future time steps to forecast.
                If None or 0, no forecasting is performed.
            externals (Optional[torch.Tensor]): External variables during training of shape
                [batch_size, seq_len, n_ext_vars]. Required if model was initialized with n_ext_vars.
            future_externals (Optional[torch.Tensor]): External variables for forecasting of shape
                [batch_size, forecast_horizon, n_ext_vars]. Required for forecasting if model uses externals.
            prob (Optional[float]): Dropout probability for partial teacher forcing. If None,
                uses target_prob during training and 0.0 during evaluation.

        Returns:
            PTFHCNNOutput: Named tuple containing:
                - expectations (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_obs_vars]
                - states (torch.Tensor): Internal states of shape [batch_size, seq_len, n_state_vars]
                - delta_terms (torch.Tensor): Full prediction errors of shape [batch_size, seq_len, n_obs_vars]
                - partial_delta_terms (torch.Tensor): Dropout-masked errors of shape [batch_size, seq_len, n_obs_vars]
                - forecasts (Optional[torch.Tensor]): Future predictions of shape
                  [batch_size, forecast_horizon, n_obs_vars] if forecast_horizon > 0, else None
                - future_states (Optional[torch.Tensor]): Future states of shape
                  [batch_size, forecast_horizon, n_state_vars] if forecast_horizon > 0, else None
        """
        if prob is None:
            prob = self.target_prob if self.training else 0.0

        batch_size, seq_len, _ = data_window.shape
        device = data_window.device

        # Initialize lists to collect results (avoid in-place operations)
        states_list = []
        expectations_list = []
        delta_terms_list = []
        partial_delta_terms_list = []

        # Set initial state for all sequences in batch
        current_state = self.s0.to(device).expand(batch_size, -1).clone()
        states_list.append(current_state)

        # Process observed data with partial teacher forcing
        for t in range(seq_len - 1):
            ext_t = externals[:, t, :] if externals is not None else None
            expectation, next_state, delta_term, partial_delta_term = self.cell(
                state=current_state,
                teacher_forcing=True,
                observation=data_window[:, t, :],
                externals=ext_t
            )

            expectations_list.append(expectation)
            delta_terms_list.append(delta_term)
            partial_delta_terms_list.append(partial_delta_term)
            current_state = next_state
            states_list.append(current_state)

        # Final time step
        ext_final = externals[:, seq_len - 1, :] if externals is not None else None
        final_expectation, _, final_delta, final_partial_delta = self.cell(
            state=current_state,
            teacher_forcing=True,
            observation=data_window[:, seq_len - 1, :],
            externals=ext_final
        )
        expectations_list.append(final_expectation)
        delta_terms_list.append(final_delta)
        partial_delta_terms_list.append(final_partial_delta)

        # Stack results into tensors
        states = torch.stack(states_list, dim=1)
        expectations = torch.stack(expectations_list, dim=1)
        delta_terms = torch.stack(delta_terms_list, dim=1)
        partial_delta_terms = torch.stack(partial_delta_terms_list, dim=1)

        # Forecasting
        forecasts = None
        future_states = None

        if forecast_horizon and forecast_horizon > 0:
            forecasts_list = []
            future_states_list = []

            # Initialize forecasting from last state
            current_state = states[:, seq_len - 1, :]

            for t in range(forecast_horizon):
                # Get external variables for forecasting if available
                ext_forecast = future_externals[:, t, :] if future_externals is not None else None

                expectation, next_state, _, _ = self.cell(
                    state=current_state,
                    teacher_forcing=False,
                    externals=ext_forecast
                )

                forecasts_list.append(expectation)
                future_states_list.append(next_state)
                current_state = next_state

            # Stack results into tensors
            forecasts = torch.stack(forecasts_list, dim=1)
            future_states = torch.stack(future_states_list, dim=1)

        return PTFHCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            partial_delta_terms=partial_delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )


# Temporary aliases for LForm_Model and LSpa_Model until fully implemented
# These use the Vanilla_Model as a base with modified names
class LForm_Model(BaseHCNNModel):
    """
    HCNN with LSTM formulation.

    This model implements the LSTM-inspired variant of HCNN that incorporates
    memory-preserving mechanisms through a learnable diagonal matrix and residual
    connections. The LForm-HCNN combines nonlinear state transformations with
    diagonal modulation to regulate information flow and improve long-term memory
    retention.

    The model uses an LSTM-like residual block where the diagonal matrix acts as
    a learnable memory gate, controlling how much of the transformed residual
    contributes to the next state. This design helps preserve important information
    while allowing for complex nonlinear dynamics.

    Key Features:
    - LSTM-inspired residual connections
    - Learnable diagonal matrix for memory modulation
    - Improved gradient flow through residual blocks
    - Enhanced long-term memory retention
    - Support for external variables
    - Constrained diagonal elements for numerical stability
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_diag: float = 1.0,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None
    ):
        # Call parent constructor first
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars
        )

        # Store LForm-specific attributes
        self.init_diag = init_diag
        self.name = "LForm_Model"

        # Create the LFormHCNNCell with diagonal matrix
        self.cell = LFormHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            init_diag=init_diag,
            n_ext_vars=n_ext_vars
        )

    @property
    def model_type(self) -> str:
        """Return the type of HCNN model."""
        return "lform_hcnn"

    def get_diagonal_values(self) -> torch.Tensor:
        """Get current diagonal values from the cell's diagonal matrix."""
        return self.cell.get_diagonal_values()

    def get_diagonal_matrix(self) -> torch.Tensor:
        """Get the current diagonal matrix weights."""
        return self.cell.get_diagonal_matrix()

    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        **kwargs
    ) -> HCNNOutput:
        """
        Forward pass through the HCNN with LSTM formulation.

        Processes a sequence of observations through the LSTM-inspired HCNN with
        diagonal matrix modulation and residual connections. The model applies
        teacher forcing during training and can generate autonomous forecasts.

        Args:
            data_window (torch.Tensor): Input observation sequence of shape
                [batch_size, seq_len, n_obs_vars]
            forecast_horizon (Optional[int]): Number of future time steps to forecast.
                If None or 0, no forecasting is performed.
            externals (Optional[torch.Tensor]): External variables during training of shape
                [batch_size, seq_len, n_ext_vars]. Required if model was initialized with n_ext_vars.
            future_externals (Optional[torch.Tensor]): External variables for forecasting of shape
                [batch_size, forecast_horizon, n_ext_vars]. Required for forecasting if model uses externals.

        Returns:
            HCNNOutput: Named tuple containing:
                - expectations (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_obs_vars]
                - states (torch.Tensor): Internal states of shape [batch_size, seq_len, n_state_vars]
                - delta_terms (torch.Tensor): Prediction errors of shape [batch_size, seq_len, n_obs_vars]
                - forecasts (Optional[torch.Tensor]): Future predictions of shape
                  [batch_size, forecast_horizon, n_obs_vars] if forecast_horizon > 0, else None
                - future_states (Optional[torch.Tensor]): Future states of shape
                  [batch_size, forecast_horizon, n_state_vars] if forecast_horizon > 0, else None
        """
        batch_size, seq_len, _ = data_window.shape
        device = data_window.device

        # Initialize lists to collect results (avoid in-place operations)
        states_list = []
        expectations_list = []
        delta_terms_list = []

        # Set initial state for all sequences in batch
        current_state = self.s0.to(device).expand(batch_size, -1).clone()
        states_list.append(current_state)

        # Process observed data with teacher forcing
        for t in range(seq_len - 1):
            ext_t = externals[:, t, :] if externals is not None else None
            expectation, next_state, delta_term = self.cell(
                state=current_state,
                teacher_forcing=True,
                observation=data_window[:, t, :],
                externals=ext_t
            )

            expectations_list.append(expectation)
            delta_terms_list.append(delta_term)
            current_state = next_state
            states_list.append(current_state)

        # Final time step
        ext_final = externals[:, seq_len - 1, :] if externals is not None else None
        final_expectation, _, final_delta = self.cell(
            state=current_state,
            teacher_forcing=True,
            observation=data_window[:, seq_len - 1, :],
            externals=ext_final
        )
        expectations_list.append(final_expectation)
        delta_terms_list.append(final_delta)

        # Stack results into tensors
        states = torch.stack(states_list, dim=1)
        expectations = torch.stack(expectations_list, dim=1)
        delta_terms = torch.stack(delta_terms_list, dim=1)

        # Forecasting
        forecasts = None
        future_states = None

        if forecast_horizon and forecast_horizon > 0:
            forecasts_list = []
            future_states_list = []

            # Initialize forecasting from last state
            current_state = states[:, seq_len - 1, :]

            for t in range(forecast_horizon):
                # Get external variables for forecasting if available
                ext_forecast = future_externals[:, t, :] if future_externals is not None else None

                expectation, next_state, _ = self.cell(
                    state=current_state,
                    teacher_forcing=False,
                    externals=ext_forecast
                )

                forecasts_list.append(expectation)
                future_states_list.append(next_state)
                current_state = next_state

            # Stack results into tensors
            forecasts = torch.stack(forecasts_list, dim=1)
            future_states = torch.stack(future_states_list, dim=1)

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )


class LSpa_Model(BaseHCNNModel):
    """
    HCNN with large sparse transition matrices.

    This model implements the Large Sparse variant of HCNN designed for high-dimensional
    dynamical systems. It uses structured sparsity patterns in the state transition
    matrix to enable efficient computation while maintaining modeling capability for
    systems with large state spaces (100-200 variables).

    The model supports two sparsity patterns: random block sparsity for uniform
    sparsification across the entire matrix, and non-observable block sparsity
    that preserves connections related to observable variables while sparsifying
    hidden-to-hidden connections.

    Key Features:
    - Structured sparsity for computational efficiency
    - Multiple sparsity patterns (random_block, non_obs_block)
    - Configurable sparsity ratios (0.0 to 1.0)
    - Significant memory and computational savings
    - Support for high-dimensional state spaces
    - Gradient preservation through non-masked connections
    - Support for external variables
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
        n_ext_vars: Optional[int] = None
    ):
        # Call parent constructor first
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars
        )

        # Store LSpa-specific attributes
        self.sparsity_ratio = sparsity_ratio
        self.mask_type = mask_type
        self.bias = bias
        self.name = "LSpa_Model"

        # Create the LSpaHCNNCell with sparsity configuration
        self.cell = LSpaHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            bias=bias,
            mask_type=mask_type,
            sparsity_ratio=sparsity_ratio,
            n_ext_vars=n_ext_vars
        )

    @property
    def model_type(self) -> str:
        """Return the type of HCNN model."""
        return "lspa_hcnn"

    def get_sparsity_info(self) -> dict:
        """Get information about the current sparsity pattern from the cell."""
        return self.cell.get_sparsity_info()

    def visualize_sparsity_pattern(self) -> torch.Tensor:
        """Get the current sparsity pattern for visualization from the cell."""
        return self.cell.visualize_sparsity_pattern()

    def get_sparse_transition_matrix(self) -> torch.Tensor:
        """Get the current sparse state transition matrix weights."""
        return self.cell.get_sparse_transition_matrix()

    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        **kwargs
    ) -> HCNNOutput:
        """
        Forward pass through the HCNN with large sparse transition matrices.

        Processes a sequence of observations through the sparse HCNN using structured
        sparsity patterns for efficient computation. The model applies teacher forcing
        during training and can generate autonomous forecasts while maintaining
        computational efficiency through sparse matrix operations.

        Args:
            data_window (torch.Tensor): Input observation sequence of shape
                [batch_size, seq_len, n_obs_vars]
            forecast_horizon (Optional[int]): Number of future time steps to forecast.
                If None or 0, no forecasting is performed.
            externals (Optional[torch.Tensor]): External variables during training of shape
                [batch_size, seq_len, n_ext_vars]. Required if model was initialized with n_ext_vars.
            future_externals (Optional[torch.Tensor]): External variables for forecasting of shape
                [batch_size, forecast_horizon, n_ext_vars]. Required for forecasting if model uses externals.

        Returns:
            HCNNOutput: Named tuple containing:
                - expectations (torch.Tensor): Model predictions of shape [batch_size, seq_len, n_obs_vars]
                - states (torch.Tensor): Internal states of shape [batch_size, seq_len, n_state_vars]
                - delta_terms (torch.Tensor): Prediction errors of shape [batch_size, seq_len, n_obs_vars]
                - forecasts (Optional[torch.Tensor]): Future predictions of shape
                  [batch_size, forecast_horizon, n_obs_vars] if forecast_horizon > 0, else None
                - future_states (Optional[torch.Tensor]): Future states of shape
                  [batch_size, forecast_horizon, n_state_vars] if forecast_horizon > 0, else None
        """
        batch_size, seq_len, _ = data_window.shape
        device = data_window.device

        # Initialize lists to collect results (avoid in-place operations)
        states_list = []
        expectations_list = []
        delta_terms_list = []

        # Set initial state for all sequences in batch
        current_state = self.s0.to(device).expand(batch_size, -1).clone()
        states_list.append(current_state)

        # Process observed data with teacher forcing
        for t in range(seq_len - 1):
            ext_t = externals[:, t, :] if externals is not None else None
            expectation, next_state, delta_term = self.cell(
                state=current_state,
                teacher_forcing=True,
                observation=data_window[:, t, :],
                externals=ext_t
            )

            expectations_list.append(expectation)
            delta_terms_list.append(delta_term)
            current_state = next_state
            states_list.append(current_state)

        # Final time step
        ext_final = externals[:, seq_len - 1, :] if externals is not None else None
        final_expectation, _, final_delta = self.cell(
            state=current_state,
            teacher_forcing=True,
            observation=data_window[:, seq_len - 1, :],
            externals=ext_final
        )
        expectations_list.append(final_expectation)
        delta_terms_list.append(final_delta)

        # Stack results into tensors
        states = torch.stack(states_list, dim=1)
        expectations = torch.stack(expectations_list, dim=1)
        delta_terms = torch.stack(delta_terms_list, dim=1)

        # Forecasting
        forecasts = None
        future_states = None

        if forecast_horizon and forecast_horizon > 0:
            forecasts_list = []
            future_states_list = []

            # Initialize forecasting from last state
            current_state = states[:, seq_len - 1, :]

            for t in range(forecast_horizon):
                # Get external variables for forecasting if available
                ext_forecast = future_externals[:, t, :] if future_externals is not None else None

                expectation, next_state, _ = self.cell(
                    state=current_state,
                    teacher_forcing=False,
                    externals=ext_forecast
                )

                forecasts_list.append(expectation)
                future_states_list.append(next_state)
                current_state = next_state

            # Stack results into tensors
            forecasts = torch.stack(forecasts_list, dim=1)
            future_states = torch.stack(future_states_list, dim=1)

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )
