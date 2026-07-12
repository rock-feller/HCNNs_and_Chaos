"""
Classic RNN and LSTM model implementations.

This module provides wrapper classes for standard RNN and LSTM models
for comparison with HCNN variants.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from datetime import datetime
from collections import namedtuple

from ..base import ClassicOutput


class RNNModel(nn.Module):
    """
    Standard RNN Model for time series prediction.

    This model provides a standard RNN implementation for comparison
    with HCNN variants in chaotic systems modeling.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden units
    output_size : int
        Number of output features
    num_layers : int, default=1
        Number of RNN layers
    nonlinearity : Literal['tanh', 'relu'], default='tanh'
        Nonlinearity to use
    bias : bool, default=True
        Whether to use bias
    batch_first : bool, default=True
        Whether batch dimension is first
    dropout : float, default=0.0
        Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection layer
        self.output_layer = nn.Linear(hidden_size, output_size, bias=bias)

        # Generate model name
        self.name = self._generate_model_name()

    def _generate_model_name(self) -> str:
        """Generate a unique model name based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"RNN_in{self.input_size}_hid{self.hidden_size}_out{self.output_size}"
            f"_layers{self.num_layers}_{self.nonlinearity}_{timestamp}"
        )

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> ClassicOutput:
        """
        Forward pass through the RNN model.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_size)
        hidden_state : Optional[torch.Tensor], default=None
            Initial hidden state of shape (num_layers, batch_size, hidden_size)

        Returns
        -------
        ClassicOutput
            Named tuple containing outputs, hidden_states, and forecasts
        """
        batch_size = input_sequence.shape[0]

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, input_sequence.device)

        # Forward pass through RNN
        rnn_output, final_hidden = self.rnn(input_sequence, hidden_state)

        # Project to output space
        outputs = self.output_layer(rnn_output)

        return ClassicOutput(
            outputs=outputs,
            hidden_states=rnn_output,
            forecasts=None
        )

    def forecast(
        self,
        input_sequence: torch.Tensor,
        forecast_horizon: int,
        hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate forecasts using auto-regressive prediction.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input sequence for initialization
        forecast_horizon : int
            Number of steps to forecast
        hidden_state : Optional[torch.Tensor], default=None
            Initial hidden state

        Returns
        -------
        torch.Tensor
            Forecasted outputs of shape (batch_size, forecast_horizon, output_size)
        """
        batch_size = input_sequence.shape[0]
        device = input_sequence.device

        # Get final state from input sequence
        with torch.no_grad():
            _, final_hidden = self.rnn(input_sequence, hidden_state)

        # Initialize forecast tensor
        forecasts = torch.zeros(batch_size, forecast_horizon, self.output_size, device=device)

        # Get last output as starting point
        current_input = self.output_layer(final_hidden[-1:].transpose(0, 1))  # (batch, 1, output_size)
        current_hidden = final_hidden

        # Generate forecasts auto-regressively
        for t in range(forecast_horizon):
            rnn_out, current_hidden = self.rnn(current_input, current_hidden)
            forecast = self.output_layer(rnn_out)
            forecasts[:, t:t+1, :] = forecast
            current_input = forecast

        return forecasts

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class LSTMModel(nn.Module):
    """
    Standard LSTM Model for time series prediction.

    This model provides a standard LSTM implementation for comparison
    with HCNN variants in chaotic systems modeling.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden units
    output_size : int
        Number of output features
    num_layers : int, default=1
        Number of LSTM layers
    bias : bool, default=True
        Whether to use bias
    batch_first : bool, default=True
        Whether batch dimension is first
    dropout : float, default=0.0
        Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection layer
        self.output_layer = nn.Linear(hidden_size, output_size, bias=bias)

        # Generate model name
        self.name = self._generate_model_name()

    def _generate_model_name(self) -> str:
        """Generate a unique model name based on configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"LSTM_in{self.input_size}_hid{self.hidden_size}_out{self.output_size}"
            f"_layers{self.num_layers}_{timestamp}"
        )

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> ClassicOutput:
        """
        Forward pass through the LSTM model.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_size)
        hidden_state : Optional[Tuple[torch.Tensor, torch.Tensor]], default=None
            Initial (hidden, cell) state tuple

        Returns
        -------
        ClassicOutput
            Named tuple containing outputs, hidden_states, and forecasts
        """
        batch_size = input_sequence.shape[0]

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, input_sequence.device)

        # Forward pass through LSTM
        lstm_output, final_state = self.lstm(input_sequence, hidden_state)

        # Project to output space
        outputs = self.output_layer(lstm_output)

        return ClassicOutput(
            outputs=outputs,
            hidden_states=lstm_output,
            forecasts=None
        )

    def forecast(
        self,
        input_sequence: torch.Tensor,
        forecast_horizon: int,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate forecasts using auto-regressive prediction.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input sequence for initialization
        forecast_horizon : int
            Number of steps to forecast
        hidden_state : Optional[Tuple[torch.Tensor, torch.Tensor]], default=None
            Initial (hidden, cell) state tuple

        Returns
        -------
        torch.Tensor
            Forecasted outputs of shape (batch_size, forecast_horizon, output_size)
        """
        batch_size = input_sequence.shape[0]
        device = input_sequence.device

        # Get final state from input sequence
        with torch.no_grad():
            _, final_state = self.lstm(input_sequence, hidden_state)

        # Initialize forecast tensor
        forecasts = torch.zeros(batch_size, forecast_horizon, self.output_size, device=device)

        # Get last output as starting point
        current_input = self.output_layer(final_state[0][-1:].transpose(0, 1))  # (batch, 1, output_size)
        current_state = final_state

        # Generate forecasts auto-regressively
        for t in range(forecast_horizon):
            lstm_out, current_state = self.lstm(current_input, current_state)
            forecast = self.output_layer(lstm_out)
            forecasts[:, t:t+1, :] = forecast
            current_input = forecast

        return forecasts

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states."""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (hidden, cell)
