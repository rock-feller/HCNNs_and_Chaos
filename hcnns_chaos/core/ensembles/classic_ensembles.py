"""
Classic model ensemble implementations.

This module provides ensemble classes for RNN and LSTM models
for comparison with HCNN ensembles.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Literal, Tuple
from datetime import datetime

from ..base import BaseEnsemble, ClassicOutput
from ..models.classic_models import RNNModel, LSTMModel


class RNNEnsemble(BaseEnsemble):
    """
    Ensemble of RNN models.

    Parameters
    ----------
    n_ensemble : int
        Number of models in the ensemble
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
        n_ensemble: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        nonlinearity: Literal['tanh', 'relu'] = 'tanh',
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super().__init__(n_ensemble)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # Create ensemble models
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            model = RNNModel(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout
            )
            self.models[f"model_{i}"] = model

        self.name = self._generate_ensemble_name()

    @property
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        return "rnn_ensemble"

    def _generate_ensemble_name(self) -> str:
        """Generate a unique ensemble name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"RNNEnsemble_n{self.n_ensemble}_in{self.input_size}_hid{self.hidden_size}"
            f"_out{self.output_size}_layers{self.num_layers}_{self.nonlinearity}_{timestamp}"
        )

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean"
    ) -> ClassicOutput:
        """
        Forward pass through the RNN ensemble.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_size)
        hidden_state : Optional[torch.Tensor], default=None
            Initial hidden state
        aggregation_method : str, default="mean"
            Method to aggregate ensemble outputs

        Returns
        -------
        ClassicOutput
            Aggregated ensemble output
        """
        outputs = []
        for model in self.models.values():
            output = model(input_sequence, hidden_state)
            outputs.append(output)

        return self.aggregate_outputs(outputs, aggregation_method)

    def aggregate_outputs(self, outputs: List[ClassicOutput], method: str = "mean") -> ClassicOutput:
        """
        Aggregate outputs from ensemble members.

        Parameters
        ----------
        outputs : List[ClassicOutput]
            List of outputs from ensemble members
        method : str, default="mean"
            Aggregation method

        Returns
        -------
        ClassicOutput
            Aggregated output
        """
        if method == "mean":
            aggregated_outputs = torch.stack([out.outputs for out in outputs]).mean(dim=0)
            aggregated_hidden = torch.stack([out.hidden_states for out in outputs]).mean(dim=0)
            
            return ClassicOutput(
                outputs=aggregated_outputs,
                hidden_states=aggregated_hidden,
                forecasts=None
            )
        else:
            raise ValueError(f"Aggregation method {method} not implemented")

    def forecast_ensemble(
        self,
        input_sequence: torch.Tensor,
        forecast_horizon: int,
        hidden_state: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean"
    ) -> torch.Tensor:
        """
        Generate ensemble forecasts.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Input sequence for initialization
        forecast_horizon : int
            Number of steps to forecast
        hidden_state : Optional[torch.Tensor], default=None
            Initial hidden state
        aggregation_method : str, default="mean"
            Method to aggregate forecasts

        Returns
        -------
        torch.Tensor
            Aggregated forecasts
        """
        forecasts = []
        for model in self.models.values():
            forecast = model.forecast(input_sequence, forecast_horizon, hidden_state)
            forecasts.append(forecast)

        if aggregation_method == "mean":
            return torch.stack(forecasts).mean(dim=0)
        elif aggregation_method == "median":
            return torch.stack(forecasts).median(dim=0)[0]
        else:
            raise ValueError(f"Aggregation method {aggregation_method} not implemented")

    def get_model(self, index: int) -> RNNModel:
        """Get a specific model from the ensemble."""
        return self.models[f"model_{index}"]


class LSTMEnsemble(BaseEnsemble):
    """
    Ensemble of LSTM models.

    Parameters
    ----------
    n_ensemble : int
        Number of models in the ensemble
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
        n_ensemble: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super().__init__(n_ensemble)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # Create ensemble models
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout
            )
            self.models[f"model_{i}"] = model

        self.name = self._generate_ensemble_name()

    @property
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        return "lstm_ensemble"

    def _generate_ensemble_name(self) -> str:
        """Generate a unique ensemble name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"LSTMEnsemble_n{self.n_ensemble}_in{self.input_size}_hid{self.hidden_size}"
            f"_out{self.output_size}_layers{self.num_layers}_{timestamp}"
        )

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        aggregation_method: str = "mean"
    ) -> ClassicOutput:
        """Forward pass through the LSTM ensemble."""
        outputs = []
        for model in self.models.values():
            output = model(input_sequence, hidden_state)
            outputs.append(output)

        return self.aggregate_outputs(outputs, aggregation_method)

    def aggregate_outputs(self, outputs: List[ClassicOutput], method: str = "mean") -> ClassicOutput:
        """Aggregate outputs from ensemble members."""
        if method == "mean":
            aggregated_outputs = torch.stack([out.outputs for out in outputs]).mean(dim=0)
            aggregated_hidden = torch.stack([out.hidden_states for out in outputs]).mean(dim=0)
            
            return ClassicOutput(
                outputs=aggregated_outputs,
                hidden_states=aggregated_hidden,
                forecasts=None
            )
        else:
            raise ValueError(f"Aggregation method {method} not implemented")

    def forecast_ensemble(
        self,
        input_sequence: torch.Tensor,
        forecast_horizon: int,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        aggregation_method: str = "mean"
    ) -> torch.Tensor:
        """Generate ensemble forecasts."""
        forecasts = []
        for model in self.models.values():
            forecast = model.forecast(input_sequence, forecast_horizon, hidden_state)
            forecasts.append(forecast)

        if aggregation_method == "mean":
            return torch.stack(forecasts).mean(dim=0)
        elif aggregation_method == "median":
            return torch.stack(forecasts).median(dim=0)[0]
        else:
            raise ValueError(f"Aggregation method {aggregation_method} not implemented")

    def get_model(self, index: int) -> LSTMModel:
        """Get a specific model from the ensemble."""
        return self.models[f"model_{index}"]
