"""
HCNN ensemble implementations.


This module provides ensemble classes for all HCNN variants,
enabling improved prediction accuracy through model averaging.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Literal, Union
from datetime import datetime

from ..base import BaseEnsemble, HCNNOutput, PTFHCNNOutput
from ..models.hcnn_models import Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model


class VanillaHCNNEnsemble(BaseEnsemble):
    """
    Ensemble of Vanilla HCNN models.

    This class manages multiple Vanilla HCNN models and provides
    ensemble prediction capabilities with various aggregation methods.

    Parameters
    ----------
    n_ensemble : int
        Number of models in the ensemble
    n_obs_vars : int
        Number of observable variables
    n_hid_vars : int
        Number of hidden variables
    s0_nature : Literal['zeros_', 'random_']
        Initial state initialization strategy
    train_s0 : bool
        Whether initial state is trainable
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Weight initialization range
    n_ext_vars : Optional[int], default=None
        Number of external variables
    """

    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'],
        train_s0: bool,
        init_range: tuple = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_ensemble)
        
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.init_range = init_range
        self.n_ext_vars = n_ext_vars

        # Create ensemble models
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            model = Vanilla_Model(
                n_obs_vars=n_obs_vars,
                n_hid_vars=n_hid_vars,
                s0_nature=s0_nature,
                train_s0=train_s0,
                init_range=init_range,
                n_ext_vars=n_ext_vars
            )
            self.models[f"model_{i}"] = model

        # Generate ensemble name
        self.name = self._generate_ensemble_name()

    @property
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        return "vanilla_hcnn_ensemble"

    def _generate_ensemble_name(self) -> str:
        """Generate a unique ensemble name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s0_str = "zeroInit" if self.s0_nature == "zeros_" else "randInit"
        trainable_str = "trainable" if self.train_s0 else "fixed"
        ext_str = f"_ext{self.n_ext_vars}" if self.n_ext_vars else ""
        
        return (
            f"VanillaHCNNEnsemble_n{self.n_ensemble}_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
            f"_{s0_str}_{trainable_str}{ext_str}_{timestamp}"
        )

    def forward(
        self,
        data_window: torch.Tensor,
        ext_data_window: Optional[torch.Tensor] = None,
        forecast_horizon: Optional[int] = None,
        future_externals: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean"
    ) -> HCNNOutput:
        """
        Forward pass through the ensemble.

        Parameters
        ----------
        data_window : torch.Tensor
            Input observation sequence
        ext_data_window : Optional[torch.Tensor], default=None
            External variables for input sequence
        forecast_horizon : Optional[int], default=None
            Number of future steps to forecast
        future_externals : Optional[torch.Tensor], default=None
            External variables for forecasting
        aggregation_method : str, default="mean"
            Method to aggregate ensemble outputs

        Returns
        -------
        HCNNOutput
            Aggregated ensemble output
        """
        # Collect outputs from all models
        outputs = []
        for model in self.models.values():
            output = model(
                data_window=data_window,
                ext_data_window=ext_data_window,
                forecast_horizon=forecast_horizon,
                future_externals=future_externals
            )
            outputs.append(output)

        # Aggregate outputs
        return self.aggregate_outputs(outputs, aggregation_method)

    def aggregate_outputs(self, outputs: List[HCNNOutput], method: str = "mean") -> HCNNOutput:
        """
        Aggregate outputs from ensemble members.

        Parameters
        ----------
        outputs : List[HCNNOutput]
            List of outputs from ensemble members
        method : str, default="mean"
            Aggregation method ('mean', 'median', 'weighted_mean')

        Returns
        -------
        HCNNOutput
            Aggregated output
        """
        if method == "mean":
            return self._aggregate_mean(outputs)
        elif method == "median":
            return self._aggregate_median(outputs)
        elif method == "weighted_mean":
            return self._aggregate_weighted_mean(outputs)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _aggregate_mean(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using mean."""
        expectations = torch.stack([out.expectations for out in outputs]).mean(dim=0)
        states = torch.stack([out.states for out in outputs]).mean(dim=0)
        delta_terms = torch.stack([out.delta_terms for out in outputs]).mean(dim=0)
        
        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).mean(dim=0)
            future_states = torch.stack([out.future_states for out in outputs]).mean(dim=0)

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_median(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using median."""
        expectations = torch.stack([out.expectations for out in outputs]).median(dim=0)[0]
        states = torch.stack([out.states for out in outputs]).median(dim=0)[0]
        delta_terms = torch.stack([out.delta_terms for out in outputs]).median(dim=0)[0]
        
        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).median(dim=0)[0]
            future_states = torch.stack([out.future_states for out in outputs]).median(dim=0)[0]

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_weighted_mean(self, outputs: List[HCNNOutput], weights: Optional[torch.Tensor] = None) -> HCNNOutput:
        """Aggregate using weighted mean."""
        if weights is None:
            # Create weights on the same device as the outputs
            device = outputs[0].expectations.device
            weights = torch.ones(len(outputs), device=device) / len(outputs)

        weights = weights.view(-1, 1, 1, 1)  # Reshape for broadcasting
        
        expectations = torch.stack([out.expectations for out in outputs])
        expectations = (expectations * weights).sum(dim=0)
        
        states = torch.stack([out.states for out in outputs])
        states = (states * weights).sum(dim=0)
        
        delta_terms = torch.stack([out.delta_terms for out in outputs])
        delta_terms = (delta_terms * weights).sum(dim=0)
        
        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs])
            forecasts = (forecasts * weights).sum(dim=0)
            
            future_states = torch.stack([out.future_states for out in outputs])
            future_states = (future_states * weights).sum(dim=0)

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def get_model(self, index: int) -> Vanilla_Model:
        """Get a specific model from the ensemble."""
        return self.models[f"model_{index}"]

    def get_ensemble_predictions_variance(
        self,
        data_window: torch.Tensor,
        ext_data_window: Optional[torch.Tensor] = None,
        forecast_horizon: Optional[int] = None,
        future_externals: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get prediction variance across ensemble members.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing mean predictions and variances
        """
        outputs = []
        for model in self.models.values():
            output = model(
                data_window=data_window,
                ext_data_window=ext_data_window,
                forecast_horizon=forecast_horizon,
                future_externals=future_externals
            )
            outputs.append(output)

        # Calculate statistics
        expectations_stack = torch.stack([out.expectations for out in outputs])
        expectations_mean = expectations_stack.mean(dim=0)
        expectations_var = expectations_stack.var(dim=0)

        result = {
            "expectations_mean": expectations_mean,
            "expectations_var": expectations_var,
        }

        if outputs[0].forecasts is not None:
            forecasts_stack = torch.stack([out.forecasts for out in outputs])
            result["forecasts_mean"] = forecasts_stack.mean(dim=0)
            result["forecasts_var"] = forecasts_stack.var(dim=0)

        return result


class PTFHCNNEnsemble(BaseEnsemble):
    """
    Ensemble of Partial Teacher Forcing HCNN models.

    Parameters
    ----------
    n_ensemble : int
        Number of models in the ensemble
    n_obs_vars : int
        Number of observable variables
    n_hid_vars : int
        Number of hidden variables
    s0_nature : Literal['zeros_', 'random_']
        Initial state initialization strategy
    train_s0 : bool
        Whether initial state is trainable
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Weight initialization range
    target_prob : float, default=0.25
        Default dropout probability for partial teacher forcing
    n_ext_vars : Optional[int], default=None
        Number of external variables
    """

    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'],
        train_s0: bool,
        init_range: tuple = (-0.75, 0.75),
        target_prob: float = 0.25,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_ensemble)

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.init_range = init_range
        self.target_prob = target_prob
        self.n_ext_vars = n_ext_vars

        # Create ensemble models
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            model = PTF_Model(
                n_obs_vars=n_obs_vars,
                n_hid_vars=n_hid_vars,
                s0_nature=s0_nature,
                train_s0=train_s0,
                init_range=init_range,
                target_prob=target_prob,
                n_ext_vars=n_ext_vars
            )
            self.models[f"model_{i}"] = model

        self.name = self._generate_ensemble_name()

    @property
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        return "ptf_hcnn_ensemble"

    def _generate_ensemble_name(self) -> str:
        """Generate a unique ensemble name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s0_str = "zeroInit" if self.s0_nature == "zeros_" else "randInit"
        trainable_str = "trainable" if self.train_s0 else "fixed"
        ext_str = f"_ext{self.n_ext_vars}" if self.n_ext_vars else ""

        return (
            f"PTFHCNNEnsemble_n{self.n_ensemble}_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
            f"_{s0_str}_{trainable_str}_prob{self.target_prob}{ext_str}_{timestamp}"
        )

    def forward(
        self,
        data_window: torch.Tensor,
        prob: float = 0.0,
        ext_data_window: Optional[torch.Tensor] = None,
        forecast_horizon: Optional[int] = None,
        future_externals: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean"
    ) -> PTFHCNNOutput:
        """Forward pass through the PTF HCNN ensemble."""
        outputs = []
        for model in self.models.values():
            output = model(
                data_window=data_window,
                prob=prob,
                ext_data_window=ext_data_window,
                forecast_horizon=forecast_horizon,
                future_externals=future_externals
            )
            outputs.append(output)

        return self.aggregate_outputs(outputs, aggregation_method)

    def aggregate_outputs(self, outputs: List[PTFHCNNOutput], method: str = "mean") -> PTFHCNNOutput:
        """Aggregate PTF HCNN outputs from ensemble members."""
        if method == "mean":
            return self._aggregate_mean(outputs)
        elif method == "median":
            return self._aggregate_median(outputs)
        elif method == "weighted_mean":
            return self._aggregate_weighted_mean(outputs)
        else:
            raise ValueError(f"Aggregation method {method} not implemented for PTF ensemble")

    def _aggregate_mean(self, outputs: List[PTFHCNNOutput]) -> PTFHCNNOutput:
        """Aggregate using mean."""
        expectations = torch.stack([out.expectations for out in outputs]).mean(dim=0)
        states = torch.stack([out.states for out in outputs]).mean(dim=0)
        delta_terms = torch.stack([out.delta_terms for out in outputs]).mean(dim=0)
        partial_delta_terms = torch.stack([out.partial_delta_terms for out in outputs]).mean(dim=0)

        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).mean(dim=0)
            future_states = torch.stack([out.future_states for out in outputs]).mean(dim=0)

        return PTFHCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            partial_delta_terms=partial_delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_median(self, outputs: List[PTFHCNNOutput]) -> PTFHCNNOutput:
        """Aggregate using median."""
        expectations = torch.stack([out.expectations for out in outputs]).median(dim=0)[0]
        states = torch.stack([out.states for out in outputs]).median(dim=0)[0]
        delta_terms = torch.stack([out.delta_terms for out in outputs]).median(dim=0)[0]
        partial_delta_terms = torch.stack([out.partial_delta_terms for out in outputs]).median(dim=0)[0]

        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).median(dim=0)[0]
            future_states = torch.stack([out.future_states for out in outputs]).median(dim=0)[0]

        return PTFHCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            partial_delta_terms=partial_delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_weighted_mean(self, outputs: List[PTFHCNNOutput]) -> PTFHCNNOutput:
        """Aggregate using weighted mean (equal weights for now)."""
        # For now, use equal weights (same as mean)
        return self._aggregate_mean(outputs)

    def get_model(self, index: int) -> PTF_Model:
        """Get a specific model from the ensemble."""
        return self.models[f"model_{index}"]


class LFormHCNNEnsemble(BaseEnsemble):
    """
    Ensemble of LSTM Formulation HCNN models.

    Parameters
    ----------
    n_ensemble : int
        Number of models in the ensemble
    n_obs_vars : int
        Number of observable variables
    n_hid_vars : int
        Number of hidden variables
    s0_nature : Literal['zeros_', 'random_']
        Initial state initialization strategy
    train_s0 : bool
        Whether initial state is trainable
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Weight initialization range
    init_diag : float, default=1.0
        Initial value for diagonal elements
    n_ext_vars : Optional[int], default=None
        Number of external variables
    """

    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'],
        train_s0: bool,
        init_range: tuple = (-0.75, 0.75),
        init_diag: float = 1.0,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_ensemble)

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.init_range = init_range
        self.init_diag = init_diag
        self.n_ext_vars = n_ext_vars

        # Create ensemble models
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            model = LForm_Model(
                n_obs_vars=n_obs_vars,
                n_hid_vars=n_hid_vars,
                s0_nature=s0_nature,
                train_s0=train_s0,
                init_range=init_range,
                init_diag=init_diag,
                n_ext_vars=n_ext_vars
            )
            self.models[f"model_{i}"] = model

        self.name = self._generate_ensemble_name()

    @property
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        return "lform_hcnn_ensemble"

    def _generate_ensemble_name(self) -> str:
        """Generate a unique ensemble name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s0_str = "zeroInit" if self.s0_nature == "zeros_" else "randInit"
        trainable_str = "trainable" if self.train_s0 else "fixed"
        ext_str = f"_ext{self.n_ext_vars}" if self.n_ext_vars else ""

        return (
            f"LFormHCNNEnsemble_n{self.n_ensemble}_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
            f"_{s0_str}_{trainable_str}_diag{self.init_diag}{ext_str}_{timestamp}"
        )

    def forward(
        self,
        data_window: torch.Tensor,
        ext_data_window: Optional[torch.Tensor] = None,
        forecast_horizon: Optional[int] = None,
        future_externals: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean"
    ) -> HCNNOutput:
        """Forward pass through the LForm HCNN ensemble."""
        outputs = []
        for model in self.models.values():
            output = model(
                data_window=data_window,
                ext_data_window=ext_data_window,
                forecast_horizon=forecast_horizon,
                future_externals=future_externals
            )
            outputs.append(output)

        return self.aggregate_outputs(outputs, aggregation_method)

    def aggregate_outputs(self, outputs: List[HCNNOutput], method: str = "mean") -> HCNNOutput:
        """Aggregate HCNN outputs from ensemble members."""
        if method == "mean":
            return self._aggregate_mean(outputs)
        elif method == "median":
            return self._aggregate_median(outputs)
        elif method == "weighted_mean":
            return self._aggregate_weighted_mean(outputs)
        else:
            raise ValueError(f"Aggregation method {method} not implemented for LForm ensemble")

    def _aggregate_mean(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using mean."""
        expectations = torch.stack([out.expectations for out in outputs]).mean(dim=0)
        states = torch.stack([out.states for out in outputs]).mean(dim=0)
        delta_terms = torch.stack([out.delta_terms for out in outputs]).mean(dim=0)

        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).mean(dim=0)
            future_states = torch.stack([out.future_states for out in outputs]).mean(dim=0)

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_median(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using median."""
        expectations = torch.stack([out.expectations for out in outputs]).median(dim=0)[0]
        states = torch.stack([out.states for out in outputs]).median(dim=0)[0]
        delta_terms = torch.stack([out.delta_terms for out in outputs]).median(dim=0)[0]

        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).median(dim=0)[0]
            future_states = torch.stack([out.future_states for out in outputs]).median(dim=0)[0]

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_weighted_mean(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using weighted mean (equal weights for now)."""
        # For now, use equal weights (same as mean)
        return self._aggregate_mean(outputs)

    def get_model(self, index: int) -> LForm_Model:
        """Get a specific model from the ensemble."""
        return self.models[f"model_{index}"]


class LSpaHCNNEnsemble(BaseEnsemble):
    """
    Ensemble of Large Sparse HCNN models.

    Parameters
    ----------
    n_ensemble : int
        Number of models in the ensemble
    n_obs_vars : int
        Number of observable variables
    n_hid_vars : int
        Number of hidden variables
    s0_nature : Literal['zeros_', 'random_']
        Initial state initialization strategy
    train_s0 : bool
        Whether initial state is trainable
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Weight initialization range
    bias : bool, default=False
        Whether to include bias terms
    mask_type : Literal['non_obs_block', 'random_block'], default='random_block'
        Type of sparsity mask to apply
    sparsity_ratio : float, default=0.25
        Proportion of weights to set to zero
    n_ext_vars : Optional[int], default=None
        Number of external variables
    """

    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'],
        train_s0: bool,
        init_range: tuple = (-0.75, 0.75),
        bias: bool = False,
        mask_type: Literal['non_obs_block', 'random_block'] = 'random_block',
        sparsity_ratio: float = 0.25,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__(n_ensemble)

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.init_range = init_range
        self.bias = bias
        self.mask_type = mask_type
        self.sparsity_ratio = sparsity_ratio
        self.n_ext_vars = n_ext_vars

        # Create ensemble models
        self.models = nn.ModuleDict()
        for i in range(n_ensemble):
            model = LSpa_Model(
                n_obs_vars=n_obs_vars,
                n_hid_vars=n_hid_vars,
                s0_nature=s0_nature,
                train_s0=train_s0,
                init_range=init_range,
                bias=bias,
                mask_type=mask_type,
                sparsity_ratio=sparsity_ratio,
                n_ext_vars=n_ext_vars
            )
            self.models[f"model_{i}"] = model

        self.name = self._generate_ensemble_name()

    @property
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        return "lspa_hcnn_ensemble"

    def _generate_ensemble_name(self) -> str:
        """Generate a unique ensemble name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s0_str = "zeroInit" if self.s0_nature == "zeros_" else "randInit"
        trainable_str = "trainable" if self.train_s0 else "fixed"
        ext_str = f"_ext{self.n_ext_vars}" if self.n_ext_vars else ""

        return (
            f"LSpaHCNNEnsemble_n{self.n_ensemble}_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
            f"_{s0_str}_{trainable_str}_sparse{self.sparsity_ratio}_{self.mask_type}{ext_str}_{timestamp}"
        )

    def forward(
        self,
        data_window: torch.Tensor,
        ext_data_window: Optional[torch.Tensor] = None,
        forecast_horizon: Optional[int] = None,
        future_externals: Optional[torch.Tensor] = None,
        aggregation_method: str = "mean"
    ) -> HCNNOutput:
        """Forward pass through the LSpa HCNN ensemble."""
        outputs = []
        for model in self.models.values():
            output = model(
                data_window=data_window,
                ext_data_window=ext_data_window,
                forecast_horizon=forecast_horizon,
                future_externals=future_externals
            )
            outputs.append(output)

        return self.aggregate_outputs(outputs, aggregation_method)

    def aggregate_outputs(self, outputs: List[HCNNOutput], method: str = "mean") -> HCNNOutput:
        """Aggregate HCNN outputs from ensemble members."""
        if method == "mean":
            return self._aggregate_mean(outputs)
        elif method == "median":
            return self._aggregate_median(outputs)
        elif method == "weighted_mean":
            return self._aggregate_weighted_mean(outputs)
        else:
            raise ValueError(f"Aggregation method {method} not implemented for LSpa ensemble")

    def _aggregate_mean(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using mean."""
        expectations = torch.stack([out.expectations for out in outputs]).mean(dim=0)
        states = torch.stack([out.states for out in outputs]).mean(dim=0)
        delta_terms = torch.stack([out.delta_terms for out in outputs]).mean(dim=0)

        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).mean(dim=0)
            future_states = torch.stack([out.future_states for out in outputs]).mean(dim=0)

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_median(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using median."""
        expectations = torch.stack([out.expectations for out in outputs]).median(dim=0)[0]
        states = torch.stack([out.states for out in outputs]).median(dim=0)[0]
        delta_terms = torch.stack([out.delta_terms for out in outputs]).median(dim=0)[0]

        forecasts = None
        future_states = None
        if outputs[0].forecasts is not None:
            forecasts = torch.stack([out.forecasts for out in outputs]).median(dim=0)[0]
            future_states = torch.stack([out.future_states for out in outputs]).median(dim=0)[0]

        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states
        )

    def _aggregate_weighted_mean(self, outputs: List[HCNNOutput]) -> HCNNOutput:
        """Aggregate using weighted mean (equal weights for now)."""
        # For now, use equal weights (same as mean)
        return self._aggregate_mean(outputs)

    def get_model(self, index: int) -> LSpa_Model:
        """Get a specific model from the ensemble."""
        return self.models[f"model_{index}"]

    def get_ensemble_sparsity_info(self) -> Dict[str, Dict]:
        """Get sparsity information for all models in the ensemble."""
        sparsity_info = {}
        for i, model in enumerate(self.models.values()):
            sparsity_info[f"model_{i}"] = model.get_sparsity_info()
        return sparsity_info
