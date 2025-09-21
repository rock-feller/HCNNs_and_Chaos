"""
Base classes and interfaces for HCNN models and components.

This module defines the fundamental abstractions and interfaces that all
HCNN components should implement, ensuring consistency and extensibility
across the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, Literal
from collections import namedtuple
import torch
import torch.nn as nn


# Named tuples for model outputs
HCNNOutput = namedtuple(
    "HCNNOutput",
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)

PTFHCNNOutput = namedtuple(
    "PTFHCNNOutput", 
    ["expectations", "states", "delta_terms", "partial_delta_terms", "forecasts", "future_states"]
)

ClassicOutput = namedtuple(
    "ClassicOutput",
    ["outputs", "hidden_states", "forecasts"]
)


class BaseHCNNCell(nn.Module, ABC):
    """
    Abstract base class for all HCNN cell implementations.
    
    This class defines the interface that all HCNN cells must implement,
    ensuring consistency across different variants (Vanilla, PTF, LForm, LSpa).
    
    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables ( or observables) in the system
    n_hid_vars : int  
        Number of hidden variables (or unobservables) in the state
    init_range : Tuple[float, float]
        Range for weight initialization
    n_ext_vars : Optional[int]
        Number of external variables (if any).
    """
    
    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int, 
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None
    ):
        super().__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = n_obs_vars + n_hid_vars
        self.init_range = init_range
        self.n_ext_vars = n_ext_vars
        
    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the HCNN cell.
        
        Parameters
        ----------
        state : torch.Tensor
            Current state tensor
        teacher_forcing : bool
            Whether to use teacher forcing
        observation : Optional[torch.Tensor]
            Ground truth observation for teacher forcing
        externals : Optional[torch.Tensor]
            External variables
        **kwargs
            Additional cell-specific parameters
            
        Returns
        -------
        Tuple[torch.Tensor, ...]
            Cell-specific output tuple
        """
        pass
        
    @property
    @abstractmethod
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""
        pass


class BaseHCNNModel(nn.Module, ABC):
    """
    Abstract base class for HCNN model wrappers.
    
    This class provides the common interface for all HCNN models,
    handling initialization, state management, and forward passes.
    
    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables ( or observables)
    n_hid_vars : int
        Number of hidden variables (or unobservables)
    s0_nature : Literal['zeros_', 'random_']
        Initial state initialization strategy.
    train_s0 : bool
        Whether initial state is trainable.
    init_range : Tuple[float, float]
        Weight initialization range.
    n_ext_vars : Optional[int]
        Number of external variables (if any).
    """
    
    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'],
        train_s0: bool,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None
    ):
        super().__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = n_obs_vars + n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.init_range = init_range
        self.n_ext_vars = n_ext_vars
        
        # Initialize the cell (to be implemented by subclasses)
        self.cell = None
        
        # Initialize state
        self._init_state()
        
    def _init_state(self):
        """Initialize the initial state s0."""
        if self.s0_nature == 'zeros_':
            s0 = torch.zeros(self.n_state_vars)
        elif self.s0_nature == 'random_':
            s0 = torch.empty(self.n_state_vars).uniform_(*self.init_range)
        else:
            raise ValueError(f"Unknown s0_nature: {self.s0_nature}")
            
        if self.train_s0:
            self.s0 = nn.Parameter(s0)
        else:
            self.register_buffer('s0', s0)
    
    @abstractmethod
    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[HCNNOutput, PTFHCNNOutput]:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        data_window : torch.Tensor
            Input data window
        forecast_horizon : Optional[int]
            Number of steps to forecast
        externals : Optional[torch.Tensor]
            External variables
        **kwargs
            Model-specific parameters
            
        Returns
        -------
        Union[HCNNOutput, PTFHCNNOutput]
            Model output
        """
        pass
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the type of HCNN model."""
        pass
        
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
        add_stuffs: str = "",
        cleanup: bool = True
    ):
        """Save model checkpoint."""
        from ..utils.checkpoints import save_checkpoint
        save_checkpoint(
            model=self,
            epoch=epoch,
            loss=loss,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            model_name=f"{self.model_type}{add_stuffs}",
            cleanup=cleanup
        )

    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model checkpoint."""
        from ..utils.checkpoints import load_checkpoint
        return load_checkpoint(checkpoint_path, model=self, optimizer=optimizer)


class BaseEnsemble(nn.Module, ABC):
    """
    Abstract base class for ensemble implementations.
    
    This class defines the interface for ensemble models,
    providing common functionality for managing multiple models.
    """
    
    def __init__(self, n_ensemble: int):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.models = {}
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the ensemble."""
        pass
        
    @abstractmethod
    def aggregate_outputs(self, outputs, method: str = "mean"):
        """Aggregate outputs from ensemble members."""
        pass
        
    @property
    @abstractmethod
    def ensemble_type(self) -> str:
        """Return the type of ensemble."""
        pass
