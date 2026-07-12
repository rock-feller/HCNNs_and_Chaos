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

# Unified per-step output returned by every HCNN cell. Keeping a single, common
# signature across all variants lets the rollout loop live once in BaseHCNNModel.
#   - expectation : predicted observation y_hat_t = C s_t          (batch, n_obs)
#   - next_state  : s_{t+1}                                        (batch, n_state)
#   - delta_term  : teacher-forcing error y_t - y_hat_t (or None)  (batch, n_obs)
#   - extras      : dict of variant-specific tensors (e.g. PTF's
#                   partial_delta_terms); {} for variants with none.
CellOutput = namedtuple("CellOutput", ["expectation", "next_state", "delta_term", "extras"])
CellOutput.__new__.__defaults__ = ({},)  # extras optional, defaults to empty dict


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
    
    def forward(
        self,
        data_window: torch.Tensor,
        forecast_horizon: Optional[int] = None,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        **cell_kwargs
    ) -> Union[HCNNOutput, PTFHCNNOutput]:
        """
        Generic HCNN forward pass shared by every variant.

        The recurrence itself is delegated to ``self.cell`` (which must return a
        :class:`CellOutput`); this method only orchestrates the two-phase rollout
        that is identical across Vanilla / PTF / LForm / LSpa:

        1. **Calibration** over the observed ``data_window`` with teacher forcing
           (the cell sees the ground-truth observation at each step).
        2. **Forecast** (optional) for ``forecast_horizon`` steps in autonomous
           mode (``teacher_forcing=False``) - the model is fed only its own state,
           so there is no target leakage.

        Variant-specific outputs (e.g. PTF's ``partial_delta_terms``) are carried
        through ``CellOutput.extras`` and assembled by :meth:`_pack_output`.

        Parameters
        ----------
        data_window : torch.Tensor
            Observed sequence, shape ``(batch, seq_len, n_obs_vars)``.
        forecast_horizon : Optional[int]
            Number of autonomous steps to forecast. ``None``/0 -> no forecast.
        externals : Optional[torch.Tensor]
            External inputs over the calibration window,
            shape ``(batch, seq_len, n_ext_vars)``.
        future_externals : Optional[torch.Tensor]
            External inputs over the forecast window,
            shape ``(batch, forecast_horizon, n_ext_vars)``.
        **cell_kwargs
            Extra keyword arguments forwarded verbatim to the cell.

        Returns
        -------
        Union[HCNNOutput, PTFHCNNOutput]
            The variant's output named tuple (see :meth:`_pack_output`).
        """
        batch_size, seq_len, _ = data_window.shape
        device = data_window.device

        current_state = self.s0.to(device).expand(batch_size, -1).clone()
        states_list = [current_state]
        expectations_list = []
        delta_terms_list = []
        extras_lists: Dict[str, list] = {}

        # --- Calibration: teacher forcing over the observed window ---
        for t in range(seq_len):
            ext_t = externals[:, t, :] if externals is not None else None
            co = self.cell(
                state=current_state,
                teacher_forcing=True,
                observation=data_window[:, t, :],
                externals=ext_t,
                **cell_kwargs
            )
            expectations_list.append(co.expectation)
            delta_terms_list.append(co.delta_term)
            if co.extras:
                for k, v in co.extras.items():
                    extras_lists.setdefault(k, []).append(v)
            # state[t] already recorded; advance to state[t+1] for t < seq_len-1
            if t < seq_len - 1:
                current_state = co.next_state
                states_list.append(current_state)

        states = torch.stack(states_list, dim=1)
        expectations = torch.stack(expectations_list, dim=1)
        delta_terms = torch.stack(delta_terms_list, dim=1)
        extras = {k: torch.stack(v, dim=1) for k, v in extras_lists.items()}

        # --- Forecast: autonomous rollout (no teacher forcing) ---
        forecasts = None
        future_states = None
        if forecast_horizon and forecast_horizon > 0:
            forecasts_list = []
            future_states_list = []
            current_state = states[:, seq_len - 1, :]
            for t in range(forecast_horizon):
                ext_f = future_externals[:, t, :] if future_externals is not None else None
                co = self.cell(
                    state=current_state,
                    teacher_forcing=False,
                    externals=ext_f,
                    **cell_kwargs
                )
                forecasts_list.append(co.expectation)
                future_states_list.append(co.next_state)
                current_state = co.next_state
            forecasts = torch.stack(forecasts_list, dim=1)
            future_states = torch.stack(future_states_list, dim=1)

        return self._pack_output(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states,
            extras=extras,
        )

    def _pack_output(self, expectations, states, delta_terms, forecasts, future_states, extras):
        """
        Assemble the variant's output named tuple from the rolled-out tensors.

        Default packs a :class:`HCNNOutput` (Vanilla / LForm / LSpa). Variants
        with extra fields (e.g. PTF) override this to include them.
        """
        return HCNNOutput(
            expectations=expectations,
            states=states,
            delta_terms=delta_terms,
            forecasts=forecasts,
            future_states=future_states,
        )

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
