"""
Base classes and interfaces for HCNN models and components.

This module defines the fundamental abstractions and interfaces that all
HCNN components should implement, ensuring consistency and extensibility
across the framework.
"""

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Dict, Mapping, NamedTuple, Optional, Tuple, Union, Literal
from collections import namedtuple
import torch
import torch.nn as nn

from .defaults import DEFAULT_INIT_RANGE
from .initial_state import InitialState, SharedInitialState
from .layers import CustomLinear


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

class CellOutput(NamedTuple):
    """
    Per-step output of every HCNN cell. One signature for all variants lets the rollout
    live once, in :class:`BaseHCNNModel`.

    - ``expectation``: the prediction ``y_hat_t = C s_t``, shape ``(batch, n_obs)``;
    - ``next_state``: ``s_{t+1}``, shape ``(batch, n_state)``;
    - ``delta_term``: the teacher-forcing error ``y_t - y_hat_t``, or ``None`` in the
      autonomous mode, shape ``(batch, n_obs)``;
    - ``extras``: variant-specific tensors (e.g. PTF's ``partial_delta_terms``). The
      default is a read-only empty mapping, so it can never be shared and mutated.
    """
    expectation: torch.Tensor
    next_state: torch.Tensor
    delta_term: Optional[torch.Tensor]
    extras: Mapping[str, torch.Tensor] = MappingProxyType({})


class BaseHCNNCell(nn.Module, ABC):
    r"""
    Base class of every HCNN cell. It implements the HCNN step **once**; a subclass only
    says what is specific to its architecture.

    With ``C = [I | 0]`` (the first ``n_obs_vars`` coordinates), one step is:

    .. code-block:: text

        y_hat_t = C s_t                          expectation (read before any correction)
        delta_t = y_t - y_hat_t                  teacher forcing only
        r_t     = s_t + C^T g(delta_t)           teacher forcing; r_t = s_t otherwise
        s_{t+1} = f(r_t) [+ B u_t]

    With the default ``g`` (identity) the observed part of ``r_t`` is the data ``y_t``.

    A subclass implements:

    - ``transition(r)``: ``f``, e.g. ``A tanh(r)`` (required);
    - ``correction(delta)``: ``g``, e.g. PTF's dropout mask (optional; default identity);
    - ``step_extras(delta, correction)``: extra per-step tensors (optional; default none);
    - ``cell_type``.

    It creates its own parameters, then calls :meth:`_init_external_matrix` to create
    ``B``, in that order, so the random draws keep their order. It must not override
    ``forward``: the sign of the correction and ``y_hat = C s`` then exist in one place.
    A cell that genuinely needs its own step sets ``_custom_forward = True`` in its body
    and is responsible for the contract (``tests/test_architecture_contract.py``).

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables.
    n_hid_vars : int
        Number of hidden variables.
    init_range : Tuple[float, float]
        Range of the uniform weight initialisation (default ``DEFAULT_INIT_RANGE``).
    n_ext_vars : Optional[int]
        Number of external input variables (``None``: no external inputs).
    """

    _custom_forward: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "forward" in cls.__dict__ and not cls.__dict__.get("_custom_forward", False):
            raise TypeError(
                f"{cls.__name__} overrides forward(). Implement transition() (and optionally "
                f"correction() / step_extras()) instead; the HCNN step lives in BaseHCNNCell. "
                f"Set _custom_forward = True in the class body only if the step must differ."
            )

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = DEFAULT_INIT_RANGE,
        n_ext_vars: Optional[int] = None
    ):
        super().__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = n_obs_vars + n_hid_vars
        self.init_range = init_range
        self.n_ext_vars = n_ext_vars
        self.B: Optional[nn.Module] = None
        # C = [I | 0]: reads the observed coordinates out of the state.
        self.register_buffer("ConMat", torch.eye(n_obs_vars, self.n_state_vars), persistent=False)

    def _init_external_matrix(self) -> None:
        """Create ``B`` (``n_ext_vars -> n_state``) if the cell has external inputs.
        Call it after creating the transition parameters, so the random draws keep their order."""
        if self.n_ext_vars is not None:
            self.B = CustomLinear(
                in_features=self.n_ext_vars,
                out_features=self.n_state_vars,
                bias=False,
                init_range=self.init_range,
            )

    # ---- what a subclass provides ------------------------------------ #
    @abstractmethod
    def transition(self, r: torch.Tensor) -> torch.Tensor:
        """``f(r_t)``: the next state before external inputs, shape ``(batch, n_state)``."""

    def correction(self, delta: torch.Tensor) -> torch.Tensor:
        """``g(delta_t)``: the part of the error fed back into the observed coordinates."""
        return delta

    def step_extras(self, delta: torch.Tensor, correction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extra per-step tensors returned in ``CellOutput.extras`` (teacher forcing only)."""
        return {}

    @property
    @abstractmethod
    def cell_type(self) -> str:
        """Return the type of HCNN cell."""

    # ---- the HCNN step (shared; do not override) --------------------- #
    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None,
    ) -> CellOutput:
        """
        One HCNN step.

        Parameters
        ----------
        state : torch.Tensor
            ``s_t``, shape ``(batch, n_state)``.
        teacher_forcing : bool
            Correct the observed coordinates with ``observation`` (calibration) or not
            (autonomous forecast; ``observation`` is then ignored).
        observation : Optional[torch.Tensor]
            ``y_t``, shape ``(batch, n_obs)``; required with ``teacher_forcing=True``.
        externals : Optional[torch.Tensor]
            ``u_t``, shape ``(batch, n_ext)``; required if and only if the cell has ``B``.
        """
        expectation = state @ self.ConMat.T                       # y_hat_t = C s_t

        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            if observation.shape[-1] != self.n_obs_vars:
                raise ValueError(
                    f"Expected {self.n_obs_vars} observation variables, got {observation.shape[-1]}"
                )
            delta_term = observation - expectation                 # delta_t = y_t - y_hat_t
            correction = self.correction(delta_term)
            r = state + correction @ self.ConMat                    # r_t = s_t + C^T g(delta_t)
            extras = self.step_extras(delta_term, correction)
        else:
            delta_term, r, extras = None, state, {}

        next_state = self.transition(r)
        if self.n_ext_vars is not None:
            if self.B is None:
                raise RuntimeError(
                    f"{type(self).__name__} has n_ext_vars={self.n_ext_vars} but no B; "
                    "call self._init_external_matrix() in __init__."
                )
            if externals is None:
                raise ValueError(
                    "External variables expected but not provided. "
                    f"Cell was initialized with n_ext_vars={self.n_ext_vars}"
                )
            if externals.shape[-1] != self.n_ext_vars:
                raise ValueError(
                    f"Expected {self.n_ext_vars} external variables, got {externals.shape[-1]}"
                )
            next_state = next_state + self.B(externals)
        elif externals is not None:
            raise ValueError(
                "External variables provided but cell doesn't support them. "
                "Initialize cell with n_ext_vars parameter."
            )
        return CellOutput(expectation, next_state, delta_term, extras)

    # ---- accessors shared by every cell ------------------------------ #
    def get_observation_matrix(self) -> torch.Tensor:
        """``C = [I | 0]``, shape ``(n_obs, n_state)``."""
        return self.ConMat

    def get_external_input_matrix(self) -> Optional[torch.Tensor]:
        """The weights of ``B``, or ``None`` without external inputs."""
        return self.B.weight if self.B is not None else None

    def extra_repr(self) -> str:
        return (
            f"n_obs_vars={self.n_obs_vars}, n_hid_vars={self.n_hid_vars}, "
            f"n_ext_vars={self.n_ext_vars}, init_range={self.init_range}"
        )


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
        init_range: Tuple[float, float] = DEFAULT_INIT_RANGE,
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
        """Build the default initial-state module: one shared ``s0`` (see ``set_initial_state``)."""
        self.initial_state = SharedInitialState(
            self.n_obs_vars, self.n_hid_vars,
            s0_nature=self.s0_nature, train_s0=self.train_s0, init_range=self.init_range,
        )

    def set_initial_state(self, initial_state: InitialState) -> "BaseHCNNModel":
        """
        Replace how the first state of each window is built, e.g. with a
        ``SharedInitialState(..., noise_std=0.1)`` or a ``DelayEmbeddingInitialState``.
        Returns the model, so calls can be chained.
        """
        if not isinstance(initial_state, InitialState):
            raise TypeError(f"Expected an InitialState, got {type(initial_state).__name__}")
        if (initial_state.n_obs_vars, initial_state.n_hid_vars) != (self.n_obs_vars, self.n_hid_vars):
            raise ValueError(
                f"Initial state is sized for ({initial_state.n_obs_vars}, {initial_state.n_hid_vars}) "
                f"observed/hidden variables; the model has ({self.n_obs_vars}, {self.n_hid_vars})."
            )
        self.initial_state = initial_state
        return self

    @property
    def s0(self) -> torch.Tensor:
        """The shared initial state, when the initial-state module has one."""
        return self.initial_state.s0

    @property
    def n_warmup(self) -> int:
        """Leading observations of each window used to build the initial state (not predicted)."""
        return self.initial_state.n_warmup

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the initial-state module stored s0 as `<prefix>s0`.
        old, new = prefix + "s0", prefix + "initial_state.s0"
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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

        0. **Initial state**: ``self.initial_state(data_window)`` gives one state per
           window (default: a shared ``s0``; see :mod:`hcnn.core.initial_state`). A module
           that infers the state from the first ``n_warmup`` observations starts the
           calibration at ``t = n_warmup``; those observations are not predicted.
        1. **Calibration** over the observed ``data_window`` with teacher forcing
           (the cell sees the ground-truth observation at each step).
           ``expectations[:, i]`` predicts ``data_window[:, n_warmup + i]``.
        2. **Forecast** (optional) for ``forecast_horizon`` steps in autonomous
           mode (``teacher_forcing=False``), starting from ``s_T`` (the state after the
           last teacher-forced step) - the model is fed only its own state, so there is
           no target leakage. ``forecasts[:, k]`` predicts the observation ``k`` steps
           after the end of ``data_window``.

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
        seq_len = data_window.shape[1]
        n_warmup = self.initial_state.n_warmup
        if seq_len <= n_warmup:
            raise ValueError(
                f"data_window has {seq_len} time steps; the initial state uses the first "
                f"{n_warmup}, so at least {n_warmup + 1} are needed."
            )

        # One initial state per window, built from at most data_window[:, :n_warmup].
        current_state = self.initial_state(data_window)
        states_list = [current_state]
        expectations_list = []
        delta_terms_list = []
        extras_lists: Dict[str, list] = {}

        # --- Calibration: teacher forcing over the observed window ---
        for t in range(n_warmup, seq_len):
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
            # states[t] = s_t is already recorded; advance to s_{t+1}. The step after the
            # last observation, s_T = f(r_{T-1}), is where the forecast starts.
            current_state = co.next_state
            if t < seq_len - 1:
                states_list.append(current_state)

        states = torch.stack(states_list, dim=1)
        expectations = torch.stack(expectations_list, dim=1)
        delta_terms = torch.stack(delta_terms_list, dim=1)
        extras = {k: torch.stack(v, dim=1) for k, v in extras_lists.items()}

        # --- Forecast: autonomous rollout (no teacher forcing) ---
        # Starts from s_T, the state produced by the LAST teacher-forced step, so the last
        # observation is used and forecasts[:, k] = C s_{T+k} predicts y_{T+k}.
        # future_states[:, k] = s_{T+k} is aligned with forecasts[:, k] (as states with
        # expectations).
        forecasts = None
        future_states = None
        if forecast_horizon and forecast_horizon > 0:
            forecasts_list = []
            future_states_list = []
            for t in range(forecast_horizon):
                ext_f = future_externals[:, t, :] if future_externals is not None else None
                co = self.cell(
                    state=current_state,
                    teacher_forcing=False,
                    externals=ext_f,
                    **cell_kwargs
                )
                forecasts_list.append(co.expectation)
                future_states_list.append(current_state)
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
