"""
Initial-state modules: how the first state of the rollout is built for each window.

``BaseHCNNModel.forward`` calls ``self.initial_state(data_window)`` to get one state per
window, shape ``(batch, n_state)``, and starts the teacher-forced calibration at
``t = n_warmup``. Every module works row by row, so windows that start at different
points of the attractor can get different states, and batching needs nothing special.

- :class:`SharedInitialState` - one ``s0`` for every window (the original HCNN; default).
  Optional per-window noise on its hidden part, and an option to take its observed part
  from the data.
- :class:`DelayEmbeddingInitialState` - ``s_k = g(y_0, ..., y_{k-1})``, inferred from the
  first ``k`` observations of each window (a delay embedding).

Swap modules with ``model.set_initial_state(...)``. To write a new one, subclass
:class:`InitialState`, implement ``forward`` and set ``n_warmup``.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class InitialState(nn.Module, ABC):
    """
    Base class: maps a window ``(batch, T, n_obs)`` to the state the rollout starts from,
    ``(batch, n_obs + n_hid)``.

    ``n_warmup`` is the number of leading observations the module uses to build the state.
    The rollout starts at ``t = n_warmup``, so those observations are not predicted and the
    model returns ``T - n_warmup`` expectations. A module that uses no observations keeps
    ``n_warmup = 0``. A module must only read ``data_window[:, :n_warmup]``; reading later
    observations would leak them into the predictions.
    """

    n_warmup: int = 0

    def __init__(self, n_obs_vars: int, n_hid_vars: int):
        super().__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = n_obs_vars + n_hid_vars

    @abstractmethod
    def forward(self, data_window: torch.Tensor) -> torch.Tensor:
        """Return the initial state of each window, shape ``(batch, n_state)``."""


class SharedInitialState(InitialState):
    """
    One initial state ``s0`` shared by every window: the original HCNN, and the default.

    Parameters
    ----------
    n_obs_vars, n_hid_vars : int
        State sizes.
    s0_nature : {'zeros_', 'random_'}
        ``s0`` starts at zero, or uniform in ``init_range``.
    train_s0 : bool
        ``s0`` is a learned parameter (``True``) or a fixed buffer.
    init_range : tuple of float
        Range of the uniform draw for ``s0_nature='random_'``.
    noise_std : float
        Standard deviation of Gaussian noise added to the **hidden** part of ``s0``, drawn
        separately for every window, in training mode only. It makes the dynamics learn
        to forget the initial state. ``0`` (default) adds none.
    observed_from_data : bool
        If ``True``, the observed part of the initial state is ``y_0`` of each window, so
        ``y_hat_0 = y_0`` and the ``t = 0`` loss term is exactly 0. The teacher-forced
        state ``r_0 = [y_0 ; h_0]`` is the same either way, so the gradients of every
        other parameter and the forecasts do not change; only the reported loss does.
        ``y_hat_0`` is then not a prediction. ``False`` (default) uses the learned
        observed part.
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        noise_std: float = 0.0,
        observed_from_data: bool = False,
    ):
        super().__init__(n_obs_vars, n_hid_vars)
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self.noise_std = noise_std
        self.observed_from_data = observed_from_data

        if s0_nature == 'zeros_':
            s0 = torch.zeros(self.n_state_vars)
        elif s0_nature == 'random_':
            s0 = torch.empty(self.n_state_vars).uniform_(*init_range)
        else:
            raise ValueError(f"Unknown s0_nature: {s0_nature}")
        if train_s0:
            self.s0 = nn.Parameter(s0)
        else:
            self.register_buffer('s0', s0)

    def forward(self, data_window: torch.Tensor) -> torch.Tensor:
        batch_size = data_window.shape[0]
        state = self.s0.to(data_window.device).expand(batch_size, -1)
        if self.observed_from_data:
            state = torch.cat([data_window[:, 0, :], state[:, self.n_obs_vars:]], dim=-1)
        if self.training and self.noise_std > 0:
            noise = torch.cat([
                torch.zeros(batch_size, self.n_obs_vars, device=state.device, dtype=state.dtype),
                self.noise_std * torch.randn(batch_size, self.n_hid_vars,
                                             device=state.device, dtype=state.dtype),
            ], dim=-1)
            state = state + noise
        return state.clone()

    def extra_repr(self) -> str:
        return (f"n_obs_vars={self.n_obs_vars}, n_hid_vars={self.n_hid_vars}, "
                f"noise_std={self.noise_std}, observed_from_data={self.observed_from_data}")


class DelayEmbeddingInitialState(InitialState):
    """
    Infer the state at ``t = k`` from the first ``k`` observations of each window:

    ``s_k = g(y_0, ..., y_{k-1})``

    By Takens' theorem the hidden state of a deterministic system is (generically) a
    function of enough past observations, so this replaces the shared guess with a
    per-window estimate and behaves the same on training, validation and test windows.
    The rollout starts at ``t = k``: the model returns ``T - k`` expectations, for
    ``y_k ... y_{T-1}``, and ``HCNNTrainer`` aligns the targets through
    ``model.n_warmup``. ``y_hat_k = C s_k`` is a genuine prediction from the past.

    Parameters
    ----------
    n_obs_vars, n_hid_vars : int
        State sizes.
    n_delays : int
        ``k``, the number of observations used (and not predicted).
    hidden_size : int, optional
        ``None`` (default): ``g`` is linear. Otherwise ``g`` has one tanh hidden layer of
        this size.
    """

    def __init__(self, n_obs_vars: int, n_hid_vars: int, n_delays: int,
                 hidden_size: Optional[int] = None):
        super().__init__(n_obs_vars, n_hid_vars)
        if n_delays < 1:
            raise ValueError(f"n_delays must be >= 1, got {n_delays}")
        self.n_warmup = n_delays
        in_features = n_delays * n_obs_vars
        if hidden_size is None:
            self.net = nn.Linear(in_features, self.n_state_vars)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, self.n_state_vars),
            )

    def forward(self, data_window: torch.Tensor) -> torch.Tensor:
        delays = data_window[:, :self.n_warmup, :].reshape(data_window.shape[0], -1)
        return self.net(delays)

    def extra_repr(self) -> str:
        return f"n_obs_vars={self.n_obs_vars}, n_hid_vars={self.n_hid_vars}, n_delays={self.n_warmup}"
