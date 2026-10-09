"""
<Name> HCNN cell - TEMPLATE.

Copy this folder to ``hcnn/architectures/<your_name>/`` and replace every TODO.
As shipped, the template reproduces the vanilla recurrence so that it passes the
contract tests from the very first commit; change ``_transition`` to your idea.

Contract (checked by ``tests/test_architecture_contract.py``):
- ``forward`` returns a :class:`CellOutput` ``(expectation, next_state, delta_term, extras)``;
- ``expectation = C s_t`` with ``C = [I | 0]`` (first ``n_obs_vars`` coordinates);
- with ``teacher_forcing=True`` the state is corrected by ``-C^T delta_t`` with
  ``delta_t = y_t - expectation``; with ``teacher_forcing=False`` no truth is used;
- parameters are created on the default device (no ``.to("mps")`` inside the cell).
"""

import torch
from typing import Optional, Tuple

from ...core.base import BaseHCNNCell, CellOutput
from ...core.layers import CustomLinear


class TemplateHCNNCell(BaseHCNNCell):  # TODO: rename, e.g. MyIdeaHCNNCell
    """TODO: one paragraph on the idea and the transition equation."""

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None,
        # TODO: add your architecture's hyper-parameters here (with defaults).
    ):
        super().__init__(n_obs_vars, n_hid_vars, init_range, n_ext_vars)
        self.A = CustomLinear(self.n_state_vars, self.n_state_vars, bias=False, init_range=init_range)
        self.B = (
            CustomLinear(n_ext_vars, self.n_state_vars, bias=False, init_range=init_range)
            if n_ext_vars is not None else None
        )
        self.register_buffer("ConMat", torch.eye(n_obs_vars, self.n_state_vars), persistent=False)

    @property
    def cell_type(self) -> str:
        return "template_hcnn_cell"  # TODO

    def _transition(self, r: torch.Tensor) -> torch.Tensor:
        """TODO: your state transition s_{t+1} = f(r_t). Vanilla: A tanh(r_t)."""
        return self.A(torch.tanh(r))

    def forward(
        self,
        state: torch.Tensor,
        teacher_forcing: bool = False,
        observation: Optional[torch.Tensor] = None,
        externals: Optional[torch.Tensor] = None,
    ) -> CellOutput:
        expectation = state @ self.ConMat.T

        delta_term = None
        r = state
        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            delta_term = observation - expectation
            r = state - delta_term @ self.ConMat

        next_state = self._transition(r)
        if self.B is not None:
            if externals is None:
                raise ValueError(f"Cell expects {self.n_ext_vars} external variables.")
            next_state = next_state + self.B(externals)

        # Put any extra per-step tensors you want returned in `extras` (see PTF).
        return CellOutput(expectation, next_state, delta_term, {})
