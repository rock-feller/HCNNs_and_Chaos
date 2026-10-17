"""
<Name> HCNN cell - TEMPLATE.

Copy this folder to ``hcnn/architectures/<your_name>/`` and replace every TODO.
As shipped, the template reproduces the vanilla recurrence so that it passes the
contract tests from the very first commit; change ``transition`` to your idea.

The HCNN step itself (``y_hat = C s``, the teacher-forcing correction, external inputs)
is :meth:`BaseHCNNCell.forward`, shared by every architecture. Do not write a ``forward``
(defining one raises ``TypeError``). Implement:

- ``transition(r)``: your state transition ``s_{t+1} = f(r_t)`` (required);
- ``correction(delta)``: what of the error ``delta_t = y_t - y_hat_t`` is fed back into the
  observed coordinates (optional; default: all of it, so ``r_t``'s observed part is ``y_t``);
- ``step_extras(delta, correction)``: extra per-step tensors to return (optional; see PTF).

Contract (checked by ``tests/test_architecture_contract.py``): parameters are created on
the default device (no ``.to("mps")`` inside the cell), and the cell can be built as
``Cell(n_obs_vars, n_hid_vars)``.
"""

import torch
from typing import Optional, Tuple

from ...core.base import BaseHCNNCell
from ...core.defaults import DEFAULT_INIT_RANGE
from ...core.layers import CustomLinear


class TemplateHCNNCell(BaseHCNNCell):  # TODO: rename, e.g. MyIdeaHCNNCell
    """TODO: one paragraph on the idea and the transition equation."""

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        init_range: Tuple[float, float] = DEFAULT_INIT_RANGE,
        n_ext_vars: Optional[int] = None,
        # TODO: add your architecture's hyper-parameters here (with defaults).
    ):
        super().__init__(n_obs_vars, n_hid_vars, init_range, n_ext_vars)
        # TODO: your transition parameters.
        self.A = CustomLinear(self.n_state_vars, self.n_state_vars, bias=False, init_range=init_range)
        # B (external inputs) last, so the random draws keep their order.
        self._init_external_matrix()

    @property
    def cell_type(self) -> str:
        return "template_hcnn_cell"  # TODO

    def transition(self, r: torch.Tensor) -> torch.Tensor:
        """TODO: your state transition s_{t+1} = f(r_t). Vanilla: A tanh(r_t)."""
        return self.A(torch.tanh(r))
