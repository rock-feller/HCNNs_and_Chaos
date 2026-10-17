"""BaseHCNNCell: the shared HCNN step, its guard against per-cell forward(), and CellOutput."""
from types import MappingProxyType

import pytest
import torch

from hcnn.core.base import BaseHCNNCell, CellOutput
from hcnn.core.defaults import DEFAULT_INIT_RANGE
from hcnn.core.layers import CustomLinear
from hcnn.architectures.vanilla import VanillaHCNNCell

N_OBS, N_HID = 2, 3
N_STATE = N_OBS + N_HID


class _Linear(BaseHCNNCell):
    """Minimal cell: s_{t+1} = A r_t."""

    def __init__(self, n_obs_vars, n_hid_vars, n_ext_vars=None, build_b=True):
        super().__init__(n_obs_vars, n_hid_vars, n_ext_vars=n_ext_vars)
        self.A = CustomLinear(self.n_state_vars, self.n_state_vars, bias=False, init_range=self.init_range)
        if build_b:
            self._init_external_matrix()

    @property
    def cell_type(self):
        return "linear_test_cell"

    def transition(self, r):
        return self.A(r)


def test_defining_forward_in_a_cell_raises():
    with pytest.raises(TypeError, match="overrides forward"):
        class _Bad(_Linear):
            def forward(self, state, teacher_forcing=False, observation=None, externals=None):
                return None


def test_custom_forward_is_an_explicit_opt_in():
    class _Custom(_Linear):
        _custom_forward = True

        def forward(self, state, teacher_forcing=False, observation=None, externals=None):
            return CellOutput(state[:, :N_OBS], state, None)

    out = _Custom(N_OBS, N_HID)(torch.randn(4, N_STATE))
    assert out.next_state.shape == (4, N_STATE)


def test_shared_step_implements_the_equations():
    cell = _Linear(N_OBS, N_HID)
    s, y = torch.randn(4, N_STATE), torch.randn(4, N_OBS)
    co = cell(s, teacher_forcing=True, observation=y)
    r = torch.cat([y, s[:, N_OBS:]], dim=-1)                 # observed part replaced by the data
    assert torch.equal(co.expectation, s[:, :N_OBS])
    assert torch.allclose(co.delta_term, y - s[:, :N_OBS])
    assert torch.allclose(co.next_state, r @ cell.A.weight.T, atol=1e-6)
    auto = cell(s, teacher_forcing=False, observation=y)       # observation ignored
    assert auto.delta_term is None
    assert torch.allclose(auto.next_state, s @ cell.A.weight.T, atol=1e-6)


def test_celloutput_default_extras_are_read_only():
    co = CellOutput(torch.zeros(1), torch.zeros(1), None)
    assert isinstance(co.extras, MappingProxyType) and len(co.extras) == 0
    with pytest.raises(TypeError):
        co.extras["leak"] = torch.zeros(1)


def test_cells_return_a_fresh_extras_dict_every_step():
    cell = _Linear(N_OBS, N_HID)
    s = torch.randn(2, N_STATE)
    a = cell(s, teacher_forcing=False).extras
    b = cell(s, teacher_forcing=False).extras
    assert a is not b


def test_external_inputs_are_checked():
    s = torch.randn(2, N_STATE)
    with pytest.raises(ValueError, match="doesn't support"):
        _Linear(N_OBS, N_HID)(s, externals=torch.randn(2, 2))
    with_b = _Linear(N_OBS, N_HID, n_ext_vars=2)
    with pytest.raises(ValueError, match="expected but not provided"):
        with_b(s)
    with pytest.raises(ValueError, match="Expected 2 external"):
        with_b(s, externals=torch.randn(2, 3))
    with pytest.raises(RuntimeError, match="_init_external_matrix"):
        _Linear(N_OBS, N_HID, n_ext_vars=2, build_b=False)(s, externals=torch.randn(2, 2))


def test_observation_is_required_and_checked_under_teacher_forcing():
    cell = _Linear(N_OBS, N_HID)
    s = torch.randn(2, N_STATE)
    with pytest.raises(ValueError, match="must be provided"):
        cell(s, teacher_forcing=True)
    with pytest.raises(ValueError, match="observation variables"):
        cell(s, teacher_forcing=True, observation=torch.randn(2, N_OBS + 1))


def test_default_init_range_is_shared():
    assert _Linear(N_OBS, N_HID).init_range == DEFAULT_INIT_RANGE
    cell = VanillaHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    lo, hi = DEFAULT_INIT_RANGE
    assert cell.init_range == DEFAULT_INIT_RANGE
    assert lo <= cell.A.weight.min() and cell.A.weight.max() <= hi
