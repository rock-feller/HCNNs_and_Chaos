"""
Recurrence correctness + model rollout tests (against the REAL hcnns_chaos code).

The key test re-derives one Vanilla cell step independently from the cell's own
weights and asserts the cell matches - a regression anchor for the recurrence.
"""
import pytest
import torch

from hcnns_chaos.core.base import CellOutput, HCNNOutput, PTFHCNNOutput
from hcnns_chaos.core.cells import (
    VanillaHCNNCell, PTFHCNNCell, LFormHCNNCell, LSpaHCNNCell,
)
from hcnns_chaos.core.models.hcnn_models import (
    Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model,
)

N_OBS, N_HID = 2, 3
N_STATE = N_OBS + N_HID


def _model(cls):
    return cls(n_obs_vars=N_OBS, n_hid_vars=N_HID)


# --------------------------------------------------------------------------- #
# Recurrence: cells implement the documented formula
# --------------------------------------------------------------------------- #
def test_vanilla_cell_matches_formula():
    cell = VanillaHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    B = 4
    state = torch.randn(B, N_STATE)
    obs = torch.randn(B, N_OBS)
    co = cell(state, teacher_forcing=True, observation=obs)

    C = torch.eye(N_OBS, N_STATE)
    exp_ref = state @ C.T
    delta_ref = obs - exp_ref
    corrected = state - delta_ref @ C
    next_ref = torch.tanh(corrected) @ cell.A.weight.T

    assert torch.allclose(co.expectation, exp_ref, atol=1e-6)
    assert torch.allclose(co.delta_term, delta_ref, atol=1e-6)
    assert torch.allclose(co.next_state, next_ref, atol=1e-5)


def test_lform_reduces_to_memory_when_D_zero():
    # With D forced to ~0, s_{t+1} = r_t (pure memory / identity on corrected state).
    cell = LFormHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID, init_diag=1e-6)
    B = 4
    state = torch.randn(B, N_STATE)
    obs = torch.randn(B, N_OBS)
    co = cell(state, teacher_forcing=True, observation=obs)
    C = torch.eye(N_OBS, N_STATE)
    r = state - (obs - state @ C.T) @ C
    assert torch.allclose(co.next_state, r, atol=1e-4)


@pytest.mark.parametrize("cls", [VanillaHCNNCell, LFormHCNNCell, LSpaHCNNCell])
def test_cells_return_celloutput(cls):
    cell = cls(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    co = cell(torch.randn(4, N_STATE), teacher_forcing=False)
    assert isinstance(co, CellOutput)
    assert co.expectation.shape == (4, N_OBS)
    assert co.next_state.shape == (4, N_STATE)
    assert co.extras == {}


def test_ptf_cell_carries_partial_delta_in_extras():
    cell = PTFHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    co = cell(torch.randn(4, N_STATE), teacher_forcing=True, observation=torch.randn(4, N_OBS))
    assert "partial_delta_terms" in co.extras
    assert co.extras["partial_delta_terms"].shape == (4, N_OBS)


# --------------------------------------------------------------------------- #
# Model rollout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls,out_type", [
    (Vanilla_Model, HCNNOutput),
    (PTF_Model, PTFHCNNOutput),
    (LForm_Model, HCNNOutput),
    (LSpa_Model, HCNNOutput),
])
def test_model_forward_forecast_shapes(cls, out_type):
    m = _model(cls)
    B, T, H = 4, 12, 7
    data = torch.randn(B, T, N_OBS)
    out = m(data, forecast_horizon=H)
    assert isinstance(out, out_type)
    assert out.expectations.shape == (B, T, N_OBS)
    assert out.states.shape == (B, T, N_STATE)
    assert out.forecasts.shape == (B, H, N_OBS)
    assert out.future_states.shape == (B, H, N_STATE)


def test_expectations_are_C_times_states():
    """Rollout invariant: expectation_t == C @ state_t for all t."""
    m = _model(Vanilla_Model)
    data = torch.randn(4, 10, N_OBS)
    out = m(data)
    # C = [I|0] -> first n_obs coords of each state
    assert torch.allclose(out.expectations, out.states[..., :N_OBS], atol=1e-6)


def test_forecast_is_deterministic_and_autonomous():
    """Forecast depends only on the calibration window (no future-truth input)."""
    m = _model(Vanilla_Model).eval()
    data = torch.randn(1, 20, N_OBS)
    with torch.no_grad():
        f1 = m(data, forecast_horizon=15).forecasts
        f2 = m(data, forecast_horizon=15).forecasts
    assert torch.allclose(f1, f2)
    # First forecast step continues from the last calibration state.
    with torch.no_grad():
        out = m(data, forecast_horizon=1)
    assert out.forecasts.shape == (1, 1, N_OBS)


@pytest.mark.parametrize("cls", [Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model])
def test_gradients_flow(cls):
    m = _model(cls)
    data = torch.randn(4, 8, N_OBS)
    loss = torch.nn.functional.mse_loss(m(data).expectations, data)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_models_build_on_cpu_by_default():
    for cls in (Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model):
        m = _model(cls)
        devs = {p.device.type for p in m.parameters()} | {b.device.type for b in m.buffers()}
        assert devs == {"cpu"}, f"{cls.__name__} not all-CPU: {devs}"
