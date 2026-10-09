"""Vanilla HCNN: s_{t+1} = A tanh(r_t), r_t = s_t - C^T (y_pred - y_true)."""
import torch

from hcnn.architectures.vanilla import VanillaHCNNCell

N_OBS, N_HID = 2, 3
N_STATE = N_OBS + N_HID
C = torch.eye(N_OBS, N_STATE)


def test_teacher_forced_step_matches_formula():
    """Regression anchor: re-derive one step from the cell's own weights."""
    cell = VanillaHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    state, obs = torch.randn(4, N_STATE), torch.randn(4, N_OBS)
    co = cell(state, teacher_forcing=True, observation=obs)

    exp_ref = state @ C.T
    delta_ref = obs - exp_ref
    r_ref = state - (exp_ref - obs) @ C          # observed part of r_t equals the data
    assert torch.allclose(r_ref[:, :N_OBS], obs, atol=1e-6)
    next_ref = torch.tanh(r_ref) @ cell.A.weight.T

    assert torch.allclose(co.expectation, exp_ref, atol=1e-6)
    assert torch.allclose(co.delta_term, delta_ref, atol=1e-6)
    assert torch.allclose(co.next_state, next_ref, atol=1e-5)


def test_autonomous_step_matches_formula():
    cell = VanillaHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    state = torch.randn(4, N_STATE)
    co = cell(state, teacher_forcing=False)
    assert torch.allclose(co.next_state, torch.tanh(state) @ cell.A.weight.T, atol=1e-5)


def test_external_inputs_enter_additively():
    cell = VanillaHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID, n_ext_vars=2)
    state, u = torch.randn(4, N_STATE), torch.randn(4, 2)
    co = cell(state, teacher_forcing=False, externals=u)
    ref = torch.tanh(state) @ cell.A.weight.T + u @ cell.B.weight.T
    assert torch.allclose(co.next_state, ref, atol=1e-5)
