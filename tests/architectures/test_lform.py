"""LForm HCNN: s_{t+1} = r_t + D (A tanh(r_t) - r_t) with a diagonal gate D in (0, 1)."""
import torch

from hcnn.architectures.lform import LFormHCNNCell, LForm_Model

N_OBS, N_HID = 2, 3
N_STATE = N_OBS + N_HID
C = torch.eye(N_OBS, N_STATE)


def _corrected(state, obs):
    return state - (obs - state @ C.T) @ C


def test_reduces_to_memory_when_gate_closed():
    """D -> 0: s_{t+1} = r_t (the corrected state is carried over unchanged)."""
    cell = LFormHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID, init_diag=1e-6)
    state, obs = torch.randn(4, N_STATE), torch.randn(4, N_OBS)
    co = cell(state, teacher_forcing=True, observation=obs)
    assert torch.allclose(co.next_state, _corrected(state, obs), atol=1e-4)


def test_step_matches_formula():
    cell = LFormHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID, init_diag=0.3)
    state, obs = torch.randn(4, N_STATE), torch.randn(4, N_OBS)
    co = cell(state, teacher_forcing=True, observation=obs)
    r = _corrected(state, obs)
    d = cell.get_diagonal_values()
    ref = r + d * (torch.tanh(r) @ cell.A.weight.T - r)
    assert torch.allclose(co.next_state, ref, atol=1e-5)


def test_gate_stays_diagonal_and_in_unit_interval_after_training_step():
    m = LForm_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, init_diag=0.9)
    opt = torch.optim.SGD(m.parameters(), lr=5.0)
    data = torch.randn(4, 8, N_OBS)
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(data).expectations, data).backward()
        opt.step()
    m(data)  # the gate is re-projected on the forward pass
    D = m.get_diagonal_matrix()
    assert torch.count_nonzero(D - torch.diag(torch.diagonal(D))) == 0
    vals = m.get_diagonal_values()
    assert (vals > 0).all() and (vals < 1).all()
