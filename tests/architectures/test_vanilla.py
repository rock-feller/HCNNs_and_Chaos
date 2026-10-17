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


def test_rollout_matches_an_independent_reference_with_gradients():
    """The whole model (shared rollout + cell), against the HCNN equations written as a plain
    loop that shares only the weights, in float64, with external inputs:

        y_hat_t = [I 0] s_t,  r_t = [y_t ; h_t],  s_{t+1} = A tanh(r_t) + B u_t   (calibration)
        s_{T+k+1} = A tanh(s_{T+k}) + B u'_k,  forecasts[k] = [I 0] s_{T+k}        (forecast)

    The reference writes r_t by replacing the observed part with the data; it does not reuse
    the code's correction formula, so a sign error in s_t + C^T delta_t cannot cancel out."""
    from hcnn import Vanilla_Model

    torch.manual_seed(0)
    T, H, N_EXT = 9, 4, 2
    m = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, n_ext_vars=N_EXT).double()
    g = torch.Generator().manual_seed(1)
    y = torch.randn(3, T, N_OBS, generator=g, dtype=torch.float64)
    u = torch.randn(3, T, N_EXT, generator=g, dtype=torch.float64)
    fu = torch.randn(3, H, N_EXT, generator=g, dtype=torch.float64)
    target = torch.randn(3, H, N_OBS, generator=g, dtype=torch.float64)

    out = m(y, forecast_horizon=H, externals=u, future_externals=fu)

    A, B, s = m.cell.A.weight, m.cell.B.weight, m.s0.expand(3, -1)
    exps, fcs = [], []
    for t in range(T):
        exps.append(s[:, :N_OBS])
        r = torch.cat([y[:, t], s[:, N_OBS:]], dim=-1)
        s = torch.tanh(r) @ A.T + u[:, t] @ B.T
    for k in range(H):
        fcs.append(s[:, :N_OBS])
        s = torch.tanh(s) @ A.T + fu[:, k] @ B.T
    exps, fcs = torch.stack(exps, 1), torch.stack(fcs, 1)

    assert torch.allclose(out.expectations, exps, atol=1e-12)
    assert torch.allclose(out.delta_terms, y - exps, atol=1e-12)
    assert torch.allclose(out.forecasts, fcs, atol=1e-12)

    def loss(e, f):
        return ((e - y) ** 2).mean() + ((f - target) ** 2).mean()

    params = [m.cell.A.weight, m.cell.B.weight, m.s0]
    g_model = torch.autograd.grad(loss(out.expectations, out.forecasts), params, retain_graph=True)
    g_ref = torch.autograd.grad(loss(exps, fcs), params)
    for a, b in zip(g_model, g_ref):
        assert torch.allclose(a, b, atol=1e-12)
