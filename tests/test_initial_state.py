"""Initial-state modules: the default reproduces the shared s0, the options change only what
they claim to, and every architecture runs with a delay-embedding initial state."""
import tempfile

import pytest
import torch

import hcnn
from hcnn import DelayEmbeddingInitialState, SharedInitialState, Vanilla_Model
from hcnn.core.training import HCNNTrainer

N_OBS, N_HID, T = 3, 5, 12


def _window(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, T, N_OBS, generator=g) * 0.5


def _a_grad(model, y):
    model.zero_grad()
    HCNNTrainer(model, save_dir=tempfile.mkdtemp())._batch_loss(y).backward()
    return model.cell.A.weight.grad.clone()


def test_default_is_the_shared_s0_with_the_same_random_draw():
    torch.manual_seed(0)
    m = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    torch.manual_seed(0)
    expected = torch.empty(N_OBS + N_HID).uniform_(-0.5, 0.5)
    assert isinstance(m.initial_state, SharedInitialState)
    assert torch.equal(m.s0.detach(), expected)
    out = m(_window())
    assert torch.equal(out.states[:, 0], m.s0.detach().expand(2, -1))


@pytest.mark.parametrize("train_s0", [True, False])
def test_checkpoints_with_the_old_s0_key_still_load(train_s0):
    a = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, train_s0=train_s0)
    old = {("s0" if k == "initial_state.s0" else k): v for k, v in a.state_dict().items()}
    assert "s0" in old and "initial_state.s0" not in old
    b = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, train_s0=train_s0)
    b.load_state_dict(old)  # strict
    assert torch.equal(a.s0, b.s0)


def test_observed_from_data_zeroes_t0_and_changes_nothing_else():
    torch.manual_seed(0)
    a = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    b = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    b.load_state_dict(a.state_dict())
    b.set_initial_state(SharedInitialState(N_OBS, N_HID, observed_from_data=True))
    b.initial_state.s0.data.copy_(a.s0.data)
    y = _window()
    oa, ob = a(y, forecast_horizon=4), b(y, forecast_horizon=4)
    assert torch.equal(ob.expectations[:, 0], y[:, 0])                  # t = 0 term is 0
    assert torch.allclose(oa.expectations[:, 1:], ob.expectations[:, 1:])
    assert torch.allclose(oa.forecasts, ob.forecasts)
    assert torch.allclose(_a_grad(a, y), _a_grad(b, y))                  # same gradient for A


def test_noise_is_per_window_on_the_hidden_part_and_training_only():
    torch.manual_seed(0)
    m = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    m.set_initial_state(SharedInitialState(N_OBS, N_HID, noise_std=0.1))
    y = _window(batch=1).expand(4, -1, -1)                               # 4 identical windows
    s = m.train()(y).states[:, 0]
    assert torch.equal(s[:, :N_OBS], m.s0.detach()[:N_OBS].expand(4, -1))  # observed part untouched
    assert not torch.allclose(s[0, N_OBS:], s[1, N_OBS:])                # a draw per window
    s_eval = m.eval()(y).states[:, 0]
    assert torch.equal(s_eval, m.s0.detach().expand(4, -1))              # no noise in eval


def test_window_must_be_longer_than_the_warmup():
    m = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 0, N_OBS))                                      # empty window, default
    m.set_initial_state(DelayEmbeddingInitialState(N_OBS, N_HID, n_delays=4))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 4, N_OBS))


def test_set_initial_state_checks_type_and_sizes():
    m = Vanilla_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    with pytest.raises(TypeError):
        m.set_initial_state(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError):
        m.set_initial_state(SharedInitialState(N_OBS, N_HID + 1))


@pytest.mark.parametrize("arch", hcnn.list_architectures())
@pytest.mark.parametrize("hidden_size", [None, 16])
def test_delay_embedding_on_every_architecture(arch, hidden_size):
    k = 4
    torch.manual_seed(0)
    m = hcnn.build_model(arch, n_obs_vars=N_OBS, n_hid_vars=N_HID)
    m.set_initial_state(DelayEmbeddingInitialState(N_OBS, N_HID, n_delays=k, hidden_size=hidden_size))
    m.eval()
    y = _window()
    out = m(y, forecast_horizon=3)
    assert m.n_warmup == k
    assert out.expectations.shape == (2, T - k, N_OBS)                   # y_k .. y_{T-1}
    assert torch.allclose(out.delta_terms, y[:, k:] - out.expectations)
    # causal: the predictions never see the observation they predict
    y_last = y.clone(); y_last[:, -1] += 1.0
    assert torch.allclose(m(y_last).expectations, out.expectations)
    # the warm-up observations do shape the state
    y_first = y.clone(); y_first[:, 0] += 1.0
    assert not torch.allclose(m(y_first).expectations[:, 0], out.expectations[:, 0])
    # trains with HCNNTrainer, and the gradient reaches the encoder
    m.train()
    m.zero_grad()
    loss = HCNNTrainer(m, save_dir=tempfile.mkdtemp())._batch_loss(y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.initial_state.parameters())
