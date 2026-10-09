"""
Architecture contract - runs against EVERY registered architecture.

A new package under ``hcnn/architectures/`` is picked up automatically: if these
tests pass, the architecture works with the shared rollout, trainers, ensembles
and checkpoints. The ``_template`` package is checked too, so a fresh copy of it
is always a valid starting point.

Architecture-specific behaviour (e.g. "the cell implements *this* formula") is
tested in ``tests/architectures/test_<name>.py``.
"""
import importlib
import math
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

import hcnn
from hcnn.core import registry
from hcnn.core.base import BaseHCNNCell, BaseHCNNModel, CellOutput
from hcnn.core.ensembles import HCNNEnsemble
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_preprocessing import SlidingWindowDataset

N_OBS, N_HID = 2, 3
N_STATE = N_OBS + N_HID
TEMPLATE = "_template"

# Known contract violations, tracked here until fixed (strict: the xfail turns
# into a failure once the bug is fixed, so this list cannot go stale).
KNOWN_FAILURES = {}


def _template_spec():
    mod = importlib.import_module(f"hcnn.architectures.{TEMPLATE}")
    registry._REGISTRY.pop(mod.SPEC.name, None)  # keep the template out of the real registry
    return mod.SPEC


@pytest.fixture(params=hcnn.list_architectures() + [TEMPLATE])
def spec(request):
    return _template_spec() if request.param == TEMPLATE else hcnn.get_architecture(request.param)


def _xfail_if_known(spec, check):
    reason = KNOWN_FAILURES.get((spec.name, check))
    if reason:
        pytest.xfail(reason)


def _model(spec):
    return spec.model_cls(n_obs_vars=N_OBS, n_hid_vars=N_HID)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_spec_is_consistent(spec):
    package = spec.model_cls.__module__.split(".")[2]  # hcnn.architectures.<package>.model
    assert package.lstrip("_") == spec.name, "spec name must equal the package folder name"
    assert issubclass(spec.model_cls, BaseHCNNModel)
    assert issubclass(spec.cell_cls, BaseHCNNCell)
    assert spec.ensemble_cls is None or issubclass(spec.ensemble_cls, HCNNEnsemble)
    assert spec.summary, "give the architecture a one-line summary"
    assert isinstance(_model(spec).cell, spec.cell_cls)


# --------------------------------------------------------------------------- #
# Cell
# --------------------------------------------------------------------------- #
def test_cell_returns_celloutput_in_both_modes(spec):
    cell = spec.cell_cls(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    state, obs = torch.randn(4, N_STATE), torch.randn(4, N_OBS)

    tf = cell(state, teacher_forcing=True, observation=obs)
    assert isinstance(tf, CellOutput)
    assert tf.expectation.shape == (4, N_OBS)
    assert tf.next_state.shape == (4, N_STATE)
    assert torch.allclose(tf.expectation, state[:, :N_OBS], atol=1e-6), "expectation must be C s_t"
    assert torch.allclose(tf.delta_term, obs - tf.expectation, atol=1e-6)
    assert isinstance(tf.extras, dict)

    auto = cell(state, teacher_forcing=False)
    assert auto.next_state.shape == (4, N_STATE)
    assert auto.delta_term is None


def test_teacher_forcing_replaces_observed_coordinates(spec):
    """r_t = s_t - C^T (y_pred - y_true): the observed part of r_t IS the data, so under
    teacher forcing the next state cannot depend on the model's own observed coordinates."""
    cell = spec.cell_cls(n_obs_vars=N_OBS, n_hid_vars=N_HID).eval()  # eval: no PTF dropout
    obs = torch.randn(4, N_OBS)
    a = torch.randn(4, N_STATE)
    b = a.clone()
    b[:, :N_OBS] = torch.randn(4, N_OBS)  # different predictions, same hidden state
    na = cell(a, teacher_forcing=True, observation=obs).next_state
    nb = cell(b, teacher_forcing=True, observation=obs).next_state
    assert torch.allclose(na, nb, atol=1e-6)


def test_forecast_starts_after_the_window(spec):
    """forecasts[:, 0] predicts the step AFTER the window, not the last observed step."""
    m = _model(spec).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 10, N_OBS), forecast_horizon=2)
    assert not torch.allclose(out.forecasts[:, 0], out.expectations[:, -1])
    assert torch.allclose(out.forecasts, out.future_states[..., :N_OBS], atol=1e-6)


def test_teacher_forcing_uses_the_observation(spec):
    cell = spec.cell_cls(n_obs_vars=N_OBS, n_hid_vars=N_HID).eval()  # eval: no PTF dropout
    state = torch.randn(4, N_STATE)
    a = cell(state, teacher_forcing=True, observation=torch.randn(4, N_OBS)).next_state
    b = cell(state, teacher_forcing=True, observation=torch.randn(4, N_OBS)).next_state
    assert not torch.allclose(a, b)


def test_autonomous_step_ignores_the_observation(spec):
    cell = spec.cell_cls(n_obs_vars=N_OBS, n_hid_vars=N_HID).eval()
    state = torch.randn(4, N_STATE)
    a = cell(state, teacher_forcing=False).next_state
    b = cell(state, teacher_forcing=False, observation=torch.randn(4, N_OBS)).next_state
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# Model / shared rollout
# --------------------------------------------------------------------------- #
def test_model_rollout_shapes(spec):
    B, T, H = 4, 12, 7
    out = _model(spec)(torch.randn(B, T, N_OBS), forecast_horizon=H)
    assert out.expectations.shape == (B, T, N_OBS)
    assert out.states.shape == (B, T, N_STATE)
    assert out.delta_terms.shape == (B, T, N_OBS)
    assert out.forecasts.shape == (B, H, N_OBS)
    assert out.future_states.shape == (B, H, N_STATE)


def test_no_forecast_without_horizon(spec):
    out = _model(spec)(torch.randn(2, 5, N_OBS))
    assert out.forecasts is None and out.future_states is None


def test_expectations_are_C_times_states(spec):
    out = _model(spec)(torch.randn(4, 10, N_OBS))
    assert torch.allclose(out.expectations, out.states[..., :N_OBS], atol=1e-6)


def test_forecast_is_deterministic_in_eval(spec):
    m = _model(spec).eval()
    data = torch.randn(1, 20, N_OBS)
    with torch.no_grad():
        f1 = m(data, forecast_horizon=15).forecasts
        f2 = m(data, forecast_horizon=15).forecasts
    assert torch.allclose(f1, f2)


def test_gradients_flow(spec):
    m = _model(spec)
    data = torch.randn(4, 8, N_OBS)
    torch.nn.functional.mse_loss(m(data).expectations, data).backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_forecast_uses_the_last_observation(spec):
    m = _model(spec).eval()
    data = torch.randn(1, 10, N_OBS)
    shifted = data.clone()
    shifted[:, -1] += 1.0  # change only the last observed point
    with torch.no_grad():
        a = m(data, forecast_horizon=3).forecasts
        b = m(shifted, forecast_horizon=3).forecasts
    assert not torch.allclose(a, b)


def test_builds_on_cpu_by_default(spec):
    m = _model(spec)
    devices = {t.device.type for t in [*m.parameters(), *m.buffers()]}
    assert devices == {"cpu"}


def test_state_dict_roundtrip(spec):
    _xfail_if_known(spec, "state_dict_roundtrip")
    a = _model(spec).eval()
    data = torch.randn(2, 8, N_OBS)
    with torch.no_grad():
        before = a(data, forecast_horizon=5).forecasts
    b = _model(spec).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        after = b(data, forecast_horizon=5).forecasts
    assert torch.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def test_ensemble(spec):
    if spec.ensemble_cls is None:
        pytest.skip("architecture has no ensemble")
    ens = spec.ensemble_cls(n_ensemble=3, n_obs_vars=N_OBS, n_hid_vars=N_HID, seed=7)
    data = torch.randn(2, 6, N_OBS)
    out = ens(data, forecast_horizon=4, aggregation_method="median")
    assert out.forecasts.shape == (2, 4, N_OBS)
    u = ens.predict_with_uncertainty(data, forecast_horizon=4)
    assert u["forecasts_var"].shape == (2, 4, N_OBS)
    assert 0 < u["model_agreement"] <= 1
    w0, w1 = (torch.nn.utils.parameters_to_vector(ens.get_model(i).parameters()) for i in (0, 1))
    assert not torch.equal(w0, w1), "seeded members must differ"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_trains_with_hcnn_trainer(spec):
    t = torch.linspace(0, 12, 120)
    series = 0.5 * torch.stack([torch.sin(t), torch.cos(1.3 * t)], dim=1)
    loader = DataLoader(SlidingWindowDataset(series[:80], window_size=10), batch_size=8)
    m = _model(spec)
    trainer = HCNNTrainer(m, learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    best = trainer.train_and_validate(loader, num_epochs=2, calibration_window=series[80:100],
                                      val_data=series[100:110], verbose=False)
    assert math.isfinite(best)
    trainer.restore_best()
