"""Ensemble aggregation, uncertainty, diversity, and parameter registration."""
import pytest
import torch

from hcnns_chaos.core.ensembles.hcnn_ensembles import (
    VanillaHCNNEnsemble, PTFHCNNEnsemble, LFormHCNNEnsemble, LSpaHCNNEnsemble,
)

N_OBS, N_HID, N = 3, 6, 4
ALL = [VanillaHCNNEnsemble, PTFHCNNEnsemble, LFormHCNNEnsemble, LSpaHCNNEnsemble]


def _ens(cls, seed=42):
    return cls(N, N_OBS, N_HID, seed=seed)


@pytest.mark.parametrize("cls", ALL)
def test_params_registered(cls):
    ens = _ens(cls)
    assert sum(p.numel() for p in ens.parameters()) > 0  # ModuleDict registers members


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("method", ["mean", "median", "weighted_mean"])
def test_aggregation_shapes(cls, method):
    ens = _ens(cls)
    data = torch.randn(4, 10, N_OBS)
    out = ens(data, forecast_horizon=5, aggregation_method=method)
    assert out.expectations.shape == (4, 10, N_OBS)
    assert out.forecasts.shape == (4, 5, N_OBS)


def test_weighted_mean_equals_mean_at_equal_weights():
    ens = _ens(VanillaHCNNEnsemble)
    data = torch.randn(4, 10, N_OBS)
    m = ens(data, aggregation_method="mean").expectations
    w = ens(data, aggregation_method="weighted_mean", weights=torch.ones(N)).expectations
    assert torch.allclose(m, w, atol=1e-6)


def test_weights_are_normalized():
    ens = _ens(VanillaHCNNEnsemble)
    data = torch.randn(2, 6, N_OBS)
    # unnormalized weights must be normalized internally (result stays in range)
    a = ens(data, aggregation_method="weighted_mean", weights=torch.tensor([2., 2., 2., 2.])).expectations
    b = ens(data, aggregation_method="mean").expectations
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("cls", ALL)
def test_predict_with_uncertainty(cls):
    ens = _ens(cls)
    data = torch.randn(4, 10, N_OBS)
    uq = ens.predict_with_uncertainty(data, forecast_horizon=5, return_individual=True)
    for k in ("predictions_mean", "predictions_var", "model_agreement",
              "forecasts_mean", "forecasts_var", "individual_predictions"):
        assert k in uq
    assert (uq["predictions_var"] >= 0).all()
    assert 0.0 < uq["model_agreement"] <= 1.0
    assert uq["individual_predictions"].shape[0] == N


def test_single_member_variance_is_finite():
    ens = VanillaHCNNEnsemble(1, N_OBS, N_HID, seed=1)
    uq = ens.predict_with_uncertainty(torch.randn(2, 5, N_OBS))
    assert torch.isfinite(uq["predictions_var"]).all()


def test_diversity_reproducible():
    e1 = VanillaHCNNEnsemble(N, N_OBS, N_HID, seed=7)
    e2 = VanillaHCNNEnsemble(N, N_OBS, N_HID, seed=7)
    e3 = VanillaHCNNEnsemble(N, N_OBS, N_HID, seed=99)
    w1 = e1.get_model(0).cell.A.weight
    # members within an ensemble differ (diversity)
    assert not torch.allclose(e1.get_model(0).cell.A.weight, e1.get_model(1).cell.A.weight)
    # same seed -> reproducible; different seed -> different
    assert torch.allclose(w1, e2.get_model(0).cell.A.weight)
    assert not torch.allclose(w1, e3.get_model(0).cell.A.weight)
