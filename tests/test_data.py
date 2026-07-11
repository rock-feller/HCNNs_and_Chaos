"""Data-pipeline correctness: burn-in, leakage-safe normalization, windows."""
import torch

from hcnns_chaos.utils.data_generation import LorenzSolver, ChaoticSystemGenerator
from hcnns_chaos.utils.data_preprocessing import (
    NormalizationStrategy, NoisificationStrategy, SlidingWindowDataset, prepare_chaotic_data,
)


def test_burn_in_preserves_length_and_moves_off_transient():
    a, _ = LorenzSolver(0, 10, (1., 0., 1.25), 0.01, burn_in=0.0)
    b, _ = LorenzSolver(0, 10, (1., 0., 1.25), 0.01, burn_in=5.0)
    assert len(a) == len(b)
    assert not torch.allclose(a[0], b[0])  # burn-in starts elsewhere on the attractor


def test_generator_burn_in():
    gen = ChaoticSystemGenerator()
    traj, t = gen.generate_lorenz(start=0.0, stop=10.0, time_grid=0.01, burn_in=5.0)
    assert traj.shape[0] == len(t) == len(torch.arange(0, 10, 0.01))


def test_normalization_fit_on_train_no_leakage():
    data = LorenzSolver(0, 20, (1., 0., 1.25), 0.01, burn_in=10.0)[0]
    tr, te = data[:1200], data[1200:]
    norm = NormalizationStrategy().fit(tr, 0.02)
    # fitted mean is the TRAIN mean, not the full-series mean
    assert torch.allclose(norm.mean_, tr.mean(0))
    assert not torch.allclose(tr.mean(0), data.mean(0), atol=1e-3)
    # centered train has ~zero mean; inverse recovers
    assert torch.allclose(norm.transform(tr).mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(norm.inverse_transform(norm.transform(tr)), tr, atol=1e-4)


def test_prepare_chaotic_data_uses_train_mean():
    data = LorenzSolver(0, 20, (1., 0., 1.25), 0.01, burn_in=10.0)[0]
    res = prepare_chaotic_data(data, scaling_factor=0.02, train_ratio=0.7)
    split = int(len(data) * 0.7)
    assert torch.allclose(res["data_averages"], data[:split].mean(0), atol=1e-5)


def test_uniform_noise_runs():
    x = torch.zeros(50, 3)
    y = NoisificationStrategy().add_uniform_noise(x, 0.1)
    assert y.shape == x.shape
    assert y.abs().max() <= 0.1 + 1e-6


def test_sliding_window_shapes_cpu():
    data = torch.randn(100, 3)
    ds = SlidingWindowDataset(data, window_size=20)
    assert len(ds) == 100 - 20 + 1
    assert ds[0].shape == (20, 3)
    assert ds[0].device.type == "cpu"
