"""HCNNTrainer: loss decreases, both backprop modes validate, PTF schedule advances."""
import math
import tempfile
import torch
from torch.utils.data import DataLoader

from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import NormalizationStrategy, SlidingWindowDataset
from hcnn.core.models.hcnn_models import Vanilla_Model, PTF_Model
from hcnn.core.training.hcnn_trainer import HCNNTrainer


def _data():
    traj = LorenzSolver(0, 20, (1., 0., 1.25), 0.01, burn_in=10.0)[0]
    norm = NormalizationStrategy().fit(traj[:1500], 0.02)
    train = norm.transform(traj[:1500])
    test = norm.transform(traj[1500:])
    loader = DataLoader(SlidingWindowDataset(train, 100), batch_size=16, shuffle=False)
    return loader, test[:200], test[200:400]


def test_train_only_reduces_loss():
    loader, _, _ = _data()
    m = Vanilla_Model(n_obs_vars=3, n_hid_vars=10)
    tr = HCNNTrainer(m, learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    first = tr._run_train_epoch(loader, 0)
    best = tr.train_only(loader, num_epochs=5, verbose=False)
    assert best <= first  # best epoch no worse than the first


def test_train_and_validate_per_epoch_finite():
    """per_epoch mode previously fell through and did nothing; now it must run."""
    loader, cal, val = _data()
    m = Vanilla_Model(n_obs_vars=3, n_hid_vars=10)
    tr = HCNNTrainer(m, loss_fn="logcosh", backprop_mode="per_epoch",
                     learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    best_val = tr.train_and_validate(loader, num_epochs=4, calibration_window=cal,
                                     val_data=val, verbose=False)
    assert math.isfinite(best_val)


def test_logcosh_stable_on_large_errors():
    big = torch.tensor([[100.0, -100.0]])
    zero = torch.zeros_like(big)
    loss = HCNNTrainer._logcosh_loss(big, zero)
    assert torch.isfinite(loss)  # naive log(cosh(100)) would overflow to inf


def test_ptf_dropout_schedule_advances():
    loader, _, _ = _data()
    m = PTF_Model(n_obs_vars=3, n_hid_vars=10, dropout_strategy="linear",
                  dropout_params={"start_p": 0.0, "end_p": 0.5, "total_epochs": 5})
    before = m.cell.dropout_module.p
    tr = HCNNTrainer(m, learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    tr.train_only(loader, num_epochs=5, verbose=False)
    assert m.cell.dropout_module.p > before
