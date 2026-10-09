"""HCNNTrainer: loss decreases, both backprop modes validate, best weights restore.

(The PTF dropout-schedule test lives in tests/architectures/test_ptf.py.)
"""
import math
import tempfile
import torch
from torch.utils.data import DataLoader

from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import NormalizationStrategy, SlidingWindowDataset, forecast_windows
from hcnn import Vanilla_Model
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


def test_restore_best_loads_the_selected_epoch():
    loader, cal, val = _data()
    m = Vanilla_Model(n_obs_vars=3, n_hid_vars=10)
    tr = HCNNTrainer(m, learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    best_val = tr.train_and_validate(loader, num_epochs=3, calibration_window=cal,
                                     val_data=val, verbose=False)
    tr.restore_best()
    assert math.isclose(tr._validate(cal, val), best_val, rel_tol=1e-5)


def test_validation_over_several_windows():
    loader, cal, val = _data()
    test_like = torch.cat([cal, val])
    vcal, vtgt = forecast_windows(test_like, context=100, horizon=20, n_windows=4)
    assert vcal.shape == (4, 100, 3) and vtgt.shape == (4, 20, 3)
    assert torch.equal(vtgt[0], test_like[100:120])          # target follows its window
    tr = HCNNTrainer(Vanilla_Model(n_obs_vars=3, n_hid_vars=10), learning_rate=1e-2,
                     save_dir=tempfile.mkdtemp())
    assert math.isfinite(tr.train_and_validate(loader, 2, vcal, vtgt, verbose=False))


def test_cosine_schedule_anneals_learning_rate():
    loader, _, _ = _data()
    tr = HCNNTrainer(Vanilla_Model(n_obs_vars=3, n_hid_vars=10), learning_rate=1e-2,
                     lr_schedule="cosine", min_lr_ratio=0.1, save_dir=tempfile.mkdtemp())
    tr.train_only(loader, num_epochs=3, verbose=False)
    assert math.isclose(tr.optimizer.param_groups[0]["lr"], 1e-3, rel_tol=1e-6)


def test_early_stopping_with_patience():
    loader, cal, val = _data()
    tr = HCNNTrainer(Vanilla_Model(n_obs_vars=3, n_hid_vars=10), learning_rate=0.0,  # never improves
                     patience=2, save_dir=tempfile.mkdtemp())
    tr.train_and_validate(loader, num_epochs=20, calibration_window=cal, val_data=val, verbose=False)
    import json, glob
    history = json.load(open(glob.glob(tr.save_dir + "/*_losses.json")[0]))
    assert len(history) == 3  # best at epoch 1, then 2 epochs of patience


def test_per_epoch_mode_matches_full_batch_gradient():
    """Batch-wise gradient accumulation == one backward on the mean loss."""
    loader, _, _ = _data()
    torch.manual_seed(0)
    a = Vanilla_Model(n_obs_vars=3, n_hid_vars=6)
    b = Vanilla_Model(n_obs_vars=3, n_hid_vars=6)
    b.load_state_dict(a.state_dict())
    ta = HCNNTrainer(a, backprop_mode="per_epoch", learning_rate=0.0, save_dir=tempfile.mkdtemp())
    ta._run_train_epoch(loader, 0)
    losses = [ta.loss_fn(b(data_window=batch).expectations, batch) for batch in loader]
    (sum(losses) / len(losses)).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)
