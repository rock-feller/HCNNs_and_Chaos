"""RNN/LSTM baseline models, ensembles, and trainers."""
import math
import tempfile
import torch
from torch.utils.data import DataLoader

from hcnn.baselines import (
    RNNModel, LSTMModel, RNNEnsemble, LSTMEnsemble,
    SequenceModelTrainer, SequenceEnsembleTrainer,
)
from hcnn.utils.data_generation import LorenzSolver
from hcnn.utils.data_preprocessing import NormalizationStrategy, SlidingWindowDataset


def _data():
    traj = LorenzSolver(0, 20, (1., 0., 1.25), 0.01, burn_in=10.0)[0]
    norm = NormalizationStrategy().fit(traj[:1500], 0.02)
    train = norm.transform(traj[:1500])
    test = norm.transform(traj[1500:])
    loader = DataLoader(SlidingWindowDataset(train, 50), batch_size=16, shuffle=False)
    return loader, test[:100], test[100:200]


def test_rnn_forward_and_forecast_shapes():
    m = RNNModel(input_size=3, hidden_size=16, output_size=3)
    assert m(torch.randn(4, 10, 3)).outputs.shape == (4, 10, 3)
    assert m.forecast(torch.randn(4, 10, 3), 5).shape == (4, 5, 3)


def test_baselines_build_on_cpu():
    for m in (RNNModel(3, 16, 3), LSTMModel(3, 16, 3)):
        assert {p.device.type for p in m.parameters()} == {"cpu"}


def test_sequence_trainer_reduces_loss():
    loader, _, _ = _data()
    m = RNNModel(input_size=3, hidden_size=32, output_size=3)
    tr = SequenceModelTrainer(m, learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    first = tr._run_train_epoch(loader, 0)
    best = tr.train_only(loader, num_epochs=5, verbose=False)
    assert best <= first


def test_sequence_trainer_validate_finite():
    loader, cal, val = _data()
    m = LSTMModel(input_size=3, hidden_size=32, output_size=3)
    tr = SequenceModelTrainer(m, loss_fn="logcosh", backprop_mode="per_epoch",
                              learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    best_val = tr.train_and_validate(loader, 3, cal, val, verbose=False)
    assert math.isfinite(best_val)


def test_sequence_ensemble_trainer_trains_all_members():
    loader, _, _ = _data()
    ens = RNNEnsemble(n_ensemble=3, input_size=3, hidden_size=16, output_size=3)
    tr = SequenceEnsembleTrainer(ens, learning_rate=1e-2, save_dir=tempfile.mkdtemp())
    losses = tr.train_only(loader, num_epochs=2, verbose=False)
    assert len(losses) == 3 and all(math.isfinite(x) for x in losses)


def test_deprecated_shims_reexport():
    from hcnn.core.models.classic_models import RNNModel as R2, LSTMModel as L2
    from hcnn.core.ensembles.classic_ensembles import RNNEnsemble as RE2, LSTMEnsemble as LE2
    assert (R2, L2, RE2, LE2) == (RNNModel, LSTMModel, RNNEnsemble, LSTMEnsemble)
