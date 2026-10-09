"""PTF HCNN: dropout on the correction term and its per-epoch schedule."""
import tempfile

import torch
from torch.utils.data import DataLoader

from hcnn.architectures.ptf import PTFHCNNCell, PTF_Model
from hcnn.core.base import PTFHCNNOutput
from hcnn.core.training import HCNNTrainer
from hcnn.utils.data_preprocessing import SlidingWindowDataset

N_OBS, N_HID = 2, 3
N_STATE = N_OBS + N_HID


def test_partial_delta_is_carried_in_extras_and_output():
    cell = PTFHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID)
    co = cell(torch.randn(4, N_STATE), teacher_forcing=True, observation=torch.randn(4, N_OBS))
    assert co.extras["partial_delta_terms"].shape == (4, N_OBS)

    out = PTF_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID)(torch.randn(3, 6, N_OBS))
    assert isinstance(out, PTFHCNNOutput)
    assert out.partial_delta_terms.shape == (3, 6, N_OBS)


def test_no_dropout_in_eval_mode():
    cell = PTFHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID, dropout_strategy="constant",
                       dropout_params={"p": 0.9}).eval()
    co = cell(torch.randn(4, N_STATE), teacher_forcing=True, observation=torch.randn(4, N_OBS))
    assert torch.allclose(co.extras["partial_delta_terms"], co.delta_term)


def test_kept_corrections_are_not_rescaled():
    """Partial TF keeps or drops each correction; kept ones must equal delta (no 1/(1-p))."""
    cell = PTFHCNNCell(n_obs_vars=N_OBS, n_hid_vars=N_HID, dropout_strategy="constant",
                       dropout_params={"p": 0.5}).train()
    co = cell(torch.randn(256, N_STATE), teacher_forcing=True, observation=torch.randn(256, N_OBS))
    partial, full = co.extras["partial_delta_terms"], co.delta_term
    kept = partial != 0
    assert 0.3 < kept.float().mean() < 0.7
    assert torch.equal(partial[kept], full[kept])


def test_dropout_schedule_advances_during_training():
    t = torch.linspace(0, 12, 120)
    series = 0.5 * torch.stack([torch.sin(t), torch.cos(t)], dim=1)
    loader = DataLoader(SlidingWindowDataset(series, window_size=10), batch_size=16)
    m = PTF_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, dropout_strategy="linear",
                  dropout_params={"start_p": 0.0, "end_p": 0.5, "total_epochs": 5})
    before = m.cell.dropout_module.p
    HCNNTrainer(m, learning_rate=1e-2, save_dir=tempfile.mkdtemp()).train_only(
        loader, num_epochs=5, verbose=False)
    assert m.cell.dropout_module.p > before
