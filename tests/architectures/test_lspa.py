"""LSpa HCNN: masked (sparse) transition matrix."""
import torch

from hcnn.architectures.lspa import LSpa_Model

N_OBS, N_HID = 3, 17
N_STATE = N_OBS + N_HID


def test_actual_sparsity_matches_target():
    m = LSpa_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, sparsity_ratio=0.6)
    info = m.get_sparsity_info()
    assert abs(info["actual_sparsity"] - 0.6) < 0.02


def test_masked_weights_stay_zero_after_training():
    m = LSpa_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, sparsity_ratio=0.5)
    pattern = m.visualize_sparsity_pattern().clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    data = torch.randn(4, 8, N_OBS)
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(data).expectations, data).backward()
        opt.step()
    m(data)  # mask is re-applied on the forward pass
    W = m.get_sparse_transition_matrix()
    assert torch.count_nonzero(W[pattern == 0]) == 0


def test_non_obs_block_keeps_observed_columns_dense():
    m = LSpa_Model(n_obs_vars=N_OBS, n_hid_vars=N_HID, sparsity_ratio=0.7,
                   mask_type="non_obs_block")
    W = m.get_sparse_transition_matrix()
    assert torch.count_nonzero(W[:, :N_OBS]) == N_STATE * N_OBS

