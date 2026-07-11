"""Custom-layer invariants: diagonal gate and structured sparsity."""
import torch

from hcnns_chaos.core.layers import CustomLinear, DiagonalMatrix, CustomSparseLinear


def test_custom_linear_init_range_and_device():
    layer = CustomLinear(4, 6, init_range=(-0.3, 0.3))
    assert layer.weight.device.type == "cpu"
    assert layer.weight.min() >= -0.3 - 1e-6 and layer.weight.max() <= 0.3 + 1e-6


def test_diagonal_matrix_stays_diagonal_and_bounded():
    D = DiagonalMatrix(n_features=6, init_diag=0.5)
    x = torch.randn(8, 6)
    _ = D(x)  # forward enforces structure + clamp
    off_diag = D.weight - torch.diag(torch.diag(D.weight))
    assert torch.count_nonzero(off_diag) == 0
    vals = D.get_diagonal_values()
    assert (vals > 0).all() and (vals < 1).all()


def test_diagonal_gradient_is_masked_to_diagonal():
    D = DiagonalMatrix(n_features=5, init_diag=0.5)
    out = D(torch.randn(4, 5))
    out.pow(2).mean().backward()
    off = D.weight.grad - torch.diag(torch.diag(D.weight.grad))
    assert torch.count_nonzero(off) == 0  # off-diagonal grads zeroed by the hook


def test_diagonal_grad_clip_configurable():
    D = DiagonalMatrix(n_features=5, init_diag=0.5, grad_clip=None)
    assert D.grad_clip is None  # clipping can be disabled for gate studies


def test_sparse_linear_sparsity_ratio_random_block():
    L = CustomSparseLinear(n_obs_vars=3, n_hid_vars=17, sparsity_ratio=0.5,
                           mask_type="random_block")
    info = L.get_sparsity_info()
    # roughly half the entries zero
    assert abs(info["actual_sparsity"] - 0.5) < 0.05


def test_sparse_non_obs_block_keeps_obs_columns_dense():
    n_obs, n_hid = 3, 7
    L = CustomSparseLinear(n_obs_vars=n_obs, n_hid_vars=n_hid, sparsity_ratio=0.5,
                           mask_type="non_obs_block")
    m = L.sparsity_mask
    assert (m[:, :n_obs] == 1).all()          # observable columns dense
    assert (m[:, n_obs:] == 0).any()          # hidden columns sparsified


def test_sparse_mask_maintained_after_optimizer_step():
    L = CustomSparseLinear(n_obs_vars=2, n_hid_vars=8, sparsity_ratio=0.5)
    mask = L.sparsity_mask.bool()
    opt = torch.optim.SGD(L.parameters(), lr=1.0)
    L(torch.randn(4, 10)).pow(2).mean().backward()
    opt.step()
    _ = L(torch.randn(4, 10))  # forward re-applies the mask
    assert torch.count_nonzero(L.weight[~mask]) == 0
