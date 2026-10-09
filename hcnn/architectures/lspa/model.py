"""
Large-Sparse (LSpa) HCNN model.

The two-phase rollout (teacher-forced calibration, then autonomous forecast) is
inherited from :class:`hcnn.core.base.BaseHCNNModel`; this wrapper only builds
the cell. See ``README.md`` in this folder for the math.
"""

import torch
from typing import Optional, Tuple, Literal

from ...core.base import BaseHCNNModel
from .cell import LSpaHCNNCell


class LSpa_Model(BaseHCNNModel):
    """
    Large-Sparse HCNN: vanilla dynamics with a structured-sparse transition matrix
    ``A_sparse`` for scaling to high-dimensional state spaces. Two mask types are
    supported: ``random_block`` (uniform random) and ``non_obs_block`` (sparsify
    only the hidden-related blocks).
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        sparsity_ratio: float = 0.5,
        mask_type: Literal['non_obs_block', 'random_block'] = 'random_block',
        bias: bool = False,
        s0_nature: Literal['zeros_', 'random_'] = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        n_ext_vars: Optional[int] = None,
    ):
        super().__init__(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            n_ext_vars=n_ext_vars,
        )
        self.sparsity_ratio = sparsity_ratio
        self.mask_type = mask_type
        self.bias = bias
        self.cell = LSpaHCNNCell(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            init_range=init_range,
            bias=bias,
            mask_type=mask_type,
            sparsity_ratio=sparsity_ratio,
            n_ext_vars=n_ext_vars,
        )
        self.name = "LSpa_Model"

    @property
    def model_type(self) -> str:
        return "lspa_hcnn"

    def get_sparsity_info(self) -> dict:
        """Statistics about the current sparsity pattern."""
        return self.cell.get_sparsity_info()

    def visualize_sparsity_pattern(self) -> torch.Tensor:
        """Binary (non-zero) pattern of the sparse transition matrix."""
        return self.cell.visualize_sparsity_pattern()

    def get_sparse_transition_matrix(self) -> torch.Tensor:
        """Weights of the sparse transition matrix ``A_sparse``."""
        return self.cell.get_sparse_transition_matrix()
