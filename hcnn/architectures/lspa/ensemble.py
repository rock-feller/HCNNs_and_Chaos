"""
Ensemble of LSpa_Model members.

Aggregation, uncertainty and seeding live in :class:`hcnn.core.ensembles.base.HCNNEnsemble`;
this subclass only fixes the member type and spells out its constructor.
"""

from typing import Dict

from ...core.ensembles.base import HCNNEnsemble
from .model import LSpa_Model


class LSpaHCNNEnsemble(HCNNEnsemble):
    """Ensemble of Large-Sparse HCNN models."""
    model_cls = LSpa_Model
    variant_label = "lspa"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), bias=False,
                 mask_type="random_block", sparsity_ratio=0.25, n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         bias=bias, mask_type=mask_type, sparsity_ratio=sparsity_ratio,
                         n_ext_vars=n_ext_vars)

    def get_ensemble_sparsity_info(self) -> Dict[str, Dict]:
        """Sparsity statistics for every member."""
        return {f"model_{i}": m.get_sparsity_info() for i, m in enumerate(self.models.values())}
