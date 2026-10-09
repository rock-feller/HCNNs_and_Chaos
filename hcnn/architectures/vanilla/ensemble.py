"""
Ensemble of Vanilla_Model members.

Aggregation, uncertainty and seeding live in :class:`hcnn.core.ensembles.base.HCNNEnsemble`;
this subclass only fixes the member type and spells out its constructor.
"""

from ...core.ensembles.base import HCNNEnsemble
from .model import Vanilla_Model


class VanillaHCNNEnsemble(HCNNEnsemble):
    """Ensemble of Vanilla HCNN models."""
    model_cls = Vanilla_Model
    variant_label = "vanilla"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         n_ext_vars=n_ext_vars)
