"""
Ensemble of PTF_Model members.

Aggregation, uncertainty and seeding live in :class:`hcnn.core.ensembles.base.HCNNEnsemble`;
this subclass only fixes the member type and spells out its constructor.
"""

from ...core.ensembles.base import HCNNEnsemble
from .model import PTF_Model


class PTFHCNNEnsemble(HCNNEnsemble):
    """Ensemble of Partial Teacher Forcing HCNN models."""
    model_cls = PTF_Model
    variant_label = "ptf"

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), target_prob=0.25,
                 dropout_strategy="adaptive", dropout_params=None, n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         target_prob=target_prob, dropout_strategy=dropout_strategy,
                         dropout_params=dropout_params, n_ext_vars=n_ext_vars)

    def update_dropout_epoch(self, epoch: int):
        """Advance the dropout schedule of every member (for PTF training loops)."""
        for m in self.models.values():
            m.update_dropout_epoch(epoch)
