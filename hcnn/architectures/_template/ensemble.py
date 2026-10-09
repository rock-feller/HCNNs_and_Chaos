"""<Name> HCNN ensemble - TEMPLATE. Aggregation/uncertainty come from HCNNEnsemble."""

from ...core.ensembles.base import HCNNEnsemble
from .model import Template_Model


class TemplateHCNNEnsemble(HCNNEnsemble):  # TODO: rename
    """Ensemble of Template_Model members."""
    model_cls = Template_Model
    variant_label = "template"  # TODO

    def __init__(self, n_ensemble, n_obs_vars, n_hid_vars, s0_nature="random_",
                 train_s0=True, init_range=(-0.75, 0.75), n_ext_vars=None, seed=None):
        super().__init__(n_ensemble, seed=seed, n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars,
                         s0_nature=s0_nature, train_s0=train_s0, init_range=init_range,
                         n_ext_vars=n_ext_vars)
