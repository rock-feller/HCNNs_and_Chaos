"""Architecture registry, build-by-name helpers, and legacy import paths."""
import pytest
import torch

import hcnn
from hcnn.core import registry
from hcnn.core.registry import ArchitectureSpec


def test_builtin_architectures_are_registered():
    assert {"vanilla", "ptf", "lform", "lspa"} <= set(hcnn.list_architectures())
    assert "template" not in hcnn.list_architectures()


def test_build_model_and_ensemble_by_name():
    m = hcnn.build_model("lspa", n_obs_vars=3, n_hid_vars=5, sparsity_ratio=0.4)
    assert isinstance(m, hcnn.LSpa_Model) and m.sparsity_ratio == 0.4
    ens = hcnn.build_ensemble("ptf", n_ensemble=2, n_obs_vars=3, n_hid_vars=5, seed=0)
    assert isinstance(ens, hcnn.PTFHCNNEnsemble) and ens.n_ensemble == 2


def test_unknown_name_lists_available():
    with pytest.raises(KeyError, match="vanilla"):
        hcnn.get_architecture("does-not-exist")


def test_conflicting_registration_is_rejected():
    spec = hcnn.get_architecture("vanilla")
    assert registry.register_architecture(spec) is spec  # idempotent
    clash = ArchitectureSpec(name="vanilla", model_cls=hcnn.PTF_Model, cell_cls=spec.cell_cls,
                             summary="x")
    with pytest.raises(ValueError, match="already registered"):
        registry.register_architecture(clash)


def test_spec_validates_its_classes():
    with pytest.raises(TypeError):
        ArchitectureSpec(name="bad", model_cls=torch.nn.Linear, cell_cls=torch.nn.Linear)
    with pytest.raises(ValueError):
        ArchitectureSpec(name="Bad Name", model_cls=hcnn.Vanilla_Model,
                         cell_cls=hcnn.get_architecture("vanilla").cell_cls)


def test_legacy_import_paths_resolve_to_the_same_classes():
    from hcnn.core.cells import VanillaHCNNCell, LSpaHCNNCell
    from hcnn.core.models import PTF_Model, RNNModel
    from hcnn.core.models.hcnn_models import LForm_Model
    from hcnn.core.ensembles import VanillaHCNNEnsemble, LSTMEnsemble
    from hcnn.core.ensembles.hcnn_ensembles import LSpaHCNNEnsemble, _HCNNEnsembleBase
    from hcnn.architectures import vanilla, lspa, lform

    assert VanillaHCNNCell is vanilla.VanillaHCNNCell
    assert LSpaHCNNCell is lspa.LSpaHCNNCell
    assert PTF_Model is hcnn.PTF_Model and LForm_Model is lform.LForm_Model
    assert VanillaHCNNEnsemble is hcnn.VanillaHCNNEnsemble
    assert LSpaHCNNEnsemble is hcnn.LSpaHCNNEnsemble
    assert _HCNNEnsembleBase is hcnn.HCNNEnsemble
    assert RNNModel is hcnn.RNNModel and LSTMEnsemble is hcnn.LSTMEnsemble
