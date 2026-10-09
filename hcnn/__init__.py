"""
HCNNs & Chaos: Historical Consistent Neural Networks for Chaotic Systems Modeling

A research package for modeling chaotic dynamical systems using Historical
Consistent Neural Networks (HCNNs) and related variants.

This package provides:
- ``hcnn.core``          - the shared framework (base classes + the one rollout,
                           registry, layers, generic ensemble, trainers)
- ``hcnn.architectures`` - one self-contained package per HCNN variant
                           (Vanilla, PTF, LForm, LSpa, ...)
- ``hcnn.baselines``     - RNN/LSTM comparison models
- ``hcnn.utils``         - chaotic-system data generation and preprocessing

Any registered architecture can be built by name::

    import hcnn
    hcnn.list_architectures()            # ['lform', 'lspa', 'ptf', 'vanilla']
    model = hcnn.build_model("lform", n_obs_vars=3, n_hid_vars=10)

Public model classes are exposed under two equivalent names:
  Vanilla_Model  == VanillaHCNN
  PTF_Model      == PTFHCNN
  LForm_Model    == LFormHCNN
  LSpa_Model     == LSpaHCNN
The ``*_Model`` names are the historical API (used by the notebooks); the
``*HCNN`` names are the preferred, cleaner aliases going forward.

Author: Rockefeller
"""

__version__ = "0.2.0"
__author__ = "Rockefeller"

# Core subpackages
from . import core
from . import utils
from . import config
from . import architectures  # discovers + registers every architecture package
from . import baselines  # RNN/LSTM comparison models (not part of the HCNN method)

# Architecture registry
from .core.registry import (
    ArchitectureSpec,
    register_architecture,
    get_architecture,
    list_architectures,
    describe_architectures,
    build_model,
    build_ensemble,
)

# Single-model classes
from .architectures import (
    Vanilla_Model,
    PTF_Model,
    LForm_Model,
    LSpa_Model,
)

# Clean public aliases
VanillaHCNN = Vanilla_Model
PTFHCNN = PTF_Model
LFormHCNN = LForm_Model
LSpaHCNN = LSpa_Model

# Baseline models / ensembles (see hcnn.baselines)
from .baselines import RNNModel, LSTMModel, RNNEnsemble, LSTMEnsemble

# Ensemble classes
from .core.ensembles import HCNNEnsemble
from .architectures import (
    VanillaHCNNEnsemble,
    PTFHCNNEnsemble,
    LFormHCNNEnsemble,
    LSpaHCNNEnsemble,
)

# Training
from .core.training import HCNNTrainer, BaseTrainer, EnsembleTrainer

# Utility functions
from .utils.device import get_device, set_device, get_device_info
from .utils.checkpoints import save_checkpoint, load_checkpoint

# Configuration
from .config.base import Config, load_config, save_config

__all__ = [
    "__version__",
    "__author__",
    # subpackages
    "core",
    "utils",
    "config",
    "architectures",
    "baselines",
    # registry
    "ArchitectureSpec",
    "register_architecture",
    "get_architecture",
    "list_architectures",
    "describe_architectures",
    "build_model",
    "build_ensemble",
    # HCNN models (historical names)
    "Vanilla_Model",
    "PTF_Model",
    "LForm_Model",
    "LSpa_Model",
    # HCNN models (preferred aliases)
    "VanillaHCNN",
    "PTFHCNN",
    "LFormHCNN",
    "LSpaHCNN",
    # classic models
    "RNNModel",
    "LSTMModel",
    # ensembles
    "HCNNEnsemble",
    "VanillaHCNNEnsemble",
    "PTFHCNNEnsemble",
    "LFormHCNNEnsemble",
    "LSpaHCNNEnsemble",
    "RNNEnsemble",
    "LSTMEnsemble",
    # training
    "HCNNTrainer",
    "BaseTrainer",
    "EnsembleTrainer",
    # config
    "Config",
    "load_config",
    "save_config",
    # utils
    "get_device",
    "set_device",
    "get_device_info",
    "save_checkpoint",
    "load_checkpoint",
]
