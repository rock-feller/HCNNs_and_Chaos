"""
HCNNs & Chaos: Historical Consistent Neural Networks for Chaotic Systems Modeling

A research package for modeling chaotic dynamical systems using Historical
Consistent Neural Networks (HCNNs) and related variants.

This package provides:
- Core HCNN implementations (Vanilla, PTF, LForm, LSpa)
- Training frameworks for single models and ensembles
- Chaotic systems data generation and preprocessing
- Utilities for research and experimentation

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

# Single-model classes
from .core.models.hcnn_models import (
    Vanilla_Model,
    PTF_Model,
    LForm_Model,
    LSpa_Model,
)
from .core.models.classic_models import RNNModel, LSTMModel

# Clean public aliases
VanillaHCNN = Vanilla_Model
PTFHCNN = PTF_Model
LFormHCNN = LForm_Model
LSpaHCNN = LSpa_Model

# Ensemble classes
from .core.ensembles.hcnn_ensembles import (
    VanillaHCNNEnsemble,
    PTFHCNNEnsemble,
    LFormHCNNEnsemble,
    LSpaHCNNEnsemble,
)
from .core.ensembles.classic_ensembles import RNNEnsemble, LSTMEnsemble

# Training
from .core.training import HCNNTrainer

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
    "VanillaHCNNEnsemble",
    "PTFHCNNEnsemble",
    "LFormHCNNEnsemble",
    "LSpaHCNNEnsemble",
    "RNNEnsemble",
    "LSTMEnsemble",
    # training
    "HCNNTrainer",
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
