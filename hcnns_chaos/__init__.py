"""
HCNNs & Chaos: Historical Consistent Neural Networks for Chaotic Systems Modeling

A comprehensive research package for modeling chaotic dynamical systems using 
Historical Consistent Neural Networks (HCNNs) and related variants.

This package provides:
- Core HCNN implementations (Vanilla, PTF, LForm, LSpa)
- Training frameworks for single models and ensembles
- Chaotic systems data generation and preprocessing
- Comprehensive utilities for research and experimentation

Author: Rockefeller
Email: rockefeller@moraeglobal.com
"""

__version__ = "0.1.0"
__author__ = "Rockefeller"
__email__ = "rockefeller@moraeglobal.com"

# Core imports
from . import core
from . import training
from . import data
from . import utils
from . import config

# Main model classes
try:
    from .core.models.hcnn_models import (
        VanillaHCNN,
        PTFHCNN,
        LFormHCNN,
        LSpaHCNN,
    )
except ImportError:
    VanillaHCNN = None
    PTFHCNN = None
    LFormHCNN = None
    LSpaHCNN = None

try:
    from .core.models.classic_models import (
        RNNModel,
        LSTMModel,
    )
except ImportError:
    RNNModel = None
    LSTMModel = None

# Ensemble classes
try:
    from .core.ensembles.hcnn_ensembles import (
        VanillaHCNNEnsemble,
        PTFHCNNEnsemble,
        LFormHCNNEnsemble,
        LSpaHCNNEnsemble,
    )
except ImportError:
    VanillaHCNNEnsemble = None
    PTFHCNNEnsemble = None
    LFormHCNNEnsemble = None
    LSpaHCNNEnsemble = None

try:
    from .core.ensembles.classic_ensembles import (
        RNNEnsemble,
        LSTMEnsemble,
    )
except ImportError:
    RNNEnsemble = None
    LSTMEnsemble = None

# Utility functions
from .utils.device import get_device, set_device
from .utils.checkpoints import save_checkpoint, load_checkpoint

# Configuration
from .config.base import Config, load_config, save_config

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Core modules
    "core",
    "training", 
    "data",
    "utils",
    "config",
    
    # HCNN Models
    "VanillaHCNN",
    "PTFHCNN",
    "LFormHCNN",
    "LSpaHCNN",

    # Classic Models
    "RNNModel",
    "LSTMModel",

    # Ensembles
    "VanillaHCNNEnsemble",
    "PTFHCNNEnsemble",
    "LFormHCNNEnsemble",
    "LSpaHCNNEnsemble",
    "RNNEnsemble",
    "LSTMEnsemble",

    # Config
    "Config",
    "load_config",
    "save_config",

    # Utils
    "get_device",
    "set_device",
    "save_checkpoint",
    "load_checkpoint",
]
