"""
DEPRECATED shim. The classic RNN/LSTM baseline models moved to
``hcnn.baselines.models``. This re-export is kept so existing imports
(``from hcnn.core.models.classic_models import RNNModel, LSTMModel``) keep working;
prefer ``from hcnn.baselines import RNNModel, LSTMModel`` going forward.
"""

from ...baselines.models import RNNModel, LSTMModel  # noqa: F401

__all__ = ["RNNModel", "LSTMModel"]
