"""
DEPRECATED shim. The classic RNN/LSTM baseline ensembles moved to
``hcnn.baselines.ensembles``. This re-export is kept so existing imports
(``from hcnn.core.ensembles.classic_ensembles import RNNEnsemble, LSTMEnsemble``)
keep working; prefer ``from hcnn.baselines import RNNEnsemble, LSTMEnsemble``.
"""

from ...baselines.ensembles import RNNEnsemble, LSTMEnsemble  # noqa: F401

__all__ = ["RNNEnsemble", "LSTMEnsemble"]
