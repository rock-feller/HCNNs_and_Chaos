"""
Baseline sequence models for comparison with HCNNs.

These classic RNN / LSTM models, ensembles, and trainers are **not part of the
HCNN method** - they live here, in a clearly separated namespace, purely so that
comparative experiments run against the same data pipeline and training
infrastructure without a second package.

    from hcnn.baselines import LSTMModel, SequenceModelTrainer
"""

from .models import RNNModel, LSTMModel
from .ensembles import RNNEnsemble, LSTMEnsemble
from .trainers import SequenceModelTrainer, SequenceEnsembleTrainer

__all__ = [
    "RNNModel",
    "LSTMModel",
    "RNNEnsemble",
    "LSTMEnsemble",
    "SequenceModelTrainer",
    "SequenceEnsembleTrainer",
]
