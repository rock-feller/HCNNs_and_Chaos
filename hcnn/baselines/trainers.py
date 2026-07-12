"""
Baseline trainers for the RNN/LSTM comparison models.

These reuse the shared :class:`hcnn.core.training.base_trainer.BaseTrainer` loop
and differ from the HCNN trainer only in the two hooks:

- ``_batch_loss``: standard next-step prediction - predict ``seq[1:]`` from
  ``seq[:-1]``;
- ``_forecast``: the model's own autoregressive ``forecast`` method.
"""

import torch

from ..core.training.base_trainer import BaseTrainer, EnsembleTrainer


class SequenceModelTrainer(BaseTrainer):
    """Single RNN/LSTM baseline trainer (next-step prediction)."""

    def _batch_loss(self, batch: torch.Tensor) -> torch.Tensor:
        inp, target = batch[:, :-1, :], batch[:, 1:, :]
        return self.loss_fn(self.model(inp).outputs, target)

    def _forecast(self, calibration_window: torch.Tensor, horizon: int) -> torch.Tensor:
        return self.model.forecast(calibration_window.unsqueeze(0), horizon).squeeze(0)


class SequenceEnsembleTrainer(EnsembleTrainer):
    """Ensemble trainer for RNN/LSTM baselines (each member via SequenceModelTrainer)."""

    def __init__(self, ensemble, save_dir: str = "./checkpoints", **trainer_kwargs):
        super().__init__(ensemble, SequenceModelTrainer, save_dir=save_dir, **trainer_kwargs)
