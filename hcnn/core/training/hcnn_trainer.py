"""
HCNN single-model trainer.

Thin subclass of :class:`hcnn.core.training.base_trainer.BaseTrainer`: the shared
epoch loop, best-model selection, checkpointing and logging live in the base; here
we only say how an HCNN computes a training-batch loss (reconstruct the observed
window) and how it forecasts (autonomous rollout via ``forecast_horizon``).
"""

import torch

from .base_trainer import BaseTrainer


class HCNNTrainer(BaseTrainer):
    """Trainer for a single HCNN model (Vanilla / PTF / LForm / LSpa)."""

    def _batch_loss(self, batch: torch.Tensor) -> torch.Tensor:
        # HCNN reconstructs the observed window under teacher forcing.
        return self.loss_fn(self.model(data_window=batch).expectations, batch)

    def _forecast(self, calibration_window: torch.Tensor, horizon: int) -> torch.Tensor:
        # Autonomous rollout: forecast `horizon` steps from the calibration window.
        out = self.model(data_window=calibration_window.unsqueeze(0), forecast_horizon=horizon)
        return out.forecasts.squeeze(0)
