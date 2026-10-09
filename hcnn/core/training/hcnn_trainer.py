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
        # HCNN reconstructs the observed window under teacher forcing; expectations[:, t]
        # = C s_t is the one-step-ahead prediction of batch[:, t].
        return self.loss_fn(self.model(data_window=batch).expectations, batch)

    def _forecast(self, calibration_window: torch.Tensor, horizon: int) -> torch.Tensor:
        # Autonomous rollout: forecast `horizon` steps after one window (T, n_obs) or a
        # batch of windows (n_windows, T, n_obs).
        single = calibration_window.dim() == 2
        cal = calibration_window.unsqueeze(0) if single else calibration_window
        out = self.model(data_window=cal, forecast_horizon=horizon)
        return out.forecasts.squeeze(0) if single else out.forecasts
