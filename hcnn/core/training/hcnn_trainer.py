"""
HCNN Trainer.

Single-model trainer for the HCNN variants. Provides ``train_only`` (train on the
training loader, checkpoint the best epoch by training loss) and
``train_and_validate`` (checkpoint the best epoch by a forecast loss on a held-out
calibration+forecast window).

Design notes / fixes vs the original:
- Epoch-level best-model selection (not per-batch), initialized once so it tracks
  the genuine best across the whole run.
- ``per_batch`` and ``per_epoch`` backprop are supported by BOTH train methods
  (the original ``train_and_validate`` silently did nothing in per_epoch mode).
- Validation forecast is computed once per epoch (not once per batch).
- Numerically stable log-cosh loss.
- PTF models have their dropout schedule advanced each epoch.
- Model is always returned to train() mode after validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from typing import Optional, Literal
import numpy as np


class HCNNTrainer:
    """Trainer for a single HCNN model."""

    def __init__(
        self,
        model,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        backprop_mode: Literal["per_batch", "per_epoch"] = "per_batch",
        optimizer_type: Literal["adam", "sgd"] = "adam",
        learning_rate: float = 1e-4,
        grad_clip: Optional[float] = None,
        save_dir: str = "./checkpoints",
    ):
        """
        Args:
            model: HCNN model to train.
            loss_fn: "mse" or "logcosh".
            backprop_mode: "per_batch" (step every batch) or "per_epoch"
                (accumulate the mean batch loss and step once per epoch).
            optimizer_type: "adam" or "sgd".
            learning_rate: Optimizer learning rate.
            grad_clip: If set, clip gradient norm to this value before stepping.
            save_dir: Directory for checkpoints and the loss log.
        """
        self.model = model
        self.backprop_mode = backprop_mode
        self.grad_clip = grad_clip
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "logcosh":
            self.loss_fn = self._logcosh_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logcosh_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable log-cosh loss.

        ``log(cosh(x)) = |x| + softplus(-2|x|) - log(2)`` avoids the overflow of
        ``log(cosh(x))`` computed directly (cosh overflows to inf for |x| ~> 88).
        """
        diff = predictions - targets
        ax = diff.abs()
        return torch.mean(ax + torch.nn.functional.softplus(-2.0 * ax) - np.log(2.0))

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _step(self, loss: torch.Tensor):
        """One optimizer step with optional grad clipping."""
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def _run_train_epoch(self, data_loader, epoch: int) -> float:
        """Run one training epoch; returns the average training loss."""
        # Advance PTF dropout schedule if present (1-based epoch).
        if hasattr(self.model, "update_dropout_epoch"):
            self.model.update_dropout_epoch(epoch + 1)

        self.model.train()
        device = self._device
        total_loss, batch_count = 0.0, 0

        if self.backprop_mode == "per_batch":
            for batch in data_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                results = self.model(data_window=batch)
                loss = self.loss_fn(results.expectations, batch)
                self._step(loss)
                total_loss += loss.item()
                batch_count += 1

        elif self.backprop_mode == "per_epoch":
            self.optimizer.zero_grad()
            accumulated = 0.0
            for batch in data_loader:
                batch = batch.to(device)
                results = self.model(data_window=batch)
                loss = self.loss_fn(results.expectations, batch)
                accumulated = accumulated + loss
                total_loss += loss.item()
                batch_count += 1
            (accumulated / max(batch_count, 1)).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        else:
            raise ValueError(f"Unknown backprop_mode: {self.backprop_mode}")

        return total_loss / max(batch_count, 1)

    @torch.no_grad()
    def _validate(self, calibration_window: torch.Tensor, val_data: torch.Tensor) -> float:
        """Forecast ``len(val_data)`` steps from the calibration window; return loss."""
        was_training = self.model.training
        self.model.eval()
        device = self._device
        cal = calibration_window.to(device)
        val = val_data.to(device)
        results = self.model(data_window=cal.unsqueeze(0), forecast_horizon=val.shape[0])
        forecast = results.forecasts.squeeze(0)
        val_loss = self.loss_fn(forecast, val).item()
        if was_training:
            self.model.train()
        return val_loss

    def _save_best(self, epoch: int, loss: float, tag: str):
        self.model.save_checkpoint(
            epoch=epoch,
            loss=loss,
            optimizer=self.optimizer,
            checkpoint_dir=self.save_dir,
            add_stuffs=tag,
            cleanup=True,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train_only(self, data_loader, num_epochs: int = 10, verbose: bool = True):
        """Train on ``data_loader``; checkpoint the best epoch by avg training loss."""
        best_loss = float("inf")
        epoch_loss_summary = []

        for epoch in tqdm(range(num_epochs), desc="Training Only"):
            avg_train_loss = self._run_train_epoch(data_loader, epoch)

            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                self._save_best(epoch + 1, avg_train_loss, "_single")
                if verbose:
                    print(f"✅ Epoch {epoch+1}: new best train loss {avg_train_loss:.6f}")

            epoch_loss_summary.append({f"epoch_{epoch+1}": {"avg_train_loss": avg_train_loss}})
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.6f}")

        self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)
        return best_loss

    def train_and_validate(
        self,
        data_loader,
        num_epochs: int,
        calibration_window: torch.Tensor,
        val_data: torch.Tensor,
        verbose: bool = True,
    ):
        """
        Train and, once per epoch, forecast from ``calibration_window`` and score
        against ``val_data``; checkpoint the best epoch by validation loss.

        NOTE: ``val_data`` is used for model selection, so it must NOT be reused as
        the final test set (that would be selection-on-test leakage).
        """
        best_val_loss = float("inf")
        epoch_loss_summary = []

        for epoch in tqdm(range(num_epochs), desc="Training and Validating"):
            avg_train_loss = self._run_train_epoch(data_loader, epoch)
            avg_val_loss = self._validate(calibration_window, val_data)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_best(epoch + 1, avg_val_loss, "_validated")
                if verbose:
                    print(f"✅ Epoch {epoch+1}: new best val loss {avg_val_loss:.6f}")

            epoch_loss_summary.append({
                f"epoch_{epoch+1}": {
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                }
            })
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")

        self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)
        return best_val_loss

    def save_epochs_losses_to_json(self, epoch_losses):
        """Save the per-epoch loss log to JSON."""
        loss_file = os.path.join(self.save_dir, f"{self.model.name}_losses.json")
        with open(loss_file, "w") as f:
            json.dump(epoch_losses, f, indent=2)
