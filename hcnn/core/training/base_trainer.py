"""
Shared training infrastructure.

``BaseTrainer`` holds everything that is identical across model families - the
epoch loop, per_batch / per_epoch backprop, epoch-level best-model selection,
checkpointing, JSON loss logging, device handling, gradient clipping, and a
numerically stable log-cosh loss. Concrete trainers implement just two hooks:

- ``_batch_loss(batch)`` - forward + loss for one training batch;
- ``_forecast(calibration_window, horizon)`` - an autonomous forecast used for
  validation.

This is what keeps the HCNN trainer and the RNN/LSTM baseline trainer from
duplicating (and independently rotting) the same 150 lines of loop code.

``EnsembleTrainer`` is a thin generic wrapper that trains each ensemble member
independently with its own single-model trainer, so one class serves both HCNN
and baseline ensembles.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Literal


class BaseTrainer:
    """Family-agnostic training loop. Subclass and implement the two hooks."""

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

    # ---- hooks implemented by concrete trainers ---------------------- #
    def _batch_loss(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _forecast(self, calibration_window: torch.Tensor, horizon: int) -> torch.Tensor:
        raise NotImplementedError

    def _on_epoch_start(self, epoch: int):
        """Per-epoch hook. Default advances a PTF dropout schedule if present."""
        if hasattr(self.model, "update_dropout_epoch"):
            self.model.update_dropout_epoch(epoch + 1)

    # ---- shared helpers ---------------------------------------------- #
    @staticmethod
    def _logcosh_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Stable log-cosh: |x| + softplus(-2|x|) - log 2 (no cosh overflow)."""
        diff = predictions - targets
        ax = diff.abs()
        return torch.mean(ax + torch.nn.functional.softplus(-2.0 * ax) - np.log(2.0))

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _step(self, loss: torch.Tensor):
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def _run_train_epoch(self, data_loader, epoch: int) -> float:
        self._on_epoch_start(epoch)
        self.model.train()
        device = self._device
        total_loss, batch_count = 0.0, 0

        if self.backprop_mode == "per_batch":
            for batch in data_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                loss = self._batch_loss(batch)
                self._step(loss)
                total_loss += loss.item()
                batch_count += 1

        elif self.backprop_mode == "per_epoch":
            self.optimizer.zero_grad()
            accumulated = 0.0
            for batch in data_loader:
                batch = batch.to(device)
                loss = self._batch_loss(batch)
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
        was_training = self.model.training
        self.model.eval()
        cal = calibration_window.to(self._device)
        val = val_data.to(self._device)
        forecast = self._forecast(cal, val.shape[0])
        val_loss = self.loss_fn(forecast, val).item()
        if was_training:
            self.model.train()
        return val_loss

    def _save_best(self, epoch: int, loss: float, tag: str):
        # HCNN models carry a rich save_checkpoint; plain nn.Modules (RNN/LSTM
        # baselines) fall back to a standard torch.save.
        if hasattr(self.model, "save_checkpoint"):
            self.model.save_checkpoint(
                epoch=epoch, loss=loss, optimizer=self.optimizer,
                checkpoint_dir=self.save_dir, add_stuffs=tag, cleanup=True,
            )
        else:
            name = getattr(self.model, "name", self.model.__class__.__name__)
            torch.save(
                {"epoch": epoch, "loss": loss,
                 "model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.save_dir, f"{name}{tag}_best.pth"),
            )

    # ---- public API --------------------------------------------------- #
    def train_only(self, data_loader, num_epochs: int = 10, verbose: bool = True):
        """Train on ``data_loader``; checkpoint the best epoch by avg training loss."""
        best_loss = float("inf")
        summary = []
        for epoch in tqdm(range(num_epochs), desc="Training Only"):
            avg = self._run_train_epoch(data_loader, epoch)
            if avg < best_loss:
                best_loss = avg
                self._save_best(epoch + 1, avg, "_single")
                if verbose:
                    print(f"✅ Epoch {epoch+1}: new best train loss {avg:.6f}")
            summary.append({f"epoch_{epoch+1}": {"avg_train_loss": avg}})
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg:.6f}")
        self.save_epochs_losses_to_json(summary)
        return best_loss

    def train_and_validate(self, data_loader, num_epochs, calibration_window,
                           val_data, verbose: bool = True):
        """
        Train and, once per epoch, forecast from ``calibration_window`` and score
        against ``val_data``; checkpoint the best epoch by validation loss.

        NOTE: ``val_data`` drives model selection, so it must NOT double as the
        final test set (that would be selection-on-test leakage).
        """
        best_val = float("inf")
        summary = []
        for epoch in tqdm(range(num_epochs), desc="Training and Validating"):
            avg = self._run_train_epoch(data_loader, epoch)
            val = self._validate(calibration_window, val_data)
            if val < best_val:
                best_val = val
                self._save_best(epoch + 1, val, "_validated")
                if verbose:
                    print(f"✅ Epoch {epoch+1}: new best val loss {val:.6f}")
            summary.append({f"epoch_{epoch+1}": {"avg_train_loss": avg, "avg_val_loss": val}})
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg:.6f}, Val: {val:.6f}")
        self.save_epochs_losses_to_json(summary)
        return best_val

    def save_epochs_losses_to_json(self, epoch_losses):
        name = getattr(self.model, "name", self.model.__class__.__name__)
        with open(os.path.join(self.save_dir, f"{name}_losses.json"), "w") as f:
            json.dump(epoch_losses, f, indent=2)


class EnsembleTrainer:
    """
    Train each ensemble member independently, each with its own single-model
    trainer of type ``single_trainer_cls`` (e.g. ``HCNNTrainer`` or
    ``SequenceModelTrainer``). Members are checkpointed into per-member
    subdirectories to avoid filename collisions.
    """

    def __init__(self, ensemble, single_trainer_cls, save_dir: str = "./checkpoints",
                 **trainer_kwargs):
        self.ensemble = ensemble
        self.trainers = [
            single_trainer_cls(
                ensemble.get_model(i),
                save_dir=os.path.join(save_dir, f"member_{i}"),
                **trainer_kwargs,
            )
            for i in range(ensemble.n_ensemble)
        ]

    def train_only(self, data_loader, num_epochs: int = 10, verbose: bool = False):
        return [t.train_only(data_loader, num_epochs, verbose) for t in self.trainers]

    def train_and_validate(self, data_loader, num_epochs, calibration_window,
                           val_data, verbose: bool = False):
        return [
            t.train_and_validate(data_loader, num_epochs, calibration_window, val_data, verbose)
            for t in self.trainers
        ]
