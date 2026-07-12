"""
Fully Unfolded Training Mode for HCNN Models.

This module implements the fully unfolded training mode where the entire training
dataset is used as one long sequence. The HCNN processes the complete training
sequence at once, providing maximum context and long-term dependency information.

Key Features:
- Sequence length equals the entire training set size
- Single forward pass per epoch with the complete sequence
- Maximum context for capturing long-term dependencies
- Entire training sequence serves as calibration window
- Memory-intensive but provides rich temporal context

Author: HCNN Framework
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List, Literal
import numpy as np
from tqdm import tqdm
import time
import warnings
import os
import json

from ..core.base import BaseHCNNModel


class FullyUnfoldedTrainer:
    """
    Unified Fully Unfolded Training Mode for HCNN Models.

    This class combines the functionality of both the original FullyUnfoldedTrainer
    and FullyUnfoldedHCNNTrainer, providing train_only and train_and_validate methods
    with full gradient clipping and optimization logic.

    In this mode, the entire training dataset is processed as a single long sequence
    during each epoch. This provides maximum context for the model but requires
    significant computational resources.

    Parameters
    ----------
    model : nn.Module
        HCNN model to train (Vanilla, PTF, LForm, or LSpa)
    loss_fn : Literal["mse", "logcosh"], default="mse"
        Loss function type
    optimizer_type : Literal["adam", "sgd"], default="adam"
        Optimizer type
    learning_rate : float, default=1e-4
        Learning rate for optimizer
    device : torch.device
        Device for computation (CPU, CUDA, MPS)
    gradient_clip_value : Optional[float], default=None
        Maximum gradient norm for clipping
    accumulation_steps : int, default=1
        Number of steps for gradient accumulation (for memory management)
    save_dir : str, default="./checkpoints"
        Directory to save checkpoints and loss files
    """

    def __init__(
        self,
        model: BaseHCNNModel,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        optimizer_type: Literal["adam", "sgd"] = "adam",
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
        gradient_clip_value: Optional[float] = None,
        accumulation_steps: int = 1,
        save_dir: str = "./checkpoints"
    ):
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: BaseHCNNModel = model.to(device)
        self.device = device
        self.gradient_clip_value = gradient_clip_value
        self.accumulation_steps = accumulation_steps
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup loss function
        if loss_fn == "mse":
            self.loss_function = nn.MSELoss()
        elif loss_fn == "logcosh":
            self.loss_function = self._logcosh_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Setup optimizer
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Training history
        self.training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'epoch_times': [],
            'gradient_norms': [],
            'memory_usage': []
        }

    def _logcosh_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute log-cosh loss (more robust to outliers than MSE)."""
        diff = predictions - targets
        return torch.mean(torch.log(torch.cosh(diff)))
    
    def _train_epoch_core(
        self,
        training_data: torch.Tensor,
        externals: Optional[torch.Tensor] = None,
        epoch: int = 0,
        verbose: bool = True
    ) -> Tuple[float, float, float, float]:
        """
        Core training logic for one epoch with full optimization features.

        Returns
        -------
        Tuple[float, float, float, float]
            (avg_loss, epoch_time, memory_usage, gradient_norm)
        """
        start_time = time.time()

        # Validate input
        if training_data.dim() != 3:
            raise ValueError(f"Training data must be 3D [batch_size, seq_len, n_obs_vars], got {training_data.shape}")

        if training_data.size(0) != 1:
            warnings.warn(f"Fully unfolded mode expects batch_size=1, got {training_data.size(0)}. "
                         f"Consider using abridged mode for batch processing.")

        # Move data to device
        training_data = training_data.to(self.device)
        if externals is not None:
            externals = externals.to(self.device)

        # Initialize metrics
        total_loss = 0.0
        num_accumulation_steps = 0

        # Process the entire sequence (potentially with gradient accumulation)
        seq_len = training_data.size(1)
        chunk_size = max(1, seq_len // self.accumulation_steps)

        self.optimizer.zero_grad()

        for step in range(self.accumulation_steps):
            # Define chunk boundaries
            start_idx = step * chunk_size
            end_idx = min((step + 1) * chunk_size, seq_len)

            if start_idx >= seq_len:
                break

            # Extract chunk (maintaining the sequence structure)
            if step == 0:
                # First chunk: process from beginning
                chunk_data = training_data[:, :end_idx, :]
                chunk_externals = externals[:, :end_idx, :] if externals is not None else None
            else:
                # Subsequent chunks: include some overlap for continuity
                overlap = min(10, start_idx)  # Small overlap for state continuity
                chunk_start = max(0, start_idx - overlap)
                chunk_data = training_data[:, chunk_start:end_idx, :]
                chunk_externals = externals[:, chunk_start:end_idx, :] if externals is not None else None

            # Forward pass
            output = self.model(chunk_data, externals=chunk_externals)

            # Compute loss (only on the non-overlapping part for subsequent chunks)
            if step == 0:
                loss_data = chunk_data
                loss_expectations = output.expectations
            else:
                # Skip overlap region for loss computation
                skip_overlap = min(10, start_idx) if start_idx > 0 else 0
                loss_data = chunk_data[:, skip_overlap:, :]
                loss_expectations = output.expectations[:, skip_overlap:, :]

            loss = self.loss_function(loss_expectations, loss_data)

            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()
            num_accumulation_steps += 1

        # Gradient clipping
        gradient_norm = 0.0
        if self.gradient_clip_value is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_value
            )
            gradient_norm = grad_norm.item()
            self.training_history['gradient_norms'].append(gradient_norm)

        # Optimizer step
        self.optimizer.step()

        # Calculate metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_accumulation_steps)

        # Memory usage (if CUDA)
        memory_usage = 0.0
        if torch.cuda.is_available() and self.device.type == 'cuda':
            memory_usage = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            torch.cuda.reset_peak_memory_stats(self.device)

        return avg_loss, epoch_time, memory_usage, gradient_norm

    def train_only(
        self,
        training_data: torch.Tensor,
        num_epochs: int = 10,
        externals: Optional[torch.Tensor] = None,
        verbose: bool = True
    ):
        """
        Train the model using only training data (no validation) in fully unfolded mode.
        Saves best model based on training loss and tracks all metrics.

        Args:
            training_data: Complete training sequence [1, seq_len, n_obs_vars]
            num_epochs: Number of training epochs
            externals: External variables [1, seq_len, n_ext_vars]
            verbose: Whether to print training progress
        """
        best_loss = float('inf')
        best_epoch = -1
        epoch_loss_summary = []

        for epoch in tqdm(range(num_epochs), desc="Training Only (Fully Unfolded)"):
            self.model.train()

            # Core training logic with all optimization features
            avg_loss, epoch_time, memory_usage, gradient_norm = self._train_epoch_core(
                training_data=training_data,
                externals=externals,
                epoch=epoch + 1,
                verbose=False  # We'll handle printing here
            )

            # Store history
            self.training_history['epoch_losses'].append(avg_loss)
            self.training_history['val_losses'].append(None)  # No validation
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['memory_usage'].append(memory_usage)

            # Save best model based on training loss
            if avg_loss < best_loss:
                best_epoch = epoch + 1
                self.model.save_checkpoint(
                    epoch=best_epoch,
                    loss=avg_loss,
                    optimizer=self.optimizer,
                    checkpoint_dir=self.save_dir,
                    add_stuffs="_fully_unfolded_single",
                    cleanup=True
                )
                best_loss = avg_loss

                if verbose:
                    print(f"✅ New best train loss: {avg_loss:.6f}")
            elif verbose:
                print(f"ℹ️ Current train loss {avg_loss:.6f} not better than best {best_loss:.6f}")

            # Store epoch summary
            epoch_summary = {
                'train_loss': avg_loss,
                'val_loss': None,
                'epoch_time': epoch_time,
                'memory_usage': memory_usage,
                'gradient_norm': gradient_norm,
                'sequence_length': training_data.size(1)
            }
            epoch_loss_summary.append({f'epoch_{epoch+1}': epoch_summary})

            if verbose:
                seq_len = training_data.size(1)
                print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s | "
                      f"Seq Len: {seq_len} | Memory: {memory_usage:.2f}GB")

        # Save epoch losses to JSON
        self._save_losses_to_json(epoch_loss_summary)

    def train_and_validate(
        self,
        training_data: torch.Tensor,
        num_epochs: int,
        calibration_window: torch.Tensor,
        val_data: torch.Tensor,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None,
        verbose: bool = True
    ):
        """
        Train the model with validation after each epoch in fully unfolded mode.
        Saves best model based on validation loss and tracks all metrics.

        Args:
            training_data: Complete training sequence [1, seq_len, n_obs_vars]
            num_epochs: Number of training epochs
            calibration_window: Context window for forecasting [1, cal_len, n_obs_vars]
            val_data: Validation target data [forecast_horizon, n_obs_vars]
            externals: External variables for training [1, seq_len, n_ext_vars]
            future_externals: Future external variables [1, forecast_horizon, n_ext_vars]
            verbose: Whether to print training progress
        """
        best_val_loss = float('inf')
        best_epoch = -1
        epoch_loss_summary = []

        # Move validation data to device
        calibration_window = calibration_window.to(self.device)
        val_data = val_data.to(self.device)
        if future_externals is not None:
            future_externals = future_externals.to(self.device)

        for epoch in tqdm(range(num_epochs), desc="Training and Validating (Fully Unfolded)"):
            # Training step with full optimization logic
            self.model.train()
            avg_train_loss, epoch_time, memory_usage, gradient_norm = self._train_epoch_core(
                training_data=training_data,
                externals=externals,
                epoch=epoch + 1,
                verbose=False  # We'll handle printing here
            )

            # Validation step
            self.model.eval()
            with torch.no_grad():
                results = self.model.forward(
                    data_window=calibration_window,
                    forecast_horizon=val_data.shape[0],
                    externals=future_externals
                )
                forecast = results.forecasts.squeeze(0) if results.forecasts.dim() > 2 else results.forecasts
                val_loss = self.loss_function(forecast, val_data).item()

            # Store history
            self.training_history['epoch_losses'].append(avg_train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['epoch_times'].append(epoch_time)
            self.training_history['memory_usage'].append(memory_usage)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_epoch = epoch + 1
                self.model.save_checkpoint(
                    epoch=best_epoch,
                    loss=val_loss,
                    optimizer=self.optimizer,
                    checkpoint_dir=self.save_dir,
                    add_stuffs="_fully_unfolded_validated",
                    cleanup=True
                )
                best_val_loss = val_loss

                if verbose:
                    print(f"✅ New best validation loss: {val_loss:.6f}")

            # Store epoch summary
            epoch_summary = {
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'memory_usage': memory_usage,
                'gradient_norm': gradient_norm,
                'sequence_length': training_data.size(1)
            }
            epoch_loss_summary.append({f'epoch_{epoch+1}': epoch_summary})

            if verbose:
                seq_len = training_data.size(1)
                print(f"Epoch {epoch+1:4d} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Time: {epoch_time:.2f}s | Seq Len: {seq_len} | Memory: {memory_usage:.2f}GB")

        # Save epoch losses to JSON
        self._save_losses_to_json(epoch_loss_summary)

    def _save_losses_to_json(self, epoch_losses):
        """Save epoch losses to JSON file."""
        loss_file = os.path.join(self.save_dir, f"{self.model.name}_fully_unfolded_losses.json")
        with open(loss_file, 'w') as f:
            json.dump(epoch_losses, f, indent=2)
        print(f"📊 Loss history saved to: {loss_file}")

    def forecast(
        self,
        calibration_data: torch.Tensor,
        forecast_horizon: int,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, int, None]]:
        """
        Generate forecasts using the trained model.
        
        In fully unfolded mode, the entire training sequence serves as the
        calibration window for teacher forcing.
        
        Parameters
        ----------
        calibration_data : torch.Tensor
            Calibration sequence (typically the entire training data) [1, seq_len, n_obs_vars]
        forecast_horizon : int
            Number of future time steps to predict
        externals : Optional[torch.Tensor]
            External variables for calibration [1, seq_len, n_ext_vars]
        future_externals : Optional[torch.Tensor]
            Future external variables [1, forecast_horizon, n_ext_vars]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing forecasts and related information
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            calibration_data = calibration_data.to(self.device)
            if externals is not None:
                externals = externals.to(self.device)
            if future_externals is not None:
                future_externals = future_externals.to(self.device)
            
            # Forward pass with forecasting
            output = self.model(
                calibration_data,
                forecast_horizon=forecast_horizon,
                externals=externals,
                future_externals=future_externals
            )
            
            return {
                'calibration_expectations': output.expectations.cpu(),
                'calibration_states': output.states.cpu(),
                'forecasts': output.forecasts.cpu() if output.forecasts is not None else None,
                'future_states': output.future_states.cpu() if output.future_states is not None else None,
                'calibration_length': calibration_data.size(1),
                'forecast_horizon': forecast_horizon
            }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training history."""
        if not self.training_history['epoch_losses']:
            return {'message': 'No training history available'}

        return {
            'total_epochs': len(self.training_history['epoch_losses']),
            'final_loss': self.training_history['epoch_losses'][-1],
            'best_loss': min(self.training_history['epoch_losses']),
            'total_training_time': sum(self.training_history['epoch_times']),
            'avg_epoch_time': np.mean(self.training_history['epoch_times']),
            'max_memory_usage': max(self.training_history['memory_usage']) if self.training_history['memory_usage'] else 0.0,
            'avg_gradient_norm': np.mean(self.training_history['gradient_norms']) if self.training_history['gradient_norms'] else 0.0
        }


def train_fully_unfolded(
    model: BaseHCNNModel,
    training_data: torch.Tensor,
    num_epochs: int,
    loss_fn: Literal["mse", "logcosh"] = "mse",
    optimizer_type: Literal["adam", "sgd"] = "adam",
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
    externals: Optional[torch.Tensor] = None,
    gradient_clip_value: Optional[float] = None,
    accumulation_steps: int = 1,
    verbose: bool = True,
    save_dir: str = "./checkpoints"
) -> Tuple[BaseHCNNModel, Dict[str, Any]]:
    """
    Convenience function for fully unfolded training using train_only method.

    Parameters
    ----------
    model : BaseHCNNModel
        HCNN model to train
    training_data : torch.Tensor
        Complete training sequence [1, seq_len, n_obs_vars]
    num_epochs : int
        Number of training epochs
    loss_fn : Literal["mse", "logcosh"], default="mse"
        Loss function type
    optimizer_type : Literal["adam", "sgd"], default="adam"
        Optimizer type
    learning_rate : float, default=1e-4
        Learning rate for optimizer
    device : Optional[torch.device]
        Device for computation (auto-detected if None)
    externals : Optional[torch.Tensor]
        External variables [1, seq_len, n_ext_vars]
    gradient_clip_value : Optional[float]
        Maximum gradient norm for clipping
    accumulation_steps : int
        Number of steps for gradient accumulation
    verbose : bool
        Whether to print progress
    save_dir : str
        Directory to save checkpoints

    Returns
    -------
    Tuple[BaseHCNNModel, Dict[str, Any]]
        Trained model and training history
    """
    trainer = FullyUnfoldedTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        device=device,
        gradient_clip_value=gradient_clip_value,
        accumulation_steps=accumulation_steps,
        save_dir=save_dir
    )

    if verbose:
        print(f"Starting Fully Unfolded Training")
        print(f"Sequence Length: {training_data.size(1)}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {trainer.device}")
        print("=" * 60)

    # Use the train_only method
    trainer.train_only(
        training_data=training_data,
        num_epochs=num_epochs,
        externals=externals,
        verbose=verbose
    )

    training_summary = trainer.get_training_summary()

    if verbose:
        print("=" * 60)
        print("Training Complete!")
        print(f"Best Loss: {training_summary['best_loss']:.6f}")
        print(f"Total Time: {training_summary['total_training_time']:.2f}s")

    return model, training_summary
