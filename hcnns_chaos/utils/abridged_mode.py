"""
Abridged Training Mode for HCNN Models.

This module implements the abridged training mode where the training data is split
into smaller chunks of fixed size. This approach is computationally efficient and
allows for batch processing, making it suitable for long sequences and limited
computational resources.

Key Features:
- Fixed chunk size for sequence processing
- Batch processing of multiple chunks
- Optional overlapping between chunks
- Computationally efficient with lower memory requirements
- Last L chronological samples used as calibration window for forecasting

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


class AbridgedTrainer:
    """
    Abridged Training Mode for HCNN Models.
    
    In this mode, the training data is split into smaller chunks of fixed size,
    allowing for efficient batch processing and reduced memory requirements.
    
    Parameters
    ----------
    model : nn.Module
        HCNN model to train (Vanilla, PTF, LForm, or LSpa)
    loss_function : nn.Module
        Loss function for training (e.g., MSELoss, LogCoshLoss)
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    chunk_size : int
        Fixed size for each training chunk
    device : torch.device
        Device for computation (CPU, CUDA, MPS)
    overlap_size : int, default=0
        Number of overlapping time steps between chunks
    shuffle_chunks : bool, default=True
        Whether to shuffle chunks during training
    gradient_clip_value : Optional[float], default=None
        Maximum gradient norm for clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        chunk_size: int,
        device: torch.device,
        overlap_size: int = 0,
        shuffle_chunks: bool = True,
        gradient_clip_value: Optional[float] = None
    ):
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.chunk_size = chunk_size
        self.device = device
        self.overlap_size = overlap_size
        self.shuffle_chunks = shuffle_chunks
        self.gradient_clip_value = gradient_clip_value
        
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {chunk_size}")
        if overlap_size < 0:
            raise ValueError(f"Overlap size must be non-negative, got {overlap_size}")
        if overlap_size >= chunk_size:
            raise ValueError(f"Overlap size ({overlap_size}) must be less than chunk size ({chunk_size})")
        
        # Training history
        self.training_history = {
            'epoch_losses': [],
            'batch_losses': [],
            'epoch_times': [],
            'gradient_norms': [],
            'num_chunks_per_epoch': []
        }
    
    def create_chunks(
        self,
        data: torch.Tensor,
        externals: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Split data into overlapping chunks.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data [batch_size, seq_len, n_obs_vars]
        externals : Optional[torch.Tensor]
            External variables [batch_size, seq_len, n_ext_vars]
            
        Returns
        -------
        Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]
            List of data chunks and optional list of external chunks
        """
        batch_size, seq_len, n_obs_vars = data.shape
        
        if seq_len < self.chunk_size:
            warnings.warn(f"Sequence length ({seq_len}) is smaller than chunk size ({self.chunk_size}). "
                         f"Using the entire sequence as a single chunk.")
            return [data], [externals] if externals is not None else None
        
        # Calculate step size (accounting for overlap)
        step_size = self.chunk_size - self.overlap_size
        
        # Generate chunk start indices
        start_indices = list(range(0, seq_len - self.chunk_size + 1, step_size))
        
        # Create chunks
        data_chunks = []
        external_chunks = [] if externals is not None else None
        
        for start_idx in start_indices:
            end_idx = start_idx + self.chunk_size
            
            # Extract data chunk
            chunk = data[:, start_idx:end_idx, :]
            data_chunks.append(chunk)
            
            # Extract external chunk if available
            if externals is not None:
                ext_chunk = externals[:, start_idx:end_idx, :]
                external_chunks.append(ext_chunk)
        
        return data_chunks, external_chunks
    
    def train_epoch(
        self,
        training_data: torch.Tensor,
        externals: Optional[torch.Tensor] = None,
        epoch: int = 0,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch using abridged mode.
        
        Parameters
        ----------
        training_data : torch.Tensor
            Training data [batch_size, seq_len, n_obs_vars]
        externals : Optional[torch.Tensor]
            External variables [batch_size, seq_len, n_ext_vars]
        epoch : int
            Current epoch number
        verbose : bool
            Whether to print progress information
            
        Returns
        -------
        Dict[str, float]
            Training metrics for this epoch
        """
        self.model.train()
        start_time = time.time()
        
        # Move data to device
        training_data = training_data.to(self.device)
        if externals is not None:
            externals = externals.to(self.device)
        
        # Create chunks
        data_chunks, external_chunks = self.create_chunks(training_data, externals)
        num_chunks = len(data_chunks)
        
        # Create chunk indices for shuffling
        chunk_indices = list(range(num_chunks))
        if self.shuffle_chunks:
            np.random.shuffle(chunk_indices)
        
        # Training loop over chunks
        total_loss = 0.0
        batch_losses = []
        
        if verbose:
            chunk_iterator = tqdm(chunk_indices, desc=f"Epoch {epoch}", leave=False)
        else:
            chunk_iterator = chunk_indices
        
        for chunk_idx in chunk_iterator:
            self.optimizer.zero_grad()
            
            # Get chunk data
            chunk_data = data_chunks[chunk_idx]
            chunk_externals = external_chunks[chunk_idx] if external_chunks else None
            
            # Forward pass
            output = self.model(chunk_data, externals=chunk_externals)
            
            # Compute loss
            loss = self.loss_function(output.expectations, chunk_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_value is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_value
                )
                self.training_history['gradient_norms'].append(grad_norm.item())
            
            # Optimizer step
            self.optimizer.step()
            
            # Record loss
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Update progress bar
            if verbose and hasattr(chunk_iterator, 'set_postfix'):
                chunk_iterator.set_postfix({'Loss': f'{batch_loss:.6f}'})
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_chunks
        
        # Store history
        self.training_history['epoch_losses'].append(avg_loss)
        self.training_history['batch_losses'].extend(batch_losses)
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['num_chunks_per_epoch'].append(num_chunks)
        
        # Print epoch summary
        if verbose:
            print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s | "
                  f"Chunks: {num_chunks} | Chunk Size: {self.chunk_size}")
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'num_chunks': num_chunks,
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'gradient_norm': np.mean(self.training_history['gradient_norms'][-num_chunks:]) if self.gradient_clip_value else 0.0
        }
    
    def forecast(
        self,
        calibration_data: torch.Tensor,
        forecast_horizon: int,
        externals: Optional[torch.Tensor] = None,
        future_externals: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate forecasts using the trained model.
        
        In abridged mode, the last L chronological samples from training data
        are used as the calibration window, where L is the chunk size.
        
        Parameters
        ----------
        calibration_data : torch.Tensor
            Calibration sequence (last L samples from training) [1, L, n_obs_vars]
        forecast_horizon : int
            Number of future time steps to predict
        externals : Optional[torch.Tensor]
            External variables for calibration [1, L, n_ext_vars]
        future_externals : Optional[torch.Tensor]
            Future external variables [1, forecast_horizon, n_ext_vars]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing forecasts and related information
        """
        self.model.eval()
        
        # Validate calibration data length
        if calibration_data.size(1) != self.chunk_size:
            warnings.warn(f"Calibration data length ({calibration_data.size(1)}) "
                         f"differs from chunk size ({self.chunk_size}). "
                         f"Consider using the last {self.chunk_size} samples.")
        
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
                'forecast_horizon': forecast_horizon,
                'chunk_size': self.chunk_size
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
            'total_chunks_processed': sum(self.training_history['num_chunks_per_epoch']),
            'avg_chunks_per_epoch': np.mean(self.training_history['num_chunks_per_epoch']),
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'avg_gradient_norm': np.mean(self.training_history['gradient_norms']) if self.training_history['gradient_norms'] else 0.0
        }


class AbridgedHCNNTrainer:
    """
    HCNN Trainer for Abridged Mode implementing train_only and train_and_validate methods.

    This trainer follows the patterns from HCNNTrainer but adapts them for abridged mode
    where data is processed in chunks.
    """

    def __init__(
        self,
        model,
        chunk_size: int,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        backprop_mode: Literal["per_batch", "per_epoch"] = "per_batch",
        optimizer_type: Literal["adam", "sgd"] = "adam",
        learning_rate: float = 1e-4,
        save_dir: str = "./checkpoints",
        overlap_size: int = 0,
        shuffle_chunks: bool = True
    ):
        """
        Initialize the Abridged HCNN Trainer.

        Args:
            model: HCNN model to train
            chunk_size: Size of each training chunk
            loss_fn: Loss function type ("mse" or "logcosh")
            backprop_mode: When to update weights ("per_batch" or "per_epoch")
            optimizer_type: Optimizer type ("adam" or "sgd")
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save checkpoints
            overlap_size: Number of overlapping time steps between chunks
            shuffle_chunks: Whether to shuffle chunks during training
        """
        self.model = model
        self.chunk_size = chunk_size
        self.backprop_mode = backprop_mode
        self.save_dir = save_dir
        self.overlap_size = overlap_size
        self.shuffle_chunks = shuffle_chunks

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup loss function
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "logcosh":
            self.loss_fn = self._logcosh_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Setup optimizer
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _logcosh_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute log-cosh loss (more robust to outliers than MSE)."""
        diff = predictions - targets
        return torch.mean(torch.log(torch.cosh(diff)))

    def create_chunks(
        self,
        data: torch.Tensor,
        externals: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Split data into overlapping chunks.

        Parameters
        ----------
        data : torch.Tensor
            Input data [batch_size, seq_len, n_obs_vars]
        externals : Optional[torch.Tensor]
            External variables [batch_size, seq_len, n_ext_vars]

        Returns
        -------
        Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]
            List of data chunks and optional list of external chunks
        """
        batch_size, seq_len, n_obs_vars = data.shape

        if seq_len < self.chunk_size:
            warnings.warn(f"Sequence length ({seq_len}) is smaller than chunk size ({self.chunk_size}). "
                         f"Using the entire sequence as a single chunk.")
            return [data], [externals] if externals is not None else None

        # Calculate step size (accounting for overlap)
        step_size = self.chunk_size - self.overlap_size

        # Generate chunk start indices
        start_indices = list(range(0, seq_len - self.chunk_size + 1, step_size))

        # Create chunks
        data_chunks = []
        external_chunks = [] if externals is not None else None

        for start_idx in start_indices:
            end_idx = start_idx + self.chunk_size

            # Extract data chunk
            chunk = data[:, start_idx:end_idx, :]
            data_chunks.append(chunk)

            # Extract external chunk if available
            if externals is not None:
                ext_chunk = externals[:, start_idx:end_idx, :]
                external_chunks.append(ext_chunk)

        return data_chunks, external_chunks

    def train_only(
        self,
        training_data: torch.Tensor,
        num_epochs: int = 10,
        externals: Optional[torch.Tensor] = None,
        verbose: bool = True
    ):
        """
        Train the model using only training data (no validation) in abridged mode.

        Args:
            training_data: Training data [batch_size, seq_len, n_obs_vars]
            num_epochs: Number of training epochs
            externals: External variables [batch_size, seq_len, n_ext_vars]
            verbose: Whether to print training progress
        """
        best_loss = float('inf')
        best_epoch = -1

        # Move data to device
        device = next(self.model.parameters()).device
        training_data = training_data.to(device)
        if externals is not None:
            externals = externals.to(device)

        if self.backprop_mode == 'per_batch':
            epoch_loss_summary = []

            for epoch in tqdm(range(num_epochs), desc="Training Only (Abridged)"):
                # Create chunks
                data_chunks, external_chunks = self.create_chunks(training_data, externals)
                num_chunks = len(data_chunks)

                # Create chunk indices for shuffling
                chunk_indices = list(range(num_chunks))
                if self.shuffle_chunks:
                    np.random.shuffle(chunk_indices)

                batch_loss_summary = []
                batch_losses = []

                self.model.train()
                total_loss, batch_count = 0.0, 0

                for chunk_idx in chunk_indices:
                    batch_count += 1
                    self.optimizer.zero_grad()

                    # Get chunk data
                    chunk_data = data_chunks[chunk_idx]
                    chunk_externals = external_chunks[chunk_idx] if external_chunks else None

                    # Forward pass
                    results = self.model.forward(data_window=chunk_data, externals=chunk_externals)
                    loss = self.loss_fn(results.expectations, chunk_data)

                    loss.backward()
                    self.optimizer.step()

                    batch_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})
                    batch_losses.append(loss.item())
                    total_loss += loss.item()

                    # Save best model
                    if loss.item() < best_loss:
                        best_epoch = epoch + 1
                        self.model.save_checkpoint(
                            epoch=best_epoch,
                            loss=loss.item(),
                            optimizer=self.optimizer,
                            checkpoint_dir=self.save_dir,
                            add_stuffs="_abridged_single",
                            cleanup=True
                        )
                        best_loss = loss.item()

                        if verbose:
                            print(f"✅ New best batch train loss: {loss:.6f}")
                    elif verbose:
                        print(f"ℹ️ Current batch train loss {loss:.6f} not better than best {best_loss:.6f}")

                avg_train_loss = total_loss / batch_count
                epoch_loss_summary.append({
                    f'epoch_{epoch+1}': {
                        'avg_train_loss': avg_train_loss,
                        'batch_losses': batch_loss_summary
                    }
                })

                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.6f}")

            # Save epoch losses
            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)

        elif self.backprop_mode == 'per_epoch':
            # Per-epoch backpropagation implementation
            epoch_loss_summary = []

            for epoch in tqdm(range(num_epochs), desc="Training Only (Abridged Per Epoch)"):
                # Create chunks
                data_chunks, external_chunks = self.create_chunks(training_data, externals)
                num_chunks = len(data_chunks)

                # Create chunk indices for shuffling
                chunk_indices = list(range(num_chunks))
                if self.shuffle_chunks:
                    np.random.shuffle(chunk_indices)

                self.model.train()
                accumulated_loss = 0.0
                batch_count = 0

                for chunk_idx in chunk_indices:
                    batch_count += 1

                    # Get chunk data
                    chunk_data = data_chunks[chunk_idx]
                    chunk_externals = external_chunks[chunk_idx] if external_chunks else None

                    # Forward pass
                    results = self.model.forward(data_window=chunk_data, externals=chunk_externals)
                    loss = self.loss_fn(results.expectations, chunk_data)

                    accumulated_loss += loss

                # Backpropagation once per epoch
                self.optimizer.zero_grad()
                avg_loss = accumulated_loss / batch_count
                avg_loss.backward()
                self.optimizer.step()

                # Save best model
                if avg_loss.item() < best_loss:
                    best_epoch = epoch + 1
                    self.model.save_checkpoint(
                        epoch=best_epoch,
                        loss=avg_loss.item(),
                        optimizer=self.optimizer,
                        checkpoint_dir=self.save_dir,
                        add_stuffs="_abridged_single",
                        cleanup=True
                    )
                    best_loss = avg_loss.item()

                    if verbose:
                        print(f"✅ New best epoch train loss: {avg_loss:.6f}")

                epoch_loss_summary.append({
                    f'epoch_{epoch+1}': {
                        'avg_train_loss': avg_loss.item()
                    }
                })

                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_loss:.6f}")

            # Save epoch losses
            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)

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
        Train the model with validation after each epoch in abridged mode.

        Args:
            training_data: Training data [batch_size, seq_len, n_obs_vars]
            num_epochs: Number of training epochs
            calibration_window: Context window for forecasting [1, cal_len, n_obs_vars]
            val_data: Validation target data [forecast_horizon, n_obs_vars]
            externals: External variables for training [batch_size, seq_len, n_ext_vars]
            future_externals: Future external variables [1, forecast_horizon, n_ext_vars]
            verbose: Whether to print training progress
        """
        best_val_loss = float('inf')
        best_epoch = -1

        # Move data to device
        device = next(self.model.parameters()).device
        training_data = training_data.to(device)
        calibration_window = calibration_window.to(device)
        val_data = val_data.to(device)
        if externals is not None:
            externals = externals.to(device)
        if future_externals is not None:
            future_externals = future_externals.to(device)

        if self.backprop_mode == 'per_batch':
            epoch_loss_summary = []

            for epoch in tqdm(range(num_epochs), desc="Training and Validating (Abridged)"):
                # Create chunks
                data_chunks, external_chunks = self.create_chunks(training_data, externals)
                num_chunks = len(data_chunks)

                # Create chunk indices for shuffling
                chunk_indices = list(range(num_chunks))
                if self.shuffle_chunks:
                    np.random.shuffle(chunk_indices)

                batch_train_loss_summary = []
                batch_val_loss_summary = []
                batch_losses = []
                batch_val_losses = []

                self.model.train()
                total_loss, batch_count = 0.0, 0

                for chunk_idx in chunk_indices:
                    batch_count += 1
                    self.optimizer.zero_grad()

                    # Get chunk data
                    chunk_data = data_chunks[chunk_idx]
                    chunk_externals = external_chunks[chunk_idx] if external_chunks else None

                    # Training step
                    results = self.model.forward(data_window=chunk_data, externals=chunk_externals)
                    loss = self.loss_fn(results.expectations, chunk_data)

                    loss.backward()
                    self.optimizer.step()

                    batch_train_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})
                    batch_losses.append(loss.item())
                    total_loss += loss.item()

                    # Validation step
                    self.model.eval()
                    with torch.no_grad():
                        results = self.model.forward(
                            data_window=calibration_window,
                            forecast_horizon=val_data.shape[0],
                            externals=future_externals
                        )
                        forecast = results.forecasts.squeeze(0) if results.forecasts.dim() > 2 else results.forecasts
                        val_loss = self.loss_fn(forecast, val_data).item()

                        batch_val_loss_summary.append({f'batch_idx_{batch_count}': val_loss})
                        batch_val_losses.append(val_loss)

                    self.model.train()  # Switch back to training mode

                # Calculate epoch averages
                avg_train_loss = total_loss / batch_count
                avg_val_loss = np.mean(batch_val_losses)

                # Save best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_epoch = epoch + 1
                    self.model.save_checkpoint(
                        epoch=best_epoch,
                        loss=avg_val_loss,
                        optimizer=self.optimizer,
                        checkpoint_dir=self.save_dir,
                        add_stuffs="_abridged_validated",
                        cleanup=True
                    )
                    best_val_loss = avg_val_loss

                    if verbose:
                        print(f"✅ New best validation loss: {avg_val_loss:.6f}")

                epoch_loss_summary.append({
                    f'epoch_{epoch+1}': {
                        'avg_train_loss': avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'batch_train_losses': batch_train_loss_summary,
                        'batch_val_losses': batch_val_loss_summary
                    }
                })

                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")

            # Save epoch losses
            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)

    def save_epochs_losses_to_json(self, epoch_losses):
        """Save epoch losses to JSON file."""
        loss_file = os.path.join(self.save_dir, f"{self.model.name}_abridged_losses.json")
        with open(loss_file, 'w') as f:
            json.dump(epoch_losses, f, indent=2)


def prepare_calibration_window(
    training_data: torch.Tensor,
    chunk_size: int,
    externals: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepare calibration window for abridged mode forecasting.

    Extracts the last L chronological samples from training data,
    where L is the chunk size used during training.

    Parameters
    ----------
    training_data : torch.Tensor
        Complete training data [batch_size, seq_len, n_obs_vars]
    chunk_size : int
        Chunk size used during training
    externals : Optional[torch.Tensor]
        External variables [batch_size, seq_len, n_ext_vars]

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        Calibration window and corresponding externals
    """
    seq_len = training_data.size(1)

    if seq_len < chunk_size:
        warnings.warn(f"Training data length ({seq_len}) is smaller than chunk size ({chunk_size}). "
                     f"Using entire sequence as calibration window.")
        return training_data, externals

    # Extract last chunk_size samples
    calibration_data = training_data[:, -chunk_size:, :]
    calibration_externals = externals[:, -chunk_size:, :] if externals is not None else None

    return calibration_data, calibration_externals


def train_abridged(
    model: nn.Module,
    training_data: torch.Tensor,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    chunk_size: int,
    device: torch.device,
    externals: Optional[torch.Tensor] = None,
    overlap_size: int = 0,
    shuffle_chunks: bool = True,
    gradient_clip_value: Optional[float] = None,
    verbose: bool = True,
    checkpoint_dir: Optional[str] = None,
    save_every: int = 10
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function for abridged training.

    Parameters
    ----------
    model : nn.Module
        HCNN model to train
    training_data : torch.Tensor
        Training data [batch_size, seq_len, n_obs_vars]
    loss_function : nn.Module
        Loss function for training
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    num_epochs : int
        Number of training epochs
    chunk_size : int
        Size of each training chunk
    device : torch.device
        Device for computation
    externals : Optional[torch.Tensor]
        External variables [batch_size, seq_len, n_ext_vars]
    overlap_size : int
        Number of overlapping time steps between chunks
    shuffle_chunks : bool
        Whether to shuffle chunks during training
    gradient_clip_value : Optional[float]
        Maximum gradient norm for clipping
    verbose : bool
        Whether to print progress
    checkpoint_dir : Optional[str]
        Directory to save checkpoints
    save_every : int
        Save checkpoint every N epochs

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        Trained model and training history
    """
    trainer = AbridgedTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        chunk_size=chunk_size,
        device=device,
        overlap_size=overlap_size,
        shuffle_chunks=shuffle_chunks,
        gradient_clip_value=gradient_clip_value
    )

    if verbose:
        print(f"Starting Abridged Training")
        print(f"Sequence Length: {training_data.size(1)}")
        print(f"Chunk Size: {chunk_size}")
        print(f"Overlap Size: {overlap_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {device}")
        print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        epoch_metrics = trainer.train_epoch(
            training_data=training_data,
            externals=externals,
            epoch=epoch,
            verbose=verbose
        )

        # Save checkpoint
        if checkpoint_dir and epoch % save_every == 0:
            model.save_checkpoint(
                epoch=epoch,
                loss=epoch_metrics['loss'],
                optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
                add_stuffs="_abridged"
            )

    training_summary = trainer.get_training_summary()

    if verbose:
        print("=" * 60)
        print("Training Complete!")
        print(f"Best Loss: {training_summary['best_loss']:.6f}")
        print(f"Total Time: {training_summary['total_training_time']:.2f}s")
        print(f"Total Chunks Processed: {training_summary['total_chunks_processed']}")

    return model, training_summary
