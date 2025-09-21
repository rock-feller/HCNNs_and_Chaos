"""
HCNN Trainer Implementation

This module provides the HCNNTrainer class that implements the train_only and train_and_validate
methods as shown in the sample notebook, adapted for the corrected HCNN implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from typing import Optional, Literal, Union
import numpy as np


class HCNNTrainer:
    """
    HCNN Trainer class implementing train_only and train_and_validate methods
    following the original sample implementation patterns.
    """
    
    def __init__(
        self,
        model,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        backprop_mode: Literal["per_batch", "per_epoch"] = "per_batch",
        optimizer_type: Literal["adam", "sgd"] = "adam",
        learning_rate: float = 1e-4,
        save_dir: str = "./checkpoints"
    ):
        """
        Initialize the HCNN Trainer.
        
        Args:
            model: HCNN model to train
            loss_fn: Loss function type ("mse" or "logcosh")
            backprop_mode: When to update weights ("per_batch" or "per_epoch")
            optimizer_type: Optimizer type ("adam" or "sgd")
            learning_rate: Learning rate for optimizer
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.backprop_mode = backprop_mode
        self.save_dir = save_dir
        
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
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _logcosh_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute log-cosh loss (more robust to outliers than MSE)."""
        diff = predictions - targets
        return torch.mean(torch.log(torch.cosh(diff)))
    
    def train_only(self, data_loader, num_epochs: int = 10, verbose: bool = True):
        """
        Train the model using only training data (no validation).
        
        Args:
            data_loader: Training data loader
            num_epochs: Number of training epochs
            verbose: Whether to print training progress
        """
        best_loss = float('inf')
        best_epoch = -1
        
        if self.backprop_mode == 'per_batch':
            epoch_loss_summary = []
            
            for epoch in tqdm(range(num_epochs), desc="Training Only"):
                batch_loss_summary = []
                batch_losses = []
                
                self.model.train()
                total_loss, batch_count = 0.0, 0
                
                for batch in data_loader:
                    batch_count += 1
                    self.optimizer.zero_grad()

                    # Move batch to model's device
                    batch = batch.to(next(self.model.parameters()).device)

                    # Forward pass - CORRECTED HCNN APPROACH
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    
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
                            add_stuffs="_single",
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
            
            for epoch in tqdm(range(num_epochs), desc="Training Only (Per Epoch)"):
                self.model.train()
                accumulated_loss = 0.0
                batch_count = 0
                
                for batch in data_loader:
                    batch_count += 1

                    # Move batch to model's device
                    batch = batch.to(next(self.model.parameters()).device)

                    # Forward pass - CORRECTED HCNN APPROACH
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    
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
                        add_stuffs="_single",
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
        data_loader,
        num_epochs: int,
        calibration_window: torch.Tensor,
        val_data: torch.Tensor,
        verbose: bool = True
    ):
        """
        Train the model with validation after each epoch.
        
        Args:
            data_loader: Training data loader
            num_epochs: Number of training epochs
            calibration_window: Context window for forecasting
            val_data: Validation target data
            verbose: Whether to print training progress
        """
        best_val_loss = float('inf')
        best_epoch = -1
        
        if self.backprop_mode == 'per_batch':
            epoch_loss_summary = []
            
            for epoch in tqdm(range(num_epochs), desc="Training and Validating"):
                batch_train_loss_summary = []
                batch_val_loss_summary = []
                batch_losses = []
                batch_val_losses = []
                
                self.model.train()
                total_loss, batch_count = 0.0, 0
                
                for batch in data_loader:
                    batch_count += 1
                    self.optimizer.zero_grad()

                    # Move batch to model's device
                    batch = batch.to(next(self.model.parameters()).device)

                    # Training step - CORRECTED HCNN APPROACH
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    batch_train_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})
                    batch_losses.append(loss.item())
                    total_loss += loss.item()
                    
                    # Validation step
                    self.model.eval()
                    with torch.no_grad():
                        # Move validation data to model's device
                        device = next(self.model.parameters()).device
                        cal_window_device = calibration_window.to(device)
                        val_data_device = val_data.to(device)

                        results = self.model.forward(
                            data_window=cal_window_device.unsqueeze(0),
                            forecast_horizon=val_data_device.shape[0]
                        )
                        forecast = results.forecasts.squeeze(0)
                        val_loss = self.loss_fn(forecast, val_data_device).item()
                        
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
                        add_stuffs="_validated",
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
        loss_file = os.path.join(self.save_dir, f"{self.model.name}_losses.json")
        with open(loss_file, 'w') as f:
            json.dump(epoch_losses, f, indent=2)
