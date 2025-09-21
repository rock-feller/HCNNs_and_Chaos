"""
HCNN Ensemble Trainer

Comprehensive trainer for HCNN ensemble models with support for different
training strategies and checkpoint management.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import pandas as pd
from typing import List, Tuple, Optional, Literal, Dict, Any
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from ..core.base import BaseEnsemble
from .custom_losses import LogCoshLoss


class HCNNEnsembleTrainer:
    """
    Trainer for HCNN ensemble models with advanced checkpoint management.
    
    Supports both individual model optimization and median-based ensemble optimization.
    """
    
    def __init__(
        self,
        ensemble: BaseEnsemble,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        optimizer_type: str = "adam",
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = "./checkpoints_ensemble",
        best_on: Literal["median", "individual"] = "individual",
        variable_names: Optional[List[str]] = None,
        gradient_clip_value: Optional[float] = None
    ):
        """
        Initialize the ensemble trainer.
        
        Parameters
        ----------
        ensemble : BaseEnsemble
            The ensemble model to train
        loss_fn : str
            Loss function type ("mse" or "logcosh")
        optimizer_type : str
            Optimizer type ("adam", "sgd", "rmsprop")
        learning_rate : float
            Learning rate for optimization
        device : torch.device, optional
            Device to use for training
        save_dir : str, optional
            Directory to save checkpoints
        best_on : str
            Strategy for saving best models ("individual" or "median")
        variable_names : List[str], optional
            Names of variables for CSV output
        gradient_clip_value : float, optional
            Value for gradient clipping
        """
        self.ensemble = ensemble
        self.device = device or ensemble.device
        self.best_on = best_on
        self.gradient_clip_value = gradient_clip_value
        
        # Move ensemble to device
        self.ensemble.to(self.device)
        
        # Set up loss function
        if loss_fn.lower() == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn.lower() == "logcosh":
            self.loss_fn = LogCoshLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
        
        # Set up optimizers for each ensemble member
        self.optimizers = []
        for i in range(ensemble.n_ensemble):
            model = ensemble.get_model(i)
            if optimizer_type.lower() == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type.lower() == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_type.lower() == "rmsprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            self.optimizers.append(optimizer)
        
        # Set up variable names
        self.variable_names = variable_names or [f"var_{i+1}" for i in range(ensemble.n_obs_vars)]
        
        # Set up save directory
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        ensemble_type = ensemble.ensemble_type
        self.save_dir = f"{save_dir}_{ensemble_type}_{self.best_on}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.best_losses = [float('inf')] * ensemble.n_ensemble
        self.best_median_loss = float('inf')
        self.training_history = []
        
        print(f"🏗️  Ensemble Trainer initialized:")
        print(f"   Ensemble type: {ensemble_type}")
        print(f"   Number of models: {ensemble.n_ensemble}")
        print(f"   Best model strategy: {self.best_on}")
        print(f"   Save directory: {self.save_dir}")
    
    def train_only(
        self,
        training_data: torch.Tensor,
        num_epochs: int = 100,
        externals: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble using only training data.
        
        Parameters
        ----------
        training_data : torch.Tensor
            Training data tensor
        num_epochs : int
            Number of training epochs
        externals : torch.Tensor, optional
            External variables
        verbose : bool
            Whether to print training progress
            
        Returns
        -------
        Dict[str, Any]
            Training summary with loss history and best models
        """
        print(f"\n🚀 Starting ensemble training (train_only mode)")
        print(f"Training data shape: {training_data.shape}")
        print(f"Number of epochs: {num_epochs}")
        
        epoch_losses = []
        
        for epoch in tqdm(range(num_epochs), desc="Training Ensemble"):
            epoch_member_losses = []
            member_loss_summary = {}
            
            # Train each ensemble member
            for idx in range(self.ensemble.n_ensemble):
                model = self.ensemble.get_model(idx)
                optimizer = self.optimizers[idx]
                
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data_window=training_data, ext_data_window=externals)
                loss = self.loss_fn(output.expectations, training_data)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_value)
                
                optimizer.step()
                
                loss_value = loss.item()
                epoch_member_losses.append(loss_value)
                member_loss_summary[f'member_{idx+1}'] = {'train_loss': loss_value}
                
                # Save best individual models
                if self.best_on == 'individual' and loss_value < self.best_losses[idx]:
                    self._save_model_checkpoint(
                        model, optimizer, epoch + 1, loss_value, 
                        f"best_model_member_{idx+1}.pth"
                    )
                    self.best_losses[idx] = loss_value
                    if verbose:
                        print(f"✅ [Model {idx+1}] New best loss: {loss_value:.6f}")
            
            # Save best median ensemble
            median_loss = float(np.median(epoch_member_losses))
            if self.best_on == 'median' and median_loss < self.best_median_loss:
                for idx in range(self.ensemble.n_ensemble):
                    model = self.ensemble.get_model(idx)
                    optimizer = self.optimizers[idx]
                    self._save_model_checkpoint(
                        model, optimizer, epoch + 1, median_loss,
                        f"best_ensemble_member_{idx+1}.pth"
                    )
                self.best_median_loss = median_loss
                if verbose:
                    print(f"✅ [Ensemble] New best median loss: {median_loss:.6f}")
            
            # Record epoch losses
            epoch_loss_data = {
                f'epoch_{epoch+1}': member_loss_summary
            }
            epoch_losses.append(epoch_loss_data)
            
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                mean_loss = np.mean(epoch_member_losses)
                print(f"Epoch {epoch+1:3d}/{num_epochs}: Mean Loss: {mean_loss:.6f}, "
                      f"Median Loss: {median_loss:.6f}")
        
        # Save training history
        self._save_training_history(epoch_losses, "train_only_losses.json")
        
        # Prepare summary
        summary = {
            'training_mode': 'train_only',
            'num_epochs': num_epochs,
            'best_individual_losses': self.best_losses,
            'best_median_loss': self.best_median_loss,
            'final_mean_loss': np.mean(epoch_member_losses),
            'final_median_loss': median_loss,
            'loss_history': epoch_losses
        }
        
        print(f"✅ Training completed!")
        print(f"   Best individual losses: {[f'{loss:.6f}' for loss in self.best_losses]}")
        print(f"   Best median loss: {self.best_median_loss:.6f}")
        
        return summary
    
    def train_and_validate(
        self,
        training_data: torch.Tensor,
        num_epochs: int = 100,
        calibration_window: torch.Tensor = None,
        val_data: torch.Tensor = None,
        externals: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble with validation.
        
        Parameters
        ----------
        training_data : torch.Tensor
            Training data tensor
        num_epochs : int
            Number of training epochs
        calibration_window : torch.Tensor
            Data for model calibration before forecasting
        val_data : torch.Tensor
            Validation data for forecasting evaluation
        externals : torch.Tensor, optional
            External variables
        verbose : bool
            Whether to print training progress
            
        Returns
        -------
        Dict[str, Any]
            Training summary with loss history and best models
        """
        print(f"\n🚀 Starting ensemble training (train_and_validate mode)")
        print(f"Training data shape: {training_data.shape}")
        if val_data is not None:
            print(f"Validation data shape: {val_data.shape}")
        print(f"Number of epochs: {num_epochs}")
        
        epoch_losses = []
        best_val_losses = [float('inf')] * self.ensemble.n_ensemble
        best_median_val_loss = float('inf')
        
        for epoch in tqdm(range(num_epochs), desc="Training and Validating"):
            epoch_member_losses = []
            epoch_val_losses = []
            member_loss_summary = {}
            
            # Train each ensemble member
            for idx in range(self.ensemble.n_ensemble):
                model = self.ensemble.get_model(idx)
                optimizer = self.optimizers[idx]
                
                # Training phase
                model.train()
                optimizer.zero_grad()
                
                output = model(data_window=training_data, ext_data_window=externals)
                train_loss = self.loss_fn(output.expectations, training_data)
                
                train_loss.backward()
                
                if self.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_value)
                
                optimizer.step()
                
                train_loss_value = train_loss.item()
                epoch_member_losses.append(train_loss_value)
                
                # Validation phase
                val_loss_value = 0.0
                if val_data is not None and calibration_window is not None:
                    model.eval()
                    with torch.no_grad():
                        # Use calibration window to forecast validation data
                        # Ensure calibration window has correct shape [batch_size, seq_len, features]
                        if calibration_window.dim() == 2:
                            cal_window = calibration_window.unsqueeze(0)
                        else:
                            cal_window = calibration_window

                        forecast_output = model(
                            data_window=cal_window,
                            forecast_horizon=val_data.shape[-2] if val_data.dim() == 3 else val_data.shape[0]
                        )
                        forecast = forecast_output.forecasts.squeeze(0)
                        # Ensure val_data has correct shape for loss calculation
                        if val_data.dim() == 3:
                            val_target = val_data.squeeze(0)
                        else:
                            val_target = val_data
                        val_loss = self.loss_fn(forecast, val_target)
                        val_loss_value = val_loss.item()
                        epoch_val_losses.append(val_loss_value)
                        
                        # Save forecasts for best models
                        if self.best_on == 'individual' and val_loss_value < best_val_losses[idx]:
                            self._save_forecast_csv(
                                output.expectations.squeeze(0),
                                forecast,
                                f"best_forecast_member_{idx+1}_epoch_{epoch+1}.csv"
                            )
                
                member_loss_summary[f'member_{idx+1}'] = {
                    'train_loss': train_loss_value,
                    'val_loss': val_loss_value
                }
                
                # Save best individual models based on validation loss
                if self.best_on == 'individual' and val_loss_value < best_val_losses[idx]:
                    self._save_model_checkpoint(
                        model, optimizer, epoch + 1, val_loss_value,
                        f"best_model_member_{idx+1}.pth"
                    )
                    best_val_losses[idx] = val_loss_value
                    if verbose:
                        print(f"✅ [Model {idx+1}] New best val loss: {val_loss_value:.6f}")
            
            # Save best median ensemble based on validation loss
            if epoch_val_losses:
                median_val_loss = float(np.median(epoch_val_losses))
                if self.best_on == 'median' and median_val_loss < best_median_val_loss:
                    for idx in range(self.ensemble.n_ensemble):
                        model = self.ensemble.get_model(idx)
                        optimizer = self.optimizers[idx]
                        self._save_model_checkpoint(
                            model, optimizer, epoch + 1, median_val_loss,
                            f"best_ensemble_member_{idx+1}.pth"
                        )
                    best_median_val_loss = median_val_loss
                    if verbose:
                        print(f"✅ [Ensemble] New best median val loss: {median_val_loss:.6f}")
            
            # Record epoch losses
            epoch_loss_data = {
                f'epoch_{epoch+1}': member_loss_summary
            }
            epoch_losses.append(epoch_loss_data)
            
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                mean_train_loss = np.mean(epoch_member_losses)
                mean_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
                print(f"Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {mean_train_loss:.6f}, "
                      f"Val Loss: {mean_val_loss:.6f}")
        
        # Save training history
        self._save_training_history(epoch_losses, "train_validate_losses.json")
        
        # Prepare summary
        summary = {
            'training_mode': 'train_and_validate',
            'num_epochs': num_epochs,
            'best_individual_val_losses': best_val_losses,
            'best_median_val_loss': best_median_val_loss,
            'final_mean_train_loss': np.mean(epoch_member_losses),
            'final_mean_val_loss': np.mean(epoch_val_losses) if epoch_val_losses else 0.0,
            'loss_history': epoch_losses
        }
        
        print(f"✅ Training completed!")
        print(f"   Best individual val losses: {[f'{loss:.6f}' for loss in best_val_losses]}")
        print(f"   Best median val loss: {best_median_val_loss:.6f}")
        
        return summary

    def _save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        filename: str
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_name': model.name if hasattr(model, 'name') else 'unknown'
        }

        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)

    def _save_training_history(self, epoch_losses: List[Dict], filename: str) -> None:
        """Save training history to JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(epoch_losses, f, indent=4)
        print(f"📊 Training history saved to: {filename}")

    def _save_forecast_csv(
        self,
        context_output: torch.Tensor,
        forecast: torch.Tensor,
        filename: str
    ) -> None:
        """Save forecast results to CSV file."""
        all_data = []

        # Context output data
        context_np = context_output.cpu().numpy()
        for t, val in enumerate(context_np):
            all_data.append({
                "type": "context_output",
                "timestep": t,
                **{f"dim_{i}": v for i, v in enumerate(val)}
            })

        # Forecast data
        forecast_np = forecast.cpu().numpy()
        for t, val in enumerate(forecast_np):
            all_data.append({
                "type": "forecast",
                "timestep": t,
                **{f"dim_{i}": v for i, v in enumerate(val)}
            })

        # Create DataFrame and rename columns
        df = pd.DataFrame(all_data)
        for i, var_name in enumerate(self.variable_names):
            if f"dim_{i}" in df.columns:
                df.rename(columns={f"dim_{i}": var_name}, inplace=True)

        # Save to CSV
        filepath = os.path.join(self.save_dir, filename)
        df.to_csv(filepath, index=False)

    def load_best_models(self) -> None:
        """Load the best saved models back into the ensemble."""
        print(f"🔄 Loading best models from {self.save_dir}")

        for idx in range(self.ensemble.n_ensemble):
            if self.best_on == 'individual':
                checkpoint_file = f"best_model_member_{idx+1}.pth"
            else:
                checkpoint_file = f"best_ensemble_member_{idx+1}.pth"

            checkpoint_path = os.path.join(self.save_dir, checkpoint_file)

            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model = self.ensemble.get_model(idx)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizers[idx].load_state_dict(checkpoint['optimizer_state_dict'])

                print(f"✅ Loaded Model {idx+1} from epoch {checkpoint['epoch']} "
                      f"with loss {checkpoint['loss']:.6f}")
            else:
                print(f"⚠️  Checkpoint not found for Model {idx+1}: {checkpoint_path}")

    def evaluate_ensemble(
        self,
        test_data: torch.Tensor,
        externals: Optional[torch.Tensor] = None,
        aggregation_methods: List[str] = ["mean", "median"]
    ) -> Dict[str, Any]:
        """
        Evaluate the ensemble on test data.

        Parameters
        ----------
        test_data : torch.Tensor
            Test data tensor
        externals : torch.Tensor, optional
            External variables
        aggregation_methods : List[str]
            Aggregation methods to evaluate

        Returns
        -------
        Dict[str, Any]
            Evaluation results
        """
        print(f"\n📊 Evaluating ensemble on test data")
        print(f"Test data shape: {test_data.shape}")

        self.ensemble.eval()
        results = {}

        with torch.no_grad():
            # Individual model evaluation
            individual_losses = []
            individual_predictions = []

            for idx in range(self.ensemble.n_ensemble):
                model = self.ensemble.get_model(idx)
                output = model(data_window=test_data, ext_data_window=externals)
                loss = self.loss_fn(output.expectations, test_data).item()

                individual_losses.append(loss)
                individual_predictions.append(output.expectations.cpu().numpy())

                print(f"Model {idx+1} test loss: {loss:.6f}")

            # Ensemble evaluation with different aggregation methods
            ensemble_results = {}
            for method in aggregation_methods:
                ensemble_output = self.ensemble(
                    data_window=test_data,
                    ext_data_window=externals,
                    aggregation_method=method
                )
                ensemble_loss = self.loss_fn(ensemble_output.expectations, test_data).item()
                ensemble_results[method] = {
                    'loss': ensemble_loss,
                    'predictions': ensemble_output.expectations.cpu().numpy()
                }
                print(f"Ensemble ({method}) test loss: {ensemble_loss:.6f}")

            # Calculate prediction uncertainty
            predictions_stack = np.stack(individual_predictions)
            prediction_std = np.std(predictions_stack, axis=0)
            mean_uncertainty = np.mean(prediction_std)

            results = {
                'individual_losses': individual_losses,
                'individual_predictions': individual_predictions,
                'ensemble_results': ensemble_results,
                'prediction_uncertainty': {
                    'std_per_timestep': prediction_std,
                    'mean_uncertainty': mean_uncertainty
                },
                'summary': {
                    'best_individual_loss': min(individual_losses),
                    'worst_individual_loss': max(individual_losses),
                    'mean_individual_loss': np.mean(individual_losses),
                    'best_ensemble_method': min(ensemble_results.keys(),
                                              key=lambda k: ensemble_results[k]['loss']),
                    'mean_uncertainty': mean_uncertainty
                }
            }

        print(f"\n📈 Evaluation Summary:")
        print(f"   Best individual: {results['summary']['best_individual_loss']:.6f}")
        print(f"   Mean individual: {results['summary']['mean_individual_loss']:.6f}")
        print(f"   Best ensemble method: {results['summary']['best_ensemble_method']}")
        print(f"   Mean uncertainty: {mean_uncertainty:.6f}")

        return results

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process."""
        return {
            'ensemble_type': self.ensemble.ensemble_type,
            'n_ensemble': self.ensemble.n_ensemble,
            'best_strategy': self.best_on,
            'best_individual_losses': self.best_losses,
            'best_median_loss': self.best_median_loss,
            'save_directory': self.save_dir,
            'device': str(self.device)
        }
