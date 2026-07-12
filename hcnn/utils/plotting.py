"""
Plotting Utilities for HCNN and Chaotic Systems

This module provides comprehensive plotting utilities for visualizing chaotic system data,
model predictions, training progress, and ensemble results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import Tuple, Literal, Union, Optional, List
from pathlib import Path


class ChaoticSystemPlotter:
    """
    Plotter for chaotic system trajectories and predictions.
    """
    
    def __init__(self, data: Union[torch.Tensor, pd.DataFrame], system_name: str = "chaotic_system"):
        """
        Initialize plotter with chaotic system data.
        
        Args:
            data: Either torch.Tensor of shape (n, 3) or DataFrame with 3 columns
            system_name: Name of the chaotic system for labeling
        """
        if isinstance(data, torch.Tensor):
            assert data.ndim == 2 and data.shape[1] == 3, "Tensor must be of shape (n, 3)"
            self.data_type = 'tensor'
            self.data = data
            self.labels = [f"{system_name}: {var} variable" for var in ["x", "y", "z"]]
            self.system_name = system_name
        elif isinstance(data, pd.DataFrame):
            assert data.shape[1] == 3, "DataFrame must have 3 columns"
            self.data_type = 'dataframe'
            self.data = data
            self.labels = list(data.columns)
            self.system_name = system_name
        else:
            raise TypeError("Input must be a torch.Tensor or pandas.DataFrame")
    
    def plot_trajectories(
        self, 
        render: Literal["vertical", "horizontal"] = "vertical",
        figsize: Tuple[int, int] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot all trajectory components.
        
        Args:
            render: Layout orientation
            figsize: Figure size (width, height)
            save_path: Path to save the plot
        """
        if figsize is None:
            figsize = (25, 10) if render == "vertical" else (25, 4)
        
        if render == "vertical":
            fig, axs = plt.subplots(3, 1, figsize=figsize)
        elif render == "horizontal":
            fig, axs = plt.subplots(1, 3, figsize=figsize)
        else:
            raise ValueError("render must be 'vertical' or 'horizontal'")
        
        colors = ['blue', 'red', 'green']
        
        for i, ax in enumerate(axs):
            y = self.data[:, i].numpy() if self.data_type == 'tensor' else self.data.iloc[:, i]
            ax.plot(y, color=colors[i], label=self.labels[i], linewidth=1.5)
            ax.set_title(self.labels[i], fontsize=12)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.system_name} Trajectories', fontsize=14)
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_phase_space(
        self, 
        pairs: List[Tuple[int, int]] = None,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None
    ):
        """
        Plot phase space projections.
        
        Args:
            pairs: List of (i, j) pairs for phase space plots
            figsize: Figure size
            save_path: Path to save the plot
        """
        if pairs is None:
            pairs = [(0, 1), (0, 2), (1, 2)]  # xy, xz, yz
        
        fig, axs = plt.subplots(1, len(pairs), figsize=figsize)
        if len(pairs) == 1:
            axs = [axs]
        
        labels = ['x', 'y', 'z']
        
        for idx, (i, j) in enumerate(pairs):
            data_i = self.data[:, i].numpy() if self.data_type == 'tensor' else self.data.iloc[:, i]
            data_j = self.data[:, j].numpy() if self.data_type == 'tensor' else self.data.iloc[:, j]
            
            axs[idx].plot(data_i, data_j, alpha=0.7, linewidth=0.8)
            axs[idx].set_xlabel(f'{labels[i]} variable')
            axs[idx].set_ylabel(f'{labels[j]} variable')
            axs[idx].set_title(f'{labels[i]}-{labels[j]} Phase Space')
            axs[idx].grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.system_name} Phase Space', fontsize=14)
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_true_vs_predicted(
        self, 
        predicted: torch.Tensor, 
        true: torch.Tensor, 
        render: Literal["vertical", "horizontal"] = "horizontal",
        figsize: Tuple[int, int] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot true vs predicted trajectories.
        
        Args:
            predicted: Predicted data of shape (n, 3)
            true: True data of shape (n, 3)
            render: Layout orientation
            figsize: Figure size
            save_path: Path to save the plot
        """
        assert predicted.shape == true.shape and predicted.shape[1] == 3, \
            "Shapes must match and be (n, 3)"
        
        if figsize is None:
            figsize = (25, 10) if render == "vertical" else (25, 4)
        
        if render == "vertical":
            fig, axs = plt.subplots(3, 1, figsize=figsize)
        elif render == "horizontal":
            fig, axs = plt.subplots(1, 3, figsize=figsize)
        else:
            raise ValueError("render must be 'vertical' or 'horizontal'")
        
        colors = ['red', 'green', 'blue']
        labels = ['x', 'y', 'z']
        
        for i, ax in enumerate(axs):
            ax.plot(true[:, i].numpy(), color=colors[i], label=f"True {labels[i]}", 
                   linewidth=2, alpha=0.8)
            ax.plot(predicted[:, i].numpy(), color=colors[i], linestyle='--', 
                   label=f"Predicted {labels[i]}", linewidth=1.5, alpha=0.9)
            ax.set_title(f"True vs Predicted - {labels[i]} variable")
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.system_name} - True vs Predicted', fontsize=14)
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class TrainingPlotter:
    """
    Plotter for training progress and metrics.
    """
    
    @staticmethod
    def plot_training_loss(
        loss_history: List[float],
        val_loss_history: Optional[List[float]] = None,
        title: str = "Training Loss",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot training loss over epochs.
        
        Args:
            loss_history: List of training losses
            val_loss_history: Optional list of validation losses
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        epochs = range(1, len(loss_history) + 1)
        plt.plot(epochs, loss_history, 'b-', label='Training Loss', linewidth=2)
        
        if val_loss_history is not None:
            plt.plot(epochs, val_loss_history, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_training_metrics(
        metrics_dict: dict,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot multiple training metrics.
        
        Args:
            metrics_dict: Dictionary with metric names as keys and lists as values
            figsize: Figure size
            save_path: Path to save the plot
        """
        n_metrics = len(metrics_dict)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        
        for idx, (metric_name, values) in enumerate(metrics_dict.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            epochs = range(1, len(values) + 1)
            axs[row, col].plot(epochs, values, linewidth=2)
            axs[row, col].set_xlabel('Epoch')
            axs[row, col].set_ylabel(metric_name)
            axs[row, col].set_title(f'{metric_name} over Epochs')
            axs[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].set_visible(False)
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_loss_from_json(
        json_path: str,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot training loss from JSON file.

        Args:
            json_path: Path to JSON file with loss history
            figsize: Figure size
            save_path: Path to save the plot
        """
        with open(json_path, 'r') as f:
            loss_data = json.load(f)

        train_losses = []
        val_losses = []

        # Handle both list and dict formats
        if isinstance(loss_data, list):
            # List format: [{"epoch_1": {...}}, {"epoch_2": {...}}, ...]
            for epoch_dict in loss_data:
                for epoch_key, epoch_data in epoch_dict.items():
                    train_losses.append(epoch_data['train_loss'])
                    if 'val_loss' in epoch_data and epoch_data['val_loss'] is not None:
                        val_losses.append(epoch_data['val_loss'])
        else:
            # Dict format: {"epoch_1": {...}, "epoch_2": {...}, ...}
            for epoch_key in sorted(loss_data.keys()):
                epoch_data = loss_data[epoch_key]
                train_losses.append(epoch_data['train_loss'])
                if 'val_loss' in epoch_data and epoch_data['val_loss'] is not None:
                    val_losses.append(epoch_data['val_loss'])

        val_losses = val_losses if len(val_losses) == len(train_losses) else None

        TrainingPlotter.plot_training_loss(
            train_losses, val_losses,
            title=f"Training Progress - {Path(json_path).stem}",
            figsize=figsize, save_path=save_path
        )


class EnsemblePlotter:
    """
    Plotter for ensemble predictions.
    """
    
    @staticmethod
    def plot_ensemble_predictions(
        ensemble: torch.Tensor,
        ground_truth: torch.Tensor,
        add_statistics: Literal['mean', 'median', 'both', None] = 'mean',
        render: Literal["vertical", "horizontal"] = "vertical",
        figsize: Tuple[int, int] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot ensemble predictions with statistics.
        
        Args:
            ensemble: Ensemble predictions of shape (n_models, n_steps, 3)
            ground_truth: Ground truth of shape (n_steps, 3)
            add_statistics: Which statistics to overlay
            render: Layout orientation
            figsize: Figure size
            save_path: Path to save the plot
        """
        assert ensemble.ndim == 3 and ensemble.shape[2] == 3, \
            "Ensemble must be (n_models, n_steps, 3)"
        assert ground_truth.shape == ensemble.shape[1:], \
            "Ground truth must be (n_steps, 3)"
        
        if figsize is None:
            figsize = (25, 10) if render == "vertical" else (25, 4)
        
        if render == "vertical":
            fig, axs = plt.subplots(3, 1, figsize=figsize)
        elif render == "horizontal":
            fig, axs = plt.subplots(1, 3, figsize=figsize)
        else:
            raise ValueError("render must be 'vertical' or 'horizontal'")
        
        colors = ['red', 'green', 'blue']
        labels = ['x', 'y', 'z']
        n_models, n_steps, _ = ensemble.shape
        
        for i, ax in enumerate(axs):
            # Plot ensemble members
            for j in range(n_models):
                ax.plot(ensemble[j, :, i].numpy(), color='grey', 
                       linestyle='-', alpha=0.3, linewidth=0.5)
            
            # Plot ground truth
            ax.plot(ground_truth[:, i].numpy(), color=colors[i], 
                   label='Ground Truth', linewidth=2)
            
            # Add statistics if requested
            if add_statistics in ['mean', 'both']:
                mean_vals = ensemble[:, :, i].mean(dim=0)
                ax.plot(mean_vals.numpy(), color=colors[i], linestyle='--', 
                       label='Ensemble Mean', linewidth=2)
            
            if add_statistics in ['median', 'both']:
                median_vals = ensemble[:, :, i].median(dim=0).values
                ax.plot(median_vals.numpy(), color=colors[i], linestyle='-.', 
                       label='Ensemble Median', linewidth=2)
            
            ax.set_title(f"Ensemble Predictions - {labels[i]} variable")
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Ensemble Predictions vs Ground Truth', fontsize=14)
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def plot_model_comparison(
    comparison_results: dict,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
):
    """
    Plot comparison between different models.
    
    Args:
        comparison_results: Dictionary with model comparison data
        figsize: Figure size
        save_path: Path to save the plot
    """
    models = list(comparison_results.keys())
    metrics = ['best_loss', 'final_loss', 'training_time', 'max_memory_gb']
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [comparison_results[model].get(metric, 0) for model in models]
        
        bars = axs[idx].bar(models, values, alpha=0.7)
        axs[idx].set_title(f'{metric.replace("_", " ").title()}')
        axs[idx].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axs[idx].text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.4f}' if metric.endswith('loss') else f'{value:.2f}',
                         ha='center', va='bottom')
    
    fig.suptitle('Model Comparison', fontsize=14)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
