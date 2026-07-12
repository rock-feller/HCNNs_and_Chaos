"""
Custom Loss Functions for HCNN Models.

This module provides custom loss functions designed for Historical Consistent Neural Networks,
with support for scalable computation across different model types and batch sizes.

All loss functions are designed to work with:
- y_pred: expectations tensor from model forward pass [batch_size, seq_len, n_obs_vars]
- y_true: data_window tensor (ground truth) [batch_size, seq_len, n_obs_vars]

Author: HCNN Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Literal
import math


class LogCoshLoss(nn.MSELoss):
    """
    Log-Cosh Loss Function for HCNN Models.
    
    The log-cosh loss is a smooth approximation to the absolute error that behaves like
    MSE for small errors and like MAE for large errors. This makes it robust to outliers
    while maintaining differentiability.
    
    Mathematical formulation:
        L(y_pred, y_true) = p * mean(log(cosh(y_pred - y_true)))
    
    Where:
        - log(cosh(x)) ≈ x²/2 for small |x| (quadratic, like MSE)
        - log(cosh(x)) ≈ |x| - log(2) for large |x| (linear, like MAE)
    
    Parameters
    ----------
    p : float, default=0.02
        Scaling factor for the loss. Controls the magnitude of the loss value.
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    eps : float, default=1e-12
        Small epsilon for numerical stability
    clip_value : Optional[float], default=None
        Maximum absolute value for error clipping to prevent overflow
    """
    
    def __init__(
        self, 
        p: float = 0.02,
        reduction: str = 'mean',
        eps: float = 1e-12,
        clip_value: Optional[float] = None
    ):
        # Initialize parent MSELoss (we override forward, so size_average and reduce don't matter)
        super(LogCoshLoss, self).__init__(reduction='none')
        
        self.p = p
        self.reduction = reduction
        self.eps = eps
        self.clip_value = clip_value
        
        # Validate parameters
        if p <= 0:
            raise ValueError(f"Parameter p must be positive, got {p}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Log-Cosh loss between predictions and targets.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions (expectations) of shape [batch_size, seq_len, n_obs_vars]
        y_true : torch.Tensor
            Ground truth targets (data_window) of shape [batch_size, seq_len, n_obs_vars]
            
        Returns
        -------
        torch.Tensor
            Computed loss value (scalar if reduction != 'none')
        """
        # Validate input shapes
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        # Compute prediction error
        error = y_pred - y_true
        
        # Optional error clipping for numerical stability
        if self.clip_value is not None:
            error = torch.clamp(error, -self.clip_value, self.clip_value)
        
        # Compute log(cosh(error)) with numerical stability
        # Use the identity: log(cosh(x)) = log((exp(x) + exp(-x))/2)
        # For numerical stability, use: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        abs_error = torch.abs(error)
        
        # For small errors, use Taylor expansion: log(cosh(x)) ≈ x²/2
        # For large errors, use asymptotic form: log(cosh(x)) ≈ |x| - log(2)
        small_error_mask = abs_error < 1.0
        
        # Small errors: use x²/2 (quadratic behavior)
        small_error_loss = 0.5 * error.pow(2)
        
        # Large errors: use |x| - log(2) + small correction
        large_error_loss = abs_error - math.log(2) + torch.log(1 + torch.exp(-2 * abs_error) + self.eps)
        
        # Combine using mask
        log_cosh_error = torch.where(small_error_mask, small_error_loss, large_error_loss)
        
        # Apply scaling factor
        loss = log_cosh_error * self.p
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss
    
    def extra_repr(self) -> str:
        """String representation of the loss function."""
        return f'p={self.p}, reduction={self.reduction}, eps={self.eps}, clip_value={self.clip_value}'


class HuberLoss(nn.MSELoss):
    """
    Huber Loss Function for HCNN Models.
    
    The Huber loss is less sensitive to outliers than MSE. It's quadratic for small errors
    and linear for large errors, with a smooth transition controlled by the delta parameter.
    
    Mathematical formulation:
        L(y_pred, y_true) = {
            0.5 * (y_pred - y_true)² / delta,           if |y_pred - y_true| <= delta
            |y_pred - y_true| - 0.5 * delta,           otherwise
        }
    
    Parameters
    ----------
    delta : float, default=1.0
        Threshold for switching between quadratic and linear loss
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super(HuberLoss, self).__init__(reduction='none')
        self.delta = delta
        self.reduction = reduction
        
        if delta <= 0:
            raise ValueError(f"Delta must be positive, got {delta}")
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss between predictions and targets."""
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        error = y_pred - y_true
        abs_error = torch.abs(error)
        
        # Quadratic for small errors, linear for large errors
        quadratic_mask = abs_error <= self.delta
        quadratic_loss = 0.5 * error.pow(2) / self.delta
        linear_loss = abs_error - 0.5 * self.delta
        
        loss = torch.where(quadratic_mask, quadratic_loss, linear_loss)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class WeightedMSELoss(nn.MSELoss):
    """
    Weighted MSE Loss for HCNN Models.
    
    Applies different weights to different time steps or variables, useful for
    emphasizing certain parts of the sequence or certain observed variables.
    
    Parameters
    ----------
    temporal_weights : Optional[torch.Tensor]
        Weights for different time steps [seq_len] or [1, seq_len, 1]
    variable_weights : Optional[torch.Tensor]
        Weights for different variables [n_obs_vars] or [1, 1, n_obs_vars]
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        temporal_weights: Optional[torch.Tensor] = None,
        variable_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(WeightedMSELoss, self).__init__(reduction='none')
        self.temporal_weights = temporal_weights
        self.variable_weights = variable_weights
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss between predictions and targets."""
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        # Compute base MSE
        mse_loss = (y_pred - y_true).pow(2)
        
        # Apply temporal weights
        if self.temporal_weights is not None:
            temp_weights = self.temporal_weights.to(y_pred.device)
            if temp_weights.dim() == 1:
                temp_weights = temp_weights.view(1, -1, 1)
            mse_loss = mse_loss * temp_weights
        
        # Apply variable weights
        if self.variable_weights is not None:
            var_weights = self.variable_weights.to(y_pred.device)
            if var_weights.dim() == 1:
                var_weights = var_weights.view(1, 1, -1)
            mse_loss = mse_loss * var_weights
        
        if self.reduction == 'mean':
            return torch.mean(mse_loss)
        elif self.reduction == 'sum':
            return torch.sum(mse_loss)
        else:
            return mse_loss


class QuantileLoss(nn.MSELoss):
    """
    Quantile Loss for HCNN Models.
    
    Asymmetric loss function that penalizes over-prediction and under-prediction differently.
    Useful for uncertainty quantification and robust regression.
    
    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile (0.5 = median, 0.9 = 90th percentile)
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, quantile: float = 0.5, reduction: str = 'mean'):
        super(QuantileLoss, self).__init__(reduction='none')
        self.quantile = quantile
        self.reduction = reduction
        
        if not 0 < quantile < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss between predictions and targets."""
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        error = y_true - y_pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


# Convenience function for creating loss functions
def get_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function for creating loss functions.
    
    Parameters
    ----------
    loss_type : str
        Type of loss function: 'mse', 'logcosh', 'huber', 'weighted_mse', 'quantile'
    **kwargs
        Additional arguments for the specific loss function
        
    Returns
    -------
    nn.Module
        Configured loss function
    """
    loss_functions = {
        'mse': nn.MSELoss,
        'logcosh': LogCoshLoss,
        'huber': HuberLoss,
        'weighted_mse': WeightedMSELoss,
        'quantile': QuantileLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)
