"""
Data Preprocessing Utilities for HCNN Training

This module provides comprehensive data preprocessing utilities including normalization,
noise addition, and sliding window dataset creation for chaotic system data.
"""

import torch
import numpy as np
import random
from typing import Tuple, Literal, Union, Optional
from torch.utils.data import Dataset, DataLoader


class NormalizationStrategy:
    """
    Mean-centering + scaling for chaotic-system data.

    Transform:  ``scaled = scaling_factor * (data - mean)``.

    IMPORTANT (leakage): the centering ``mean`` must be estimated on the TRAINING
    split only and then applied to validation/test. Use the stateful
    :meth:`fit` / :meth:`transform` / :meth:`inverse_transform` API for this - it
    stores the fitted mean so the same statistics are reused everywhere:

        norm = NormalizationStrategy()
        train_scaled = norm.fit(train, scaling_factor=0.02).transform(train)
        test_scaled  = norm.transform(test)          # uses TRAIN mean - no leakage
        recovered    = norm.inverse_transform(train_scaled)

    The older stateless :meth:`scale_to_normalize` / :meth:`scale_back_to_originals`
    helpers are retained for backward compatibility, but they estimate the mean on
    whatever array they are handed - only ever pass them the training split.
    """

    def __init__(self):
        self.data_normalization = True
        self.mean_: Optional[torch.Tensor] = None
        self.scaling_factor_: Optional[float] = None

    # -- Stateful, leakage-safe API ---------------------------------------
    def fit(self, train_data: torch.Tensor, scaling_factor: float) -> "NormalizationStrategy":
        """Estimate the centering mean on the TRAINING split and store it."""
        self.mean_ = train_data.mean(dim=0)
        self.scaling_factor_ = scaling_factor
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the fitted transform: ``scaling_factor * (data - train_mean)``."""
        if self.mean_ is None or self.scaling_factor_ is None:
            raise RuntimeError("NormalizationStrategy.transform called before fit().")
        return self.scaling_factor_ * (data - self.mean_.to(data.device))

    def fit_transform(self, train_data: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        return self.fit(train_data, scaling_factor).transform(train_data)

    def inverse_transform(self, scaled_data: torch.Tensor) -> torch.Tensor:
        """Invert the transform back to the original scale."""
        if self.mean_ is None or self.scaling_factor_ is None:
            raise RuntimeError("NormalizationStrategy.inverse_transform called before fit().")
        return (scaled_data / self.scaling_factor_) + self.mean_.to(scaled_data.device)

    # -- Stateless helpers (backward compatible; pass TRAIN only) ---------
    def scale_to_normalize(
        self,
        data: torch.Tensor,
        scaling_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stateless mean-center + scale. Returns ``(scaled_data, data_mean)``.

        WARNING: estimates the mean on ``data`` itself - only pass the training
        split, or you leak test statistics into training. Prefer fit/transform.
        """
        data_averages = data.mean(dim=0)
        scaled_data = scaling_factor * (data - data_averages)
        return scaled_data, data_averages

    def scale_back_to_originals(
        self,
        scaled_data: torch.Tensor,
        scaling_factor: float,
        data_averages: torch.Tensor
    ) -> np.ndarray:
        """Convert scaled data back to the original scale (numpy)."""
        original_data = (scaled_data / scaling_factor) + data_averages
        return original_data.detach().cpu().numpy()


class NoisificationStrategy:
    """
    Handles noise addition to chaotic system trajectories.
    """
    
    def __init__(self):
        self.add_noise = True
    
    def add_gaussian_noise(
        self, 
        trajectories: torch.Tensor, 
        sigma: float
    ) -> torch.Tensor:
        """
        Add Gaussian noise to chaotic trajectories.
        
        Args:
            trajectories: Input trajectories of shape (samples, 3)
            sigma: Standard deviation of Gaussian noise
            
        Returns:
            Noisy trajectories of same shape
        """
        n_samples = len(trajectories)
        
        # Generate noise for each dimension
        noise_x = np.random.normal(0, sigma, n_samples)
        noise_y = np.random.normal(0, sigma, n_samples)
        noise_z = np.random.normal(0, sigma, n_samples)
        
        # Stack noise and add to trajectories
        noise = torch.tensor(
            np.vstack([noise_x, noise_y, noise_z]).T, 
            dtype=torch.float32
        )
        
        noisy_trajectories = trajectories + noise
        return noisy_trajectories
    
    def add_uniform_noise(
        self, 
        trajectories: torch.Tensor, 
        noise_level: float
    ) -> torch.Tensor:
        """
        Add uniform noise to trajectories.
        
        Args:
            trajectories: Input trajectories of shape (samples, 3)
            noise_level: Maximum amplitude of uniform noise
            
        Returns:
            Noisy trajectories of same shape
        """
        # torch has no torch.uniform; sample U(-noise_level, noise_level) correctly
        noise = (torch.rand(trajectories.shape, device=trajectories.device,
                            dtype=trajectories.dtype) * 2 - 1) * noise_level
        return trajectories + noise


class SlidingWindowDataset(Dataset):
    """
    Dataset class for creating sliding windows from time series data.
    """
    
    def __init__(self, data: torch.Tensor, window_size: int):
        """
        Initialize sliding window dataset.
        
        Args:
            data: Time series data of shape (samples, features)
            window_size: Size of sliding window
        """
        self.data = data
        self.window_size = window_size
        # Keep windows on CPU by default (standard for a Dataset); the DataLoader /
        # training loop moves batches to the model's device. Auto-grabbing MPS/CUDA
        # here forces the whole windowed tensor onto the accelerator up front.
        self.device = data.device
        self.windowed_data = self._create_sliding_windows()
    
    def _create_sliding_windows(self) -> torch.Tensor:
        """
        Create sliding windows from data.
        
        Returns:
            Windowed data of shape (num_windows, window_size, features)
        """
        num_windows = len(self.data) - self.window_size + 1
        windows = []
        
        for i in range(num_windows):
            window = self.data[i:i + self.window_size]
            windows.append(window)
        
        return torch.stack(windows).float().to(self.device)
    
    def __len__(self):
        return len(self.windowed_data)
    
    def __getitem__(self, idx):
        return self.windowed_data[idx]
    
    def create_context_forecast_split(
        self, 
        context_size: int, 
        forecast_size: int,
        location: Literal["random", "last"] = "last"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create context window and forecast target from data.
        
        Args:
            context_size: Size of context window
            forecast_size: Size of forecast window
            location: Where to extract the window ("random" or "last")
            
        Returns:
            Tuple of (context_window, forecast_target)
        """
        total_size = context_size + forecast_size
        full_windows = self._create_sliding_windows_with_size(total_size)
        
        if location == "random":
            window_idx = random.randint(0, len(full_windows) - 1)
        elif location == "last":
            window_idx = -1
        else:
            raise ValueError("location must be 'random' or 'last'")
        
        selected_window = full_windows[window_idx]
        context_window = selected_window[:context_size]
        forecast_target = selected_window[context_size:]
        
        return context_window, forecast_target
    
    def _create_sliding_windows_with_size(self, window_size: int) -> torch.Tensor:
        """Helper method to create windows with specific size."""
        num_windows = len(self.data) - window_size + 1
        windows = []
        
        for i in range(num_windows):
            window = self.data[i:i + window_size]
            windows.append(window)
        
        return torch.stack(windows).float().to(self.device)


class InputTargetDataset(Dataset):
    """
    Dataset for input-target pairs in time series prediction.
    """
    
    def __init__(self, data: torch.Tensor, window_size: int):
        """
        Initialize input-target dataset.
        
        Args:
            data: Time series data of shape (samples, features)
            window_size: Size of input window
        """
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx: int):
        input_seq = self.data[idx:idx + self.window_size]
        target_seq = self.data[idx + 1:idx + self.window_size + 1]
        return input_seq, target_seq
    
    @classmethod
    def create_dataloader(
        cls, 
        data: torch.Tensor, 
        window_size: int, 
        batch_size: int, 
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create DataLoader for input-target pairs.
        
        Args:
            data: Time series data of shape (samples, features)
            window_size: Size of input window
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader object
        """
        dataset = cls(data, window_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_chaotic_data(
    raw_data: torch.Tensor,
    scaling_factor: float = 0.02,
    train_ratio: float = 0.75,
    add_noise: bool = False,
    noise_sigma: float = 0.01,
    device: Optional[torch.device] = None
) -> dict:
    """
    Comprehensive data preparation pipeline for chaotic system data.
    
    Args:
        raw_data: Raw chaotic system data of shape (samples, features)
        scaling_factor: Scaling factor for normalization
        train_ratio: Ratio of data to use for training
        add_noise: Whether to add noise to data
        noise_sigma: Standard deviation of noise (if add_noise=True)
        device: Device to move data to
        
    Returns:
        Dictionary containing processed data and metadata
    """
    if device is None:
        device = torch.device("cpu")

    # Move data to device
    raw_data = raw_data.to(device)

    # --- Split FIRST, then fit normalization on TRAIN only (no leakage) ---
    split_idx = int(len(raw_data) * train_ratio)
    raw_train = raw_data[:split_idx]
    raw_test = raw_data[split_idx:]

    normalizer = NormalizationStrategy().fit(raw_train, scaling_factor)
    data_averages = normalizer.mean_
    train_data = normalizer.transform(raw_train)
    test_data = normalizer.transform(raw_test)
    # Full normalized series, reconstructed from the two splits (train stats only)
    normalized_data = torch.cat([train_data, test_data], dim=0)

    # Add noise if requested (random perturbation; applied post-split)
    if add_noise:
        noise_adder = NoisificationStrategy()
        train_data = noise_adder.add_gaussian_noise(train_data, noise_sigma)
        test_data = noise_adder.add_gaussian_noise(test_data, noise_sigma)

    # Prepare for fully unfolded mode (add batch dimension)
    fully_unfolded_train = train_data.unsqueeze(0)  # [1, seq_len, features]

    return {
        'raw_data': raw_data,
        'normalized_data': normalized_data,
        'data_averages': data_averages,
        'normalizer': normalizer,
        'train_data': train_data,
        'test_data': test_data,
        'fully_unfolded_train': fully_unfolded_train,
        'metadata': {
            'total_length': len(raw_data),
            'train_length': len(train_data),
            'test_length': len(test_data),
            'n_features': raw_data.shape[1],
            'scaling_factor': scaling_factor,
            'train_ratio': train_ratio,
            'noise_added': add_noise,
            'noise_sigma': noise_sigma if add_noise else None,
            'device': str(device)
        }
    }
