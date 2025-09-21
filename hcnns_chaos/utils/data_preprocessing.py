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
    Handles data normalization and denormalization for chaotic system data.
    """
    
    def __init__(self):
        self.data_normalization = True
    
    def scale_to_normalize(
        self, 
        data: torch.Tensor, 
        scaling_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize data around 0 and multiply by scaling factor.
        
        Formula: scaled_data = scaling_factor * (data - data_averages)
        
        Args:
            data: Input tensor of shape (samples, features)
            scaling_factor: Scaling factor for normalization
            
        Returns:
            Tuple of (scaled_data, data_averages)
            - scaled_data: Normalized data of shape (samples, features)
            - data_averages: Mean values of shape (features,)
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
        """
        Convert scaled data back to original scale.
        
        Args:
            scaled_data: Normalized data of shape (samples, features)
            scaling_factor: Original scaling factor used
            data_averages: Original mean values of shape (features,)
            
        Returns:
            Original scale data as numpy array
        """
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
        noise = torch.uniform(-noise_level, noise_level, trajectories.shape)
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
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
        device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    
    # Move data to device
    raw_data = raw_data.to(device)
    
    # Normalize data
    normalizer = NormalizationStrategy()
    normalized_data, data_averages = normalizer.scale_to_normalize(
        raw_data, scaling_factor
    )
    
    # Add noise if requested
    if add_noise:
        noise_adder = NoisificationStrategy()
        normalized_data = noise_adder.add_gaussian_noise(
            normalized_data, noise_sigma
        )
    
    # Split into train/test
    split_idx = int(len(normalized_data) * train_ratio)
    train_data = normalized_data[:split_idx]
    test_data = normalized_data[split_idx:]
    
    # Prepare for fully unfolded mode (add batch dimension)
    fully_unfolded_train = train_data.unsqueeze(0)  # [1, seq_len, features]
    
    return {
        'raw_data': raw_data,
        'normalized_data': normalized_data,
        'data_averages': data_averages,
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
