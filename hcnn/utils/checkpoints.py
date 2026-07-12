"""
Model checkpointing and loading utilities.

This module provides utilities for saving and loading model checkpoints,
including support for optimizer states, training metadata, and cleanup.
"""

import os
import glob
import torch
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    loss: float,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    model_name: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
    cleanup: bool = True
) -> str:
    """
    Save model checkpoint with training state.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    epoch : int
        Current epoch number
    loss : float
        Current loss value
    optimizer : torch.optim.Optimizer
        Optimizer state to save
    checkpoint_dir : str
        Directory to save checkpoint
    model_name : str, default="model"
        Name prefix for checkpoint file
    metadata : Optional[Dict[str, Any]]
        Additional metadata to save
    cleanup : bool, default=True
        Whether to remove previous checkpoints
        
    Returns
    -------
    str
        Path to saved checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_epoch_{epoch:04d}_loss_{loss:.6f}_{timestamp}.pth"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_name': model_name,
        'timestamp': timestamp,
        'model_config': {
            'model_type': getattr(model, 'model_type', 'unknown'),
            'n_obs_vars': getattr(model, 'n_obs_vars', None),
            'n_hid_vars': getattr(model, 'n_hid_vars', None),
            'n_ext_vars': getattr(model, 'n_ext_vars', None),
        }
    }
    
    # Add metadata if provided
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    # Save human-readable info
    info_file = filepath.replace('.pth', '_info.json')
    with open(info_file, 'w') as f:
        json.dump({
            'epoch': epoch,
            'loss': loss,
            'model_name': model_name,
            'timestamp': timestamp,
            'filepath': filepath,
            'model_config': checkpoint['model_config'],
            'metadata': metadata or {}
        }, f, indent=2)
    
    print(f"✅ Checkpoint saved: {filename}")
    
    # Cleanup old checkpoints if requested
    if cleanup:
        cleanup_old_checkpoints(checkpoint_dir, model_name, keep_latest=1)
    
    return filepath



def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    model : Optional[torch.nn.Module]
        Model to load state into
    optimizer : Optional[torch.optim.Optimizer]
        Optimizer to load state into
    device : Optional[torch.device]
        Device to load checkpoint on
        
    Returns
    -------
    Dict[str, Any]
        Loaded checkpoint data
    """
    if device is None:
        from .device import get_device
        device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state if model provided
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model state loaded from epoch {checkpoint['epoch']}")
    
    # Load optimizer state if optimizer provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Optimizer state loaded")
    
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str, model_name: str = "model") -> Optional[str]:
    """
    Find the latest checkpoint file for a given model.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    model_name : str, default="model"
        Model name prefix to search for
        
    Returns
    -------
    Optional[str]
        Path to latest checkpoint file, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (latest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]


def find_best_checkpoint(checkpoint_dir: str, model_name: str = "model") -> Optional[str]:
    """
    Find the checkpoint with the lowest loss for a given model.

    This function looks for checkpoints with "_best_" in the filename first,
    then falls back to searching all checkpoints for the lowest loss.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    model_name : str, default="model"
        Model name prefix to search for

    Returns
    -------
    Optional[str]
        Path to best checkpoint file, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # First, look for explicitly marked "best" checkpoints
    best_pattern = os.path.join(checkpoint_dir, f"{model_name}_best_*.pth")
    best_files = glob.glob(best_pattern)

    if best_files:
        # If multiple "best" files exist, return the one with lowest loss
        best_loss = float('inf')
        best_file = None

        for filepath in best_files:
            try:
                # Extract loss from filename
                filename = os.path.basename(filepath)
                loss_str = filename.split('_loss_')[1].split('_')[0]
                loss = float(loss_str)

                if loss < best_loss:
                    best_loss = loss
                    best_file = filepath
            except (IndexError, ValueError):
                # If filename parsing fails, load checkpoint to get loss
                try:
                    checkpoint = torch.load(filepath, map_location='cpu')
                    loss = checkpoint.get('loss', float('inf'))
                    if loss < best_loss:
                        best_loss = loss
                        best_file = filepath
                except Exception:
                    continue

        return best_file

    # Fallback: search all checkpoints for lowest loss (legacy behavior)
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None

    best_loss = float('inf')
    best_file = None

    for filepath in checkpoint_files:
        try:
            # Extract loss from filename
            filename = os.path.basename(filepath)
            loss_str = filename.split('_loss_')[1].split('_')[0]
            loss = float(loss_str)

            if loss < best_loss:
                best_loss = loss
                best_file = filepath
        except (IndexError, ValueError):
            # Skip files that don't match expected format
            continue

    return best_file


def cleanup_old_checkpoints(
    checkpoint_dir: str, 
    model_name: str = "model", 
    keep_latest: int = 3
):
    """
    Remove old checkpoint files, keeping only the latest N files.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    model_name : str, default="model"
        Model name prefix to clean up
    keep_latest : int, default=3
        Number of latest checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if len(checkpoint_files) <= keep_latest:
        return
    
    # Sort by modification time (latest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old files
    files_to_remove = checkpoint_files[keep_latest:]
    for filepath in files_to_remove:
        try:
            os.remove(filepath)
            # Also remove corresponding info file
            info_file = filepath.replace('.pth', '_info.json')
            if os.path.exists(info_file):
                os.remove(info_file)
            print(f"🗑️  Removed old checkpoint: {os.path.basename(filepath)}")
        except OSError as e:
            print(f"⚠️  Failed to remove {filepath}: {e}")


def list_checkpoints(checkpoint_dir: str, model_name: str = "model") -> list:
    """
    List all available checkpoints for a model.
    
    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints
    model_name : str, default="model"
        Model name prefix to search for
        
    Returns
    -------
    list
        List of checkpoint information dictionaries
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    pattern = os.path.join(checkpoint_dir, f"{model_name}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    checkpoints = []
    for filepath in checkpoint_files:
        try:
            # Load basic info without loading full checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'loss': checkpoint.get('loss', 'unknown'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'model_type': checkpoint.get('model_config', {}).get('model_type', 'unknown'),
                'file_size': os.path.getsize(filepath),
                'modified_time': datetime.fromtimestamp(os.path.getmtime(filepath))
            }
            checkpoints.append(info)
        except Exception as e:
            print(f"⚠️  Failed to read checkpoint {filepath}: {e}")
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'] if isinstance(x['epoch'], int) else 0)
    return checkpoints
