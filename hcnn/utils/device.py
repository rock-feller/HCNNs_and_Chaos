"""
Device management utilities for CUDA, MPS, and CPU.

This module provides utilities for automatic device detection and management,
supporting NVIDIA CUDA GPUs, Apple M-series with Metal Performance Shaders (MPS),
and CPU fallback.
"""

import torch
from typing import Optional, Union


# Global device state
_current_device: Optional[torch.device] = None


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for computation.
    
    Automatically detects and returns the best available device in order:
    1. CUDA (if available and prefer_gpu=True)
    2. MPS (if available and prefer_gpu=True) 
    3. CPU (fallback)
    
    Parameters
    ----------
    prefer_gpu : bool, default=True
        Whether to prefer GPU devices over CPU
        
    Returns
    -------
    torch.device
        The best available device
    """
    global _current_device
    
    if _current_device is not None:
        return _current_device
        
    if prefer_gpu:
        # Check for CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return device
            
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon) device")
            return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    print("Using CPU device")
    return device


def set_device(device: Union[str, torch.device]) -> torch.device:
    """
    Set the global device for the framework.
    
    Parameters
    ----------
    device : Union[str, torch.device]
        Device to set ('cuda', 'mps', 'cpu', or torch.device object)
        
    Returns
    -------
    torch.device
        The set device
        
    Raises
    ------
    RuntimeError
        If the specified device is not available
    """
    global _current_device
    
    if isinstance(device, str):
        device = torch.device(device)
        
    # Validate device availability
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")
        
    if device.type == "mps" and not (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS is not available on this system")
        
    _current_device = device
    print(f"Device set to: {device}")
    return device


def reset_device():
    """Reset the global device to None, forcing re-detection."""
    global _current_device
    _current_device = None


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns
    -------
    dict
        Dictionary containing device availability and information
    """
    info = {
        "cpu": True,  # CPU is always available
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": []
        },
        "mps": {
            "available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
    }
    
    # Get CUDA device info
    if info["cuda"]["available"]:
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["cuda"]["devices"].append({
                "id": i,
                "name": device_props.name,
                "memory": device_props.total_memory,
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
    
    return info


def move_to_device(
    obj: Union[torch.Tensor, torch.nn.Module, dict, list], 
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, torch.nn.Module, dict, list]:
    """
    Move tensor, model, or collection to specified device.
    
    Parameters
    ----------
    obj : Union[torch.Tensor, torch.nn.Module, dict, list]
        Object to move to device
    device : Optional[torch.device]
        Target device. If None, uses current global device
        
    Returns
    -------
    Union[torch.Tensor, torch.nn.Module, dict, list]
        Object moved to device
    """
    if device is None:
        device = get_device()
        
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    else:
        return obj


def get_memory_usage() -> dict:
    """
    Get current memory usage for available devices.
    
    Returns
    -------
    dict
        Memory usage information for each device type
    """
    usage = {}
    
    # CUDA memory usage
    if torch.cuda.is_available():
        usage["cuda"] = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            
            usage["cuda"][f"device_{i}"] = {
                "allocated": allocated,
                "reserved": reserved, 
                "total": total,
                "allocated_gb": allocated / 1024**3,
                "reserved_gb": reserved / 1024**3,
                "total_gb": total / 1024**3
            }
    
    # MPS memory usage (limited info available)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        usage["mps"] = {
            "available": True,
            "note": "Detailed memory info not available for MPS"
        }
    
    return usage
