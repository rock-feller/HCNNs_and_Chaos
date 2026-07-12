"""
Configuration management for the HCNN framework.

This module provides a robust configuration system using YAML files
for managing model parameters, training settings, and experiment configurations.
"""

from . import base

from .base import Config, load_config, save_config

__all__ = [
    "base",
    "Config",
    "load_config",
    "save_config",
]
