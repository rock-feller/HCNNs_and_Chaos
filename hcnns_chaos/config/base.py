"""
Base configuration classes and utilities. 

This module provides the foundation for the configuration system,
including base classes, loading/saving utilities, and validation.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    """
    Base configuration class.
    
    This class provides the foundation for all configuration objects,
    with support for YAML serialization, validation, and merging.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(self.to_yaml())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str: str):
        """Create configuration from YAML string."""
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            return cls.from_yaml(f.read())
    
    def update(self, other: Union['Config', Dict[str, Any]]):
        """Update configuration with values from another config or dict."""
        if isinstance(other, Config):
            other = other.to_dict()
        
        for key, value in other.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def merge(self, other: Union['Config', Dict[str, Any]]) -> 'Config':
        """Create new configuration by merging with another config or dict."""
        merged_dict = self.to_dict()
        
        if isinstance(other, Config):
            other = other.to_dict()
        
        merged_dict.update(other)
        return self.__class__.from_dict(merged_dict)


def load_config(filepath: Union[str, Path], config_class: type = Config) -> Config:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to configuration file
    config_class : type, default=Config
        Configuration class to instantiate
        
    Returns
    -------
    Config
        Loaded configuration object
    """
    return config_class.load(filepath)


def save_config(config: Config, filepath: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : Config
        Configuration object to save
    filepath : Union[str, Path]
        Path to save configuration file
    """
    config.save(filepath)


def get_default_config_path(config_name: str) -> Path:
    """
    Get path to default configuration file.
    
    Parameters
    ----------
    config_name : str
        Name of configuration file (without extension)
        
    Returns
    -------
    Path
        Path to default configuration file
    """
    package_dir = Path(__file__).parent
    return package_dir / "defaults" / f"{config_name}.yaml"


def load_default_config(config_name: str, config_class: type = Config) -> Config:
    """
    Load default configuration.
    
    Parameters
    ----------
    config_name : str
        Name of default configuration
    config_class : type, default=Config
        Configuration class to instantiate
        
    Returns
    -------
    Config
        Loaded default configuration
    """
    config_path = get_default_config_path(config_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Default config not found: {config_path}")
    
    return load_config(config_path, config_class)


def validate_config(config: Config, required_fields: list) -> bool:
    """
    Validate that configuration has required fields.
    
    Parameters
    ----------
    config : Config
        Configuration to validate
    required_fields : list
        List of required field names
        
    Returns
    -------
    bool
        True if all required fields are present
        
    Raises
    ------
    ValueError
        If required fields are missing
    """
    config_dict = config.to_dict()
    missing_fields = []
    
    for field in required_fields:
        if field not in config_dict or config_dict[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")
    
    return True


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configurations.
    
    Parameters
    ----------
    *configs : Config
        Configuration objects to merge (later configs override earlier ones)
        
    Returns
    -------
    Config
        Merged configuration
    """
    if not configs:
        return Config()
    
    merged = configs[0]
    for config in configs[1:]:
        merged = merged.merge(config)
    
    return merged


def create_config_from_args(config_class: type, **kwargs) -> Config:
    """
    Create configuration from keyword arguments.
    
    Parameters
    ----------
    config_class : type
        Configuration class to instantiate
    **kwargs
        Configuration parameters
        
    Returns
    -------
    Config
        Created configuration object
    """
    return config_class(**kwargs)
