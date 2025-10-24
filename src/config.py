"""
Configuration management for depth estimation project.

This module provides configuration loading, validation, and management
using YAML files with sensible defaults.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for depth estimation models."""
    name: str = "small"
    device: Optional[str] = None
    force_reload: bool = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figsize: tuple = (15, 6)
    colormap: str = "inferno"
    overlay_alpha: float = 0.6
    dpi: int = 300
    save_format: str = "png"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    visualization: VisualizationConfig
    data: DataConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            visualization=VisualizationConfig(**config_dict.get("visualization", {})),
            data=DataConfig(**config_dict.get("data", {})),
            logging=LoggingConfig(**config_dict.get("logging", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            "model": asdict(self.model),
            "visualization": asdict(self.visualization),
            "data": asdict(self.data),
            "logging": asdict(self.logging)
        }


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                logger.info(f"Configuration loaded from: {self.config_path}")
                return AppConfig.from_dict(config_dict)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Config file not found at {self.config_path}, using defaults")
        
        return self._create_default_config()
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        return AppConfig(
            model=ModelConfig(),
            visualization=VisualizationConfig(),
            data=DataConfig(),
            logging=LoggingConfig()
        )
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. Uses current path if None.
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert config to dict and handle tuples
            config_dict = self.config.to_dict()
            
            # Convert tuples to lists for YAML serialization
            def convert_tuples(obj):
                if isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_tuples(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tuples(item) for item in obj]
                else:
                    return obj
            
            config_dict = convert_tuples(config_dict)
            
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration updates
        """
        config_dict = self.config.to_dict()
        
        for key, value in kwargs.items():
            if key in config_dict:
                if isinstance(value, dict) and isinstance(config_dict[key], dict):
                    config_dict[key].update(value)
                else:
                    config_dict[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        self.config = AppConfig.from_dict(config_dict)
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.config.logging
        
        # Configure logging level
        level = getattr(logging, log_config.level.upper(), logging.INFO)
        
        # Configure logging format
        logging.basicConfig(
            level=level,
            format=log_config.format,
            force=True
        )
        
        # Add file handler if specified
        if log_config.file_path:
            file_handler = logging.FileHandler(log_config.file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(log_config.format))
            
            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_config.file_path}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager(config_path)
    return manager.get_config()


def create_default_config_file(config_path: Union[str, Path] = "config/config.yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to create the config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    manager = ConfigManager()
    manager.save_config(config_path)
    
    logger.info(f"Default configuration file created: {config_path}")


if __name__ == "__main__":
    # Create default configuration file
    create_default_config_file()
    
    # Load and display configuration
    config = load_config()
    print("Current configuration:")
    print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))
