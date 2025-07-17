"""
Configuration Manager for EchoStar AI Simulator.
Handles loading, validation, and access to configuration values.
"""

import os
import json
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from pydantic import ValidationError

from .models import EchoStarConfig


class ConfigManager:
    """
    Centralized configuration manager that supports:
    - YAML/JSON file loading
    - Environment variable overrides
    - Configuration validation
    - Runtime configuration reloading
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, uses default locations.
        """
        self._config: Optional[EchoStarConfig] = None
        self._config_file = config_file
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Start with default configuration
        config_data = {}
        
        # Load from file if specified or found
        if self._config_file:
            config_data = self._load_config_file(self._config_file)
        else:
            # Try default locations
            default_files = [
                "config/default.yaml",
                "config/default.yml", 
                "config/default.json",
                "config.yaml",
                "config.yml",
                "config.json"
            ]
            
            for file_path in default_files:
                if Path(file_path).exists():
                    config_data = self._load_config_file(file_path)
                    self._config_file = file_path
                    break
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Validate and create configuration object
        try:
            self._config = EchoStarConfig(**config_data)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    # Try to detect format by content
                    content = f.read()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return yaml.safe_load(content) or {}
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {e}")
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should be prefixed with ECHOSTAR_ and use
        double underscores to separate nested keys.
        
        Examples:
            ECHOSTAR_MEMORY__TURNS_TO_SUMMARIZE=15
            ECHOSTAR_LLM__MODEL_NAME=openai:gpt-4
            ECHOSTAR_LOGGING__LEVEL=DEBUG
        
        Args:
            config_data: Base configuration dictionary
            
        Returns:
            Configuration dictionary with environment overrides applied
        """
        env_prefix = "ECHOSTAR_"
        
        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue
            
            # Remove prefix and convert to lowercase
            config_key = key[len(env_prefix):].lower()
            
            # Split nested keys
            key_parts = config_key.split('__')
            
            # Navigate/create nested structure
            current_dict = config_data
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            # Set the value with type conversion
            final_key = key_parts[-1]
            current_dict[final_key] = self._convert_env_value(value)
        
        return config_data
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value with appropriate type
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Return as string if conversion fails
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'memory.turns_to_summarize')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Examples:
            config.get('memory.turns_to_summarize')
            config.get('llm.model_name')
            config.get('logging.level', 'INFO')
        """
        if self._config is None:
            return default
        
        # Split the key by dots to handle nested access
        key_parts = key.split('.')
        current_value = self._config
        
        try:
            for part in key_parts:
                if hasattr(current_value, part):
                    current_value = getattr(current_value, part)
                else:
                    return default
            return current_value
        except (AttributeError, KeyError):
            return default
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if self._config is None:
                return False
            # Pydantic validation happens during object creation
            # Additional custom validation can be added here
            return True
        except Exception:
            return False
    
    def reload_config(self) -> None:
        """
        Reload configuration from file and environment variables.
        
        Raises:
            ValueError: If configuration validation fails after reload
        """
        self._load_config()
    
    @property
    def config(self) -> Optional[EchoStarConfig]:
        """Get the current configuration object."""
        return self._config
    
    def get_memory_config(self):
        """Get memory configuration section."""
        return self._config.memory if self._config else None
    
    def get_routing_config(self):
        """Get routing configuration section."""
        return self._config.routing if self._config else None
    
    def get_llm_config(self):
        """Get LLM configuration section."""
        return self._config.llm if self._config else None
    
    def get_logging_config(self):
        """Get logging configuration section."""
        return self._config.logging if self._config else None


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_file: Path to configuration file (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    Convenience function to get configuration values.
    
    Args:
        key: Configuration key in dot notation
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    return get_config_manager().get(key, default)