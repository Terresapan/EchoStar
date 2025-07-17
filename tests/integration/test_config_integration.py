#!/usr/bin/env python3
"""
Integration tests for the EchoStar configuration system.
Tests configuration loading, validation, environment overrides, and integration with application components.
"""

import os
import json
import yaml
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

# Import configuration system
from config.manager import ConfigManager, get_config_manager, get_config
from config.models import EchoStarConfig, MemoryConfig, RoutingConfig, LLMConfig, LoggingConfig


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear any existing global config manager
        import config.manager
        config.manager._config_manager = None
        
        # Clear environment variables
        self.original_env = {}
        for key in list(os.environ.keys()):
            if key.startswith('ECHOSTAR_'):
                self.original_env[key] = os.environ.pop(key)
    
    def teardown_method(self):
        """Clean up after each test."""
        # Restore environment variables
        for key, value in self.original_env.items():
            os.environ[key] = value
        
        # Clear global config manager
        import config.manager
        config.manager._config_manager = None
    
    def test_default_configuration_loading(self):
        """Test loading default configuration from config/default.yaml."""
        # This should load the default configuration
        config_manager = ConfigManager()
        
        assert config_manager.config is not None
        assert isinstance(config_manager.config, EchoStarConfig)
        
        # Test specific default values
        assert config_manager.get('memory.turns_to_summarize') == 10
        assert config_manager.get('memory.search_limit') == 10
        assert config_manager.get('memory.enable_condensation') is True
        assert config_manager.get('routing.roleplay_threshold') == 2
        assert config_manager.get('llm.model_name') == "openai:gpt-4.1-mini"
        assert config_manager.get('llm.temperature') == 0.7
        assert config_manager.get('logging.level') == "INFO"
    
    def test_custom_yaml_configuration(self):
        """Test loading custom YAML configuration file."""
        custom_config = {
            'memory': {
                'turns_to_summarize': 15,
                'search_limit': 20,
                'enable_condensation': False
            },
            'routing': {
                'roleplay_threshold': 5,
                'enable_fallback': False
            },
            'llm': {
                'model_name': 'openai:gpt-4',
                'temperature': 0.5,
                'max_tokens': 1000
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'text'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            temp_file = f.name
        
        try:
            config_manager = ConfigManager(temp_file)
            
            # Test custom values are loaded
            assert config_manager.get('memory.turns_to_summarize') == 15
            assert config_manager.get('memory.search_limit') == 20
            assert config_manager.get('memory.enable_condensation') is False
            assert config_manager.get('routing.roleplay_threshold') == 5
            assert config_manager.get('routing.enable_fallback') is False
            assert config_manager.get('llm.model_name') == 'openai:gpt-4'
            assert config_manager.get('llm.temperature') == 0.5
            assert config_manager.get('llm.max_tokens') == 1000
            assert config_manager.get('logging.level') == 'DEBUG'
            assert config_manager.get('logging.format') == 'text'
        finally:
            os.unlink(temp_file)
    
    def test_json_configuration(self):
        """Test loading JSON configuration file."""
        custom_config = {
            'memory': {
                'turns_to_summarize': 25,
                'search_limit': 15
            },
            'llm': {
                'model_name': 'anthropic:claude-3',
                'temperature': 0.9
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            temp_file = f.name
        
        try:
            config_manager = ConfigManager(temp_file)
            
            # Test JSON values are loaded
            assert config_manager.get('memory.turns_to_summarize') == 25
            assert config_manager.get('memory.search_limit') == 15
            assert config_manager.get('llm.model_name') == 'anthropic:claude-3'
            assert config_manager.get('llm.temperature') == 0.9
            
            # Test defaults are preserved for unspecified values
            assert config_manager.get('routing.roleplay_threshold') == 2
            assert config_manager.get('logging.level') == "INFO"
        finally:
            os.unlink(temp_file)
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ['ECHOSTAR_MEMORY__TURNS_TO_SUMMARIZE'] = '20'
        os.environ['ECHOSTAR_MEMORY__ENABLE_CONDENSATION'] = 'false'
        os.environ['ECHOSTAR_ROUTING__ROLEPLAY_THRESHOLD'] = '7'
        os.environ['ECHOSTAR_LLM__TEMPERATURE'] = '0.3'
        os.environ['ECHOSTAR_LLM__MAX_TOKENS'] = '2000'
        os.environ['ECHOSTAR_LOGGING__LEVEL'] = 'DEBUG'
        
        config_manager = ConfigManager()
        
        # Test environment overrides are applied
        assert config_manager.get('memory.turns_to_summarize') == 20
        assert config_manager.get('memory.enable_condensation') is False
        assert config_manager.get('routing.roleplay_threshold') == 7
        assert config_manager.get('llm.temperature') == 0.3
        assert config_manager.get('llm.max_tokens') == 2000
        assert config_manager.get('logging.level') == 'DEBUG'
        
        # Test non-overridden values remain default
        assert config_manager.get('memory.search_limit') == 10
        assert config_manager.get('routing.enable_fallback') is True
    
    def test_invalid_configuration_validation(self):
        """Test validation of invalid configuration values."""
        # Test invalid memory configuration
        invalid_config = {
            'memory': {
                'turns_to_summarize': 0,  # Below minimum
                'search_limit': 100,      # Above maximum
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                ConfigManager(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_invalid_logging_level(self):
        """Test validation of invalid logging level."""
        invalid_config = {
            'logging': {
                'level': 'INVALID_LEVEL'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                ConfigManager(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_missing_configuration_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager('/nonexistent/config.yaml')
    
    def test_malformed_yaml_file(self):
        """Test handling of malformed YAML file."""
        malformed_yaml = """
        memory:
          turns_to_summarize: 10
        routing:
          - invalid: yaml: structure
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid configuration file format"):
                ConfigManager(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_configuration_reload(self):
        """Test configuration reloading functionality."""
        # Create initial config
        initial_config = {
            'memory': {'turns_to_summarize': 10},
            'llm': {'temperature': 0.7}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(initial_config, f)
            temp_file = f.name
        
        try:
            config_manager = ConfigManager(temp_file)
            assert config_manager.get('memory.turns_to_summarize') == 10
            assert config_manager.get('llm.temperature') == 0.7
            
            # Update config file
            updated_config = {
                'memory': {'turns_to_summarize': 20},
                'llm': {'temperature': 0.5}
            }
            
            with open(temp_file, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Reload configuration
            config_manager.reload_config()
            
            # Test updated values
            assert config_manager.get('memory.turns_to_summarize') == 20
            assert config_manager.get('llm.temperature') == 0.5
        finally:
            os.unlink(temp_file)
    
    def test_global_config_manager_singleton(self):
        """Test that global config manager is a singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
        
        # Test convenience function
        value1 = get_config('memory.turns_to_summarize')
        value2 = manager1.get('memory.turns_to_summarize')
        
        assert value1 == value2
    
    def test_configuration_sections_access(self):
        """Test accessing configuration sections directly."""
        config_manager = ConfigManager()
        
        memory_config = config_manager.get_memory_config()
        routing_config = config_manager.get_routing_config()
        llm_config = config_manager.get_llm_config()
        logging_config = config_manager.get_logging_config()
        
        assert isinstance(memory_config, MemoryConfig)
        assert isinstance(routing_config, RoutingConfig)
        assert isinstance(llm_config, LLMConfig)
        assert isinstance(logging_config, LoggingConfig)
        
        # Test values
        assert memory_config.turns_to_summarize == 10
        assert routing_config.roleplay_threshold == 2
        assert llm_config.model_name == "openai:gpt-4.1-mini"
        assert logging_config.level == "INFO"
    
    def test_dot_notation_access(self):
        """Test accessing nested configuration values with dot notation."""
        config_manager = ConfigManager()
        
        # Test various dot notation patterns
        assert config_manager.get('memory.turns_to_summarize') == 10
        assert config_manager.get('memory.search_limit') == 10
        assert config_manager.get('routing.roleplay_threshold') == 2
        assert config_manager.get('llm.model_name') == "openai:gpt-4.1-mini"
        assert config_manager.get('logging.level') == "INFO"
        
        # Test default values
        assert config_manager.get('nonexistent.key', 'default') == 'default'
        assert config_manager.get('memory.nonexistent', 42) == 42
    
    def test_configuration_validation_method(self):
        """Test configuration validation method."""
        config_manager = ConfigManager()
        
        # Valid configuration should pass validation
        assert config_manager.validate_config() is True
        
        # Test with invalid configuration
        config_manager._config = None
        assert config_manager.validate_config() is False


def test_hardcoded_values_replacement():
    """Test that hardcoded values have been properly replaced with configuration calls."""
    # Clear any environment variables that might affect this test
    env_vars_to_clear = [key for key in os.environ.keys() if key.startswith('ECHOSTAR_')]
    original_env = {}
    for key in env_vars_to_clear:
        original_env[key] = os.environ.pop(key)
    
    try:
        # Clear global config manager to ensure fresh load
        import config.manager
        config.manager._config_manager = None
        
        # Import modules that should use configuration
        import nodes
        import graph
        import streamlit_app
        
        # Test that configuration manager is imported and used
        assert hasattr(nodes, 'get_config_manager')
        assert hasattr(graph, 'get_config_manager')
        assert hasattr(streamlit_app, 'get_config_manager')
        
        # Test that configuration values are being used
        config_manager = get_config_manager()
        
        # Check that the values match what's expected in the code
        memory_config = config_manager.get_memory_config()
        routing_config = config_manager.get_routing_config()
        llm_config = config_manager.get_llm_config()
        
        assert memory_config.turns_to_summarize == 10
        assert memory_config.search_limit == 10
        assert memory_config.procedural_search_limit == 5
        assert routing_config.roleplay_threshold == 2
        assert llm_config.model_name == "openai:gpt-4.1-mini"
        assert llm_config.temperature == 0.7
    finally:
        # Restore environment variables
        for key, value in original_env.items():
            os.environ[key] = value


def test_environment_specific_configurations():
    """Test loading environment-specific configuration files."""
    # Test development configuration
    dev_config_path = Path("config/development.yaml")
    if dev_config_path.exists():
        config_manager = ConfigManager(str(dev_config_path))
        assert config_manager.validate_config()
    
    # Test production configuration
    prod_config_path = Path("config/production.yaml")
    if prod_config_path.exists():
        config_manager = ConfigManager(str(prod_config_path))
        assert config_manager.validate_config()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])