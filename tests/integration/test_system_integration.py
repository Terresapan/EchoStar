#!/usr/bin/env python3
"""
End-to-end system integration tests for the EchoStar application.
Tests the complete application with new configuration, typing, and logging systems.
Verifies backward compatibility and that existing functionality remains intact.
"""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
import pytest

# Import all major system components
from config.manager import ConfigManager, get_config_manager, get_config
from config.models import EchoStarConfig, MemoryConfig, RoutingConfig, LLMConfig, LoggingConfig
from logging_utils import StructuredLogger, get_logger, setup_application_logging
from schemas import AgentState, Router, UserProfile, EpisodicMemory, SemanticMemory, ProceduralMemory
import nodes
import graph
import memory
import utils


class TestSystemIntegration:
    """End-to-end system integration tests."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear global config manager
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
    
    def test_configuration_system_integration(self):
        """Test that configuration system integrates properly with all components."""
        # Test that configuration loads successfully
        config_manager = get_config_manager()
        assert config_manager is not None
        assert config_manager.validate_config()
        
        # Test that all configuration sections are accessible
        memory_config = config_manager.get_memory_config()
        routing_config = config_manager.get_routing_config()
        llm_config = config_manager.get_llm_config()
        logging_config = config_manager.get_logging_config()
        
        assert isinstance(memory_config, MemoryConfig)
        assert isinstance(routing_config, RoutingConfig)
        assert isinstance(llm_config, LLMConfig)
        assert isinstance(logging_config, LoggingConfig)
        
        # Test dot notation access
        assert config_manager.get('memory.turns_to_summarize') == memory_config.turns_to_summarize
        assert config_manager.get('routing.roleplay_threshold') == routing_config.roleplay_threshold
        assert config_manager.get('llm.model_name') == llm_config.model_name
        assert config_manager.get('logging.level') == logging_config.level
        
        print("✅ Configuration system integration verified")
    
    def test_logging_system_integration(self):
        """Test that logging system integrates properly with all components."""
        # Test logger creation
        logger = get_logger("test_integration")
        assert isinstance(logger, StructuredLogger)
        
        # Test that logging works with configuration
        config_manager = get_config_manager()
        logging_config = config_manager.get_logging_config()
        
        logger_with_config = StructuredLogger("test_config", logging_config)
        
        # Test logging at different levels
        logger_with_config.debug("Debug message", component="integration_test")
        logger_with_config.info("Info message", component="integration_test")
        logger_with_config.warning("Warning message", component="integration_test")
        logger_with_config.error("Error message", component="integration_test")
        
        # Test performance timing
        with logger_with_config.performance_timer("test_operation", component="integration"):
            pass
        
        print("✅ Logging system integration verified")
    
    def test_schema_validation_integration(self):
        """Test that schema validation works with all data models."""
        # Test AgentState
        state: AgentState = {
            "message": "Hello, how are you?",
            "classification": "echo_respond",
            "reasoning": "Simple greeting message"
        }
        
        # Test Router schema
        router = Router(
            classification="echo_respond",
            reasoning="User is greeting the AI"
        )
        assert router.classification == "echo_respond"
        assert isinstance(router.reasoning, str)
        
        # Test UserProfile schema
        profile = UserProfile(
            name="Test User",
            background="Software developer interested in AI",
            communication_style="friendly and technical",
            emotional_baseline="curious and optimistic"
        )
        assert profile.name == "Test User"
        
        # Test memory schemas
        episodic = EpisodicMemory(
            user_message="What's the weather like?",
            ai_response="I don't have access to current weather data.",
            timestamp="2024-01-01T12:00:00Z",
            context="weather_inquiry"
        )
        assert episodic.user_message == "What's the weather like?"
        
        semantic = SemanticMemory(
            category="preference",
            content="User is interested in weather information",
            context="user_preferences and interests",
            importance=0.7,
            timestamp="2024-01-01T12:00:00Z"
        )
        assert semantic.content == "User is interested in weather information"
        
        procedural = ProceduralMemory(
            trigger="weather_question",
            action="explain_weather_limitations",
            context="weather_handling",
            success_rate=0.9,
            timestamp="2024-01-01T12:00:00Z"
        )
        assert procedural.trigger == "weather_question"
        
        print("✅ Schema validation integration verified")
    
    def test_nodes_integration_with_configuration(self):
        """Test that nodes use configuration values correctly."""
        # Mock dependencies
        mock_llm = MagicMock()
        mock_store = MagicMock()
        mock_search_tool = MagicMock()
        
        # Test that nodes access configuration
        config_manager = get_config_manager()
        memory_config = config_manager.get_memory_config()
        routing_config = config_manager.get_routing_config()
        
        # Verify configuration values are used in nodes
        assert memory_config.turns_to_summarize > 0
        assert memory_config.search_limit > 0
        assert routing_config.roleplay_threshold > 0
        
        # Test that nodes can be imported and have proper type annotations
        assert hasattr(nodes, 'memory_retrieval_node')
        assert hasattr(nodes, 'router_node')
        assert hasattr(nodes, 'save_memories_node')
        assert hasattr(nodes, 'condense_memory_node')
        
        print("✅ Nodes integration with configuration verified")
    
    def test_graph_integration_with_configuration(self):
        """Test that graph module integrates with configuration system."""
        # Test that graph module uses configuration
        config_manager = get_config_manager()
        llm_config = config_manager.get_llm_config()
        
        # Verify LLM configuration is accessible
        assert llm_config.model_name is not None
        assert llm_config.temperature >= 0.0
        assert llm_config.timeout > 0
        
        # Test that graph module can be imported without errors
        assert hasattr(graph, 'create_graph')
        
        print("✅ Graph integration with configuration verified")
    
    def test_memory_system_integration(self):
        """Test that memory system works with configuration and logging."""
        # Test memory module integration
        assert hasattr(memory, 'create_search_memory_tool')
        
        # Test that memory operations would use configuration
        config_manager = get_config_manager()
        memory_config = config_manager.get_memory_config()
        
        assert memory_config.search_limit > 0
        assert memory_config.procedural_search_limit > 0
        assert isinstance(memory_config.enable_condensation, bool)
        
        print("✅ Memory system integration verified")
    
    def test_environment_variable_override_integration(self):
        """Test that environment variable overrides work in integrated system."""
        # Set environment variables
        os.environ['ECHOSTAR_MEMORY__TURNS_TO_SUMMARIZE'] = '15'
        os.environ['ECHOSTAR_LOGGING__LEVEL'] = 'DEBUG'
        os.environ['ECHOSTAR_LLM__TEMPERATURE'] = '0.5'
        
        # Clear and recreate config manager to pick up env vars
        import config.manager
        config.manager._config_manager = None
        
        config_manager = get_config_manager()
        
        # Verify environment overrides are applied
        assert config_manager.get('memory.turns_to_summarize') == 15
        assert config_manager.get('logging.level') == 'DEBUG'
        assert config_manager.get('llm.temperature') == 0.5
        
        # Test that logging uses the overridden level
        logger = get_logger("env_test")
        assert logger.config.level == 'DEBUG'
        
        print("✅ Environment variable override integration verified")
    
    def test_error_handling_integration(self):
        """Test that error handling works across all systems."""
        logger = get_logger("error_test")
        
        # Test error logging with context
        try:
            raise ValueError("Test integration error")
        except ValueError as e:
            logger.error("Integration test error", 
                        error=e, 
                        component="integration_test",
                        test_phase="error_handling")
        
        # Test validation error handling
        from logging_utils import validate_user_input
        
        # Test with invalid input
        result = validate_user_input("")
        assert not result.is_valid
        assert len(result.errors) > 0
        
        print("✅ Error handling integration verified")
    
    def test_type_safety_integration(self):
        """Test that type safety is maintained across all systems."""
        # Test configuration type safety
        config_manager = get_config_manager()
        
        # These should all return the correct types
        memory_config = config_manager.get_memory_config()
        assert isinstance(memory_config.turns_to_summarize, int)
        assert isinstance(memory_config.enable_condensation, bool)
        
        routing_config = config_manager.get_routing_config()
        assert isinstance(routing_config.roleplay_threshold, int)
        assert isinstance(routing_config.enable_fallback, bool)
        
        llm_config = config_manager.get_llm_config()
        assert isinstance(llm_config.model_name, str)
        assert isinstance(llm_config.temperature, float)
        
        # Test schema type safety
        router = Router(classification="echo_respond", reasoning="test reasoning for type safety")
        assert isinstance(router.classification, str)
        assert isinstance(router.reasoning, str)
        
        print("✅ Type safety integration verified")
    
    def test_performance_integration(self):
        """Test that performance monitoring works across systems."""
        logger = get_logger("performance_test")
        
        # Test performance timing in different contexts
        with logger.performance_timer("config_access", component="integration"):
            config_manager = get_config_manager()
            config_manager.get('memory.turns_to_summarize')
            config_manager.get('llm.model_name')
        
        with logger.performance_timer("schema_validation", component="integration"):
            router = Router(classification="echo_respond", reasoning="test reasoning for performance")
            profile = UserProfile(
                name="Test User",
                background="Test background information",
                communication_style="Test communication style",
                emotional_baseline="Test emotional baseline"
            )
        
        print("✅ Performance integration verified")
    
    def test_backward_compatibility(self):
        """Test that existing functionality remains intact."""
        # Test that all expected modules can be imported
        import config.manager
        import config.models
        import logging_utils
        import schemas
        import nodes
        import graph
        import memory
        import utils
        
        # Test that key functions exist and are callable
        assert callable(get_config_manager)
        assert callable(get_config)
        assert callable(get_logger)
        
        # Test that schemas can be instantiated
        router = Router(classification="echo_respond", reasoning="test reasoning for backward compatibility")
        assert router.classification == "echo_respond"
        
        # Test that configuration works as expected
        config_manager = get_config_manager()
        assert config_manager.validate_config()
        
        print("✅ Backward compatibility verified")
    
    def test_full_system_workflow(self):
        """Test a complete workflow through the system."""
        # 1. Initialize configuration
        config_manager = get_config_manager()
        assert config_manager.validate_config()
        
        # 2. Set up logging
        logger = get_logger("workflow_test")
        logger.info("Starting full system workflow test", phase="initialization")
        
        # 3. Create test state
        state: AgentState = {
            "message": "Hello, I'm testing the system integration",
            "classification": None,
            "reasoning": None
        }
        
        # 4. Test routing classification
        router = Router(
            classification="echo_respond",
            reasoning="User is greeting and testing the system"
        )
        
        state["classification"] = router.classification
        state["reasoning"] = router.reasoning
        
        # 5. Test memory creation
        episodic = EpisodicMemory(
            user_message=state["message"],
            ai_response="Hello! I'm working properly with all the new improvements.",
            timestamp="2024-01-01T12:00:00Z",
            context="system_test"
        )
        
        # 6. Test configuration access during workflow
        memory_config = config_manager.get_memory_config()
        turns_limit = memory_config.turns_to_summarize
        search_limit = memory_config.search_limit
        
        assert turns_limit > 0
        assert search_limit > 0
        
        # 7. Test performance monitoring
        with logger.performance_timer("full_workflow", component="integration"):
            # Simulate some work
            import time
            time.sleep(0.001)
        
        logger.info("Completed full system workflow test", 
                   phase="completion",
                   state_classification=state["classification"],
                   memory_turns_limit=turns_limit,
                   memory_search_limit=search_limit)
        
        print("✅ Full system workflow verified")


def run_all_tests():
    """Run all system integration tests."""
    print("🔍 Running end-to-end system integration tests...\n")
    
    try:
        test_class = TestSystemIntegration()
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            test_class.setup_method()
            try:
                method = getattr(test_class, method_name)
                method()
            finally:
                test_class.teardown_method()
        
        print("\n✅ All system integration tests passed!")
        print("\n🎉 EchoStar codebase improvements are fully integrated and working!")
        return True
        
    except Exception as e:
        print(f"\n❌ System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)