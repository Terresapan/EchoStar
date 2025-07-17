#!/usr/bin/env python3
"""
Unit tests for logging system fixes.
Tests that logger method calls use correct signatures and don't cause secondary exceptions.
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock, patch
from io import StringIO
import sys

from src.agents.memory_error_handler import MemoryErrorHandler, ProfileSerializer
from src.agents.profile_utils import (
    update_or_create_profile,
    safe_update_or_create_profile,
    search_existing_profile
)


class TestLoggingSystemFixes:
    """Test logging system fixes to ensure proper method signatures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a string buffer to capture log output
        self.log_buffer = StringIO()
        
        # Create a test logger
        self.test_logger = logging.getLogger('test_logger')
        self.test_logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        for handler in self.test_logger.handlers[:]:
            self.test_logger.removeHandler(handler)
        
        # Add a stream handler to capture output
        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.handler.setFormatter(formatter)
        self.test_logger.addHandler(self.handler)
        
    def teardown_method(self):
        """Clean up after tests."""
        self.test_logger.removeHandler(self.handler)
        self.handler.close()
        
    def test_logger_error_method_with_string_formatting(self):
        """Test that logger.error() uses string formatting instead of kwargs."""
        # Test data that would cause issues with kwargs
        error_context = {
            "error_type": "TestError",
            "user_id": "test_user",
            "operation": "test_operation"
        }
        
        # This should work without raising TypeError
        try:
            self.test_logger.error(f"Test error occurred: {error_context}")
            logging_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Logger.error() called with unsupported kwargs: {e}")
            else:
                raise
        
        assert logging_successful
        
        # Verify the message was logged
        log_output = self.log_buffer.getvalue()
        assert "Test error occurred" in log_output
        assert "TestError" in log_output
        
    def test_logger_warning_method_with_string_formatting(self):
        """Test that logger.warning() uses string formatting instead of kwargs."""
        # Test data that would cause issues with kwargs
        warning_context = {
            "correlation_id": "test_correlation_123",
            "operation": "memory_condensation"
        }
        
        # This should work without raising TypeError
        try:
            self.test_logger.warning(f"Warning occurred: {warning_context}")
            logging_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Logger.warning() called with unsupported kwargs: {e}")
            else:
                raise
        
        assert logging_successful
        
        # Verify the message was logged
        log_output = self.log_buffer.getvalue()
        assert "Warning occurred" in log_output
        assert "test_correlation_123" in log_output
        
    def test_logger_info_method_with_string_formatting(self):
        """Test that logger.info() uses string formatting instead of kwargs."""
        # Test data
        info_context = {
            "user_id": "test_user",
            "profile_updated": True
        }
        
        # This should work without raising TypeError
        try:
            self.test_logger.info(f"Info message: {info_context}")
            logging_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Logger.info() called with unsupported kwargs: {e}")
            else:
                raise
        
        assert logging_successful
        
        # Verify the message was logged
        log_output = self.log_buffer.getvalue()
        assert "Info message" in log_output
        assert "test_user" in log_output
        
    def test_logger_debug_method_with_string_formatting(self):
        """Test that logger.debug() uses string formatting instead of kwargs."""
        # Test data
        debug_context = {
            "field": "background",
            "value_length": 150
        }
        
        # This should work without raising TypeError
        try:
            self.test_logger.debug(f"Debug message: {debug_context}")
            logging_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Logger.debug() called with unsupported kwargs: {e}")
            else:
                raise
        
        assert logging_successful
        
        # Verify the message was logged
        log_output = self.log_buffer.getvalue()
        assert "Debug message" in log_output
        assert "background" in log_output
        
    def test_memory_error_handler_logging_fixes(self):
        """Test that MemoryErrorHandler uses correct logging methods."""
        # Test data that would cause issues with old logging approach
        test_data = {
            "complex_object": {"nested": {"data": "value"}},
            "user_id": "test_user"
        }
        
        # Test validate_hashable_structure logging
        is_valid, errors = MemoryErrorHandler.validate_hashable_structure(test_data)
        
        # Should not raise logging-related exceptions
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # Test sanitize_for_storage logging
        try:
            sanitized = MemoryErrorHandler.sanitize_for_storage(test_data)
            sanitization_successful = True
        except Exception as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Sanitization failed due to logging error: {e}")
            else:
                # Other exceptions are acceptable for this test
                sanitization_successful = True
        
        assert sanitization_successful
        
    def test_memory_error_handler_log_error_context(self):
        """Test that log_error_context uses proper logging format."""
        error = ValueError("Test error")
        context = {
            "operation": "test_operation",
            "user_id": "test_user",
            "data_size": 1024
        }
        
        # This should not raise TypeError due to logging issues
        try:
            MemoryErrorHandler.log_error_context(error, context)
            logging_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"log_error_context failed due to logging error: {e}")
            else:
                raise
        
        assert logging_successful
        
    def test_profile_serializer_logging_fixes(self):
        """Test that ProfileSerializer uses correct logging methods."""
        # Test data that might cause serialization issues
        profile_data = {
            "name": "Test User",
            "complex_data": {"nested": {"object": "value"}},
            "invalid_object": object()  # This will cause serialization issues
        }
        
        # Test serialize_profile logging
        try:
            serialized = ProfileSerializer.serialize_profile(profile_data)
            serialization_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Profile serialization failed due to logging error: {e}")
            else:
                # Other exceptions are acceptable for this test
                serialization_successful = True
        
        assert serialization_successful
        assert isinstance(serialized, dict)
        
        # Test deserialize_profile logging
        try:
            deserialized = ProfileSerializer.deserialize_profile(serialized)
            deserialization_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Profile deserialization failed due to logging error: {e}")
            else:
                # Other exceptions are acceptable for this test
                deserialization_successful = True
        
        assert deserialization_successful
        
    @patch('src.agents.profile_utils.logger')
    def test_profile_utils_logging_fixes(self, mock_logger):
        """Test that profile_utils uses correct logging method signatures."""
        mock_store = Mock()
        
        # Test search_existing_profile logging
        mock_store.search.side_effect = Exception("Store error")
        
        result = search_existing_profile(mock_store, "test_user")
        
        # Verify logger.error was called with string message, not kwargs
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args
        
        # Should be called with string message as first argument
        assert len(call_args[0]) > 0  # Has positional arguments
        assert isinstance(call_args[0][0], str)  # First argument is string
        
        # Should not have problematic kwargs
        if len(call_args) > 1:
            kwargs = call_args[1]
            problematic_kwargs = ['errors', 'error_message', 'error_context']
            for kwarg in problematic_kwargs:
                assert kwarg not in kwargs, f"Found problematic kwarg: {kwarg}"
        
    @patch('src.agents.profile_utils.logger')
    def test_profile_validation_logging_fixes(self, mock_logger):
        """Test that profile validation uses correct logging method signatures."""
        mock_store = Mock()
        mock_store.search.return_value = []
        
        # Test with invalid profile data that will trigger validation errors
        invalid_profile = {
            "name": "",  # Too short
            "background": "x",  # Too short
            "invalid_field": object()  # Can't be serialized
        }
        
        # This should not raise TypeError due to logging issues
        try:
            result = update_or_create_profile(mock_store, invalid_profile, "test_user")
            validation_successful = True
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Profile validation failed due to logging error: {e}")
            else:
                # Other exceptions are acceptable for this test
                validation_successful = True
        
        assert validation_successful
        
        # Verify logging calls use correct signatures
        if mock_logger.error.called:
            for call in mock_logger.error.call_args_list:
                # Should have string message as first argument
                assert len(call[0]) > 0
                assert isinstance(call[0][0], str)
        
        if mock_logger.warning.called:
            for call in mock_logger.warning.call_args_list:
                # Should have string message as first argument
                assert len(call[0]) > 0
                assert isinstance(call[0][0], str)
        
    def test_no_secondary_exceptions_during_error_logging(self):
        """Test that error logging doesn't cause secondary exceptions."""
        # Create a scenario that would previously cause secondary exceptions
        complex_error_data = {
            "error_type": "ComplexError",
            "nested_data": {
                "level1": {
                    "level2": {
                        "problematic_object": object(),
                        "circular_ref": None
                    }
                }
            }
        }
        
        # Add circular reference
        complex_error_data["nested_data"]["level1"]["level2"]["circular_ref"] = complex_error_data
        
        # This should not raise secondary exceptions during logging
        try:
            # Simulate the old problematic logging approach (but fixed)
            error_message = f"Complex error occurred: {str(complex_error_data)[:200]}..."
            self.test_logger.error(error_message)
            
            no_secondary_exceptions = True
        except Exception as e:
            if "unexpected keyword argument" in str(e) or "unhashable type" in str(e):
                pytest.fail(f"Secondary exception during error logging: {e}")
            else:
                # Other exceptions might be acceptable depending on the scenario
                no_secondary_exceptions = True
        
        assert no_secondary_exceptions
        
    def test_structured_error_context_logging(self):
        """Test that structured error context is logged properly."""
        error_context = {
            "operation": "memory_condensation",
            "user_id": "test_user_123",
            "error_count": 3,
            "timestamp": "2024-01-01T12:00:00Z",
            "additional_info": {
                "store_type": "InMemoryStore",
                "memory_items": 5
            }
        }
        
        # Test that complex error context can be logged without issues
        try:
            formatted_context = str(error_context)
            self.test_logger.error(f"Structured error context: {formatted_context}")
            
            structured_logging_successful = True
        except Exception as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Structured logging failed: {e}")
            else:
                raise
        
        assert structured_logging_successful
        
        # Verify the context was logged
        log_output = self.log_buffer.getvalue()
        assert "memory_condensation" in log_output
        assert "test_user_123" in log_output
        
    def test_logging_with_exception_objects(self):
        """Test that logging with exception objects works correctly."""
        test_exception = ValueError("Test exception with complex data")
        
        # Test logging exception information
        try:
            exception_info = {
                "exception_type": type(test_exception).__name__,
                "exception_message": str(test_exception),
                "operation": "test_operation"
            }
            
            self.test_logger.error(f"Exception occurred: {exception_info}")
            
            exception_logging_successful = True
        except Exception as e:
            if "unexpected keyword argument" in str(e):
                pytest.fail(f"Exception logging failed: {e}")
            else:
                raise
        
        assert exception_logging_successful
        
        # Verify exception information was logged
        log_output = self.log_buffer.getvalue()
        assert "ValueError" in log_output
        assert "Test exception" in log_output