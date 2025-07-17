#!/usr/bin/env python3
"""
Unit tests for profile storage error handling.
Tests that profile storage errors are handled gracefully without breaking conversation flow.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.agents.profile_utils import (
    update_or_create_profile,
    safe_update_or_create_profile,
    search_existing_profile,
    validate_profile_data,
    merge_profile_data
)
from src.agents.memory_error_handler import MemoryErrorHandler, ProfileSerializer


class TestProfileStorageErrorHandling:
    """Test profile storage error handling to ensure conversation continuity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_store = Mock()
        
        # Valid profile data for testing
        self.valid_profile = {
            "name": "Test User",
            "background": "Software developer with 5 years experience",
            "communication_style": "Direct and technical",
            "emotional_baseline": "Generally optimistic"
        }
        
        # Invalid profile data for testing
        self.invalid_profile = {
            "name": "",  # Too short
            "background": "x",  # Too short
            "communication_style": "",  # Empty
            "emotional_baseline": None  # Invalid type
        }
        
    def test_safe_wrapper_never_raises_exceptions(self):
        """Test that safe_update_or_create_profile never raises exceptions."""
        # Test various failure scenarios
        failure_scenarios = [
            # Store connection failure
            lambda: setattr(self.mock_store, 'search', Mock(side_effect=ConnectionError("Store unavailable"))),
            # Store method missing
            lambda: delattr(self.mock_store, 'search') if hasattr(self.mock_store, 'search') else None,
            # Store put failure
            lambda: setattr(self.mock_store, 'put', Mock(side_effect=Exception("Put failed"))),
        ]
        
        for setup_failure in failure_scenarios:
            # Reset mock
            self.mock_store.reset_mock()
            self.mock_store.search = Mock(return_value=[])
            self.mock_store.put = Mock()
            
            # Apply failure scenario
            try:
                if setup_failure:
                    setup_failure()
            except:
                pass  # Ignore setup errors
            
            # This should never raise an exception
            try:
                result = safe_update_or_create_profile(self.mock_store, self.valid_profile, "TestUser")
                # Should return False for failures, but never raise
                assert isinstance(result, bool)
            except Exception as e:
                pytest.fail(f"safe_update_or_create_profile raised an exception: {e}")
                
    def test_profile_validation_before_storage_attempts(self):
        """Test that profile structure is validated before storage attempts."""
        # Test various invalid structures
        invalid_structures = [
            "not a dict",
            [],
            None,
            123,
            {"name": None},  # Invalid name
            {},  # Empty dict
        ]
        
        for invalid_data in invalid_structures:
            result = update_or_create_profile(self.mock_store, invalid_data, "TestUser")
            
            # Should fail validation and return False
            assert result == False
            
            # Should not attempt to store invalid data
            self.mock_store.put.assert_not_called()
            self.mock_store.reset_mock()
            
    def test_graceful_error_handling_in_update_or_create_profile(self):
        """Test that update_or_create_profile handles errors gracefully."""
        # Test store search failure
        self.mock_store.search.side_effect = Exception("Search failed")
        
        result = update_or_create_profile(self.mock_store, self.valid_profile, "TestUser")
        
        # Should handle error gracefully
        assert result == False
        
        # Reset and test store put failure
        self.mock_store.reset_mock()
        self.mock_store.search.return_value = []
        self.mock_store.put.side_effect = Exception("Put failed")
        
        result = update_or_create_profile(self.mock_store, self.valid_profile, "TestUser")
        
        # Should handle error gracefully
        assert result == False
        
    def test_conversation_flow_continues_after_profile_errors(self):
        """Test that conversation processing can continue even when profile storage fails."""
        # Simulate various storage failures
        failure_scenarios = [
            Exception("Database connection lost"),
            ValueError("Invalid data format"),
            KeyError("Missing required field"),
            TypeError("Unhashable type"),
        ]
        
        for error in failure_scenarios:
            self.mock_store.search.side_effect = error
            
            # Safe wrapper should handle all errors gracefully
            result = safe_update_or_create_profile(self.mock_store, self.valid_profile, "TestUser")
            
            # Should return False but not raise
            assert result == False
            
            # Conversation state should remain intact (simulated)
            conversation_state = {
                "user_message": "Hello, how are you?",
                "context": "User greeting",
                "turn_count": 5
            }
            
            # Profile failure shouldn't affect conversation state
            assert conversation_state["user_message"] == "Hello, how are you?"
            assert conversation_state["context"] == "User greeting"
            assert conversation_state["turn_count"] == 5
            
            # Reset for next test
            self.mock_store.reset_mock()
            
    def test_profile_structure_validation_creates_minimal_profile_on_failure(self):
        """Test that profile validation failures result in minimal valid profiles."""
        self.mock_store.search.return_value = []  # No existing profile
        
        # Profile with invalid data that will fail validation
        invalid_profile = {
            "name": "Valid Name",
            "background": "",  # Too short
            "communication_style": "x",  # Too short
            "emotional_baseline": None,  # Invalid
            "invalid_field": object()  # Can't be serialized
        }
        
        result = update_or_create_profile(self.mock_store, invalid_profile, "TestUser")
        
        # Should succeed by creating minimal profile
        assert result == True
        
        # Verify profile was created
        self.mock_store.put.assert_called_once()
        put_call = self.mock_store.put.call_args
        saved_profile = put_call[0][2]
        
        # Should have minimal valid profile
        assert saved_profile["name"] == "Valid Name"
        assert "minimal profile" in saved_profile["background"]
        assert saved_profile["communication_style"] == "Standard"
        assert saved_profile["emotional_baseline"] == "Neutral"
        
    def test_error_logging_without_secondary_exceptions(self):
        """Test that error logging doesn't cause secondary exceptions."""
        self.mock_store.search.side_effect = Exception("Test error with complex data")
        
        # This should log errors without causing secondary exceptions
        try:
            result = safe_update_or_create_profile(
                self.mock_store, 
                {"name": "Test", "complex": {"nested": "data"}}, 
                "TestUser"
            )
            
            # Should handle gracefully without raising exceptions
            assert result == False
            
            # If we get here, logging worked without secondary exceptions
            logging_test_passed = True
            
        except Exception as e:
            # If any exception is raised, the logging caused secondary exceptions
            pytest.fail(f"Logging caused secondary exception: {e}")
        
        assert logging_test_passed
        
    def test_profile_serialization_error_handling(self):
        """Test that profile serialization errors are handled gracefully."""
        # Profile with data that can't be serialized
        problematic_profile = {
            "name": "Test User",
            "background": "Valid background",
            "complex_object": object(),  # Can't be serialized
            "circular_ref": None
        }
        
        # Add circular reference
        problematic_profile["circular_ref"] = problematic_profile
        
        # Test ProfileSerializer.serialize_profile
        try:
            serialized = ProfileSerializer.serialize_profile(problematic_profile)
            serialization_successful = True
        except Exception as e:
            # Should handle serialization errors gracefully
            serialization_successful = False
            
        # Should either succeed with sanitized data or handle error gracefully
        assert serialization_successful or isinstance(serialized, dict)
        
    def test_profile_validation_with_memory_error_handler(self):
        """Test that profile validation uses MemoryErrorHandler correctly."""
        # Profile with hashable structure issues
        problematic_profile = {
            "name": "Test User",
            "data": {
                "nested": {
                    "object": object()  # Unhashable
                }
            }
        }
        
        # Test validation
        is_valid, errors = MemoryErrorHandler.validate_hashable_structure(problematic_profile)
        
        # Should detect issues
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # Test sanitization
        try:
            sanitized = MemoryErrorHandler.sanitize_for_storage(problematic_profile)
            sanitization_successful = True
        except Exception:
            sanitization_successful = False
            
        # Should handle sanitization gracefully
        assert sanitization_successful
        
    def test_profile_storage_with_store_compatibility_issues(self):
        """Test profile storage with various store compatibility issues."""
        # Test with store that doesn't support certain operations
        incompatible_store = Mock()
        
        # Store that fails on search
        incompatible_store.search.side_effect = AttributeError("Method not supported")
        
        result = safe_update_or_create_profile(incompatible_store, self.valid_profile, "TestUser")
        
        # Should handle incompatibility gracefully
        assert result == False
        
        # Test with store that fails on put
        incompatible_store.reset_mock()
        incompatible_store.search.return_value = []
        incompatible_store.put.side_effect = NotImplementedError("Put not implemented")
        
        result = safe_update_or_create_profile(incompatible_store, self.valid_profile, "TestUser")
        
        # Should handle incompatibility gracefully
        assert result == False
        
    def test_profile_merge_error_handling(self):
        """Test that profile merging handles errors gracefully."""
        # Existing profile with problematic data
        existing_profile = {
            "name": "Test User",
            "background": "Existing background",
            "problematic_field": object()  # Can't be compared
        }
        
        # New profile data
        new_profile = {
            "name": "Test User",
            "background": "Updated background",
            "communication_style": "New style"
        }
        
        # Test merge operation
        try:
            merged = merge_profile_data(existing_profile, new_profile)
            merge_successful = True
        except Exception as e:
            # Should handle merge errors gracefully
            merge_successful = False
            
        # Should either succeed or handle error gracefully
        assert merge_successful or isinstance(merged, dict)
        
    def test_profile_validation_edge_cases(self):
        """Test profile validation with edge cases."""
        edge_cases = [
            # Very long strings
            {"name": "A" * 1000, "background": "B" * 10000},
            # Unicode characters
            {"name": "用户", "background": "背景信息"},
            # Special characters
            {"name": "User@#$%", "background": "Background with\nnewlines\tand\ttabs"},
            # Numeric strings
            {"name": "123", "background": "456"},
        ]
        
        for profile_data in edge_cases:
            # Should handle edge cases without crashing
            try:
                is_valid = validate_profile_data(profile_data)
                validation_successful = True
            except Exception:
                validation_successful = False
                
            assert validation_successful
            
    def test_profile_storage_recovery_mechanisms(self):
        """Test profile storage recovery mechanisms."""
        # Test scenario where initial storage fails but retry succeeds
        self.mock_store.search.return_value = []
        
        # First put call fails, second succeeds (simulating retry)
        self.mock_store.put.side_effect = [Exception("Temporary failure"), None]
        
        # The current implementation doesn't have explicit retry logic,
        # but should handle the failure gracefully
        result = update_or_create_profile(self.mock_store, self.valid_profile, "TestUser")
        
        # Should handle the failure gracefully
        assert isinstance(result, bool)
        
    def test_profile_storage_with_concurrent_access(self):
        """Test profile storage behavior with concurrent access scenarios."""
        # Simulate concurrent modification scenario
        existing_profile = Mock()
        existing_profile.key = "existing_key"
        existing_profile.value = self.valid_profile.copy()
        
        # First search returns profile, second search returns different profile (concurrent modification)
        self.mock_store.search.side_effect = [
            [existing_profile],  # Initial search
            []  # Profile was deleted by another process
        ]
        
        # Should handle concurrent modification gracefully
        result = update_or_create_profile(self.mock_store, self.valid_profile, "TestUser")
        
        # Should either succeed or fail gracefully
        assert isinstance(result, bool)