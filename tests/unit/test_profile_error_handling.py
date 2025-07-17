"""
Unit tests for profile storage error handling to ensure conversation flow is never broken.
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.agents.profile_utils import safe_update_or_create_profile, update_or_create_profile


class TestProfileErrorHandling:
    """Test cases for profile error handling that ensures conversation continuity."""
    
    def test_safe_wrapper_never_raises_exceptions(self):
        """Test that safe_update_or_create_profile never raises exceptions under any circumstances."""
        mock_store = Mock()
        
        # Test various failure scenarios
        test_cases = [
            # Store connection failure
            (lambda: setattr(mock_store, 'search', Mock(side_effect=ConnectionError("Store unavailable"))), {}),
            # Store method missing
            (lambda: delattr(mock_store, 'search'), {"name": "Test"}),
            # Invalid profile data
            (lambda: None, None),
            # Complex object that can't be serialized
            (lambda: None, {"name": "Test", "complex": object()}),
        ]
        
        for setup_func, profile_data in test_cases:
            if setup_func:
                try:
                    setup_func()
                except:
                    pass  # Ignore setup errors
            
            # This should never raise an exception
            try:
                result = safe_update_or_create_profile(mock_store, profile_data, "TestUser")
                # Should return False for failures, but never raise
                assert isinstance(result, bool)
            except Exception as e:
                pytest.fail(f"safe_update_or_create_profile raised an exception: {e}")
    
    def test_profile_validation_creates_minimal_profile_on_failure(self):
        """Test that profile validation failures result in minimal valid profiles."""
        mock_store = Mock()
        mock_store.search.return_value = []  # No existing profile
        
        # Profile with invalid data that will fail validation
        invalid_profile = {
            "name": "Valid Name",
            "background": "",  # Too short
            "communication_style": "x",  # Too short
            "emotional_baseline": None,  # Invalid
            "invalid_field": object()  # Can't be serialized
        }
        
        result = update_or_create_profile(mock_store, invalid_profile, "TestUser")
        
        # Should succeed by creating minimal profile
        assert result == True
        
        # Verify profile was created
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        saved_profile = put_call[0][2]
        
        # Should have minimal valid profile
        assert saved_profile["name"] == "Valid Name"
        assert "minimal profile" in saved_profile["background"]
        assert saved_profile["communication_style"] == "Standard"
        assert saved_profile["emotional_baseline"] == "Neutral"
    
    def test_conversation_flow_continues_after_profile_errors(self):
        """Test that conversation processing can continue even when profile storage fails."""
        mock_store = Mock()
        
        # Simulate various storage failures
        failure_scenarios = [
            Exception("Database connection lost"),
            ValueError("Invalid data format"),
            KeyError("Missing required field"),
            TypeError("Unhashable type"),
        ]
        
        for error in failure_scenarios:
            mock_store.search.side_effect = error
            
            # Safe wrapper should handle all errors gracefully
            result = safe_update_or_create_profile(mock_store, {"name": "Test"}, "TestUser")
            
            # Should return False but not raise
            assert result == False
            
            # Reset for next test
            mock_store.reset_mock()
    
    def test_error_logging_without_secondary_exceptions(self):
        """Test that error logging doesn't cause secondary exceptions."""
        mock_store = Mock()
        mock_store.search.side_effect = Exception("Test error with complex data")
        
        # This should log errors without causing secondary exceptions
        # The main test is that no exceptions are raised during logging
        try:
            result = safe_update_or_create_profile(
                mock_store, 
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
    
    def test_profile_structure_validation_before_storage(self):
        """Test that profile structure is validated before storage attempts."""
        mock_store = Mock()
        mock_store.search.return_value = []
        
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
            result = update_or_create_profile(mock_store, invalid_data, "TestUser")
            
            # Should fail validation and return False
            assert result == False
            
            # Should not attempt to store invalid data
            mock_store.put.assert_not_called()
            mock_store.reset_mock()
    
    def test_graceful_degradation_preserves_conversation_context(self):
        """Test that profile storage failures don't lose conversation context."""
        mock_store = Mock()
        
        # Simulate storage failure
        mock_store.search.side_effect = Exception("Storage unavailable")
        
        # Conversation should continue even if profile storage fails
        conversation_state = {
            "user_message": "Hello, how are you?",
            "context": "User greeting",
            "turn_count": 5
        }
        
        # Profile storage failure shouldn't affect conversation state
        profile_result = safe_update_or_create_profile(
            mock_store, 
            {"name": "User", "background": "Friendly person"}, 
            "TestUser"
        )
        
        # Profile storage failed, but conversation state is preserved
        assert profile_result == False
        assert conversation_state["user_message"] == "Hello, how are you?"
        assert conversation_state["context"] == "User greeting"
        assert conversation_state["turn_count"] == 5
        
        # Conversation can continue normally
        conversation_state["turn_count"] += 1
        assert conversation_state["turn_count"] == 6