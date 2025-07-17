"""
Unit tests for profile management utilities.
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.agents.profile_utils import (
    search_existing_profile,
    merge_profile_data,
    update_or_create_profile,
    safe_update_or_create_profile,
    cleanup_duplicate_profiles,
    validate_profile_data,
    get_profile_summary
)


class TestProfileUtils:
    """Test cases for profile utility functions."""
    
    def test_merge_profile_data_preserves_existing(self):
        """Test that existing profile data is preserved when merging."""
        existing = {
            "name": "John Doe",
            "background": "Software developer with 5 years experience",
            "communication_style": "Direct and technical",
            "emotional_baseline": "Generally optimistic"
        }
        
        new_data = {
            "name": "John Doe",
            "background": "",  # Empty - should not overwrite
            "communication_style": "Prefers detailed explanations",  # More detailed
            "emotional_baseline": "Background information is not provided yet."  # Generic
        }
        
        result = merge_profile_data(existing, new_data)
        
        assert result["name"] == "John Doe"
        assert result["background"] == "Software developer with 5 years experience"  # Preserved
        assert result["communication_style"] == "Prefers detailed explanations"  # Updated with more detail
        assert result["emotional_baseline"] == "Generally optimistic"  # Preserved (new was generic)
    
    def test_merge_profile_data_adds_new_fields(self):
        """Test that new meaningful data is added to existing profile."""
        existing = {
            "name": "Jane Smith",
            "background": "Teacher"
        }
        
        new_data = {
            "communication_style": "Friendly and encouraging",
            "emotional_baseline": "Calm and patient"
        }
        
        result = merge_profile_data(existing, new_data)
        
        assert result["name"] == "Jane Smith"
        assert result["background"] == "Teacher"
        assert result["communication_style"] == "Friendly and encouraging"
        assert result["emotional_baseline"] == "Calm and patient"
    
    def test_merge_profile_data_ignores_generic_placeholders(self):
        """Test that generic placeholder text is ignored during merge."""
        existing = {
            "name": "Bob Wilson",
            "background": "Engineer at tech company"
        }
        
        new_data = {
            "background": "Background information is not provided yet.",
            "communication_style": "Not specified",
            "emotional_baseline": ""
        }
        
        result = merge_profile_data(existing, new_data)
        
        assert result["name"] == "Bob Wilson"
        assert result["background"] == "Engineer at tech company"  # Preserved
        assert "communication_style" not in result  # Generic value ignored
        assert "emotional_baseline" not in result  # Empty value ignored
    
    def test_validate_profile_data_valid(self):
        """Test validation of valid profile data."""
        valid_data = {
            "name": "Test User",
            "background": "This is a valid background with enough characters",
            "communication_style": "Friendly and helpful",
            "emotional_baseline": "Generally positive"
        }
        
        assert validate_profile_data(valid_data) == True
    
    def test_validate_profile_data_invalid(self):
        """Test validation of invalid profile data."""
        invalid_data = {
            "name": "",  # Too short
            "background": "Short",  # Too short
            "communication_style": "Hi",  # Too short
            "emotional_baseline": "Ok"  # Too short
        }
        
        assert validate_profile_data(invalid_data) == False
    
    def test_get_profile_summary(self):
        """Test profile summary generation."""
        profile_data = {
            "name": "Alice Johnson",
            "background": "A very long background story that should be truncated in the summary because it exceeds the character limit",
            "communication_style": "Very detailed and comprehensive communication style",
            "emotional_baseline": "Consistently positive and upbeat emotional state"
        }
        
        summary = get_profile_summary(profile_data)
        
        assert "Alice Johnson" in summary
        assert "background:" in summary
        assert "..." in summary  # Should be truncated
        assert "style:" in summary
        assert "emotion:" in summary
    
    def test_search_existing_profile_found(self):
        """Test searching for existing profile when one exists."""
        mock_store = Mock()
        mock_profile = Mock()
        mock_profile.value = {"name": "Test User", "background": "Test background"}
        mock_store.search.return_value = [mock_profile]
        
        result = search_existing_profile(mock_store, "TestUser")
        
        assert result == {"name": "Test User", "background": "Test background"}
        mock_store.search.assert_called_once_with(("echo_star", "TestUser", "profile"))
    
    def test_search_existing_profile_not_found(self):
        """Test searching for existing profile when none exists."""
        mock_store = Mock()
        mock_store.search.return_value = []
        
        result = search_existing_profile(mock_store, "TestUser")
        
        assert result is None
        mock_store.search.assert_called_once_with(("echo_star", "TestUser", "profile"))
    
    def test_search_existing_profile_multiple_found(self):
        """Test searching when multiple profiles exist (should return most recent)."""
        mock_store = Mock()
        
        # Create mock profiles with different timestamps
        mock_profile1 = Mock()
        mock_profile1.value = {"name": "User1", "background": "Old profile"}
        mock_profile1.created_at = "2023-01-01"
        
        mock_profile2 = Mock()
        mock_profile2.value = {"name": "User2", "background": "New profile"}
        mock_profile2.created_at = "2023-12-01"
        
        mock_store.search.return_value = [mock_profile1, mock_profile2]
        
        result = search_existing_profile(mock_store, "TestUser")
        
        # Should return the most recent profile
        assert result == {"name": "User2", "background": "New profile"}
    
    def test_cleanup_duplicate_profiles(self):
        """Test cleanup of duplicate profiles."""
        mock_store = Mock()
        
        # Create mock duplicate profiles
        mock_profile1 = Mock()
        mock_profile1.key = "key1"
        mock_profile1.created_at = "2023-01-01"
        
        mock_profile2 = Mock()
        mock_profile2.key = "key2"
        mock_profile2.created_at = "2023-12-01"
        
        mock_profile3 = Mock()
        mock_profile3.key = "key3"
        mock_profile3.created_at = "2023-06-01"
        
        mock_store.search.return_value = [mock_profile1, mock_profile2, mock_profile3]
        
        deleted_count = cleanup_duplicate_profiles(mock_store, "TestUser")
        
        # Should delete 2 profiles (keeping the most recent)
        assert deleted_count == 2
        
        # Should have called delete for the older profiles
        expected_calls = [
            (("echo_star", "TestUser", "profile"), "key1"),
            (("echo_star", "TestUser", "profile"), "key3")
        ]
        
        actual_calls = [call.args for call in mock_store.delete.call_args_list]
        assert len(actual_calls) == 2
        assert all(call in expected_calls for call in actual_calls)
    
    def test_cleanup_duplicate_profiles_no_duplicates(self):
        """Test cleanup when no duplicates exist."""
        mock_store = Mock()
        mock_profile = Mock()
        mock_store.search.return_value = [mock_profile]
        
        deleted_count = cleanup_duplicate_profiles(mock_store, "TestUser")
        
        assert deleted_count == 0
        mock_store.delete.assert_not_called()
    
    def test_update_or_create_profile_uses_replace_for_existing(self):
        """Test that update_or_create_profile uses replace operations for existing profiles."""
        from src.agents.profile_utils import update_or_create_profile
        
        mock_store = Mock()
        
        # Mock existing profile
        mock_existing_profile = Mock()
        mock_existing_profile.key = "existing_key"
        mock_existing_profile.value = {
            "name": "John Doe",
            "background": "Software developer",
            "communication_style": "Direct",
            "emotional_baseline": "Optimistic"
        }
        
        # Mock search to return existing profile
        mock_store.search.return_value = [mock_existing_profile]
        
        # New profile data to merge
        new_profile_data = {
            "name": "John Doe",
            "background": "Senior software developer with 10 years experience",
            "communication_style": "Direct and technical",
            "emotional_baseline": "Generally optimistic"
        }
        
        # Call the function
        result = update_or_create_profile(mock_store, new_profile_data, "TestUser")
        
        # Verify it succeeded
        assert result == True
        
        # Verify it called search to find existing profile
        mock_store.search.assert_called_with(("echo_star", "TestUser", "profile"))
        
        # Verify it called put to replace the existing profile (not create new)
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        
        # Check that it used the existing key for replacement
        assert put_call[0][0] == ("echo_star", "TestUser", "profile")  # namespace
        assert put_call[0][1] == "existing_key"  # key - should be the existing key
        
        # Check that the profile data was merged correctly
        saved_profile = put_call[0][2]  # value
        assert saved_profile["name"] == "John Doe"
        assert "Senior software developer" in saved_profile["background"]  # Should be updated
        
        # Verify no delete was called (since only one profile existed)
        mock_store.delete.assert_not_called()
    
    def test_update_or_create_profile_creates_new_when_none_exists(self):
        """Test that update_or_create_profile creates new profile when none exists."""
        from src.agents.profile_utils import update_or_create_profile
        
        mock_store = Mock()
        
        # Mock search to return no existing profiles
        mock_store.search.return_value = []
        
        # New profile data
        new_profile_data = {
            "name": "Jane Smith",
            "background": "Teacher with 5 years experience",
            "communication_style": "Friendly and encouraging",
            "emotional_baseline": "Patient and calm"
        }
        
        # Call the function
        result = update_or_create_profile(mock_store, new_profile_data, "NewUser")
        
        # Verify it succeeded
        assert result == True
        
        # Verify it called search to check for existing profile
        mock_store.search.assert_called()
        
        # Verify it called put to create new profile
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        
        # Check namespace
        assert put_call[0][0] == ("echo_star", "NewUser", "profile")
        
        # Check that a new UUID key was generated (not empty)
        assert put_call[0][1] is not None
        assert len(put_call[0][1]) > 0
        
        # Check profile data
        saved_profile = put_call[0][2]
        assert saved_profile["name"] == "Jane Smith"
        assert saved_profile["background"] == "Teacher with 5 years experience"
    
    def test_update_or_create_profile_removes_duplicates(self):
        """Test that update_or_create_profile removes duplicate profiles during update."""
        from src.agents.profile_utils import update_or_create_profile
        
        mock_store = Mock()
        
        # Mock multiple existing profiles (duplicates)
        mock_profile1 = Mock()
        mock_profile1.key = "key1"
        mock_profile1.value = {"name": "User1", "background": "Old profile"}
        mock_profile1.created_at = "2023-01-01"
        
        mock_profile2 = Mock()
        mock_profile2.key = "key2"
        mock_profile2.value = {"name": "User2", "background": "Newer profile"}
        mock_profile2.created_at = "2023-12-01"
        
        mock_profile3 = Mock()
        mock_profile3.key = "key3"
        mock_profile3.value = {"name": "User3", "background": "Another profile"}
        mock_profile3.created_at = "2023-06-01"
        
        # Mock search to return multiple profiles
        mock_store.search.return_value = [mock_profile1, mock_profile2, mock_profile3]
        
        # New profile data
        new_profile_data = {
            "name": "Updated User",
            "background": "Updated background information",
            "communication_style": "Updated style",
            "emotional_baseline": "Updated baseline"
        }
        
        # Call the function
        result = update_or_create_profile(mock_store, new_profile_data, "TestUser")
        
        # Verify it succeeded
        assert result == True
        
        # Verify it called put once to replace the primary profile
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        assert put_call[0][1] == "key1"  # Should use the first profile's key
        
        # Verify it called delete to remove duplicates
        assert mock_store.delete.call_count == 2  # Should delete the 2 duplicate profiles
        
        # Check that the correct profiles were deleted
        delete_calls = [call.args for call in mock_store.delete.call_args_list]
        expected_deletes = [
            (("echo_star", "TestUser", "profile"), "key2"),
            (("echo_star", "TestUser", "profile"), "key3")
        ]
        
        for expected_delete in expected_deletes:
            assert expected_delete in delete_calls
    
    def test_safe_update_or_create_profile_handles_exceptions(self):
        """Test that safe_update_or_create_profile never raises exceptions."""
        from src.agents.profile_utils import safe_update_or_create_profile
        
        mock_store = Mock()
        # Make the store raise an exception
        mock_store.search.side_effect = Exception("Store connection failed")
        
        # This should not raise an exception
        result = safe_update_or_create_profile(mock_store, {"name": "Test"}, "TestUser")
        
        # Should return False but not raise
        assert result == False
    
    def test_update_or_create_profile_handles_invalid_data_gracefully(self):
        """Test that update_or_create_profile handles invalid data gracefully."""
        from src.agents.profile_utils import update_or_create_profile
        
        mock_store = Mock()
        mock_store.search.return_value = []
        
        # Test with non-dict data
        result = update_or_create_profile(mock_store, "not a dict", "TestUser")
        assert result == False
        
        # Test with empty dict
        result = update_or_create_profile(mock_store, {}, "TestUser")
        assert result == False
        
        # Test with None
        result = update_or_create_profile(mock_store, None, "TestUser")
        assert result == False
    
    def test_update_or_create_profile_uses_minimal_profile_on_validation_failure(self):
        """Test that update_or_create_profile creates minimal profile when validation fails."""
        from src.agents.profile_utils import update_or_create_profile
        
        mock_store = Mock()
        mock_store.search.return_value = []
        
        # Profile data that will fail validation but has a name
        invalid_profile_data = {
            "name": "Valid Name",
            "background": "",  # Too short, will fail validation
            "communication_style": "",  # Too short
            "emotional_baseline": ""  # Too short
        }
        
        # Call the function
        result = update_or_create_profile(mock_store, invalid_profile_data, "TestUser")
        
        # Should succeed by creating minimal profile
        assert result == True
        
        # Verify it called put to create profile
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        
        # Check that minimal profile was created
        saved_profile = put_call[0][2]
        assert saved_profile["name"] == "Valid Name"
        assert "minimal profile" in saved_profile["background"]
        assert saved_profile["communication_style"] == "Standard"
        assert saved_profile["emotional_baseline"] == "Neutral"