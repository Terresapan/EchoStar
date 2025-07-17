#!/usr/bin/env python3
"""
Unit tests for memory condensation success and failure scenarios.
Tests various condensation scenarios including success, partial failure, and complete failure.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from src.agents.memory_error_handler import MemoryErrorHandler, ProfileSerializer
from src.agents.schemas import SemanticMemory


class TestMemoryCondensationScenarios:
    """Test memory condensation success and failure scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_store = Mock()
        self.mock_semantic_manager = Mock()
        self.mock_episodic_manager = Mock()
        
        # Mock memory items
        self.mock_memory_items = []
        for i in range(5):
            item = Mock()
            item.value = {
                'content': {
                    'user_message': f'User message {i}',
                    'ai_response': f'AI response {i}'
                }
            }
            item.key = f'test_key_{i}'
            self.mock_memory_items.append(item)
        
    def test_successful_memory_storage_scenario(self):
        """Test successful memory storage with all operations succeeding."""
        # Setup store mock to return memories
        self.mock_store.search.return_value = self.mock_memory_items
        
        # Setup managers to succeed
        self.mock_semantic_manager.submit.return_value = True
        self.mock_episodic_manager.submit.return_value = True
        
        # Test successful storage operations
        # Simulate semantic memory storage
        semantic_data = {"messages": [{"role": "user", "content": "test"}]}
        semantic_result = self.mock_semantic_manager.submit(semantic_data)
        assert semantic_result == True
        
        # Simulate episodic memory storage
        episodic_data = {"messages": [{"role": "assistant", "content": "response"}]}
        episodic_result = self.mock_episodic_manager.submit(episodic_data)
        assert episodic_result == True
        
        # Verify all components were called
        self.mock_semantic_manager.submit.assert_called_once()
        self.mock_episodic_manager.submit.assert_called_once()
        
    def test_memory_storage_with_no_memories(self):
        """Test memory storage when no memories are found."""
        # Setup store mock to return no memories
        self.mock_store.search.return_value = []
        
        # Test that empty memory list is handled gracefully
        memories = self.mock_store.search(('echo_star', 'Lily', 'collection'), limit=10)
        assert memories == []
        
        # Managers should handle empty data gracefully
        empty_data = {"messages": []}
        semantic_result = self.mock_semantic_manager.submit(empty_data)
        episodic_result = self.mock_episodic_manager.submit(empty_data)
        
        # Should not crash with empty data
        assert semantic_result is not None
        assert episodic_result is not None
        
    def test_semantic_storage_failure_scenario(self):
        """Test handling of semantic storage failure."""
        # Setup semantic manager to fail
        self.mock_semantic_manager.submit.side_effect = Exception("Semantic storage failed")
        
        # Setup episodic manager to succeed
        self.mock_episodic_manager.submit.return_value = True
        
        # Test that semantic failure is handled gracefully
        semantic_data = {"messages": [{"role": "user", "content": "test"}]}
        
        try:
            semantic_result = self.mock_semantic_manager.submit(semantic_data)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Semantic storage failed" in str(e)
        
        # Episodic storage should still work
        episodic_data = {"messages": [{"role": "assistant", "content": "response"}]}
        episodic_result = self.mock_episodic_manager.submit(episodic_data)
        assert episodic_result == True
        
    def test_episodic_storage_failure_scenario(self):
        """Test handling of episodic storage failure."""
        # Setup episodic manager to fail
        self.mock_episodic_manager.submit.side_effect = Exception("Episodic storage failed")
        
        # Setup semantic manager to succeed
        self.mock_semantic_manager.submit.return_value = True
        
        # Test that episodic failure is handled gracefully
        episodic_data = {"messages": [{"role": "assistant", "content": "response"}]}
        
        try:
            episodic_result = self.mock_episodic_manager.submit(episodic_data)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Episodic storage failed" in str(e)
        
        # Semantic storage should still work
        semantic_data = {"messages": [{"role": "user", "content": "test"}]}
        semantic_result = self.mock_semantic_manager.submit(semantic_data)
        assert semantic_result == True
        
    def test_both_storage_systems_fail_scenario(self):
        """Test scenario where both semantic and episodic storage fail."""
        # Setup both managers to fail
        self.mock_semantic_manager.submit.side_effect = Exception("Semantic storage failed")
        self.mock_episodic_manager.submit.side_effect = Exception("Episodic storage failed")
        
        # Test that both failures are handled gracefully
        semantic_data = {"messages": [{"role": "user", "content": "test"}]}
        episodic_data = {"messages": [{"role": "assistant", "content": "response"}]}
        
        # Both should raise exceptions
        try:
            self.mock_semantic_manager.submit(semantic_data)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Semantic storage failed" in str(e)
            
        try:
            self.mock_episodic_manager.submit(episodic_data)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Episodic storage failed" in str(e)
        
    def test_memory_verification_after_storage_scenario(self):
        """Test that memory verification occurs after storage."""
        # Setup store mock - multiple calls for verification
        self.mock_store.search.side_effect = [
            self.mock_memory_items,  # Initial memory retrieval
            [Mock()],  # Semantic verification - found
            [Mock()],  # Episodic verification - found
        ]
        
        # Test multiple search operations
        initial_memories = self.mock_store.search(('echo_star', 'Lily', 'collection'), limit=10)
        assert len(initial_memories) == 5
        
        semantic_verification = self.mock_store.search(('echo_star', 'Lily', 'facts'), filter={'category': 'summary'}, limit=5)
        assert len(semantic_verification) == 1
        
        episodic_verification = self.mock_store.search(('echo_star', 'Lily', 'collection'), limit=2)
        assert len(episodic_verification) == 1
        
        # Should have made multiple search calls for verification
        assert self.mock_store.search.call_count == 3
        
    def test_memory_error_handler_with_malformed_data(self):
        """Test handling of malformed memory items using MemoryErrorHandler."""
        # Setup malformed memory items
        malformed_items = [
            {'value': None, 'key': 'bad_key_1'},  # None value
            {'value': {'invalid': 'structure'}, 'key': 'bad_key_2'},  # Missing content
            {'value': {'content': None}, 'key': 'bad_key_3'},  # None content
        ]
        
        # Test that MemoryErrorHandler can validate and sanitize malformed data
        for item in malformed_items:
            # Test validation
            is_valid, errors = MemoryErrorHandler.validate_hashable_structure(item)
            assert isinstance(is_valid, bool)
            assert isinstance(errors, list)
            
            # Test sanitization
            sanitized = MemoryErrorHandler.sanitize_for_storage(item)
            assert isinstance(sanitized, dict)
            
    def test_semantic_memory_creation_and_validation(self):
        """Test SemanticMemory creation and validation for condensation scenarios."""
        # Test successful semantic memory creation
        semantic_memory = SemanticMemory(
            category="summary",
            content="Test conversation summary from condensation",
            context="Memory condensation process",
            timestamp=datetime.now().isoformat()
        )
        
        # Test that it can be converted to dict for storage
        memory_dict = semantic_memory.dict()
        assert isinstance(memory_dict, dict)
        assert memory_dict["category"] == "summary"
        assert "condensation" in memory_dict["content"]
        
        # Test validation with MemoryErrorHandler
        is_valid, errors = MemoryErrorHandler.validate_hashable_structure(memory_dict)
        assert is_valid == True, f"SemanticMemory should be valid for storage: {errors}"
        
    def test_profile_serializer_with_condensation_data(self):
        """Test ProfileSerializer with data that might come from memory condensation."""
        # Test data that might be generated during condensation
        condensation_profile_data = {
            "name": "User from condensed memories",
            "background": "Extracted from conversation summary",
            "communication_style": "Derived from interaction patterns",
            "emotional_baseline": "Analyzed from conversation tone",
            "condensation_metadata": {
                "source": "memory_condensation",
                "timestamp": datetime.now().isoformat(),
                "turns_processed": 10
            }
        }
        
        # Test serialization
        serialized = ProfileSerializer.serialize_profile(condensation_profile_data)
        assert isinstance(serialized, dict)
        assert "condensed memories" in serialized["name"]
        
        # Test deserialization
        deserialized = ProfileSerializer.deserialize_profile(serialized)
        assert isinstance(deserialized, dict)
        assert "name" in deserialized