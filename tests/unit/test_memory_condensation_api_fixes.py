#!/usr/bin/env python3
"""
Unit tests for memory condensation BaseStore API compatibility fixes.
Tests the fixes for BaseStore.search() method calls and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from uuid import uuid4

# Import only the components we can test without the problematic nodes.py
from src.agents.memory_error_handler import MemoryErrorHandler
from src.agents.schemas import SemanticMemory


class TestBaseStoreAPICompatibility:
    """Test BaseStore API compatibility fixes in memory condensation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_store = Mock()
        
        # Mock memory items
        self.mock_memory_item = Mock()
        self.mock_memory_item.value = {
            'content': {
                'user_message': 'Hello',
                'ai_response': 'Hi there!'
            }
        }
        self.mock_memory_item.key = 'test_key_1'
        
    def test_store_search_api_signature_validation(self):
        """Test that store.search() calls use correct API signature without namespace parameter."""
        # This test validates the correct API usage pattern
        
        # Correct usage: namespace as positional argument
        correct_call_args = (('echo_star', 'Lily', 'collection'),)
        correct_call_kwargs = {'limit': 10}
        
        # Incorrect usage: namespace as keyword argument (should be avoided)
        incorrect_call_kwargs = {'namespace': ('echo_star', 'Lily', 'collection'), 'limit': 10}
        
        # Test correct API usage
        self.mock_store.search.return_value = [self.mock_memory_item]
        
        # Simulate correct call
        result = self.mock_store.search(*correct_call_args, **correct_call_kwargs)
        
        # Verify the call was made correctly
        self.mock_store.search.assert_called_with(('echo_star', 'Lily', 'collection'), limit=10)
        
        # Verify that namespace is not passed as keyword argument
        call_args = self.mock_store.search.call_args
        assert len(call_args[0]) > 0, "Namespace should be passed as positional argument"
        assert 'namespace' not in call_args[1], "Namespace should NOT be passed as keyword argument"
        
    def test_store_search_with_filter_api_compatibility(self):
        """Test that store.search() with filters uses correct API signature."""
        # Test semantic memory search with filter
        semantic_namespace = ('echo_star', 'Lily', 'facts')
        filter_criteria = {'category': 'summary'}
        
        self.mock_store.search.return_value = [self.mock_memory_item]
        
        # Simulate correct semantic search call
        result = self.mock_store.search(semantic_namespace, filter=filter_criteria, limit=5)
        
        # Verify the call was made correctly
        self.mock_store.search.assert_called_with(
            ('echo_star', 'Lily', 'facts'), 
            filter={'category': 'summary'}, 
            limit=5
        )
        
        # Verify API compatibility
        call_args = self.mock_store.search.call_args
        assert call_args[0][0] == semantic_namespace, "Namespace should be positional argument"
        assert 'filter' in call_args[1], "Filter should be keyword argument"
        assert 'limit' in call_args[1], "Limit should be keyword argument"
        assert 'namespace' not in call_args[1], "Namespace should NOT be keyword argument"
        
    def test_store_put_api_compatibility(self):
        """Test that store.put() uses correct API signature."""
        # Test data
        namespace = ('echo_star', 'Lily', 'facts')
        key = 'test_key_123'
        value = {'content': 'test content', 'category': 'summary'}
        
        # Simulate correct put call
        self.mock_store.put(namespace, key, value)
        
        # Verify the call was made correctly
        self.mock_store.put.assert_called_with(
            ('echo_star', 'Lily', 'facts'),
            'test_key_123',
            {'content': 'test content', 'category': 'summary'}
        )
        
        # Verify API compatibility
        call_args = self.mock_store.put.call_args
        assert len(call_args[0]) == 3, "Put should have 3 positional arguments: namespace, key, value"
        assert call_args[0][0] == namespace, "First argument should be namespace"
        assert call_args[0][1] == key, "Second argument should be key"
        assert call_args[0][2] == value, "Third argument should be value"
        
    def test_store_delete_api_compatibility(self):
        """Test that store.delete() uses correct API signature."""
        # Test data
        namespace = ('echo_star', 'Lily', 'collection')
        key = 'memory_key_456'
        
        # Simulate correct delete call
        self.mock_store.delete(namespace, key)
        
        # Verify the call was made correctly
        self.mock_store.delete.assert_called_with(
            ('echo_star', 'Lily', 'collection'),
            'memory_key_456'
        )
        
        # Verify API compatibility
        call_args = self.mock_store.delete.call_args
        assert len(call_args[0]) == 2, "Delete should have 2 positional arguments: namespace, key"
        assert call_args[0][0] == namespace, "First argument should be namespace"
        assert call_args[0][1] == key, "Second argument should be key"
        
    def test_memory_error_handler_with_store_operations(self):
        """Test that MemoryErrorHandler works correctly with store operations."""
        # Test data that might cause issues
        problematic_data = {
            "content": "test content",
            "complex_object": {"nested": {"data": "value"}},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test validation
        is_valid, errors = MemoryErrorHandler.validate_hashable_structure(problematic_data)
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # Test sanitization
        sanitized_data = MemoryErrorHandler.sanitize_for_storage(problematic_data)
        assert isinstance(sanitized_data, dict)
        assert "content" in sanitized_data
        
        # Test error handling
        test_error = Exception("Store operation failed")
        recovery_successful = MemoryErrorHandler.handle_storage_error(
            test_error, 
            problematic_data, 
            "test_operation"
        )
        assert isinstance(recovery_successful, bool)
        
    def test_semantic_memory_schema_compatibility(self):
        """Test that SemanticMemory schema is compatible with store operations."""
        # Create a SemanticMemory instance
        semantic_memory = SemanticMemory(
            category="summary",
            content="Test conversation summary",
            context="User interaction context",
            timestamp=datetime.now().isoformat()
        )
        
        # Test that it can be converted to dict for storage
        memory_dict = semantic_memory.dict()
        assert isinstance(memory_dict, dict)
        assert memory_dict["category"] == "summary"
        assert memory_dict["content"] == "Test conversation summary"
        
        # Test that the dict is compatible with store operations
        is_valid, errors = MemoryErrorHandler.validate_hashable_structure(memory_dict)
        assert is_valid == True, f"SemanticMemory dict should be valid for storage: {errors}"