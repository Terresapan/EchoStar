#!/usr/bin/env python3
"""
Unit tests for condense_memory_node BaseStore API compatibility fixes.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from src.agents.nodes import condense_memory_node
from src.agents.schemas import SemanticMemory
from langgraph.store.memory import InMemoryStore


class TestCondenseMemoryNodeAPIFixes:
    """Test BaseStore API compatibility fixes in condense_memory_node."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock()
        self.mock_store = Mock(spec=InMemoryStore)
        self.mock_semantic_manager = Mock()
        self.mock_episodic_manager = Mock()
        
        # Mock state
        self.state = {
            "turn_count": 10,
            "messages": []
        }
        
        # Mock memory items
        self.mock_memory_item = Mock()
        self.mock_memory_item.value = {
            'content': {
                'user_message': 'Hello',
                'ai_response': 'Hi there!'
            }
        }
        self.mock_memory_item.key = 'test_key_1'
        
    @patch('src.agents.nodes.get_config_manager')
    def test_store_search_without_namespace_parameter(self, mock_config):
        """Test that store.search() calls use correct API signature without namespace parameter."""
        # Setup config mock
        mock_config_manager = Mock()
        mock_config_manager.get_memory_config.return_value.turns_to_summarize = 10
        mock_config.return_value = mock_config_manager
        
        # Setup store mock to return memories
        self.mock_store.search.return_value = [self.mock_memory_item]
        
        # Setup LLM mock
        mock_response = Mock()
        mock_response.content = "Test summary of conversation"
        self.mock_llm.invoke.return_value = mock_response
        
        # Call the function
        result = condense_memory_node(
            self.state,
            llm=self.mock_llm,
            store=self.mock_store,
            semantic_manager=self.mock_semantic_manager,
            episodic_manager=self.mock_episodic_manager
        )
        
        # Verify store.search was called with correct API signature (namespace as positional argument)
        search_calls = self.mock_store.search.call_args_list
        
        # First call should be for retrieving recent memories
        first_call = search_calls[0]
        assert first_call[0][0] == ('echo_star', 'Lily', 'collection')  # namespace as positional arg
        assert 'limit' in first_call[1]  # limit as keyword arg
        assert 'namespace' not in first_call[1]  # namespace should NOT be keyword arg
        
    @patch('src.agents.nodes.get_config_manager')
    def test_semantic_verification_search_api_compatibility(self, mock_config):
        """Test that semantic memory verification uses correct store.search API."""
        # Setup config mock
        mock_config_manager = Mock()
        mock_config_manager.get_memory_config.return_value.turns_to_summarize = 10
        mock_config.return_value = mock_config_manager
        
        # Setup store mock
        self.mock_store.search.return_value = [self.mock_memory_item]
        
        # Setup LLM mock
        mock_response = Mock()
        mock_response.content = "Test summary of conversation"
        self.mock_llm.invoke.return_value = mock_response
        
        # Call the function
        result = condense_memory_node(
            self.state,
            llm=self.mock_llm,
            store=self.mock_store,
            semantic_manager=self.mock_semantic_manager,
            episodic_manager=self.mock_episodic_manager
        )
        
        # Find the semantic verification search call
        search_calls = self.mock_store.search.call_args_list
        semantic_verification_call = None
        
        for call in search_calls:
            # Look for the call with facts namespace and category filter
            if (len(call[0]) > 0 and 
                call[0][0] == ('echo_star', 'Lily', 'facts') and
                'filter' in call[1] and 
                call[1]['filter'].get('category') == 'summary'):
                semantic_verification_call = call
                break
        
        # Verify the semantic verification call uses correct API
        assert semantic_verification_call is not None, "Semantic verification search call not found"
        assert semantic_verification_call[0][0] == ('echo_star', 'Lily', 'facts')  # namespace as positional
        assert 'filter' in semantic_verification_call[1]  # filter as keyword arg
        assert 'limit' in semantic_verification_call[1]  # limit as keyword arg
        assert 'namespace' not in semantic_verification_call[1]  # namespace should NOT be keyword arg
        
    @patch('src.agents.nodes.get_config_manager')
    def test_episodic_verification_search_api_compatibility(self, mock_config):
        """Test that episodic memory verification uses correct store.search API."""
        # Setup config mock
        mock_config_manager = Mock()
        mock_config_manager.get_memory_config.return_value.turns_to_summarize = 10
        mock_config.return_value = mock_config_manager
        
        # Setup store mock
        self.mock_store.search.return_value = [self.mock_memory_item]
        
        # Setup LLM mock
        mock_response = Mock()
        mock_response.content = "Test summary of conversation"
        self.mock_llm.invoke.return_value = mock_response
        
        # Call the function
        result = condense_memory_node(
            self.state,
            llm=self.mock_llm,
            store=self.mock_store,
            semantic_manager=self.mock_semantic_manager,
            episodic_manager=self.mock_episodic_manager
        )
        
        # Find the episodic verification search call
        search_calls = self.mock_store.search.call_args_list
        episodic_verification_call = None
        
        for call in search_calls:
            # Look for the call with collection namespace and limit 2
            if (len(call[0]) > 0 and 
                call[0][0] == ('echo_star', 'Lily', 'collection') and
                call[1].get('limit') == 2):
                episodic_verification_call = call
                break
        
        # Verify the episodic verification call uses correct API
        assert episodic_verification_call is not None, "Episodic verification search call not found"
        assert episodic_verification_call[0][0] == ('echo_star', 'Lily', 'collection')  # namespace as positional
        assert 'limit' in episodic_verification_call[1]  # limit as keyword arg
        assert 'namespace' not in episodic_verification_call[1]  # namespace should NOT be keyword arg
        
    @patch('src.agents.nodes.get_config_manager')
    def test_store_operation_error_handling(self, mock_config):
        """Test that store operation failures are properly handled."""
        # Setup config mock
        mock_config_manager = Mock()
        mock_config_manager.get_memory_config.return_value.turns_to_summarize = 10
        mock_config.return_value = mock_config_manager
        
        # Setup store mock to fail on initial search
        self.mock_store.search.side_effect = Exception("Store search failed")
        
        # Call the function
        result = condense_memory_node(
            self.state,
            llm=self.mock_llm,
            store=self.mock_store,
            semantic_manager=self.mock_semantic_manager,
            episodic_manager=self.mock_episodic_manager
        )
        
        # Verify error is handled gracefully
        assert "error" in result
        assert "Failed to retrieve memories for condensation" in result["error"]
        
    @patch('src.agents.nodes.get_config_manager')
    def test_semantic_storage_error_handling(self, mock_config):
        """Test that semantic storage failures are properly handled."""
        # Setup config mock
        mock_config_manager = Mock()
        mock_config_manager.get_memory_config.return_value.turns_to_summarize = 10
        mock_config.return_value = mock_config_manager
        
        # Setup store mock - successful search, failed put
        self.mock_store.search.return_value = [self.mock_memory_item]
        self.mock_store.put.side_effect = Exception("Store put failed")
        
        # Setup LLM mock
        mock_response = Mock()
        mock_response.content = "Test summary of conversation"
        self.mock_llm.invoke.return_value = mock_response
        
        # Call the function
        result = condense_memory_node(
            self.state,
            llm=self.mock_llm,
            store=self.mock_store,
            semantic_manager=self.mock_semantic_manager,
            episodic_manager=self.mock_episodic_manager
        )
        
        # Function should continue and try episodic storage
        # Since episodic storage should succeed, result should not be empty
        assert result != {}
        
    @patch('src.agents.nodes.get_config_manager')
    def test_episodic_storage_error_handling(self, mock_config):
        """Test that episodic storage failures are properly handled."""
        # Setup config mock
        mock_config_manager = Mock()
        mock_config_manager.get_memory_config.return_value.turns_to_summarize = 10
        mock_config.return_value = mock_config_manager
        
        # Setup store mock
        self.mock_store.search.return_value = [self.mock_memory_item]
        
        # Setup LLM mock
        mock_response = Mock()
        mock_response.content = "Test summary of conversation"
        self.mock_llm.invoke.return_value = mock_response
        
        # Setup episodic manager to fail
        self.mock_episodic_manager.submit.side_effect = Exception("Episodic submit failed")
        
        # Call the function
        result = condense_memory_node(
            self.state,
            llm=self.mock_llm,
            store=self.mock_store,
            semantic_manager=self.mock_semantic_manager,
            episodic_manager=self.mock_episodic_manager
        )
        
        # Function should continue and try semantic storage
        # Since semantic storage should succeed, result should not be empty
        assert result != {}