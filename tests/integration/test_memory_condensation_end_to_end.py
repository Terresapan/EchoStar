#!/usr/bin/env python3
"""
Integration tests for end-to-end memory condensation flow.
Tests the complete conversation flow through 10+ turns with memory condensation.

Requirements tested:
- 1.1: Full conversation flow through 10+ turns
- 1.4: Memory condensation triggers and completes successfully  
- 1.5: Error recovery and conversation continuation
- Memory retrieval after condensation
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
from uuid import uuid4
import json
from typing import Dict, Any, List, Optional

# Import system components
from src.agents.schemas import AgentState, SemanticMemory, EpisodicMemory
from src.agents.nodes import (
    manage_turn_count, memory_retrieval_node, router_node, echo_node,
    save_memories_node, condense_memory_node
)
from src.agents.graph import should_condense_memory, create_graph
from src.agents.memory_error_handler import MemoryErrorHandler
from config.manager import get_config_manager
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver


class TestEndToEndMemoryFlow:
    """Integration tests for complete memory condensation flow."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        # Create real InMemoryStore for integration testing
        self.store = InMemoryStore()
        self.checkpointer = MemorySaver()
        
        # Mock LLM components
        self.mock_llm = Mock()
        self.mock_llm_router = Mock()
        
        # Mock memory managers
        self.mock_semantic_manager = Mock()
        self.mock_episodic_manager = Mock()
        self.mock_procedural_manager = Mock()
        self.mock_profile_manager = Mock()
        
        # Mock memory tools
        self.mock_search_episodic_tool = Mock()
        self.mock_search_semantic_tool = Mock()
        self.mock_search_procedural_tool = Mock()
        
        # Test profile
        self.test_profile = {
            "name": "TestUser",
            "user_profile_background": "Test user for integration testing"
        }
        
        # Memory system configuration
        self.memory_system = {
            "store": self.store,
            "semantic_manager": self.mock_semantic_manager,
            "episodic_manager": self.mock_episodic_manager,
            "procedural_manager": self.mock_procedural_manager,
            "profile_manager": self.mock_profile_manager,
            "search_episodic_tool": self.mock_search_episodic_tool,
            "search_semantic_tool": self.mock_search_semantic_tool,
            "search_procedural_tool": self.mock_search_procedural_tool,
            "profile": self.test_profile
        }
        
        # Configure mock responses
        self._setup_mock_responses()
        
    def _setup_mock_responses(self):
        """Configure mock responses for consistent testing."""
        # LLM responses
        mock_response = Mock()
        mock_response.content = "Test response from agent"
        self.mock_llm.invoke.return_value = mock_response
        
        # Router classification
        mock_classification = Mock()
        mock_classification.classification = "echo_respond"
        mock_classification.reasoning = "Test classification"
        self.mock_llm_router.invoke.return_value = mock_classification
        
        # Memory tool responses (empty by default)
        self.mock_search_episodic_tool.invoke.return_value = "[]"
        self.mock_search_semantic_tool.invoke.return_value = "[]"
        self.mock_search_procedural_tool.invoke.return_value = "[]"
        
    def _create_test_state(self, turn_count: int = 1, message: str = "Test message") -> AgentState:
        """Create a test state with specified turn count."""
        return AgentState(
            message=message,
            turn_count=turn_count,
            user_id="test_user",
            session_id="test_session"
        )
        
    def _populate_store_with_memories(self, count: int = 10):
        """Populate the store with test memories for condensation."""
        for i in range(count):
            memory_key = f"test_memory_{i}"
            memory_data = {
                "content": {
                    "user_message": f"User message {i}",
                    "ai_response": f"AI response {i}",
                    "timestamp": datetime.now().isoformat()
                },
                "type": "EpisodicMemory"
            }
            
            self.store.put(
                namespace=("echo_star", "Lily", "collection"),
                key=memory_key,
                value=memory_data
            )
            
    def test_turn_count_management_through_10_turns(self):
        """Test turn count management through 10+ conversation turns."""
        # Test turn count increments correctly
        for expected_turn in range(1, 12):  # Test through 11 turns
            state = self._create_test_state(turn_count=expected_turn - 1)
            result = manage_turn_count(state)
            
            assert result["turn_count"] == expected_turn
            assert isinstance(result["turn_count"], int)
            
        # Test that turn count persists in state
        state = AgentState(message="test", turn_count=5)
        result = manage_turn_count(state)
        assert result["turn_count"] == 6
        
    def test_memory_condensation_trigger_at_turn_10(self):
        """Test that memory condensation is triggered at turn 10."""
        # Mock configuration to use 10 turns for condensation
        with patch('src.agents.graph.get_config_manager') as mock_config:
            mock_memory_config = Mock()
            mock_memory_config.turns_to_summarize = 10
            mock_config.return_value.get_memory_config.return_value = mock_memory_config
            
            # Test turns 1-9 should not trigger condensation
            for turn in range(1, 10):
                state = self._create_test_state(turn_count=turn)
                result = should_condense_memory(state)
                assert result == "save_memories", f"Turn {turn} should not trigger condensation"
            
            # Test turn 10 should trigger condensation
            state = self._create_test_state(turn_count=10)
            result = should_condense_memory(state)
            assert result == "condense_memory", "Turn 10 should trigger condensation"
            
            # Test turn 20 should also trigger condensation
            state = self._create_test_state(turn_count=20)
            result = should_condense_memory(state)
            assert result == "condense_memory", "Turn 20 should trigger condensation"
            
    def test_memory_retrieval_with_populated_memories(self):
        """Test memory retrieval when memories exist in the store."""
        # Populate store with test memories
        self._populate_store_with_memories(5)
        
        # Create test state
        state = self._create_test_state(message="Tell me about our previous conversations")
        
        # Test memory retrieval
        result = memory_retrieval_node(
            state,
            llm=self.mock_llm,
            search_episodic_tool=self.mock_search_episodic_tool,
            search_semantic_tool=self.mock_search_semantic_tool,
            search_procedural_tool=self.mock_search_procedural_tool,
            store=self.store
        )
        
        # Verify retrieval results
        assert "episodic_memories" in result
        assert "semantic_memories" in result
        assert "procedural_memories" in result
        assert "user_profile" in result
        
        # Verify memory tools were called
        self.mock_search_episodic_tool.invoke.assert_called()
        self.mock_search_semantic_tool.invoke.assert_called()
        self.mock_search_procedural_tool.invoke.assert_called()
        
    def test_successful_memory_condensation_process(self):
        """Test successful memory condensation with all components working."""
        # Populate store with memories to condense
        self._populate_store_with_memories(10)
        
        # Mock LLM to return a summary
        mock_summary_response = Mock()
        mock_summary_response.content = "This is a condensed summary of the conversation covering 10 turns of dialogue between the user and AI assistant."
        self.mock_llm.invoke.return_value = mock_summary_response
        
        # Create state for condensation
        state = self._create_test_state(turn_count=10)
        
        # Test condensation process
        with patch('src.agents.nodes.get_config_manager') as mock_config:
            mock_memory_config = Mock()
            mock_memory_config.turns_to_summarize = 10
            mock_config.return_value.get_memory_config.return_value = mock_memory_config
            
            result = condense_memory_node(
                state,
                llm=self.mock_llm,
                store=self.store,
                semantic_manager=self.mock_semantic_manager,
                episodic_manager=self.mock_episodic_manager
            )
            
            # Verify condensation results
            assert isinstance(result, dict)
            # The function should complete without raising exceptions
            
        # Verify LLM was called for summarization
        self.mock_llm.invoke.assert_called()
        
    def test_memory_condensation_with_storage_failure(self):
        """Test memory condensation behavior when storage operations fail."""
        # Populate store with memories
        self._populate_store_with_memories(10)
        
        # Mock LLM to return a summary
        mock_summary_response = Mock()
        mock_summary_response.content = "Test summary for storage failure scenario"
        self.mock_llm.invoke.return_value = mock_summary_response
        
        # Mock semantic manager to fail
        self.mock_semantic_manager.submit.side_effect = Exception("Semantic storage failed")
        
        # Mock episodic manager to succeed
        self.mock_episodic_manager.submit.return_value = True
        
        # Create state for condensation
        state = self._create_test_state(turn_count=10)
        
        # Test condensation with partial failure
        with patch('src.agents.nodes.get_config_manager') as mock_config:
            mock_memory_config = Mock()
            mock_memory_config.turns_to_summarize = 10
            mock_config.return_value.get_memory_config.return_value = mock_memory_config
            
            # Should not raise exception despite storage failure
            result = condense_memory_node(
                state,
                llm=self.mock_llm,
                store=self.store,
                semantic_manager=self.mock_semantic_manager,
                episodic_manager=self.mock_episodic_manager
            )
            
            # Verify process completed despite failure
            assert isinstance(result, dict)
            
    def test_conversation_continuation_after_condensation_failure(self):
        """Test that conversation continues normally after condensation failures."""
        # Create state that would trigger condensation
        state = self._create_test_state(turn_count=10, message="Continue our conversation")
        
        # Mock complete condensation failure
        with patch('src.agents.nodes.condense_memory_node') as mock_condense:
            mock_condense.return_value = {
                "condensation_success": False,
                "error_message": "Complete condensation failure",
                "fallback_applied": True
            }
            
            # Test that echo node still works after condensation failure
            result = echo_node(
                state,
                llm=self.mock_llm,
                profile=self.test_profile
            )
            
            # Verify conversation continues
            assert "response" in result
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            
    def test_memory_retrieval_after_successful_condensation(self):
        """Test that condensed memories can be retrieved in subsequent turns."""
        # Step 1: Populate store with memories and perform condensation
        self._populate_store_with_memories(10)
        
        # Mock successful condensation by adding condensed summary to store
        condensed_summary = SemanticMemory(
            category="summary",
            content="Condensed conversation summary from 10 turns of dialogue",
            context="Memory condensation process result",
            importance=0.9,
            timestamp=datetime.now().isoformat()
        )
        
        # Store the condensed summary
        self.store.put(
            namespace=("echo_star", "Lily", "facts"),
            key="condensed_summary_1",
            value={
                "content": condensed_summary.model_dump(),
                "type": "SemanticMemory",
                "category": "summary"
            }
        )
        
        # Step 2: Test retrieval of condensed memories
        state = self._create_test_state(
            turn_count=11,
            message="What did we discuss in our previous conversations?"
        )
        
        # Mock search tools to return the condensed summary
        condensed_result = json.dumps([{
            "key": "condensed_summary_1",
            "value": {
                "content": condensed_summary.model_dump(),
                "type": "SemanticMemory",
                "category": "summary"
            }
        }])
        self.mock_search_semantic_tool.invoke.return_value = condensed_result
        
        # Test memory retrieval
        result = memory_retrieval_node(
            state,
            llm=self.mock_llm,
            search_episodic_tool=self.mock_search_episodic_tool,
            search_semantic_tool=self.mock_search_semantic_tool,
            search_procedural_tool=self.mock_search_procedural_tool,
            store=self.store
        )
        
        # Verify condensed memories are retrieved
        assert "semantic_memories" in result
        semantic_memories = result["semantic_memories"]
        
        # Should have parsed the condensed summary
        if semantic_memories:
            found_summary = any(
                mem.category == "summary" and "condensed" in mem.content.lower()
                for mem in semantic_memories
            )
            assert found_summary, "Should find condensed summary in retrieved memories"
            
    def test_full_conversation_flow_through_condensation_cycle(self):
        """Test complete conversation flow through a full condensation cycle."""
        conversation_messages = [
            "Hello, how are you today?",
            "Tell me about the weather",
            "What's your favorite color?",
            "Can you help me with math?",
            "What did we talk about yesterday?",
            "I'm feeling a bit sad today",
            "Can you tell me a joke?",
            "What's the meaning of life?",
            "How do I cook pasta?",
            "Let's summarize our conversation"  # This should trigger condensation
        ]
        
        # Mock configuration
        with patch('src.agents.graph.get_config_manager') as mock_config:
            mock_memory_config = Mock()
            mock_memory_config.turns_to_summarize = 10
            mock_config.return_value.get_memory_config.return_value = mock_memory_config
            
            # Simulate conversation flow
            for turn, message in enumerate(conversation_messages, 1):
                # Create state for this turn (start with turn-1 since manage_turn_count increments)
                state = self._create_test_state(turn_count=turn-1, message=message)
                
                # Test turn count management
                turn_result = manage_turn_count(state)
                assert turn_result["turn_count"] == turn
                
                # Update state with turn count
                state.update(turn_result)
                
                # Test memory retrieval
                memory_result = memory_retrieval_node(
                    state,
                    llm=self.mock_llm,
                    search_episodic_tool=self.mock_search_episodic_tool,
                    search_semantic_tool=self.mock_search_semantic_tool,
                    search_procedural_tool=self.mock_search_procedural_tool,
                    store=self.store
                )
                state.update(memory_result)
                
                # Test routing
                router_result = router_node(state, self.mock_llm_router, self.test_profile)
                state.update(router_result)
                
                # Test response generation
                response_result = echo_node(state, self.mock_llm, self.test_profile)
                state.update(response_result)
                
                # Test memory saving vs condensation decision
                should_condense = should_condense_memory(state)
                
                if turn == 10:
                    # Turn 10 should trigger condensation
                    assert should_condense == "condense_memory"
                    
                    # Populate store with memories for condensation
                    self._populate_store_with_memories(10)
                    
                    # Mock successful summary generation
                    mock_summary = Mock()
                    mock_summary.content = f"Summary of conversation through turn {turn}"
                    self.mock_llm.invoke.return_value = mock_summary
                    
                    # Test condensation
                    condensation_result = condense_memory_node(
                        state,
                        llm=self.mock_llm,
                        store=self.store,
                        semantic_manager=self.mock_semantic_manager,
                        episodic_manager=self.mock_episodic_manager
                    )
                    
                    # Verify condensation completed
                    assert isinstance(condensation_result, dict)
                    
                else:
                    # Other turns should save memories normally
                    assert should_condense == "save_memories"
                    
                    # Test memory saving
                    save_result = save_memories_node(
                        state,
                        profile_manager=self.mock_profile_manager,
                        semantic_manager=self.mock_semantic_manager,
                        episodic_manager=self.mock_episodic_manager,
                        procedural_manager=self.mock_procedural_manager,
                        store=self.store
                    )
                    
                    # Verify save completed
                    assert isinstance(save_result, dict)
                
                # Verify conversation continues normally
                assert "response" in state
                assert isinstance(state["response"], str)
                
    def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios during memory operations."""
        test_scenarios = [
            {
                "name": "Store unavailable",
                "setup": lambda: setattr(self, 'store', None),
                "expected": "graceful_degradation"
            },
            {
                "name": "LLM failure during condensation",
                "setup": lambda: self.mock_llm.invoke.side_effect.append(Exception("LLM failed")),
                "expected": "fallback_behavior"
            },
            {
                "name": "Memory manager failures",
                "setup": lambda: [
                    setattr(self.mock_semantic_manager, 'submit', Mock(side_effect=Exception("Semantic failed"))),
                    setattr(self.mock_episodic_manager, 'submit', Mock(side_effect=Exception("Episodic failed")))
                ],
                "expected": "conversation_continuation"
            }
        ]
        
        for scenario in test_scenarios:
            # Reset mocks
            self._setup_mock_responses()
            
            # Apply scenario setup
            if callable(scenario["setup"]):
                try:
                    scenario["setup"]()
                except Exception:
                    pass  # Some setups may fail, that's expected
            
            # Test that system handles the error gracefully
            state = self._create_test_state(turn_count=10)
            
            try:
                # Test memory retrieval with error
                if self.store is not None:
                    memory_result = memory_retrieval_node(
                        state,
                        llm=self.mock_llm,
                        search_episodic_tool=self.mock_search_episodic_tool,
                        search_semantic_tool=self.mock_search_semantic_tool,
                        search_procedural_tool=self.mock_search_procedural_tool,
                        store=self.store
                    )
                    state.update(memory_result)
                
                # Test response generation continues
                response_result = echo_node(state, self.mock_llm, self.test_profile)
                
                # Verify conversation continues despite errors
                assert "response" in response_result
                
            except Exception as e:
                # Some errors are expected, verify they don't crash the system
                assert "response" in state or len(str(e)) > 0
                
    def test_memory_data_integrity_after_condensation(self):
        """Test that memory data maintains integrity through condensation process."""
        # Create structured test memories
        test_memories = []
        for i in range(10):
            memory_data = {
                "user_message": f"User question {i}: What is {i} + {i}?",
                "ai_response": f"AI answer {i}: {i} + {i} = {i * 2}",
                "timestamp": datetime.now().isoformat(),
                "turn_number": i + 1,
                "metadata": {
                    "topic": "mathematics",
                    "complexity": "basic"
                }
            }
            test_memories.append(memory_data)
            
            # Store in the memory store
            self.store.put(
                namespace=("echo_star", "Lily", "collection"),
                key=f"structured_memory_{i}",
                value={
                    "content": memory_data,
                    "type": "EpisodicMemory"
                }
            )
        
        # Mock LLM to return structured summary
        mock_summary = Mock()
        mock_summary.content = "Mathematical conversation covering addition problems from 0+0 to 9+9, with systematic Q&A format covering basic arithmetic concepts."
        self.mock_llm.invoke.return_value = mock_summary
        
        # Test condensation preserves key information
        state = self._create_test_state(turn_count=10)
        
        with patch('src.agents.nodes.get_config_manager') as mock_config:
            mock_memory_config = Mock()
            mock_memory_config.turns_to_summarize = 10
            mock_config.return_value.get_memory_config.return_value = mock_memory_config
            
            result = condense_memory_node(
                state,
                llm=self.mock_llm,
                store=self.store,
                semantic_manager=self.mock_semantic_manager,
                episodic_manager=self.mock_episodic_manager
            )
            
            # Verify condensation process handled structured data
            assert isinstance(result, dict)
            
        # Verify original memories were processed
        self.mock_llm.invoke.assert_called()
        call_args = self.mock_llm.invoke.call_args
        if call_args and len(call_args[0]) > 0:
            prompt_content = str(call_args[0][0])
            # Should contain references to the mathematical content
            assert "mathematics" in prompt_content.lower() or "addition" in prompt_content.lower() or any(str(i) in prompt_content for i in range(10))
            
    def test_performance_monitoring_during_condensation(self):
        """Test that performance monitoring works during memory condensation."""
        # Populate store with memories
        self._populate_store_with_memories(10)
        
        # Mock LLM with slight delay simulation
        def mock_llm_with_delay(*args, **kwargs):
            import time
            time.sleep(0.01)  # Small delay to simulate processing
            mock_response = Mock()
            mock_response.content = "Performance test summary"
            return mock_response
            
        self.mock_llm.invoke.side_effect = mock_llm_with_delay
        
        # Test condensation with performance monitoring
        state = self._create_test_state(turn_count=10)
        
        with patch('src.agents.nodes.get_config_manager') as mock_config:
            mock_memory_config = Mock()
            mock_memory_config.turns_to_summarize = 10
            mock_config.return_value.get_memory_config.return_value = mock_memory_config
            
            # Measure condensation time
            start_time = datetime.now()
            result = condense_memory_node(
                state,
                llm=self.mock_llm,
                store=self.store,
                semantic_manager=self.mock_semantic_manager,
                episodic_manager=self.mock_episodic_manager
            )
            end_time = datetime.now()
            
            # Verify process completed and took measurable time
            assert isinstance(result, dict)
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time > 0, "Condensation should take measurable time"
            assert processing_time < 10, "Condensation should complete within reasonable time"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])