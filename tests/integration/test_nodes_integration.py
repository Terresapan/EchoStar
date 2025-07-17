#!/usr/bin/env python3
"""
Test script to verify nodes.py integration with EmbeddingManager.

This script tests that the memory_retrieval_node function correctly uses
the EmbeddingManager for optimization.
"""

import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the logging utilities and config manager
class MockLogger:
    def info(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def performance_timer(self, *args, **kwargs):
        class MockTimer:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return MockTimer()

class MockConfig:
    def __init__(self):
        self.search_limit = 10
        self.procedural_search_limit = 5

class MockConfigManager:
    def get_memory_config(self):
        return MockConfig()

# Patch the imports before importing nodes
sys.modules['src.utils.logging_utils'] = Mock()
sys.modules['src.utils.logging_utils'].get_logger = Mock(return_value=MockLogger())
sys.modules['src.utils.logging_utils'].validate_user_input = Mock(return_value=Mock(is_valid=True))
sys.modules['src.utils.logging_utils'].validate_memory = Mock(return_value=Mock(is_valid=True, error_messages=[]))

sys.modules['config.manager'] = Mock()
sys.modules['config.manager'].get_config_manager = Mock(return_value=MockConfigManager())

# Mock the schemas
sys.modules['agents.schemas'] = Mock()
sys.modules['agents.schemas'].RetrievalClassifier = Mock()
sys.modules['agents.schemas'].EpisodicMemory = Mock()
sys.modules['agents.schemas'].SemanticMemory = Mock()
sys.modules['agents.schemas'].ProceduralMemory = Mock()

def test_nodes_integration():
    """Test that nodes.py correctly integrates with EmbeddingManager."""
    print("Testing nodes.py integration with EmbeddingManager...")
    
    # Import after mocking
    from agents.nodes import get_embedding_manager, memory_retrieval_node
    
    # Test that get_embedding_manager works
    manager1 = get_embedding_manager()
    manager2 = get_embedding_manager()
    
    # Should return the same instance (singleton pattern)
    assert manager1 is manager2, "get_embedding_manager should return the same instance"
    
    # Test that the manager has the expected methods
    assert hasattr(manager1, 'optimized_search'), "EmbeddingManager should have optimized_search method"
    assert hasattr(manager1, 'get_cache_stats'), "EmbeddingManager should have get_cache_stats method"
    assert hasattr(manager1, 'batch_search'), "EmbeddingManager should have batch_search method"
    
    print("✓ EmbeddingManager integration test passed")
    
    # Test memory_retrieval_node function signature and basic functionality
    try:
        # Create mock objects for the function parameters
        mock_state = {"message": "Test message"}
        mock_llm = Mock()
        mock_llm.with_structured_output = Mock(return_value=Mock())
        mock_llm.with_structured_output.return_value.invoke = Mock(return_value=Mock(retrieval_type="general"))
        
        mock_search_tool = Mock()
        mock_search_tool.invoke = Mock(return_value=[])
        
        mock_store = Mock()
        mock_store.search = Mock(return_value=[])
        
        # Test that the function can be called without errors
        result = memory_retrieval_node(
            mock_state,
            llm=mock_llm,
            search_episodic_tool=mock_search_tool,
            search_semantic_tool=mock_search_tool,
            search_procedural_tool=mock_search_tool,
            store=mock_store
        )
        
        # Verify the result has the expected structure
        assert isinstance(result, dict), "memory_retrieval_node should return a dictionary"
        expected_keys = ["episodic_memories", "semantic_memories", "procedural_memories", "user_profile"]
        for key in expected_keys:
            assert key in result, f"Result should contain key: {key}"
        
        print("✓ memory_retrieval_node integration test passed")
        
    except Exception as e:
        print(f"❌ memory_retrieval_node integration test failed: {str(e)}")
        raise
    
    print("✓ All nodes.py integration tests passed")


if __name__ == "__main__":
    try:
        test_nodes_integration()
        print("=" * 60)
        print("✅ All integration tests passed!")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"❌ Integration test failed: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)