#!/usr/bin/env python3
"""
Test script to verify embedding optimization functionality.

This script tests the EmbeddingManager implementation to ensure:
1. Caching functionality works correctly
2. Batch processing reduces API calls
3. Cache statistics are accurate
4. Integration with memory retrieval is functional
"""

import sys
import os
import time
from unittest.mock import Mock, MagicMock
from typing import List, Any, Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the logging utilities to avoid import issues in testing
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

# Patch the logging import before importing embedding_manager
import sys
from unittest.mock import Mock
sys.modules['src.utils.logging_utils'] = Mock()
sys.modules['src.utils.logging_utils'].get_logger = Mock(return_value=MockLogger())

from agents.embedding_manager import EmbeddingManager


class MockSearchTool:
    """Mock search tool for testing."""
    
    def __init__(self, name: str = "mock_search_tool"):
        self.name = name
        self.call_count = 0
        self.calls = []
    
    def invoke(self, query: str) -> List[Dict[str, Any]]:
        """Mock invoke method that tracks calls."""
        self.call_count += 1
        self.calls.append(query)
        
        # Simulate some processing time
        time.sleep(0.01)
        
        # Return mock search results
        return [
            {
                "value": {
                    "content": {
                        "user_message": f"Mock result for: {query}",
                        "ai_response": f"Mock AI response for: {query}",
                        "timestamp": "2024-01-01T00:00:00"
                    }
                }
            }
        ]


def test_embedding_manager_caching():
    """Test that caching works correctly."""
    print("Testing EmbeddingManager caching functionality...")
    
    # Initialize embedding manager with small cache for testing
    manager = EmbeddingManager(cache_size=10, cache_ttl=60)
    mock_tool = MockSearchTool("test_episodic_tool")
    
    # Test 1: First call should miss cache
    query = "What did we discuss yesterday?"
    result1 = manager.optimized_search(mock_tool, query, search_type="episodic")
    
    assert mock_tool.call_count == 1, f"Expected 1 API call, got {mock_tool.call_count}"
    assert len(result1) > 0, "Expected non-empty result"
    
    # Test 2: Second identical call should hit cache
    result2 = manager.optimized_search(mock_tool, query, search_type="episodic")
    
    assert mock_tool.call_count == 1, f"Expected 1 API call (cached), got {mock_tool.call_count}"
    assert result1 == result2, "Cached result should be identical"
    
    # Test 3: Different query should miss cache
    query2 = "Tell me about our conversation"
    result3 = manager.optimized_search(mock_tool, query2, search_type="episodic")
    
    assert mock_tool.call_count == 2, f"Expected 2 API calls, got {mock_tool.call_count}"
    assert len(result3) > 0, "Expected non-empty result"
    
    # Test 4: Check cache statistics
    stats = manager.get_cache_stats()
    assert stats["cache_hits"] == 1, f"Expected 1 cache hit, got {stats['cache_hits']}"
    assert stats["cache_misses"] == 2, f"Expected 2 cache misses, got {stats['cache_misses']}"
    assert stats["api_calls_saved"] == 1, f"Expected 1 API call saved, got {stats['api_calls_saved']}"
    
    print("✓ Caching functionality test passed")


def test_batch_processing():
    """Test batch processing functionality."""
    print("Testing batch processing functionality...")
    
    manager = EmbeddingManager(cache_size=10, cache_ttl=60)
    mock_tool = MockSearchTool("test_batch_tool")
    
    # Test batch processing with duplicate queries
    queries = [
        "What is the weather?",
        "How are you?", 
        "What is the weather?",  # Duplicate
        "Tell me a joke",
        "How are you?"  # Duplicate
    ]
    
    results = manager.batch_search(mock_tool, queries, search_type="semantic")
    
    # Should only make 3 unique API calls (deduplication)
    assert mock_tool.call_count == 3, f"Expected 3 unique API calls, got {mock_tool.call_count}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    
    # Results for duplicate queries should be identical
    assert results[0] == results[2], "Duplicate query results should be identical"
    assert results[1] == results[4], "Duplicate query results should be identical"
    
    # Check that API calls were saved
    stats = manager.get_cache_stats()
    assert stats["api_calls_saved"] == 2, f"Expected 2 API calls saved, got {stats['api_calls_saved']}"
    
    print("✓ Batch processing test passed")


def test_cache_expiration():
    """Test cache expiration functionality."""
    print("Testing cache expiration functionality...")
    
    # Use very short TTL for testing
    manager = EmbeddingManager(cache_size=10, cache_ttl=1)  # 1 second TTL
    mock_tool = MockSearchTool("test_expiration_tool")
    
    query = "Test expiration query"
    
    # First call
    result1 = manager.optimized_search(mock_tool, query)
    assert mock_tool.call_count == 1
    
    # Second call immediately - should hit cache
    result2 = manager.optimized_search(mock_tool, query)
    assert mock_tool.call_count == 1  # Still cached
    
    # Wait for cache to expire
    time.sleep(1.5)
    
    # Third call after expiration - should miss cache
    result3 = manager.optimized_search(mock_tool, query)
    assert mock_tool.call_count == 2  # Cache expired, new API call
    
    print("✓ Cache expiration test passed")


def test_cache_size_limit():
    """Test cache size limit functionality."""
    print("Testing cache size limit functionality...")
    
    # Use very small cache size
    manager = EmbeddingManager(cache_size=2, cache_ttl=60)
    mock_tool = MockSearchTool("test_size_limit_tool")
    
    # Fill cache beyond limit
    queries = ["query1", "query2", "query3"]  # 3 queries, cache size 2
    
    for query in queries:
        manager.optimized_search(mock_tool, query)
    
    assert mock_tool.call_count == 3
    
    # Cache should only hold 2 items
    stats = manager.get_cache_stats()
    assert stats["search_cache_size"] <= 2, f"Cache size should be <= 2, got {stats['search_cache_size']}"
    
    # First query should have been evicted, so it should cause a new API call
    manager.optimized_search(mock_tool, "query1")
    assert mock_tool.call_count == 4  # New API call because query1 was evicted
    
    print("✓ Cache size limit test passed")


def test_integration_with_memory_retrieval():
    """Test integration with memory retrieval patterns."""
    print("Testing integration with memory retrieval patterns...")
    
    manager = EmbeddingManager(cache_size=100, cache_ttl=3600)
    
    # Mock different search tools
    episodic_tool = MockSearchTool("episodic_search")
    semantic_tool = MockSearchTool("semantic_search") 
    procedural_tool = MockSearchTool("procedural_search")
    
    user_message = "How can I improve my productivity?"
    
    # Simulate memory retrieval pattern from nodes.py
    # This simulates the "general" retrieval type that searches all memory types
    
    # Step 1: Search episodic memories
    episodic_results = manager.optimized_search(
        episodic_tool, 
        user_message,
        search_type="episodic",
        limit=10
    )
    
    # Step 2: Search semantic memories  
    semantic_results = manager.optimized_search(
        semantic_tool,
        user_message, 
        search_type="semantic",
        limit=10
    )
    
    # Step 3: Search procedural memories
    procedural_results = manager.optimized_search(
        procedural_tool,
        user_message,
        search_type="procedural", 
        limit=5
    )
    
    # Verify all searches completed
    assert len(episodic_results) > 0
    assert len(semantic_results) > 0  
    assert len(procedural_results) > 0
    
    # Verify each tool was called once
    assert episodic_tool.call_count == 1
    assert semantic_tool.call_count == 1
    assert procedural_tool.call_count == 1
    
    # Simulate repeated retrieval with same message (should hit cache)
    episodic_results2 = manager.optimized_search(
        episodic_tool,
        user_message,
        search_type="episodic", 
        limit=10
    )
    
    # Should not make additional API call
    assert episodic_tool.call_count == 1
    assert episodic_results == episodic_results2
    
    # Check final statistics
    stats = manager.get_cache_stats()
    assert stats["cache_hits"] >= 1
    assert stats["api_calls_saved"] >= 1
    
    print("✓ Integration test passed")


def run_all_tests():
    """Run all embedding optimization tests."""
    print("=" * 60)
    print("Running Embedding Optimization Tests")
    print("=" * 60)
    
    try:
        test_embedding_manager_caching()
        test_batch_processing()
        test_cache_expiration()
        test_cache_size_limit()
        test_integration_with_memory_retrieval()
        
        print("=" * 60)
        print("✅ All tests passed! Embedding optimization is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Test failed: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)