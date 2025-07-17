#!/usr/bin/env python3
"""
Test script to verify that the memory condensation fixes are working properly.
This script tests the enhanced error handling and data structure fixes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.memory_error_handler import MemoryErrorHandler, ProfileSerializer
from src.agents.schemas import SemanticMemory
from langgraph.store.memory import InMemoryStore
from datetime import datetime
import json

def test_memory_error_handler():
    """Test the MemoryErrorHandler functionality."""
    print("Testing MemoryErrorHandler...")
    
    # Test hashable structure validation
    valid_data = {"name": "Lily", "age": 25, "preferences": ["fish", "swimming"]}
    is_valid, errors = MemoryErrorHandler.validate_hashable_structure(valid_data)
    print(f"Valid data test: {is_valid}, errors: {errors}")
    
    # Test invalid data with nested dict as key (this would cause the original error)
    invalid_data = {("nested", "tuple"): "value", "normal_key": {"nested": {"dict": "value"}}}
    is_valid, errors = MemoryErrorHandler.validate_hashable_structure(invalid_data)
    print(f"Invalid data test: {is_valid}, errors: {errors}")
    
    # Test sanitization
    sanitized = MemoryErrorHandler.sanitize_for_storage(invalid_data)
    print(f"Sanitized data: {sanitized}")
    
    print("MemoryErrorHandler tests completed.\n")

def test_profile_serializer():
    """Test the ProfileSerializer functionality."""
    print("Testing ProfileSerializer...")
    
    # Test profile with complex nested structures
    complex_profile = {
        "name": "Lily",
        "preferences": {
            "food": ["fish", "not cheese"],
            "activities": {"likes": "swimming", "dislikes": "running"}
        },
        "metadata": {"created": datetime.now(), "version": 1.0}
    }
    
    # Test serialization
    serialized = ProfileSerializer.serialize_profile(complex_profile)
    print(f"Serialized profile: {serialized}")
    
    # Test deserialization
    deserialized = ProfileSerializer.deserialize_profile(serialized)
    print(f"Deserialized profile: {deserialized}")
    
    # Test validation
    is_valid, errors = ProfileSerializer.validate_profile_structure(serialized)
    print(f"Profile validation: {is_valid}, errors: {errors}")
    
    print("ProfileSerializer tests completed.\n")

def test_semantic_memory_structure():
    """Test the corrected semantic memory storage structure."""
    print("Testing semantic memory storage structure...")
    
    # Create a semantic memory object like in the condensation process
    semantic_memory = SemanticMemory(
        category="summary",
        content="Test condensed conversation summary",
        context="Test context for condensed summary",
        importance=0.9,
        timestamp=datetime.now().isoformat()
    )
    
    # Test the correct data structure for storage
    semantic_memory_data = semantic_memory.model_dump()
    print(f"Semantic memory data: {semantic_memory_data}")
    
    # Test the storage structure that should work with the parsing code
    storage_structure = {
        "content": semantic_memory_data,  # This is the correct structure
        "type": "SemanticMemory",
        "category": "summary"
    }
    
    print(f"Storage structure: {storage_structure}")
    
    # Validate the structure
    is_valid, errors = MemoryErrorHandler.validate_hashable_structure(storage_structure)
    print(f"Storage structure validation: {is_valid}, errors: {errors}")
    
    print("Semantic memory structure tests completed.\n")

def test_memory_store_operations():
    """Test actual memory store operations with the fixed structure."""
    print("Testing memory store operations...")
    
    try:
        # Create an in-memory store
        store = InMemoryStore()
        
        # Test semantic memory storage with correct structure
        semantic_memory = SemanticMemory(
            category="summary",
            content="Test condensed conversation summary for storage verification",
            context="Test context to verify storage and retrieval works correctly",
            importance=0.9,
            timestamp=datetime.now().isoformat()
        )
        
        semantic_memory_data = semantic_memory.model_dump()
        
        # Store with the correct structure
        test_key = "test-semantic-key"
        store.put(
            namespace=("echo_star", "Lily", "facts"),
            key=test_key,
            value={
                "content": semantic_memory_data,
                "type": "SemanticMemory",
                "category": "summary"
            }
        )
        
        print("Semantic memory stored successfully")
        
        # Test retrieval
        retrieved = store.search(
            namespace=("echo_star", "Lily", "facts"),
            filter={"category": "summary"},
            limit=1
        )
        
        if retrieved:
            print(f"Retrieved {len(retrieved)} semantic memories")
            print(f"Retrieved data structure: {retrieved[0].value}")
            
            # Test parsing the retrieved data (simulate the parsing code)
            retrieved_value = retrieved[0].value
            if isinstance(retrieved_value, dict) and 'content' in retrieved_value:
                content_data = retrieved_value['content']
                if isinstance(content_data, dict):
                    # This should work now with the corrected structure
                    parsed_semantic = SemanticMemory(**content_data)
                    print(f"Successfully parsed semantic memory: {parsed_semantic.category}")
                else:
                    print(f"Content data is not a dict: {type(content_data)}")
            else:
                print(f"Retrieved value structure is incorrect: {retrieved_value}")
        else:
            print("No semantic memories retrieved")
        
        print("Memory store operations tests completed.\n")
        
    except Exception as e:
        print(f"Memory store operations test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("=== Memory Condensation Fix Tests ===\n")
    
    test_memory_error_handler()
    test_profile_serializer()
    test_semantic_memory_structure()
    test_memory_store_operations()
    
    print("=== All Tests Completed ===")
    print("\nThe memory condensation fixes should now:")
    print("1. Properly validate and sanitize data structures")
    print("2. Handle 'unhashable type' errors gracefully")
    print("3. Store condensed summaries in the correct format")
    print("4. Enable successful retrieval and parsing of stored memories")

if __name__ == "__main__":
    main()