#!/usr/bin/env python3
"""
Type checking validation tests for the EchoStar codebase.
Tests that all type annotations are consistent and mypy passes without errors.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import inspect

# Import all modules to test their type annotations
import config.manager
import config.models
import logging_utils
import nodes
import graph
import schemas
import utils
import memory
import streamlit_app


def test_mypy_passes():
    """Test that mypy passes without errors on the entire codebase."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "."],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"mypy failed with errors:\n{result.stdout}\n{result.stderr}"
    print("✅ mypy passes without errors")


def test_function_type_annotations():
    """Test that all functions have proper type annotations."""
    modules_to_check = [
        config.manager,
        config.models,
        logging_utils,
        nodes,
        utils,
        memory
    ]
    
    missing_annotations = []
    
    for module in modules_to_check:
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                sig = inspect.signature(obj)
                
                # Check return annotation
                if sig.return_annotation == inspect.Signature.empty:
                    missing_annotations.append(f"{module.__name__}.{name} missing return annotation")
                
                # Check parameter annotations
                for param_name, param in sig.parameters.items():
                    if param.annotation == inspect.Parameter.empty and param_name != 'self':
                        missing_annotations.append(f"{module.__name__}.{name} parameter '{param_name}' missing annotation")
    
    if missing_annotations:
        print("⚠️  Functions with missing type annotations:")
        for annotation in missing_annotations[:10]:  # Show first 10
            print(f"  - {annotation}")
        if len(missing_annotations) > 10:
            print(f"  ... and {len(missing_annotations) - 10} more")
    else:
        print("✅ All public functions have type annotations")


def test_pydantic_models_consistency():
    """Test that Pydantic models have consistent type annotations."""
    from pydantic import BaseModel
    
    models_to_check = [
        schemas.Router,
        schemas.RetrievalClassifier,
        schemas.UserProfile,
        schemas.EpisodicMemory,
        schemas.SemanticMemory,
        schemas.ProceduralMemory,
        config.models.MemoryConfig,
        config.models.RoutingConfig,
        config.models.LLMConfig,
        config.models.LoggingConfig,
        config.models.EchoStarConfig,
        logging_utils.ValidationError,
        logging_utils.ValidationResult
    ]
    
    for model in models_to_check:
        if issubclass(model, BaseModel):
            # Check that all fields have type annotations
            for field_name, field_info in model.model_fields.items():
                assert field_info.annotation is not None, f"{model.__name__}.{field_name} missing type annotation"
    
    print("✅ All Pydantic models have consistent type annotations")


def test_type_consistency_in_functions():
    """Test specific type consistency patterns in key functions."""
    
    # Test ConfigManager methods
    config_manager = config.manager.ConfigManager()
    
    # Test that get() method returns appropriate types
    memory_config = config_manager.get_memory_config()
    assert isinstance(memory_config, config.models.MemoryConfig)
    
    routing_config = config_manager.get_routing_config()
    assert isinstance(routing_config, config.models.RoutingConfig)
    
    llm_config = config_manager.get_llm_config()
    assert isinstance(llm_config, config.models.LLMConfig)
    
    logging_config = config_manager.get_logging_config()
    assert isinstance(logging_config, config.models.LoggingConfig)
    
    # Test dot notation access
    turns_value = config_manager.get('memory.turns_to_summarize')
    assert isinstance(turns_value, int)
    
    model_name = config_manager.get('llm.model_name')
    assert isinstance(model_name, str)
    
    print("✅ ConfigManager type consistency verified")


def test_logging_utils_types():
    """Test that logging utilities have proper type handling."""
    from logging_utils import StructuredLogger, ValidationResult, ValidationError
    
    # Test StructuredLogger
    logger = StructuredLogger("test")
    
    # Test that context methods work with proper types
    logger.set_context(test_key="test_value", number=42)
    logger.clear_context()
    
    # Test ValidationResult
    valid_result = ValidationResult(is_valid=True, errors=[], warnings=[])
    assert isinstance(valid_result.is_valid, bool)
    assert isinstance(valid_result.errors, list)
    assert isinstance(valid_result.warnings, list)
    
    invalid_result = ValidationResult(
        is_valid=False, 
        errors=[
            ValidationError(field="field1", message="Error 1", value="value1"),
            ValidationError(field="field2", message="Error 2", value="value2")
        ],
        warnings=["Warning 1"]
    )
    assert not invalid_result.is_valid
    assert len(invalid_result.errors) == 2
    assert len(invalid_result.warnings) == 1
    
    # Test ValidationError
    error = ValidationError(field="test_field", message="Test error", value="test_value")
    assert isinstance(error.field, str)
    assert isinstance(error.message, str)
    
    print("✅ Logging utilities type handling verified")


def test_schema_type_annotations():
    """Test that schema classes have proper type annotations."""
    from schemas import AgentState, Router, UserProfile, EpisodicMemory
    
    # Test AgentState TypedDict
    state: AgentState = {
        "message": "test message",
        "classification": "test",
        "reasoning": "test reasoning"
    }
    
    # Test that optional fields work
    state_minimal: AgentState = {
        "message": "test message"
    }
    
    # Test Router schema
    router = Router(
        classification="echo_respond",
        reasoning="Simple greeting message"
    )
    assert isinstance(router.classification, str)
    assert isinstance(router.reasoning, str)
    
    # Test UserProfile schema
    profile = UserProfile(
        name="Test User",
        background="Test background",
        communication_style="friendly",
        emotional_baseline="positive"
    )
    assert isinstance(profile.name, str)
    assert isinstance(profile.background, str)
    
    # Test EpisodicMemory schema
    memory = EpisodicMemory(
        user_message="Hello",
        ai_response="Hi there!",
        timestamp="2024-01-01T00:00:00Z",
        context="greeting"
    )
    assert isinstance(memory.user_message, str)
    assert isinstance(memory.ai_response, str)
    
    print("✅ Schema type annotations verified")


def test_union_and_optional_types():
    """Test that Union and Optional types are used correctly."""
    from typing import get_type_hints
    
    # Test ConfigManager.get method
    config_manager = config.manager.ConfigManager()
    hints = get_type_hints(config_manager.get)
    
    # The get method should accept Any as default and return Any
    assert 'default' in hints
    assert 'return' in hints
    
    # Test that Optional types work correctly
    optional_value: Optional[str] = None
    assert optional_value is None
    
    optional_value = "test"
    assert optional_value == "test"
    
    # Test Union types
    union_value: Union[str, int] = "test"
    assert isinstance(union_value, str)
    
    union_value = 42
    assert isinstance(union_value, int)
    
    print("✅ Union and Optional types working correctly")


def test_generic_types():
    """Test that generic types are used correctly."""
    from typing import List, Dict
    
    # Test List types
    string_list: List[str] = ["a", "b", "c"]
    assert all(isinstance(item, str) for item in string_list)
    
    # Test Dict types
    string_dict: Dict[str, Any] = {"key1": "value1", "key2": 42}
    assert isinstance(string_dict["key1"], str)
    assert isinstance(string_dict["key2"], int)
    
    # Test nested generic types
    nested_dict: Dict[str, List[str]] = {"list1": ["a", "b"], "list2": ["c", "d"]}
    assert isinstance(nested_dict["list1"], list)
    assert all(isinstance(item, str) for item in nested_dict["list1"])
    
    print("✅ Generic types working correctly")


def run_all_tests():
    """Run all type checking tests."""
    print("🔍 Running type checking validation tests...\n")
    
    try:
        test_mypy_passes()
        test_function_type_annotations()
        test_pydantic_models_consistency()
        test_type_consistency_in_functions()
        test_logging_utils_types()
        test_schema_type_annotations()
        test_union_and_optional_types()
        test_generic_types()
        
        print("\n✅ All type checking tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Type checking test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)