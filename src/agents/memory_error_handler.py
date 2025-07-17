"""
Enhanced error handling utilities for the memory system.
Provides comprehensive error handling, validation, and recovery mechanisms.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MemoryErrorHandler:
    """Centralized error handling for all memory operations."""
    
    @staticmethod
    def validate_hashable_structure(data: Any, path: str = "") -> Tuple[bool, List[str]]:
        """
        Recursively validate that data structure is compatible with LangGraph store.
        Returns (is_valid, error_messages).
        """
        errors = []
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    # Check if key is hashable
                    if not isinstance(key, (str, int, float, bool, type(None))):
                        errors.append(f"Non-hashable key at {path}.{key}: {type(key)}")
                    
                    # Recursively check value
                    is_valid, sub_errors = MemoryErrorHandler.validate_hashable_structure(
                        value, f"{path}.{key}" if path else str(key)
                    )
                    errors.extend(sub_errors)
                    
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    is_valid, sub_errors = MemoryErrorHandler.validate_hashable_structure(
                        item, f"{path}[{i}]" if path else f"[{i}]"
                    )
                    errors.extend(sub_errors)
                    
            # Check for other potentially problematic types
            elif hasattr(data, '__dict__') and not isinstance(data, (str, int, float, bool, type(None))):
                errors.append(f"Complex object at {path}: {type(data)}")
                
        except Exception as e:
            errors.append(f"Validation error at {path}: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_for_storage(data: Any) -> Any:
        """
        Sanitize data structure to make it compatible with LangGraph store.
        Converts complex objects to JSON-serializable format.
        """
        try:
            if isinstance(data, dict):
                return {
                    str(key): MemoryErrorHandler.sanitize_for_storage(value)
                    for key, value in data.items()
                }
            elif isinstance(data, list):
                return [MemoryErrorHandler.sanitize_for_storage(item) for item in data]
            elif isinstance(data, (str, int, float, bool, type(None))):
                return data
            elif hasattr(data, 'model_dump'):  # Pydantic models
                return MemoryErrorHandler.sanitize_for_storage(data.model_dump())
            elif hasattr(data, '__dict__'):  # Other objects
                return MemoryErrorHandler.sanitize_for_storage(data.__dict__)
            else:
                # Convert to string as fallback
                return str(data)
        except Exception as e:
            logger.warning(f"Failed to sanitize data: {e}")
            return str(data)
    
    @staticmethod
    def handle_storage_error(error: Exception, data: Dict[str, Any], operation: str) -> bool:
        """
        Handle storage errors with recovery mechanisms.
        Returns True if recovery was successful, False otherwise.
        """
        logger.error(f"Storage error in {operation}: {str(error)} (type: {type(error).__name__}, data_keys: {list(data.keys()) if isinstance(data, dict) else 'not_dict'})")
        
        # Attempt recovery based on error type
        if "unhashable type" in str(error).lower():
            logger.info("Attempting recovery from unhashable type error")
            try:
                # Validate and sanitize the data
                is_valid, errors = MemoryErrorHandler.validate_hashable_structure(data)
                if not is_valid:
                    logger.warning(f"Data validation failed: {errors}")
                    # Attempt to sanitize the data
                    sanitized_data = MemoryErrorHandler.sanitize_for_storage(data)
                    original_keys = list(data.keys()) if isinstance(data, dict) else "not_dict"
                    sanitized_keys = list(sanitized_data.keys()) if isinstance(sanitized_data, dict) else "not_dict"
                    logger.info(f"Data sanitization completed - original_keys: {original_keys}, sanitized_keys: {sanitized_keys}")
                    return True
            except Exception as recovery_e:
                logger.error(f"Recovery attempt failed: {str(recovery_e)}")
        
        return False
    
    @staticmethod
    def log_error_context(error: Exception, context: Dict[str, Any]) -> None:
        """Log detailed error information for debugging."""
        error_context = {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
            **context
        }
        
        logger.error(f"Detailed error context: {error_context}")
    
    @staticmethod
    def create_error_summary(errors: List[Exception], operation: str) -> Dict[str, Any]:
        """Create a summary of multiple errors for reporting."""
        error_summary = {
            "operation": operation,
            "total_errors": len(errors),
            "error_types": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for error in errors:
            error_type = type(error).__name__
            if error_type not in error_summary["error_types"]:
                error_summary["error_types"][error_type] = {
                    "count": 0,
                    "messages": []
                }
            error_summary["error_types"][error_type]["count"] += 1
            error_summary["error_types"][error_type]["messages"].append(str(error))
        
        return error_summary


class ProfileSerializer:
    """Handles serialization and validation of profile data."""
    
    @staticmethod
    def serialize_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex profile objects to JSON-serializable format."""
        try:
            # First, validate the structure
            is_valid, errors = MemoryErrorHandler.validate_hashable_structure(profile_data)
            if not is_valid:
                logger.warning(f"Profile validation failed, attempting sanitization: {errors}")
                profile_data = MemoryErrorHandler.sanitize_for_storage(profile_data)
            
            # Ensure all required fields are strings
            serialized = {}
            for key, value in profile_data.items():
                if isinstance(value, (dict, list)):
                    # Convert complex structures to JSON strings
                    serialized[str(key)] = json.dumps(value, default=str)
                else:
                    serialized[str(key)] = str(value) if value is not None else ""
            
            return serialized
            
        except Exception as e:
            logger.error(f"Profile serialization failed: {str(e)}")
            # Return a minimal valid profile
            return {
                "name": str(profile_data.get("name", "Unknown")),
                "error": f"Serialization failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def deserialize_profile(serialized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert serialized data back to usable format."""
        try:
            deserialized = {}
            for key, value in serialized_data.items():
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        # Try to parse JSON strings back to objects
                        deserialized[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # If parsing fails, keep as string
                        deserialized[key] = value
                else:
                    deserialized[key] = value
            
            return deserialized
            
        except Exception as e:
            logger.error(f"Profile deserialization failed: {str(e)}")
            return serialized_data  # Return original data if deserialization fails
    
    @staticmethod
    def validate_profile_structure(profile_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that profile data has the expected structure."""
        errors = []
        
        # Check for required fields (adjust based on your profile schema)
        required_fields = ["name"]  # Add other required fields as needed
        for field in required_fields:
            if field not in profile_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types
        if "name" in profile_data and not isinstance(profile_data["name"], str):
            errors.append("Profile name must be a string")
        
        # Check for hashable structure
        is_hashable, hashable_errors = MemoryErrorHandler.validate_hashable_structure(profile_data)
        errors.extend(hashable_errors)
        
        return len(errors) == 0, errors