"""
Structured logging and input validation utilities for EchoStar AI Simulator.
Provides JSON-formatted logging with context injection capabilities and comprehensive input validation.
"""

import json
import logging
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from contextlib import contextmanager

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from config.models import LoggingConfig
from config.manager import get_config_manager


class StructuredLogger:
    """
    Structured logger with JSON formatting and context injection capabilities.
    
    Features:
    - JSON formatted output for machine readability
    - Automatic context injection (timestamp, session, etc.)
    - Performance timing capabilities
    - File and console logging support
    - Configurable log levels
    """
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        """
        Initialize the structured logger.
        
        Args:
            name: Logger name (typically module or component name)
            config: Logging configuration. If None, uses global config.
        """
        self.name = name
        self.config = config or get_config_manager().get_logging_config()
        self._logger = self._setup_logger()
        self._context: Dict[str, Any] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the underlying Python logger with appropriate handlers."""
        logger = logging.getLogger(self.name)
        
        # Set log level
        log_level = getattr(logging, self.config.level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatter
        if self.config.format.lower() == "json":
            formatter: logging.Formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if enabled
        if self.config.enable_file_logging:
            # Ensure log directory exists
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def set_context(self, **context: Any) -> None:
        """
        Set persistent context that will be included in all log messages.
        
        Args:
            **context: Key-value pairs to include in log context
        """
        self._context.update(context)
    
    def clear_context(self) -> None:
        """Clear all persistent context."""
        self._context.clear()
    
    def _build_log_data(self, message: str, level: str, **extra_context: Any) -> Dict[str, Any]:
        """
        Build the structured log data dictionary.
        
        Args:
            message: Log message
            level: Log level
            **extra_context: Additional context for this log entry
            
        Returns:
            Dictionary containing all log data
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logger": self.name,
            "level": level,
            "message": message,
        }
        
        # Add persistent context
        if self._context:
            log_data["context"] = self._context.copy()
        
        # Add extra context for this log entry
        if extra_context:
            if "context" not in log_data:
                log_data["context"] = {}
            if isinstance(log_data["context"], dict):
                log_data["context"].update(extra_context)
        
        return log_data
    
    def debug(self, message: str, **context: Any) -> None:
        """
        Log a debug message.
        
        Args:
            message: Debug message
            **context: Additional context for this log entry
        """
        log_data = self._build_log_data(message, "DEBUG", **context)
        self._logger.debug(json.dumps(log_data) if self.config.format.lower() == "json" else message)
    
    def info(self, message: str, **context: Any) -> None:
        """
        Log an info message.
        
        Args:
            message: Info message
            **context: Additional context for this log entry
        """
        log_data = self._build_log_data(message, "INFO", **context)
        self._logger.info(json.dumps(log_data) if self.config.format.lower() == "json" else message)
    
    def warning(self, message: str, **context: Any) -> None:
        """
        Log a warning message.
        
        Args:
            message: Warning message
            **context: Additional context for this log entry
        """
        log_data = self._build_log_data(message, "WARNING", **context)
        self._logger.warning(json.dumps(log_data) if self.config.format.lower() == "json" else message)
    
    def error(self, message: str, error: Optional[Exception] = None, **context: Any) -> None:
        """
        Log an error message.
        
        Args:
            message: Error message
            error: Exception object (optional)
            **context: Additional context for this log entry
        """
        if error:
            context.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        
        log_data = self._build_log_data(message, "ERROR", **context)
        self._logger.error(json.dumps(log_data) if self.config.format.lower() == "json" else message)
    
    def critical(self, message: str, error: Optional[Exception] = None, **context: Any) -> None:
        """
        Log a critical message.
        
        Args:
            message: Critical message
            error: Exception object (optional)
            **context: Additional context for this log entry
        """
        if error:
            context.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        
        log_data = self._build_log_data(message, "CRITICAL", **context)
        self._logger.critical(json.dumps(log_data) if self.config.format.lower() == "json" else message)
    
    @contextmanager
    def performance_timer(self, operation: str, **context: Any):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation being timed
            **context: Additional context for the timing log
            
        Usage:
            with logger.performance_timer("database_query", query_type="user_search"):
                # perform operation
                pass
        """
        if not self.config.enable_performance_logging:
            yield
            return
        
        start_time = time.time()
        start_context = context.copy()
        start_context.update({
            "operation": operation,
            "phase": "start"
        })
        
        self.debug(f"Starting operation: {operation}", **start_context)
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            end_context = context.copy()
            end_context.update({
                "operation": operation,
                "phase": "complete",
                "duration_seconds": round(duration, 4),
                "duration_ms": round(duration * 1000, 2)
            })
            
            self.info(f"Completed operation: {operation}", **end_context)


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        # Try to parse message as JSON first (for structured logs)
        try:
            log_data = json.loads(record.getMessage())
            return json.dumps(log_data, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            # Fall back to creating JSON structure from record
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            }
            
            if record.exc_info:
                log_data["traceback"] = self.formatException(record.exc_info)
            
            return json.dumps(log_data, ensure_ascii=False)


# Global logger instances cache
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        config: Optional logging configuration
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, config)
    
    return _loggers[name]


def setup_application_logging() -> None:
    """
    Set up application-wide logging configuration.
    Should be called once at application startup.
    """
    config_manager = get_config_manager()
    logging_config = config_manager.get_logging_config()
    
    if not logging_config:
        return
    
    # Set root logger level
    root_logger = logging.getLogger()
    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create logs directory if file logging is enabled
    if logging_config.enable_file_logging:
        log_path = Path(logging_config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

# Input Validation Models and Classes

class ValidationError(BaseModel):
    """Model for individual validation errors."""
    
    field: str = Field(description="Field name that failed validation")
    message: str = Field(description="Error message describing the validation failure")
    value: Any = Field(description="The value that failed validation")
    error_code: Optional[str] = Field(default=None, description="Optional error code for programmatic handling")


class ValidationResult(BaseModel):
    """Model for validation results with errors and warnings."""
    
    is_valid: bool = Field(description="Whether the validation passed")
    errors: List[ValidationError] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    validated_data: Optional[Dict[str, Any]] = Field(default=None, description="Cleaned/validated data if successful")
    
    def add_error(self, field: str, message: str, value: Any = None, error_code: Optional[str] = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            field=field,
            message=message,
            value=value,
            error_code=error_code
        ))
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)
    
    @property
    def error_messages(self) -> List[str]:
        """Get list of error messages."""
        return [error.message for error in self.errors]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class InputValidator:
    """
    Comprehensive input validation class for EchoStar AI Simulator.
    
    Provides validation methods for:
    - User messages and input
    - Configuration data
    - API responses
    - Memory data structures
    """
    
    # Constants for validation
    MAX_MESSAGE_LENGTH = 10000
    MIN_MESSAGE_LENGTH = 1
    MAX_CONFIG_STRING_LENGTH = 1000
    ALLOWED_CONFIG_KEYS = {
        'memory', 'routing', 'llm', 'logging',
        'turns_to_summarize', 'search_limit', 'enable_condensation',
        'roleplay_threshold', 'enable_fallback', 'classification_confidence_threshold',
        'model_name', 'temperature', 'max_tokens', 'timeout',
        'level', 'format', 'enable_file_logging', 'log_file_path'
    }
    
    # Regex patterns for validation
    SAFE_TEXT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\.,!?\-_:;()\[\]{}@#$%^&*+=<>/\\|`~"\'\n\r\t]*$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    @staticmethod
    def validate_user_message(message: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate user input messages.
        
        Args:
            message: User message to validate
            context: Optional context for validation (e.g., session info)
            
        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        
        # Check if message is provided
        if message is None:
            result.add_error("message", "Message cannot be None", message, "NULL_MESSAGE")
            return result
        
        # Check if message is string
        if not isinstance(message, str):
            result.add_error("message", f"Message must be a string, got {type(message).__name__}", message, "INVALID_TYPE")
            return result
        
        # Check message length
        if len(message) < InputValidator.MIN_MESSAGE_LENGTH:
            result.add_error("message", f"Message too short (minimum {InputValidator.MIN_MESSAGE_LENGTH} characters)", message, "TOO_SHORT")
        
        if len(message) > InputValidator.MAX_MESSAGE_LENGTH:
            result.add_error("message", f"Message too long (maximum {InputValidator.MAX_MESSAGE_LENGTH} characters)", message, "TOO_LONG")
        
        # Check for potentially dangerous content
        stripped_message = message.strip()
        if not stripped_message:
            result.add_error("message", "Message cannot be empty or only whitespace", message, "EMPTY_MESSAGE")
        
        # Check for basic safety (no control characters except common ones)
        if not InputValidator.SAFE_TEXT_PATTERN.match(message):
            result.add_warning("Message contains potentially unsafe characters")
        
        # Check for extremely repetitive content (potential spam/abuse)
        if len(set(message.lower().split())) < max(1, len(message.split()) // 10):
            result.add_warning("Message appears to be highly repetitive")
        
        # If validation passed, store cleaned message
        if result.is_valid:
            result.validated_data = {
                "message": stripped_message,
                "length": len(stripped_message),
                "word_count": len(stripped_message.split())
            }
        
        return result
    
    @staticmethod
    def validate_config_data(config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration data structure.
        
        Args:
            config_data: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        
        # Check if config_data is provided and is a dict
        if config_data is None:
            result.add_error("config", "Configuration data cannot be None", config_data, "NULL_CONFIG")
            return result
        
        if not isinstance(config_data, dict):
            result.add_error("config", f"Configuration must be a dictionary, got {type(config_data).__name__}", config_data, "INVALID_TYPE")
            return result
        
        # Validate top-level structure
        valid_sections = {'memory', 'routing', 'llm', 'logging'}
        for key in config_data.keys():
            if key not in valid_sections:
                result.add_warning(f"Unknown configuration section: {key}")
        
        # Validate each section
        if 'memory' in config_data:
            InputValidator._validate_memory_config(config_data['memory'], result)
        
        if 'routing' in config_data:
            InputValidator._validate_routing_config(config_data['routing'], result)
        
        if 'llm' in config_data:
            InputValidator._validate_llm_config(config_data['llm'], result)
        
        if 'logging' in config_data:
            InputValidator._validate_logging_config(config_data['logging'], result)
        
        # If validation passed, store the validated config
        if result.is_valid:
            result.validated_data = config_data
        
        return result
    
    @staticmethod
    def validate_api_response(response_data: Any, expected_fields: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate API response data.
        
        Args:
            response_data: API response to validate
            expected_fields: Optional list of expected fields in the response
            
        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        
        # Check if response is provided
        if response_data is None:
            result.add_error("response", "API response cannot be None", response_data, "NULL_RESPONSE")
            return result
        
        # For string responses (common with LLM APIs)
        if isinstance(response_data, str):
            if not response_data.strip():
                result.add_error("response", "API response cannot be empty", response_data, "EMPTY_RESPONSE")
            elif len(response_data) > 50000:  # Reasonable limit for LLM responses
                result.add_warning("API response is very large")
            
            if result.is_valid:
                result.validated_data = {"content": response_data.strip()}
            
            return result
        
        # For dictionary responses
        if isinstance(response_data, dict):
            # Check for expected fields if provided
            if expected_fields:
                missing_fields = [field for field in expected_fields if field not in response_data]
                if missing_fields:
                    result.add_error("response", f"Missing required fields: {missing_fields}", response_data, "MISSING_FIELDS")
            
            # Check for common error indicators
            if 'error' in response_data:
                result.add_error("response", f"API returned error: {response_data['error']}", response_data, "API_ERROR")
            
            if result.is_valid:
                result.validated_data = response_data
            
            return result
        
        # For other types, basic validation
        result.validated_data = {"data": response_data, "type": type(response_data).__name__}
        return result
    
    @staticmethod
    def validate_memory_data(memory_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate memory data structures.
        
        Args:
            memory_data: Memory data to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        
        if memory_data is None:
            result.add_error("memory", "Memory data cannot be None", memory_data, "NULL_MEMORY")
            return result
        
        if not isinstance(memory_data, dict):
            result.add_error("memory", f"Memory data must be a dictionary, got {type(memory_data).__name__}", memory_data, "INVALID_TYPE")
            return result
        
        # Validate required fields for different memory types
        if 'type' in memory_data:
            memory_type = memory_data['type']
            
            if memory_type == 'episodic':
                required_fields = ['content', 'timestamp']
                for field in required_fields:
                    if field not in memory_data:
                        result.add_error("memory", f"Missing required field for episodic memory: {field}", memory_data, "MISSING_FIELD")
            
            elif memory_type == 'semantic':
                required_fields = ['content', 'summary']
                for field in required_fields:
                    if field not in memory_data:
                        result.add_error("memory", f"Missing required field for semantic memory: {field}", memory_data, "MISSING_FIELD")
            
            elif memory_type == 'procedural':
                required_fields = ['action', 'context']
                for field in required_fields:
                    if field not in memory_data:
                        result.add_error("memory", f"Missing required field for procedural memory: {field}", memory_data, "MISSING_FIELD")
        
        # Validate content length if present
        if 'content' in memory_data:
            content = memory_data['content']
            if isinstance(content, str) and len(content) > 10000:
                result.add_warning("Memory content is very large")
        
        if result.is_valid:
            result.validated_data = memory_data
        
        return result
    
    @staticmethod
    def _validate_memory_config(memory_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate memory configuration section."""
        if not isinstance(memory_config, dict):
            result.add_error("memory", "Memory config must be a dictionary", memory_config, "INVALID_TYPE")
            return
        
        # Validate turns_to_summarize
        if 'turns_to_summarize' in memory_config:
            value = memory_config['turns_to_summarize']
            if not isinstance(value, int) or value < 1 or value > 100:
                result.add_error("memory.turns_to_summarize", "Must be an integer between 1 and 100", value, "INVALID_RANGE")
        
        # Validate search_limit
        if 'search_limit' in memory_config:
            value = memory_config['search_limit']
            if not isinstance(value, int) or value < 1 or value > 50:
                result.add_error("memory.search_limit", "Must be an integer between 1 and 50", value, "INVALID_RANGE")
    
    @staticmethod
    def _validate_routing_config(routing_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate routing configuration section."""
        if not isinstance(routing_config, dict):
            result.add_error("routing", "Routing config must be a dictionary", routing_config, "INVALID_TYPE")
            return
        
        # Validate roleplay_threshold
        if 'roleplay_threshold' in routing_config:
            value = routing_config['roleplay_threshold']
            if not isinstance(value, int) or value < 1 or value > 10:
                result.add_error("routing.roleplay_threshold", "Must be an integer between 1 and 10", value, "INVALID_RANGE")
        
        # Validate classification_confidence_threshold
        if 'classification_confidence_threshold' in routing_config:
            value = routing_config['classification_confidence_threshold']
            if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                result.add_error("routing.classification_confidence_threshold", "Must be a number between 0.0 and 1.0", value, "INVALID_RANGE")
    
    @staticmethod
    def _validate_llm_config(llm_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate LLM configuration section."""
        if not isinstance(llm_config, dict):
            result.add_error("llm", "LLM config must be a dictionary", llm_config, "INVALID_TYPE")
            return
        
        # Validate temperature
        if 'temperature' in llm_config:
            value = llm_config['temperature']
            if not isinstance(value, (int, float)) or value < 0.0 or value > 2.0:
                result.add_error("llm.temperature", "Must be a number between 0.0 and 2.0", value, "INVALID_RANGE")
        
        # Validate timeout
        if 'timeout' in llm_config:
            value = llm_config['timeout']
            if not isinstance(value, int) or value < 1 or value > 300:
                result.add_error("llm.timeout", "Must be an integer between 1 and 300", value, "INVALID_RANGE")
    
    @staticmethod
    def _validate_logging_config(logging_config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate logging configuration section."""
        if not isinstance(logging_config, dict):
            result.add_error("logging", "Logging config must be a dictionary", logging_config, "INVALID_TYPE")
            return
        
        # Validate level
        if 'level' in logging_config:
            value = logging_config['level']
            valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
            if not isinstance(value, str) or value.upper() not in valid_levels:
                result.add_error("logging.level", f"Must be one of: {valid_levels}", value, "INVALID_CHOICE")
        
        # Validate format
        if 'format' in logging_config:
            value = logging_config['format']
            valid_formats = {'json', 'text'}
            if not isinstance(value, str) or value.lower() not in valid_formats:
                result.add_error("logging.format", f"Must be one of: {valid_formats}", value, "INVALID_CHOICE")


# Convenience functions for common validation scenarios

def validate_user_input(message: str) -> ValidationResult:
    """Convenience function for validating user input."""
    return InputValidator.validate_user_message(message)


def validate_config(config_data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for validating configuration data."""
    return InputValidator.validate_config_data(config_data)


def validate_api_response(response_data: Any, expected_fields: Optional[List[str]] = None) -> ValidationResult:
    """Convenience function for validating API responses."""
    return InputValidator.validate_api_response(response_data, expected_fields)


def validate_memory(memory_data: Dict[str, Any]) -> ValidationResult:
    """Convenience function for validating memory data."""
    return InputValidator.validate_memory_data(memory_data)