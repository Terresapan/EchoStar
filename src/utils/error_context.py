"""
Comprehensive error context collection and monitoring utilities for EchoStar.
Provides structured error tracking, correlation, and performance monitoring.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum

from .logging_utils import get_logger


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OperationType(Enum):
    """Types of operations for monitoring."""
    MEMORY_CONDENSATION = "memory_condensation"
    MEMORY_STORAGE = "memory_storage"
    MEMORY_RETRIEVAL = "memory_retrieval"
    PROFILE_STORAGE = "profile_storage"
    PROFILE_RETRIEVAL = "profile_retrieval"
    LLM_INVOKE = "llm_invoke"
    STORE_OPERATION = "store_operation"
    VALIDATION = "validation"
    CLEANUP = "cleanup"


@dataclass
class ErrorContext:
    """Structured error context with comprehensive metadata."""
    
    # Core error information
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat() + "Z")
    operation: str = ""
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_count: Optional[int] = None
    correlation_id: Optional[str] = None
    
    # Technical details
    component: Optional[str] = None
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    stack_trace: Optional[str] = None
    
    # Operation-specific data
    operation_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_applied: bool = False
    
    # Additional metadata
    environment: Optional[str] = None
    version: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType = OperationType.STORE_OPERATION
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Resource usage
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Operation-specific metrics
    items_processed: Optional[int] = None
    bytes_processed: Optional[int] = None
    success_count: Optional[int] = None
    failure_count: Optional[int] = None
    
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Additional metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self) -> None:
        """Mark the operation as complete and calculate duration."""
        self.end_time = time.time()
        if self.start_time:
            self.duration_seconds = round(self.end_time - self.start_time, 4)
            self.duration_ms = round(self.duration_seconds * 1000, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class ErrorTracker:
    """Centralized error tracking and correlation system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._error_history: List[ErrorContext] = []
        self._correlation_map: Dict[str, List[str]] = {}
        self._performance_history: List[PerformanceMetrics] = []
        self._max_history_size = 1000
    
    def create_error_context(
        self,
        operation: str,
        error: Optional[Exception] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **context_data: Any
    ) -> ErrorContext:
        """Create a new error context with comprehensive metadata."""
        
        error_context = ErrorContext(
            operation=operation,
            severity=severity,
            **context_data
        )
        
        # Extract error information if provided
        if error:
            error_context.error_type = type(error).__name__
            error_context.error_message = str(error)
            
            # Try to extract stack trace
            try:
                import traceback
                error_context.stack_trace = traceback.format_exc()
            except Exception:
                pass
        
        # Add to history
        self._add_to_history(error_context)
        
        return error_context
    
    def track_error(self, error_context: ErrorContext) -> None:
        """Track an error with correlation."""
        
        # Add to history first
        self._add_to_history(error_context)
        
        # Log the error with structured data
        self.logger.error(
            f"Error tracked: {error_context.operation}",
            error_id=error_context.error_id,
            error_type=error_context.error_type,
            severity=error_context.severity.value,
            correlation_id=error_context.correlation_id,
            operation_data=error_context.operation_data,
            recovery_attempted=error_context.recovery_attempted,
            fallback_applied=error_context.fallback_applied
        )
        
        # Update correlation mapping
        if error_context.correlation_id:
            if error_context.correlation_id not in self._correlation_map:
                self._correlation_map[error_context.correlation_id] = []
            self._correlation_map[error_context.correlation_id].append(error_context.error_id)
    
    def get_correlated_errors(self, correlation_id: str) -> List[ErrorContext]:
        """Get all errors correlated with a specific ID."""
        if correlation_id not in self._correlation_map:
            return []
        
        error_ids = self._correlation_map[correlation_id]
        return [
            error for error in self._error_history
            if error.error_id in error_ids
        ]
    
    def get_error_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get error summary for a time window."""
        cutoff_time = datetime.now().timestamp() - (time_window_minutes * 60)
        
        recent_errors = []
        for error in self._error_history:
            try:
                # Parse timestamp and check if within window
                timestamp_str = error.timestamp.replace('Z', '')
                error_timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                if error_timestamp > cutoff_time:
                    recent_errors.append(error)
            except (ValueError, AttributeError):
                # If timestamp parsing fails, include the error (assume it's recent)
                recent_errors.append(error)
        
        summary = {
            "time_window_minutes": time_window_minutes,
            "total_errors": len(recent_errors),
            "error_by_type": {},
            "error_by_operation": {},
            "error_by_severity": {},
            "recovery_stats": {
                "attempted": 0,
                "successful": 0,
                "fallback_applied": 0
            }
        }
        
        for error in recent_errors:
            # Count by type
            error_type = error.error_type or "unknown"
            summary["error_by_type"][error_type] = summary["error_by_type"].get(error_type, 0) + 1
            
            # Count by operation
            operation = error.operation or "unknown"
            summary["error_by_operation"][operation] = summary["error_by_operation"].get(operation, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            summary["error_by_severity"][severity] = summary["error_by_severity"].get(severity, 0) + 1
            
            # Recovery stats
            if error.recovery_attempted:
                summary["recovery_stats"]["attempted"] += 1
                if error.recovery_successful:
                    summary["recovery_stats"]["successful"] += 1
            if error.fallback_applied:
                summary["recovery_stats"]["fallback_applied"] += 1
        
        return summary
    
    def _add_to_history(self, error_context: ErrorContext) -> None:
        """Add error context to history with size management."""
        self._error_history.append(error_context)
        
        # Maintain history size limit
        if len(self._error_history) > self._max_history_size:
            self._error_history = self._error_history[-self._max_history_size:]


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._active_operations: Dict[str, PerformanceMetrics] = {}
        self._completed_operations: List[PerformanceMetrics] = []
        self._max_history_size = 1000
    
    def start_operation(
        self,
        operation_type: OperationType,
        correlation_id: Optional[str] = None,
        **context: Any
    ) -> str:
        """Start monitoring an operation."""
        
        metrics = PerformanceMetrics(
            operation_type=operation_type,
            correlation_id=correlation_id
        )
        
        # Add context to custom metrics
        if context:
            metrics.custom_metrics.update(context)
        
        self._active_operations[metrics.operation_id] = metrics
        
        self.logger.debug(
            f"Started monitoring operation: {operation_type.value}",
            operation_id=metrics.operation_id,
            correlation_id=correlation_id
        )
        
        return metrics.operation_id
    
    def complete_operation(
        self,
        operation_id: str,
        success: bool = True,
        **additional_metrics: Any
    ) -> Optional[PerformanceMetrics]:
        """Complete an operation and calculate metrics."""
        
        if operation_id not in self._active_operations:
            self.logger.warning(f"Operation ID not found: {operation_id}")
            return None
        
        metrics = self._active_operations.pop(operation_id)
        metrics.complete()
        
        # Add additional metrics
        for key, value in additional_metrics.items():
            metrics.custom_metrics[key] = value
        
        # Add to completed operations
        self._completed_operations.append(metrics)
        
        # Maintain history size
        if len(self._completed_operations) > self._max_history_size:
            self._completed_operations = self._completed_operations[-self._max_history_size:]
        
        # Log completion
        self.logger.info(
            f"Completed operation: {metrics.operation_type.value}",
            operation_id=operation_id,
            duration_ms=metrics.duration_ms,
            success=success,
            correlation_id=metrics.correlation_id,
            custom_metrics=metrics.custom_metrics
        )
        
        return metrics
    
    @contextmanager
    def monitor_operation(
        self,
        operation_type: OperationType,
        correlation_id: Optional[str] = None,
        **context: Any
    ):
        """Context manager for monitoring operations."""
        
        operation_id = self.start_operation(operation_type, correlation_id, **context)
        success = True
        additional_metrics = {}
        
        try:
            yield {
                'operation_id': operation_id,
                'add_metric': lambda k, v: additional_metrics.update({k: v})
            }
        except Exception as e:
            success = False
            additional_metrics['error'] = str(e)
            additional_metrics['error_type'] = type(e).__name__
            raise
        finally:
            self.complete_operation(operation_id, success, **additional_metrics)
    
    def get_performance_summary(
        self,
        operation_type: Optional[OperationType] = None,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get performance summary for operations."""
        
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        # Filter operations
        operations = [
            op for op in self._completed_operations
            if op.start_time > cutoff_time and (
                operation_type is None or op.operation_type == operation_type
            )
        ]
        
        if not operations:
            return {
                "time_window_minutes": time_window_minutes,
                "operation_type": operation_type.value if operation_type else "all",
                "total_operations": 0
            }
        
        # Calculate statistics
        durations = [op.duration_ms for op in operations if op.duration_ms is not None]
        
        summary = {
            "time_window_minutes": time_window_minutes,
            "operation_type": operation_type.value if operation_type else "all",
            "total_operations": len(operations),
            "performance_stats": {
                "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "total_duration_ms": sum(durations) if durations else 0
            },
            "throughput": {
                "operations_per_minute": round(len(operations) / time_window_minutes, 2)
            }
        }
        
        # Add operation-specific metrics
        if operations:
            items_processed = [op.items_processed for op in operations if op.items_processed is not None]
            if items_processed:
                summary["items_stats"] = {
                    "total_items": sum(items_processed),
                    "avg_items_per_operation": round(sum(items_processed) / len(items_processed), 2)
                }
        
        return summary


# Global instances
_error_tracker = ErrorTracker()
_performance_monitor = PerformanceMonitor()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _error_tracker


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def create_correlation_id() -> str:
    """Create a new correlation ID for tracking related operations."""
    return str(uuid.uuid4())


@contextmanager
def error_context(
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    correlation_id: Optional[str] = None,
    **context_data: Any
):
    """Context manager for comprehensive error handling and tracking."""
    
    tracker = get_error_tracker()
    error_ctx = None
    
    try:
        yield
    except Exception as e:
        error_ctx = tracker.create_error_context(
            operation=operation,
            error=e,
            severity=severity,
            correlation_id=correlation_id,
            **context_data
        )
        
        # Track the error
        tracker.track_error(error_ctx)
        
        # Re-raise the exception
        raise
    finally:
        # Log successful completion if no error occurred
        if error_ctx is None:
            logger = get_logger(__name__)
            logger.debug(
                f"Operation completed successfully: {operation}",
                correlation_id=correlation_id,
                **context_data
            )


@contextmanager
def performance_context(
    operation_type: OperationType,
    correlation_id: Optional[str] = None,
    **context: Any
):
    """Context manager for performance monitoring."""
    
    monitor = get_performance_monitor()
    with monitor.monitor_operation(operation_type, correlation_id, **context) as op_context:
        yield op_context