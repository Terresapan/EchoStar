"""
Memory operation monitoring and error context utilities.
Provides specialized monitoring for memory condensation and storage operations.
"""

import time
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

from .error_context import (
    get_error_tracker, get_performance_monitor, create_correlation_id,
    ErrorSeverity, OperationType, ErrorContext, PerformanceMonitor
)
from .logging_utils import get_logger
from .metrics_collector import (
    get_metrics_collector, MetricType, 
    increment_memory_counter, set_memory_gauge, start_memory_timer, stop_memory_timer
)
from .alerting_system import get_alerting_system

logger = get_logger(__name__)


class MemoryOperationMonitor:
    """Specialized monitoring for memory operations with detailed context collection."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or create_correlation_id()
        self.error_tracker = get_error_tracker()
        # Create a fresh performance monitor instance for this monitor
        self.performance_monitor = PerformanceMonitor()
        self.metrics_collector = get_metrics_collector()
        self.alerting_system = get_alerting_system()
        self.operation_context = {}
        
        # Track active timers for this monitor
        self._active_timers: Dict[str, str] = {}
        
    def set_context(self, **context: Any) -> None:
        """Set context information for all operations."""
        self.operation_context.update(context)
        
    @contextmanager
    def monitor_memory_condensation(self, **context: Any):
        """Monitor the entire memory condensation process."""
        operation_id = self.performance_monitor.start_operation(
            OperationType.MEMORY_CONDENSATION,
            correlation_id=self.correlation_id,
            **{**self.operation_context, **context}
        )
        
        # Start timer for condensation duration
        timer_id = start_memory_timer("memory_condensation", self.correlation_id)
        
        condensation_metrics = {
            'memories_retrieved': 0,
            'dialogue_turns_processed': 0,
            'semantic_storage_success': False,
            'episodic_storage_success': False,
            'cleanup_success': False,
            'errors_encountered': 0,
            'fallback_applied': False
        }
        
        # Increment condensation attempt counter
        increment_memory_counter(
            "memory_condensation_attempts",
            tags={"user_id": context.get("user_id", "unknown")},
            correlation_id=self.correlation_id
        )
        
        try:
            logger.info(
                "Starting memory condensation monitoring",
                correlation_id=self.correlation_id,
                operation_id=operation_id,
                **context
            )
            
            yield {
                'operation_id': operation_id,
                'correlation_id': self.correlation_id,
                'add_metric': lambda k, v: self._add_metric_with_collection(k, v, condensation_metrics),
                'track_error': self._create_enhanced_error_tracker(operation_id),
                'log_milestone': self._create_milestone_logger(operation_id)
            }
            
        except Exception as e:
            condensation_metrics['errors_encountered'] += 1
            condensation_metrics['fallback_applied'] = True
            
            # Record failure metrics
            increment_memory_counter(
                "memory_condensation_failures",
                tags={"user_id": context.get("user_id", "unknown"), "error_type": type(e).__name__},
                correlation_id=self.correlation_id
            )
            
            error_ctx = self.error_tracker.create_error_context(
                operation="memory_condensation_process",
                error=e,
                severity=ErrorSeverity.HIGH,
                correlation_id=self.correlation_id,
                operation_data={
                    'operation_id': operation_id,
                    'metrics': condensation_metrics,
                    **context
                }
            )
            self.error_tracker.track_error(error_ctx)
            raise
            
        finally:
            # Stop timer and record duration
            duration = stop_memory_timer(
                timer_id,
                tags={"user_id": context.get("user_id", "unknown")},
                correlation_id=self.correlation_id
            )
            
            # Record success/failure metrics
            if condensation_metrics['errors_encountered'] == 0:
                increment_memory_counter(
                    "memory_condensation_successes",
                    tags={"user_id": context.get("user_id", "unknown")},
                    correlation_id=self.correlation_id
                )
            
            # Record detailed metrics
            set_memory_gauge(
                "memory_condensation_memories_retrieved",
                condensation_metrics['memories_retrieved'],
                tags={"user_id": context.get("user_id", "unknown")},
                correlation_id=self.correlation_id
            )
            
            set_memory_gauge(
                "memory_condensation_dialogue_turns",
                condensation_metrics['dialogue_turns_processed'],
                tags={"user_id": context.get("user_id", "unknown")},
                correlation_id=self.correlation_id
            )
            
            # Complete the operation with collected metrics
            self.performance_monitor.complete_operation(
                operation_id,
                success=condensation_metrics['errors_encountered'] == 0,
                **condensation_metrics
            )
            
            logger.info(
                "Memory condensation monitoring completed",
                correlation_id=self.correlation_id,
                operation_id=operation_id,
                final_metrics=condensation_metrics,
                duration_seconds=duration
            )
    
    @contextmanager
    def monitor_memory_retrieval(self, namespace: str, limit: int, **context: Any):
        """Monitor memory retrieval operations."""
        operation_id = self.performance_monitor.start_operation(
            OperationType.MEMORY_RETRIEVAL,
            correlation_id=self.correlation_id,
            namespace=namespace,
            limit=limit,
            **{**self.operation_context, **context}
        )
        
        # Start timer for retrieval duration
        timer_id = start_memory_timer("memory_retrieval", self.correlation_id)
        
        retrieval_metrics = {
            'namespace': namespace,
            'limit': limit,
            'memories_found': 0,
            'retrieval_time_ms': 0
        }
        
        # Increment retrieval attempt counter
        increment_memory_counter(
            "memory_retrieval_attempts",
            tags={"namespace": namespace, "user_id": context.get("user_id", "unknown")},
            correlation_id=self.correlation_id
        )
        
        start_time = time.time()
        
        try:
            logger.debug(
                "Starting memory retrieval monitoring",
                correlation_id=self.correlation_id,
                operation_id=operation_id,
                namespace=namespace,
                limit=limit
            )
            
            yield {
                'operation_id': operation_id,
                'add_metric': lambda k, v: self._add_metric_with_collection(k, v, retrieval_metrics),
                'track_error': self._create_enhanced_error_tracker(operation_id)
            }
            
            # Record successful retrieval
            increment_memory_counter(
                "memory_retrieval_successes",
                tags={"namespace": namespace, "user_id": context.get("user_id", "unknown")},
                correlation_id=self.correlation_id
            )
            
        except Exception as e:
            # Record failure metrics
            increment_memory_counter(
                "memory_retrieval_failures",
                tags={
                    "namespace": namespace, 
                    "user_id": context.get("user_id", "unknown"),
                    "error_type": type(e).__name__
                },
                correlation_id=self.correlation_id
            )
            
            error_ctx = self.error_tracker.create_error_context(
                operation="memory_retrieval",
                error=e,
                severity=ErrorSeverity.MEDIUM,
                correlation_id=self.correlation_id,
                operation_data={
                    'operation_id': operation_id,
                    'namespace': namespace,
                    'limit': limit,
                    **context
                }
            )
            self.error_tracker.track_error(error_ctx)
            raise
            
        finally:
            # Stop timer and record duration
            duration = stop_memory_timer(
                timer_id,
                tags={"namespace": namespace, "user_id": context.get("user_id", "unknown")},
                correlation_id=self.correlation_id
            )
            
            retrieval_metrics['retrieval_time_ms'] = round((time.time() - start_time) * 1000, 2)
            
            # Record detailed metrics
            set_memory_gauge(
                "memory_retrieval_memories_found",
                retrieval_metrics['memories_found'],
                tags={"namespace": namespace, "user_id": context.get("user_id", "unknown")},
                correlation_id=self.correlation_id
            )
            
            self.performance_monitor.complete_operation(
                operation_id,
                success=True,
                **retrieval_metrics
            )
    
    @contextmanager
    def monitor_memory_storage(self, storage_type: str, namespace: str, **context: Any):
        """Monitor memory storage operations."""
        operation_id = self.performance_monitor.start_operation(
            OperationType.MEMORY_STORAGE,
            correlation_id=self.correlation_id,
            storage_type=storage_type,
            namespace=namespace,
            **{**self.operation_context, **context}
        )
        
        # Start timer for storage duration
        timer_id = start_memory_timer("memory_storage", self.correlation_id)
        
        storage_metrics = {
            'storage_type': storage_type,
            'namespace': namespace,
            'verification_attempted': False,
            'verification_successful': False,
            'data_size_bytes': 0
        }
        
        # Increment storage attempt counter
        increment_memory_counter(
            "memory_storage_attempts",
            tags={
                "storage_type": storage_type,
                "namespace": namespace,
                "user_id": context.get("user_id", "unknown")
            },
            correlation_id=self.correlation_id
        )
        
        try:
            logger.debug(
                "Starting memory storage monitoring",
                correlation_id=self.correlation_id,
                operation_id=operation_id,
                storage_type=storage_type,
                namespace=namespace
            )
            
            yield {
                'operation_id': operation_id,
                'add_metric': lambda k, v: self._add_metric_with_collection(k, v, storage_metrics),
                'track_error': self._create_enhanced_error_tracker(operation_id)
            }
            
            # Record successful storage
            increment_memory_counter(
                "memory_storage_successes",
                tags={
                    "storage_type": storage_type,
                    "namespace": namespace,
                    "user_id": context.get("user_id", "unknown")
                },
                correlation_id=self.correlation_id
            )
            
        except Exception as e:
            # Record failure metrics
            increment_memory_counter(
                "memory_storage_failures",
                tags={
                    "storage_type": storage_type,
                    "namespace": namespace,
                    "user_id": context.get("user_id", "unknown"),
                    "error_type": type(e).__name__
                },
                correlation_id=self.correlation_id
            )
            
            error_ctx = self.error_tracker.create_error_context(
                operation="memory_storage",
                error=e,
                severity=ErrorSeverity.HIGH,
                correlation_id=self.correlation_id,
                operation_data={
                    'operation_id': operation_id,
                    'storage_type': storage_type,
                    'namespace': namespace,
                    **context
                }
            )
            self.error_tracker.track_error(error_ctx)
            raise
            
        finally:
            # Stop timer and record duration
            duration = stop_memory_timer(
                timer_id,
                tags={
                    "storage_type": storage_type,
                    "namespace": namespace,
                    "user_id": context.get("user_id", "unknown")
                },
                correlation_id=self.correlation_id
            )
            
            # Record detailed metrics
            if storage_metrics.get('verification_attempted'):
                set_memory_gauge(
                    "memory_storage_verification_success",
                    1 if storage_metrics.get('verification_successful') else 0,
                    tags={
                        "storage_type": storage_type,
                        "namespace": namespace,
                        "user_id": context.get("user_id", "unknown")
                    },
                    correlation_id=self.correlation_id
                )
            
            if storage_metrics.get('data_size_bytes', 0) > 0:
                set_memory_gauge(
                    "memory_storage_data_size",
                    storage_metrics['data_size_bytes'],
                    tags={
                        "storage_type": storage_type,
                        "namespace": namespace,
                        "user_id": context.get("user_id", "unknown")
                    },
                    correlation_id=self.correlation_id
                )
            
            self.performance_monitor.complete_operation(
                operation_id,
                success=True,
                **storage_metrics
            )
    
    @contextmanager
    def monitor_llm_operation(self, operation_type: str, **context: Any):
        """Monitor LLM operations like summary generation."""
        operation_id = self.performance_monitor.start_operation(
            OperationType.LLM_INVOKE,
            correlation_id=self.correlation_id,
            llm_operation=operation_type,
            **{**self.operation_context, **context}
        )
        
        llm_metrics = {
            'operation_type': operation_type,
            'input_length': 0,
            'output_length': 0,
            'tokens_used': 0
        }
        
        try:
            logger.debug(
                "Starting LLM operation monitoring",
                correlation_id=self.correlation_id,
                operation_id=operation_id,
                operation_type=operation_type
            )
            
            yield {
                'operation_id': operation_id,
                'add_metric': lambda k, v: llm_metrics.update({k: v}),
                'track_error': self._create_error_tracker(operation_id)
            }
            
        except Exception as e:
            error_ctx = self.error_tracker.create_error_context(
                operation="llm_invoke",
                error=e,
                severity=ErrorSeverity.MEDIUM,
                correlation_id=self.correlation_id,
                operation_data={
                    'operation_id': operation_id,
                    'operation_type': operation_type,
                    **context
                }
            )
            self.error_tracker.track_error(error_ctx)
            raise
            
        finally:
            self.performance_monitor.complete_operation(
                operation_id,
                success=True,
                **llm_metrics
            )
    
    def _create_error_tracker(self, operation_id: str):
        """Create an error tracking function for a specific operation."""
        def track_error(error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **error_context: Any):
            error_ctx = self.error_tracker.create_error_context(
                operation=f"operation_{operation_id}",
                error=error,
                severity=severity,
                correlation_id=self.correlation_id,
                operation_data={
                    'operation_id': operation_id,
                    **self.operation_context,
                    **error_context
                }
            )
            self.error_tracker.track_error(error_ctx)
            
        return track_error
    
    def _create_enhanced_error_tracker(self, operation_id: str):
        """Create an enhanced error tracking function with metrics collection."""
        def track_error(error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, **error_context: Any):
            # Record error metrics
            increment_memory_counter(
                "memory_operation_errors",
                tags={
                    "operation_id": operation_id,
                    "error_type": type(error).__name__,
                    "severity": severity.value
                },
                correlation_id=self.correlation_id
            )
            
            # Create and track error context
            error_ctx = self.error_tracker.create_error_context(
                operation=f"operation_{operation_id}",
                error=error,
                severity=severity,
                correlation_id=self.correlation_id,
                operation_data={
                    'operation_id': operation_id,
                    **self.operation_context,
                    **error_context
                }
            )
            self.error_tracker.track_error(error_ctx)
            
        return track_error
    
    def _add_metric_with_collection(self, key: str, value: Any, metrics_dict: Dict[str, Any]) -> None:
        """Add metric to both local collection and global metrics system."""
        # Update local metrics
        metrics_dict[key] = value
        
        # Record in global metrics system
        if isinstance(value, (int, float)):
            set_memory_gauge(
                f"memory_operation_{key}",
                value,
                tags={"correlation_id": self.correlation_id},
                correlation_id=self.correlation_id
            )
    
    def _create_milestone_logger(self, operation_id: str):
        """Create a milestone logging function for a specific operation."""
        def log_milestone(milestone: str, **milestone_data: Any):
            # Record milestone as metric
            increment_memory_counter(
                "memory_operation_milestones",
                tags={
                    "operation_id": operation_id,
                    "milestone": milestone
                },
                correlation_id=self.correlation_id
            )
            
            logger.info(
                f"Memory operation milestone: {milestone}",
                correlation_id=self.correlation_id,
                operation_id=operation_id,
                **milestone_data
            )
            
        return log_milestone
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get a summary of all operations for this correlation ID."""
        return {
            'correlation_id': self.correlation_id,
            'context': self.operation_context,
            'error_summary': self.error_tracker.get_error_summary(time_window_minutes=5),
            'performance_summary': self.performance_monitor.get_performance_summary(time_window_minutes=5)
        }


def create_memory_monitor(correlation_id: Optional[str] = None, **context: Any) -> MemoryOperationMonitor:
    """Create a new memory operation monitor with context."""
    monitor = MemoryOperationMonitor(correlation_id)
    if context:
        monitor.set_context(**context)
    return monitor


@contextmanager
def enhanced_memory_operation(
    operation_name: str,
    correlation_id: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **context: Any
):
    """
    Context manager for enhanced memory operations with comprehensive monitoring.
    
    Usage:
        with enhanced_memory_operation("memory_condensation", user_id="123") as monitor:
            # Perform memory operations
            monitor.track_error(exception) if needed
            monitor.add_metric("items_processed", 5)
    """
    monitor = create_memory_monitor(correlation_id, **context)
    
    operation_id = monitor.performance_monitor.start_operation(
        OperationType.STORE_OPERATION,
        correlation_id=monitor.correlation_id,
        operation_name=operation_name,
        **context
    )
    
    operation_metrics = {
        'operation_name': operation_name,
        'success': True,
        'errors_count': 0
    }
    
    try:
        logger.info(
            f"Starting enhanced memory operation: {operation_name}",
            correlation_id=monitor.correlation_id,
            operation_id=operation_id,
            **context
        )
        
        yield {
            'monitor': monitor,
            'operation_id': operation_id,
            'correlation_id': monitor.correlation_id,
            'add_metric': lambda k, v: operation_metrics.update({k: v}),
            'track_error': lambda e, sev=ErrorSeverity.MEDIUM: monitor._create_error_tracker(operation_id)(e, sev),
            'log_milestone': monitor._create_milestone_logger(operation_id)
        }
        
    except Exception as e:
        operation_metrics['success'] = False
        operation_metrics['errors_count'] += 1
        
        error_ctx = monitor.error_tracker.create_error_context(
            operation=operation_name,
            error=e,
            severity=severity,
            correlation_id=monitor.correlation_id,
            operation_data={
                'operation_id': operation_id,
                **context
            }
        )
        monitor.error_tracker.track_error(error_ctx)
        raise
        
    finally:
        # Remove 'success' from operation_metrics to avoid duplicate parameter
        metrics_to_pass = {k: v for k, v in operation_metrics.items() if k != 'success'}
        monitor.performance_monitor.complete_operation(
            operation_id,
            success=operation_metrics['success'],
            **metrics_to_pass
        )
        
        logger.info(
            f"Enhanced memory operation completed: {operation_name}",
            correlation_id=monitor.correlation_id,
            operation_id=operation_id,
            success=operation_metrics['success'],
            final_metrics=operation_metrics
        )