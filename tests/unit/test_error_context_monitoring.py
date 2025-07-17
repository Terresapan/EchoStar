#!/usr/bin/env python3
"""
Unit tests for comprehensive error context and monitoring utilities.
Tests the error tracking, performance monitoring, and memory operation monitoring.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.error_context import (
    ErrorContext, PerformanceMetrics, ErrorTracker, PerformanceMonitor,
    ErrorSeverity, OperationType, create_correlation_id, error_context, performance_context,
    get_error_tracker, get_performance_monitor
)
from utils.memory_monitoring import (
    MemoryOperationMonitor, create_memory_monitor, enhanced_memory_operation
)


class TestErrorContext:
    """Test ErrorContext data structure and functionality."""
    
    def test_error_context_creation(self):
        """Test creating error context with default values."""
        error_ctx = ErrorContext(
            operation="test_operation",
            error_type="TestError",
            error_message="Test error message"
        )
        
        assert error_ctx.operation == "test_operation"
        assert error_ctx.error_type == "TestError"
        assert error_ctx.error_message == "Test error message"
        assert error_ctx.severity == ErrorSeverity.MEDIUM
        assert error_ctx.error_id is not None
        assert error_ctx.timestamp is not None
        
    def test_error_context_to_dict(self):
        """Test converting error context to dictionary."""
        error_ctx = ErrorContext(
            operation="test_operation",
            error_type="TestError",
            error_message="Test error message",
            user_id="test_user",
            correlation_id="test_correlation"
        )
        
        error_dict = error_ctx.to_dict()
        
        assert isinstance(error_dict, dict)
        assert error_dict["operation"] == "test_operation"
        assert error_dict["error_type"] == "TestError"
        assert error_dict["user_id"] == "test_user"
        assert error_dict["correlation_id"] == "test_correlation"
        
    def test_error_context_to_json(self):
        """Test converting error context to JSON."""
        error_ctx = ErrorContext(
            operation="test_operation",
            error_type="TestError",
            error_message="Test error message"
        )
        
        json_str = error_ctx.to_json()
        
        assert isinstance(json_str, str)
        assert "test_operation" in json_str
        assert "TestError" in json_str


class TestPerformanceMetrics:
    """Test PerformanceMetrics data structure and functionality."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            operation_type=OperationType.MEMORY_CONDENSATION
        )
        
        assert metrics.operation_type == OperationType.MEMORY_CONDENSATION
        assert metrics.operation_id is not None
        assert metrics.start_time is not None
        assert metrics.end_time is None
        assert metrics.duration_seconds is None
        
    def test_performance_metrics_completion(self):
        """Test completing performance metrics calculation."""
        metrics = PerformanceMetrics(
            operation_type=OperationType.MEMORY_STORAGE
        )
        
        # Simulate some processing time
        time.sleep(0.01)
        
        metrics.complete()
        
        assert metrics.end_time is not None
        assert metrics.duration_seconds is not None
        assert metrics.duration_ms is not None
        assert metrics.duration_seconds > 0
        assert metrics.duration_ms > 0
        
    def test_performance_metrics_to_dict(self):
        """Test converting performance metrics to dictionary."""
        metrics = PerformanceMetrics(
            operation_type=OperationType.LLM_INVOKE,
            user_id="test_user",
            items_processed=5
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["operation_type"] == OperationType.LLM_INVOKE
        assert metrics_dict["user_id"] == "test_user"
        assert metrics_dict["items_processed"] == 5


class TestErrorTracker:
    """Test ErrorTracker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_tracker = ErrorTracker()
        
    def test_create_error_context(self):
        """Test creating error context through tracker."""
        test_error = ValueError("Test error")
        
        error_ctx = self.error_tracker.create_error_context(
            operation="test_operation",
            error=test_error,
            severity=ErrorSeverity.HIGH,
            user_id="test_user"
        )
        
        assert error_ctx.operation == "test_operation"
        assert error_ctx.error_type == "ValueError"
        assert error_ctx.error_message == "Test error"
        assert error_ctx.severity == ErrorSeverity.HIGH
        assert error_ctx.user_id == "test_user"
        
    def test_track_error(self):
        """Test tracking errors with correlation."""
        correlation_id = "test_correlation"
        
        error_ctx = ErrorContext(
            operation="test_operation",
            error_type="TestError",
            error_message="Test error",
            correlation_id=correlation_id
        )
        
        with patch.object(self.error_tracker, 'logger') as mock_logger:
            self.error_tracker.track_error(error_ctx)
            
            # Verify logging was called
            mock_logger.error.assert_called_once()
            
        # Verify correlation mapping
        correlated_errors = self.error_tracker.get_correlated_errors(correlation_id)
        assert len(correlated_errors) == 1
        assert correlated_errors[0].error_id == error_ctx.error_id
        
    def test_get_error_summary(self):
        """Test getting error summary."""
        # Create some test errors
        for i in range(3):
            error_ctx = ErrorContext(
                operation=f"test_operation_{i}",
                error_type="TestError",
                error_message=f"Test error {i}",
                severity=ErrorSeverity.MEDIUM
            )
            self.error_tracker._add_to_history(error_ctx)
        
        summary = self.error_tracker.get_error_summary(time_window_minutes=60)
        
        assert summary["total_errors"] == 3
        assert "TestError" in summary["error_by_type"]
        assert summary["error_by_type"]["TestError"] == 3
        assert "medium" in summary["error_by_severity"]


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.performance_monitor = PerformanceMonitor()
        
    def test_start_operation(self):
        """Test starting operation monitoring."""
        operation_id = self.performance_monitor.start_operation(
            OperationType.MEMORY_RETRIEVAL,
            correlation_id="test_correlation",
            user_id="test_user"
        )
        
        assert operation_id is not None
        assert operation_id in self.performance_monitor._active_operations
        
        metrics = self.performance_monitor._active_operations[operation_id]
        assert metrics.operation_type == OperationType.MEMORY_RETRIEVAL
        assert metrics.correlation_id == "test_correlation"
        # Context data is stored in custom_metrics, not as direct attributes
        assert metrics.custom_metrics["user_id"] == "test_user"
        
    def test_complete_operation(self):
        """Test completing operation monitoring."""
        operation_id = self.performance_monitor.start_operation(
            OperationType.MEMORY_STORAGE,
            items_processed=10
        )
        
        # Simulate some processing time
        time.sleep(0.01)
        
        completed_metrics = self.performance_monitor.complete_operation(
            operation_id,
            success=True,
            bytes_processed=1024
        )
        
        assert completed_metrics is not None
        assert completed_metrics.duration_seconds is not None
        assert completed_metrics.duration_seconds > 0
        assert completed_metrics.custom_metrics["bytes_processed"] == 1024
        assert operation_id not in self.performance_monitor._active_operations
        
    def test_monitor_operation_context_manager(self):
        """Test operation monitoring context manager."""
        with self.performance_monitor.monitor_operation(
            OperationType.LLM_INVOKE,
            correlation_id="test_correlation"
        ) as op_context:
            
            assert "operation_id" in op_context
            assert "add_metric" in op_context
            
            # Add some metrics
            op_context["add_metric"]("tokens_used", 150)
            
            # Simulate some work
            time.sleep(0.01)
        
        # Verify operation was completed
        assert len(self.performance_monitor._completed_operations) == 1
        completed_op = self.performance_monitor._completed_operations[0]
        assert completed_op.operation_type == OperationType.LLM_INVOKE
        assert completed_op.custom_metrics["tokens_used"] == 150
        
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Create some completed operations
        for i in range(3):
            operation_id = self.performance_monitor.start_operation(
                OperationType.MEMORY_CONDENSATION,
                items_processed=i + 1
            )
            time.sleep(0.001)  # Small delay for different durations
            self.performance_monitor.complete_operation(operation_id, success=True)
        
        summary = self.performance_monitor.get_performance_summary(
            operation_type=OperationType.MEMORY_CONDENSATION,
            time_window_minutes=60
        )
        
        assert summary["total_operations"] == 3
        assert summary["operation_type"] == "memory_condensation"
        assert "performance_stats" in summary
        assert summary["performance_stats"]["avg_duration_ms"] > 0


class TestMemoryOperationMonitor:
    """Test MemoryOperationMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = MemoryOperationMonitor()
        
    def test_monitor_creation(self):
        """Test creating memory operation monitor."""
        assert self.monitor.correlation_id is not None
        assert self.monitor.error_tracker is not None
        assert self.monitor.performance_monitor is not None
        
    def test_set_context(self):
        """Test setting operation context."""
        self.monitor.set_context(
            user_id="test_user",
            session_id="test_session"
        )
        
        assert self.monitor.operation_context["user_id"] == "test_user"
        assert self.monitor.operation_context["session_id"] == "test_session"
        
    def test_monitor_memory_condensation(self):
        """Test monitoring memory condensation process."""
        with self.monitor.monitor_memory_condensation(
            user_id="test_user",
            turn_count=5
        ) as condensation_ctx:
            
            assert "operation_id" in condensation_ctx
            assert "correlation_id" in condensation_ctx
            assert "add_metric" in condensation_ctx
            assert "track_error" in condensation_ctx
            assert "log_milestone" in condensation_ctx
            
            # Test adding metrics
            condensation_ctx["add_metric"]("memories_retrieved", 10)
            condensation_ctx["add_metric"]("dialogue_turns_processed", 8)
            
            # Test milestone logging
            condensation_ctx["log_milestone"]("test_milestone", test_data="test_value")
        
        # Verify operation was completed
        assert len(self.monitor.performance_monitor._completed_operations) == 1
        
    def test_monitor_memory_retrieval(self):
        """Test monitoring memory retrieval operations."""
        # Create a fresh monitor for this test
        fresh_monitor = MemoryOperationMonitor()
        
        with fresh_monitor.monitor_memory_retrieval(
            namespace="test_namespace",
            limit=10,
            user_id="test_user"
        ) as retrieval_ctx:
            
            assert "operation_id" in retrieval_ctx
            assert "add_metric" in retrieval_ctx
            assert "track_error" in retrieval_ctx
            
            # Test adding metrics
            retrieval_ctx["add_metric"]("memories_found", 5)
        
        # Verify operation was completed
        completed_ops = fresh_monitor.performance_monitor._completed_operations
        assert len(completed_ops) == 1
        assert completed_ops[0].operation_type == OperationType.MEMORY_RETRIEVAL
        
    def test_monitor_memory_storage(self):
        """Test monitoring memory storage operations."""
        # Create a fresh monitor for this test
        fresh_monitor = MemoryOperationMonitor()
        
        with fresh_monitor.monitor_memory_storage(
            storage_type="semantic",
            namespace="test_namespace",
            user_id="test_user"
        ) as storage_ctx:
            
            assert "operation_id" in storage_ctx
            assert "add_metric" in storage_ctx
            assert "track_error" in storage_ctx
            
            # Test adding metrics
            storage_ctx["add_metric"]("verification_successful", True)
            storage_ctx["add_metric"]("data_size_bytes", 1024)
        
        # Verify operation was completed
        completed_ops = fresh_monitor.performance_monitor._completed_operations
        assert len(completed_ops) == 1
        assert completed_ops[0].operation_type == OperationType.MEMORY_STORAGE
        
    def test_monitor_llm_operation(self):
        """Test monitoring LLM operations."""
        # Create a fresh monitor for this test
        fresh_monitor = MemoryOperationMonitor()
        
        with fresh_monitor.monitor_llm_operation(
            operation_type="summary_generation",
            user_id="test_user"
        ) as llm_ctx:
            
            assert "operation_id" in llm_ctx
            assert "add_metric" in llm_ctx
            assert "track_error" in llm_ctx
            
            # Test adding metrics
            llm_ctx["add_metric"]("input_length", 500)
            llm_ctx["add_metric"]("output_length", 150)
            llm_ctx["add_metric"]("tokens_used", 200)
        
        # Verify operation was completed
        completed_ops = fresh_monitor.performance_monitor._completed_operations
        assert len(completed_ops) == 1
        assert completed_ops[0].operation_type == OperationType.LLM_INVOKE
        
    def test_get_operation_summary(self):
        """Test getting operation summary."""
        self.monitor.set_context(user_id="test_user", session_id="test_session")
        
        # Perform some operations
        with self.monitor.monitor_memory_retrieval("test_namespace", 10):
            pass
            
        with self.monitor.monitor_memory_storage("semantic", "test_namespace"):
            pass
        
        summary = self.monitor.get_operation_summary()
        
        assert summary["correlation_id"] == self.monitor.correlation_id
        assert "context" in summary
        assert summary["context"]["user_id"] == "test_user"
        assert "error_summary" in summary
        assert "performance_summary" in summary


class TestContextManagers:
    """Test context manager utilities."""
    
    def test_error_context_manager_success(self):
        """Test error context manager with successful operation."""
        with error_context(
            operation="test_operation",
            severity=ErrorSeverity.LOW,
            user_id="test_user"
        ):
            # Successful operation
            result = 1 + 1
            assert result == 2
        
        # No exception should be raised
        
    def test_error_context_manager_with_exception(self):
        """Test error context manager with exception."""
        with pytest.raises(ValueError):
            with error_context(
                operation="test_operation",
                severity=ErrorSeverity.HIGH,
                user_id="test_user"
            ):
                raise ValueError("Test error")
        
        # Error should be tracked
        error_tracker = get_error_tracker()
        recent_errors = error_tracker._error_history[-1:]
        assert len(recent_errors) >= 1
        assert recent_errors[-1].operation == "test_operation"
        
    def test_performance_context_manager(self):
        """Test performance context manager."""
        with performance_context(
            OperationType.MEMORY_CONDENSATION,
            user_id="test_user"
        ) as perf_ctx:
            
            assert "operation_id" in perf_ctx
            assert "add_metric" in perf_ctx
            
            # Add some metrics
            perf_ctx["add_metric"]("test_metric", 42)
            
            # Simulate some work
            time.sleep(0.01)
        
        # Verify operation was completed
        performance_monitor = get_performance_monitor()
        completed_ops = performance_monitor._completed_operations
        assert len(completed_ops) >= 1
        
    def test_enhanced_memory_operation_context(self):
        """Test enhanced memory operation context manager."""
        with enhanced_memory_operation(
            operation_name="test_memory_operation",
            user_id="test_user",
            session_id="test_session"
        ) as op_ctx:
            
            assert "monitor" in op_ctx
            assert "operation_id" in op_ctx
            assert "correlation_id" in op_ctx
            assert "add_metric" in op_ctx
            assert "track_error" in op_ctx
            assert "log_milestone" in op_ctx
            
            # Test functionality
            op_ctx["add_metric"]("items_processed", 5)
            op_ctx["log_milestone"]("processing_complete")
        
        # Verify operation was completed
        performance_monitor = get_performance_monitor()
        completed_ops = performance_monitor._completed_operations
        assert len(completed_ops) >= 1


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_correlation_id(self):
        """Test creating correlation IDs."""
        correlation_id1 = create_correlation_id()
        correlation_id2 = create_correlation_id()
        
        assert correlation_id1 is not None
        assert correlation_id2 is not None
        assert correlation_id1 != correlation_id2
        assert len(correlation_id1) > 0
        assert len(correlation_id2) > 0
        
    def test_create_memory_monitor(self):
        """Test creating memory monitor with context."""
        monitor = create_memory_monitor(
            user_id="test_user",
            session_id="test_session",
            turn_count=5
        )
        
        assert monitor is not None
        assert monitor.operation_context["user_id"] == "test_user"
        assert monitor.operation_context["session_id"] == "test_session"
        assert monitor.operation_context["turn_count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])