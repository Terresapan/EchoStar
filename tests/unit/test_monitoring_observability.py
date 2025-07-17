"""
Comprehensive tests for monitoring and observability improvements.
Tests metrics collection, alerting, performance monitoring, and dashboard functionality.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.utils.metrics_collector import (
    MemoryMetricsCollector, MetricType, AlertSeverity, MetricPoint
)
from src.utils.alerting_system import (
    AlertingSystem, AlertRule, AlertEvent, AlertChannel, AlertState
)
from src.utils.monitoring_dashboard import (
    MemoryMonitoringDashboard, SystemHealthStatus
)
from src.utils.monitoring_integration import (
    MemoryMonitoringIntegration, initialize_monitoring, shutdown_monitoring,
    monitor_memory_operation, record_memory_metric, get_health_summary, export_metrics
)
from src.utils.memory_monitoring import MemoryOperationMonitor


class TestMetricsCollector:
    """Test the metrics collection system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.collector = MemoryMetricsCollector(max_points_per_metric=100)
    
    def test_record_metric(self):
        """Test basic metric recording."""
        self.collector.record_metric(
            "test_metric",
            42.5,
            MetricType.GAUGE,
            tags={"component": "test"},
            correlation_id="test-123"
        )
        
        # Verify metric was recorded
        assert "test_metric" in self.collector._metrics
        assert len(self.collector._metrics["test_metric"]) == 1
        
        metric_point = self.collector._metrics["test_metric"][0]
        assert metric_point.name == "test_metric"
        assert metric_point.value == 42.5
        assert metric_point.metric_type == MetricType.GAUGE
        assert metric_point.tags["component"] == "test"
        assert metric_point.correlation_id == "test-123"
    
    def test_increment_counter(self):
        """Test counter increment functionality."""
        # Increment counter multiple times
        self.collector.increment_counter("test_counter", 1, {"user": "test1"})
        self.collector.increment_counter("test_counter", 2, {"user": "test2"})
        self.collector.increment_counter("test_counter", 3, {"user": "test1"})
        
        # Verify all increments were recorded
        assert len(self.collector._metrics["test_counter"]) == 3
        
        # Verify values
        values = [point.value for point in self.collector._metrics["test_counter"]]
        assert values == [1, 2, 3]
    
    def test_timer_operations(self):
        """Test timer start/stop functionality."""
        # Start timer
        timer_id = self.collector.start_timer("test_operation", "corr-123")
        assert timer_id in self.collector._operation_timers
        
        # Simulate some work
        time.sleep(0.1)
        
        # Stop timer
        duration = self.collector.stop_timer(timer_id, {"component": "test"}, "corr-123")
        
        # Verify timer was stopped and duration recorded
        assert timer_id not in self.collector._operation_timers
        assert duration is not None
        assert duration >= 0.1
        
        # Verify duration metric was recorded
        duration_metric = "test_operation_duration"
        assert duration_metric in self.collector._metrics
        assert len(self.collector._metrics[duration_metric]) == 1
    
    def test_metric_summary(self):
        """Test metric summary calculation."""
        # Record multiple data points
        values = [10, 20, 30, 40, 50]
        for value in values:
            self.collector.record_metric("test_summary", value, MetricType.HISTOGRAM)
        
        # Get summary
        summary = self.collector.get_metric_summary("test_summary", 60)
        
        # Verify summary statistics
        assert summary["count"] == 5
        assert summary["min"] == 10
        assert summary["max"] == 50
        assert summary["avg"] == 30
        assert summary["sum"] == 150
        
        # Verify percentiles for histogram
        assert "p50" in summary
        assert "p90" in summary
        assert "p95" in summary
        assert "p99" in summary
    
    def test_alert_evaluation(self):
        """Test alert evaluation functionality."""
        # Create a fresh collector without default alerts
        collector = MemoryMetricsCollector(max_points_per_metric=100)
        collector._alerts = {}  # Clear default alerts
        
        # Add test alert
        collector.add_alert(
            "test_alert",
            "Test Alert",
            "test_metric > 100",
            AlertSeverity.WARNING,
            100,
            window_minutes=5
        )
        
        # Record metrics below threshold
        collector.record_metric("test_metric", 50)
        triggered = collector.evaluate_alerts()
        assert len(triggered) == 0
        
        # Record metrics above threshold - need multiple points for average
        collector.record_metric("test_metric", 150)
        collector.record_metric("test_metric", 120)
        collector.record_metric("test_metric", 130)
        
        triggered = collector.evaluate_alerts()
        assert len(triggered) == 1
        assert triggered[0]["name"] == "Test Alert"
    
    def test_metrics_cleanup(self):
        """Test old metrics cleanup."""
        # Record metrics with old timestamps
        old_time = time.time() - 25 * 3600  # 25 hours ago
        
        # Manually add old metric points
        old_point = MetricPoint("old_metric", 100, MetricType.GAUGE, old_time)
        self.collector._metrics["old_metric"].append(old_point)
        
        # Add recent metric
        self.collector.record_metric("recent_metric", 200)
        
        # Perform cleanup (keep 24 hours)
        cleared_count = self.collector.clear_old_metrics(24)
        
        # Verify old metrics were cleared
        assert cleared_count >= 1
        assert "old_metric" not in self.collector._metrics
        assert "recent_metric" in self.collector._metrics


class TestAlertingSystem:
    """Test the alerting system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.alerting = AlertingSystem()
        # Don't start the processing thread for tests
    
    def test_add_alert_rule(self):
        """Test adding alert rules."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test alert rule",
            metric_name="test_metric",
            condition="test_metric > 50",
            threshold=50,
            severity=AlertSeverity.WARNING
        )
        
        self.alerting.add_alert_rule(rule)
        
        # Verify rule was added
        assert "test_rule" in self.alerting._alert_rules
        assert self.alerting._alert_rules["test_rule"].name == "Test Rule"
    
    def test_alert_evaluation(self):
        """Test alert evaluation logic."""
        # Create a fresh alerting system without default rules
        alerting = AlertingSystem()
        alerting._alert_rules = {}  # Clear default rules
        
        # Mock metrics collector
        with patch.object(alerting, 'metrics_collector') as mock_collector:
            mock_collector.get_metric_summary.return_value = {
                "count": 5,
                "sum": 300,
                "avg": 60,
                "max": 80,
                "min": 40
            }
            
            # Add alert rule
            rule = AlertRule(
                rule_id="test_eval",
                name="Test Evaluation",
                description="Test evaluation",
                metric_name="test_metric",
                condition="test_metric > 50",
                threshold=50,
                severity=AlertSeverity.WARNING
            )
            alerting.add_alert_rule(rule)
            
            # Evaluate alerts
            events = alerting.evaluate_alerts()
            
            # Should trigger because avg (60) > threshold (50)
            assert len(events) == 1
            assert events[0].event_type == "trigger"
            assert events[0].severity == AlertSeverity.WARNING
    
    def test_alert_suppression(self):
        """Test alert suppression functionality."""
        rule_id = "suppress_test"
        
        # Add rule
        rule = AlertRule(
            rule_id=rule_id,
            name="Suppression Test",
            description="Test suppression",
            metric_name="test_metric",
            condition="test_metric > 10",
            threshold=10,
            severity=AlertSeverity.WARNING
        )
        self.alerting.add_alert_rule(rule)
        
        # Suppress the alert
        result = self.alerting.suppress_alert(rule_id, 30)
        assert result is True
        assert self.alerting._alert_rules[rule_id].state == AlertState.SUPPRESSED
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        rule_id = "ack_test"
        
        # Add rule and set to active
        rule = AlertRule(
            rule_id=rule_id,
            name="Acknowledgment Test",
            description="Test acknowledgment",
            metric_name="test_metric",
            condition="test_metric > 10",
            threshold=10,
            severity=AlertSeverity.WARNING,
            state=AlertState.ACTIVE
        )
        self.alerting.add_alert_rule(rule)
        
        # Acknowledge the alert
        result = self.alerting.acknowledge_alert(rule_id)
        assert result is True
        assert self.alerting._alert_rules[rule_id].state == AlertState.ACKNOWLEDGED
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Add multiple rules with different states
        rules = [
            AlertRule("active1", "Active 1", "desc", "metric1", "metric1 > 10", 10, AlertSeverity.WARNING, state=AlertState.ACTIVE),
            AlertRule("active2", "Active 2", "desc", "metric2", "metric2 > 20", 20, AlertSeverity.CRITICAL, state=AlertState.ACTIVE),
            AlertRule("resolved", "Resolved", "desc", "metric3", "metric3 > 30", 30, AlertSeverity.WARNING, state=AlertState.RESOLVED)
        ]
        
        for rule in rules:
            self.alerting.add_alert_rule(rule)
        
        # Get active alerts
        active = self.alerting.get_active_alerts()
        
        # Should only return active alerts
        assert len(active) == 2
        active_names = [alert["name"] for alert in active]
        assert "Active 1" in active_names
        assert "Active 2" in active_names
        assert "Resolved" not in active_names


class TestMonitoringDashboard:
    """Test the monitoring dashboard."""
    
    def setup_method(self):
        """Setup test environment."""
        self.dashboard = MemoryMonitoringDashboard()
    
    def test_system_health_assessment(self):
        """Test system health assessment."""
        # Mock dependencies
        with patch.object(self.dashboard, 'metrics_collector') as mock_metrics, \
             patch.object(self.dashboard, 'alerting_system') as mock_alerting:
            
            # Mock healthy metrics
            mock_metrics.get_metric_summary.return_value = {
                "count": 10,
                "sum": 8,  # 8 successes out of 10 = 80% success rate
                "avg": 0.8
            }
            
            mock_alerting.get_active_alerts.return_value = []
            
            # Get health status
            health = self.dashboard.get_system_health(force_refresh=True)
            
            # Verify health assessment
            assert isinstance(health, SystemHealthStatus)
            assert health.overall_status in ["healthy", "degraded", "critical"]
            assert health.active_alerts == 0
            assert isinstance(health.recommendations, list)
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        with patch.object(self.dashboard, 'metrics_collector') as mock_metrics:
            # Mock the get_all_metrics_summary method
            mock_metrics.get_all_metrics_summary.return_value = {
                "window_minutes": 60,
                "timestamp": time.time(),
                "metrics": {
                    "test_metric": {
                        "count": 5,
                        "avg": 25.0,
                        "sum": 125
                    }
                },
                "total_metrics": 1,
                "active_alerts": 0
            }
            
            # Mock the get_metric_summary method for derived metrics calculations
            def mock_get_metric_summary(metric_name, window_minutes):
                return {
                    "count": 5,
                    "sum": 10,
                    "avg": 2.0,
                    "min": 1,
                    "max": 5
                }
            
            mock_metrics.get_metric_summary.side_effect = mock_get_metric_summary
            
            summary = self.dashboard.get_metrics_summary(60)
            
            # Verify summary structure
            assert "derived_metrics" in summary
            assert "trends" in summary
            assert summary["window_minutes"] == 60
    
    def test_alert_dashboard(self):
        """Test alert dashboard generation."""
        with patch.object(self.dashboard, 'alerting_system') as mock_alerting:
            mock_alerting.get_active_alerts.return_value = [
                {
                    "rule_id": "test1",
                    "name": "Test Alert 1",
                    "severity": "warning"
                }
            ]
            
            mock_alerting.get_alert_statistics.return_value = {
                "total_rules": 5,
                "active_alerts": 1
            }
            
            # Mock alert history
            mock_alerting._alert_history = []
            
            dashboard = self.dashboard.get_alert_dashboard()
            
            # Verify dashboard structure
            assert "active_alerts" in dashboard
            assert "alerts_by_severity" in dashboard
            assert "alert_statistics" in dashboard
            assert "recent_events" in dashboard
    
    def test_performance_dashboard(self):
        """Test performance dashboard generation."""
        with patch.object(self.dashboard, 'performance_monitor') as mock_perf:
            mock_perf.get_performance_summary.return_value = {
                "total_operations": 100,
                "performance_stats": {
                    "avg_duration_ms": 250.5,
                    "max_duration_ms": 500.0
                }
            }
            
            dashboard = self.dashboard.get_performance_dashboard(60)
            
            # Verify dashboard structure
            assert "performance_by_operation" in dashboard
            assert "performance_trends" in dashboard
            assert "bottlenecks" in dashboard
            assert "recommendations" in dashboard


class TestMemoryOperationMonitor:
    """Test the memory operation monitor."""
    
    def setup_method(self):
        """Setup test environment."""
        self.monitor = MemoryOperationMonitor("test-correlation-123")
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.correlation_id == "test-correlation-123"
        assert hasattr(self.monitor, 'metrics_collector')
        assert hasattr(self.monitor, 'alerting_system')
        assert hasattr(self.monitor, '_active_timers')
    
    def test_memory_condensation_monitoring(self):
        """Test memory condensation monitoring context."""
        with patch.object(self.monitor, 'performance_monitor') as mock_perf, \
             patch('src.utils.memory_monitoring.start_memory_timer') as mock_start_timer, \
             patch('src.utils.memory_monitoring.stop_memory_timer') as mock_stop_timer, \
             patch('src.utils.memory_monitoring.increment_memory_counter') as mock_counter:
            
            mock_perf.start_operation.return_value = "op-123"
            mock_start_timer.return_value = "timer-123"
            mock_stop_timer.return_value = 1.5
            
            # Test successful condensation monitoring
            with self.monitor.monitor_memory_condensation(user_id="test-user") as ctx:
                assert "operation_id" in ctx
                assert "correlation_id" in ctx
                assert "add_metric" in ctx
                assert "track_error" in ctx
                assert "log_milestone" in ctx
                
                # Test metric addition
                ctx["add_metric"]("test_key", 42)
            
            # Verify timer operations were called
            mock_start_timer.assert_called_once()
            mock_stop_timer.assert_called_once()
            mock_counter.assert_called()
    
    def test_memory_retrieval_monitoring(self):
        """Test memory retrieval monitoring context."""
        with patch.object(self.monitor, 'performance_monitor') as mock_perf, \
             patch('src.utils.memory_monitoring.start_memory_timer') as mock_start_timer, \
             patch('src.utils.memory_monitoring.stop_memory_timer') as mock_stop_timer:
            
            mock_perf.start_operation.return_value = "op-456"
            mock_start_timer.return_value = "timer-456"
            mock_stop_timer.return_value = 0.5
            
            # Test retrieval monitoring
            with self.monitor.monitor_memory_retrieval("test-namespace", 10, user_id="test-user") as ctx:
                assert "operation_id" in ctx
                assert "add_metric" in ctx
                assert "track_error" in ctx
                
                # Test adding retrieval metrics
                ctx["add_metric"]("memories_found", 5)
            
            # Verify operations were called
            mock_start_timer.assert_called_once()
            mock_stop_timer.assert_called_once()
    
    def test_error_tracking(self):
        """Test enhanced error tracking."""
        with patch.object(self.monitor, 'performance_monitor') as mock_perf, \
             patch('src.utils.memory_monitoring.increment_memory_counter') as mock_counter:
            
            mock_perf.start_operation.return_value = "op-error"
            
            # Test error tracking in condensation monitoring
            try:
                with self.monitor.monitor_memory_condensation(user_id="test-user") as ctx:
                    # Simulate an error
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected
            
            # Verify error metrics were recorded
            mock_counter.assert_called()
            
            # Check that failure counter was called
            calls = mock_counter.call_args_list
            failure_calls = [call for call in calls if "failures" in str(call)]
            assert len(failure_calls) > 0


class TestMonitoringIntegration:
    """Test the monitoring integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.integration = MemoryMonitoringIntegration()
    
    def test_initialization(self):
        """Test integration initialization."""
        with patch('src.utils.monitoring_integration.start_alerting') as mock_start:
            self.integration.initialize(start_background_monitoring=False)
            
            assert self.integration._initialized is True
            mock_start.assert_called_once()
    
    def test_record_memory_operation(self):
        """Test recording memory operations."""
        with patch.object(self.integration, 'metrics_collector') as mock_collector:
            self.integration.record_memory_operation(
                "test_operation",
                success=True,
                duration_seconds=1.5,
                user_id="test-user",
                correlation_id="test-corr"
            )
            
            # Verify metrics were recorded
            assert mock_collector.increment_counter.call_count >= 2  # attempts + successes
            mock_collector.record_histogram.assert_called_once()
    
    def test_monitor_operation_context(self):
        """Test operation monitoring context manager."""
        with patch.object(self.integration, 'record_memory_operation') as mock_record:
            # Test successful operation
            with self.integration.monitor_operation("test_op", user_id="test") as ctx:
                assert "correlation_id" in ctx
                assert "operation_name" in ctx
                assert "start_time" in ctx
                
                # Simulate some work
                time.sleep(0.01)
            
            # Verify operation was recorded as successful
            mock_record.assert_called_once()
            args, kwargs = mock_record.call_args
            assert args[0] == "test_op"  # operation_name
            assert args[1] is True  # success
            assert args[2] is not None  # duration
    
    def test_monitor_operation_with_error(self):
        """Test operation monitoring with errors."""
        with patch.object(self.integration, 'metrics_collector') as mock_collector, \
             patch.object(self.integration, 'record_memory_operation') as mock_record:
            
            # Test operation with error
            try:
                with self.integration.monitor_operation("error_op", user_id="test"):
                    raise RuntimeError("Test error")
            except RuntimeError:
                pass  # Expected
            
            # Verify error metrics were recorded
            mock_collector.increment_counter.assert_called()
            
            # Verify operation was recorded as failed
            mock_record.assert_called_once()
            args, kwargs = mock_record.call_args
            assert args[1] is False  # success = False
    
    def test_health_report(self):
        """Test health report generation."""
        with patch.object(self.integration, 'monitoring_dashboard') as mock_dashboard, \
             patch.object(self.integration, 'alerting_system') as mock_alerting:
            
            # Mock health status
            mock_health = SystemHealthStatus(
                overall_status="healthy",
                timestamp=time.time(),
                memory_condensation_health="healthy",
                memory_storage_health="healthy",
                memory_retrieval_health="healthy",
                profile_storage_health="healthy",
                active_alerts=0,
                error_rate_5min=0.0,
                avg_response_time_5min=0.5,
                success_rate_5min=100.0,
                recommendations=["System is healthy"]
            )
            
            mock_dashboard.get_system_health.return_value = mock_health
            mock_dashboard.get_metrics_summary.return_value = {"test": "data"}
            mock_dashboard.get_error_dashboard.return_value = {"recent_errors": []}
            mock_alerting.get_active_alerts.return_value = []
            
            # Get health report
            report = self.integration.get_health_report()
            
            # Verify report structure
            assert "timestamp" in report
            assert "system_health" in report
            assert "metrics_summary" in report
            assert "active_alerts" in report
            assert "recent_errors" in report
            assert "recommendations" in report
    
    def test_export_monitoring_data(self):
        """Test monitoring data export."""
        with patch.object(self.integration, 'monitoring_dashboard') as mock_dashboard:
            mock_dashboard.export_dashboard_data.return_value = '{"test": "data"}'
            
            # Test detailed export
            result = self.integration.export_monitoring_data("json", include_detailed_metrics=True)
            assert result == '{"test": "data"}'
            mock_dashboard.export_dashboard_data.assert_called_once_with("json")
    
    def test_trigger_health_check(self):
        """Test manual health check trigger."""
        with patch.object(self.integration, 'monitoring_dashboard') as mock_dashboard, \
             patch.object(self.integration, 'alerting_system') as mock_alerting:
            
            mock_health = SystemHealthStatus(
                overall_status="healthy",
                timestamp=time.time(),
                memory_condensation_health="healthy",
                memory_storage_health="healthy",
                memory_retrieval_health="healthy",
                profile_storage_health="healthy",
                active_alerts=0,
                error_rate_5min=0.0,
                avg_response_time_5min=0.5,
                success_rate_5min=100.0,
                recommendations=[]
            )
            
            mock_dashboard.get_system_health.return_value = mock_health
            mock_alerting.evaluate_alerts.return_value = []
            
            # Trigger health check
            result = self.integration.trigger_health_check()
            
            # Verify health check results
            assert "health_status" in result
            assert "triggered_alerts" in result
            assert result["triggered_alerts"] == 0
            
            # Verify force refresh was called
            mock_dashboard.get_system_health.assert_called_once_with(force_refresh=True)


class TestIntegrationFunctions:
    """Test integration convenience functions."""
    
    def test_initialize_monitoring(self):
        """Test monitoring initialization function."""
        with patch('src.utils.monitoring_integration._monitoring_integration') as mock_integration:
            initialize_monitoring(start_background=False)
            mock_integration.initialize.assert_called_once_with(False)
    
    def test_shutdown_monitoring(self):
        """Test monitoring shutdown function."""
        with patch('src.utils.monitoring_integration._monitoring_integration') as mock_integration:
            shutdown_monitoring()
            mock_integration.shutdown.assert_called_once()
    
    def test_monitor_memory_operation_context(self):
        """Test memory operation monitoring context function."""
        with patch('src.utils.monitoring_integration._monitoring_integration') as mock_integration:
            mock_integration.monitor_operation.return_value.__enter__ = Mock(return_value={"test": "context"})
            mock_integration.monitor_operation.return_value.__exit__ = Mock(return_value=None)
            
            with monitor_memory_operation("test_op", user_id="test") as ctx:
                assert ctx == {"test": "context"}
            
            mock_integration.monitor_operation.assert_called_once()
    
    def test_record_memory_metric(self):
        """Test memory metric recording function."""
        with patch('src.utils.monitoring_integration._monitoring_integration') as mock_integration:
            record_memory_metric("test_op", True, 1.5, user_id="test")
            
            mock_integration.record_memory_operation.assert_called_once_with(
                "test_op", True, 1.5, user_id="test"
            )
    
    def test_get_health_summary(self):
        """Test health summary function."""
        with patch('src.utils.monitoring_integration._monitoring_integration') as mock_integration:
            mock_integration.get_health_report.return_value = {"status": "healthy"}
            
            result = get_health_summary()
            assert result == {"status": "healthy"}
            mock_integration.get_health_report.assert_called_once()
    
    def test_export_metrics(self):
        """Test metrics export function."""
        with patch('src.utils.monitoring_integration._monitoring_integration') as mock_integration:
            mock_integration.export_monitoring_data.return_value = '{"metrics": "data"}'
            
            result = export_metrics("json")
            assert result == '{"metrics": "data"}'
            mock_integration.export_monitoring_data.assert_called_once_with("json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])