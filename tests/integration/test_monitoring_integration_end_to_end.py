"""
End-to-end integration test for monitoring and observability improvements.
Demonstrates the complete monitoring system working together.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from src.utils.monitoring_integration import (
    initialize_monitoring, shutdown_monitoring, 
    monitor_memory_operation, record_memory_metric, get_health_summary
)
from src.utils.metrics_collector import get_metrics_collector, MetricType, AlertSeverity
from src.utils.alerting_system import get_alerting_system
from src.utils.monitoring_dashboard import get_monitoring_dashboard


class TestMonitoringIntegrationEndToEnd:
    """End-to-end integration tests for the monitoring system."""
    
    def setup_method(self):
        """Setup test environment."""
        # Initialize monitoring without background thread for testing
        initialize_monitoring(start_background=False)
        
        # Clear any existing metrics and alerts for clean testing
        metrics_collector = get_metrics_collector()
        metrics_collector._metrics.clear()
        metrics_collector._metric_metadata.clear()
        
        alerting_system = get_alerting_system()
        # Reset alert states
        for alert in alerting_system._alert_rules.values():
            alert.state = alert.state.__class__.RESOLVED
            alert.trigger_count = 0
            alert.consecutive_triggers = 0
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutdown_monitoring()
    
    def test_complete_memory_operation_monitoring_flow(self):
        """Test complete flow of memory operation monitoring."""
        
        # Test successful memory operation
        with monitor_memory_operation("test_condensation", user_id="test-user-123") as ctx:
            # Simulate memory condensation work
            time.sleep(0.01)  # Small delay to measure duration
            
            # Verify context contains expected keys
            assert "correlation_id" in ctx
            assert "operation_name" in ctx
            assert ctx["operation_name"] == "test_condensation"
        
        # Verify metrics were recorded
        metrics_collector = get_metrics_collector()
        
        # Check that attempt and success counters were incremented
        attempts_summary = metrics_collector.get_metric_summary("test_condensation_attempts", 1)
        assert attempts_summary["count"] > 0
        
        successes_summary = metrics_collector.get_metric_summary("test_condensation_successes", 1)
        assert successes_summary["count"] > 0
        
        # Check that duration was recorded
        duration_summary = metrics_collector.get_metric_summary("test_condensation_duration", 1)
        assert duration_summary["count"] > 0
        assert duration_summary["avg"] > 0
    
    def test_memory_operation_error_handling_and_metrics(self):
        """Test error handling and metrics collection for failed operations."""
        
        # Test failed memory operation
        try:
            with monitor_memory_operation("test_error_operation", user_id="test-user-456"):
                # Simulate an error
                raise ValueError("Simulated memory operation error")
        except ValueError:
            pass  # Expected error
        
        # Verify error metrics were recorded
        metrics_collector = get_metrics_collector()
        
        # Check that failure counter was incremented
        failures_summary = metrics_collector.get_metric_summary("test_error_operation_failures", 1)
        assert failures_summary["count"] > 0
        
        # Check that error metrics were recorded
        error_summary = metrics_collector.get_metric_summary("memory_operation_errors", 1)
        assert error_summary["count"] > 0
    
    def test_alert_triggering_and_resolution(self):
        """Test alert triggering and resolution flow."""
        
        alerting_system = get_alerting_system()
        metrics_collector = get_metrics_collector()
        
        # Record multiple failure metrics to trigger an alert
        # The alert looks for "memory_condensation_failures" metric with threshold 5
        # Record 7 failures to ensure we exceed the threshold
        for i in range(7):
            metrics_collector.increment_counter("memory_condensation_failures", 1, {"user_id": "test-user"})
        
        # Evaluate alerts
        triggered_alerts = alerting_system.evaluate_alerts()
        
        # Verify alert was triggered
        assert len(triggered_alerts) > 0
        
        # Find the critical failures alert
        critical_alert = None
        for alert in triggered_alerts:
            if "condensation" in alert.rule_id.lower() and "critical" in alert.rule_id.lower():
                critical_alert = alert
                break
        
        assert critical_alert is not None
        assert critical_alert.event_type == "trigger"
        assert critical_alert.severity == AlertSeverity.CRITICAL
        
        # Get active alerts
        active_alerts = alerting_system.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Verify the alert is in active state
        active_alert_names = [alert["name"] for alert in active_alerts]
        assert any("Critical Memory Condensation" in name for name in active_alert_names)
    
    def test_system_health_monitoring(self):
        """Test system health monitoring and assessment."""
        
        # Record some successful operations
        for i in range(10):
            record_memory_metric("memory_condensation", True, 0.5, user_id="test-user")
            record_memory_metric("memory_storage", True, 0.2, user_id="test-user")
            record_memory_metric("memory_retrieval", True, 0.1, user_id="test-user")
        
        # Record a few failures
        for i in range(2):
            record_memory_metric("memory_condensation", False, user_id="test-user")
        
        # Get health summary
        health_report = get_health_summary()
        
        # Verify health report structure
        assert "timestamp" in health_report
        assert "system_health" in health_report
        assert "metrics_summary" in health_report
        assert "active_alerts" in health_report
        assert "recommendations" in health_report
        
        # Verify system health details
        system_health = health_report["system_health"]
        assert "overall_status" in system_health
        assert system_health["overall_status"] in ["healthy", "degraded", "critical"]
        assert "memory_condensation_health" in system_health
        assert "active_alerts" in system_health
        assert "success_rate_5min" in system_health
        assert "error_rate_5min" in system_health
    
    def test_metrics_collection_and_aggregation(self):
        """Test comprehensive metrics collection and aggregation."""
        
        metrics_collector = get_metrics_collector()
        
        # Record various types of metrics
        metrics_collector.increment_counter("test_counter", 5, {"component": "test"})
        metrics_collector.set_gauge("test_gauge", 42.5, {"component": "test"})
        metrics_collector.record_histogram("test_histogram", 1.5, {"component": "test"})
        
        # Start and stop a timer
        timer_id = metrics_collector.start_timer("test_timer")
        time.sleep(0.01)
        duration = metrics_collector.stop_timer(timer_id, {"component": "test"})
        
        # Verify all metrics were recorded
        assert "test_counter" in metrics_collector._metrics
        assert "test_gauge" in metrics_collector._metrics
        assert "test_histogram" in metrics_collector._metrics
        assert "test_timer_duration" in metrics_collector._metrics
        
        # Verify metric summaries
        counter_summary = metrics_collector.get_metric_summary("test_counter", 1)
        assert counter_summary["sum"] == 5
        
        gauge_summary = metrics_collector.get_metric_summary("test_gauge", 1)
        assert gauge_summary["avg"] == 42.5
        
        histogram_summary = metrics_collector.get_metric_summary("test_histogram", 1)
        assert histogram_summary["avg"] == 1.5
        assert "p50" in histogram_summary  # Percentiles for histograms
        
        timer_summary = metrics_collector.get_metric_summary("test_timer_duration", 1)
        assert timer_summary["count"] == 1
        assert timer_summary["avg"] > 0
    
    def test_monitoring_dashboard_integration(self):
        """Test monitoring dashboard integration and data export."""
        
        dashboard = get_monitoring_dashboard()
        
        # Record some test data
        for i in range(5):
            record_memory_metric("memory_condensation", True, 1.0 + i * 0.1, user_id="test-user")
            record_memory_metric("memory_storage", True, 0.5 + i * 0.05, user_id="test-user")
        
        # Get system health
        health_status = dashboard.get_system_health(force_refresh=True)
        
        # Verify health status structure
        assert hasattr(health_status, 'overall_status')
        assert hasattr(health_status, 'memory_condensation_health')
        assert hasattr(health_status, 'active_alerts')
        assert hasattr(health_status, 'recommendations')
        
        # Get metrics summary
        metrics_summary = dashboard.get_metrics_summary(5)
        
        # Verify metrics summary structure
        assert "window_minutes" in metrics_summary
        assert "derived_metrics" in metrics_summary
        assert "trends" in metrics_summary
        
        # Test data export
        json_export = dashboard.export_dashboard_data("json")
        assert isinstance(json_export, str)
        assert "system_health" in json_export
        assert "metrics_summary" in json_export
        
        # Test Prometheus export
        prometheus_export = dashboard.export_dashboard_data("prometheus")
        assert isinstance(prometheus_export, str)
        assert "memory_system_health" in prometheus_export
        assert "memory_active_alerts" in prometheus_export
    
    def test_correlation_id_tracking(self):
        """Test correlation ID tracking across operations."""
        
        correlation_id = "test-correlation-12345"
        
        # Perform multiple operations with the same correlation ID
        with monitor_memory_operation("test_op_1", correlation_id=correlation_id, user_id="test"):
            pass
        
        with monitor_memory_operation("test_op_2", correlation_id=correlation_id, user_id="test"):
            pass
        
        record_memory_metric("test_op_3", True, 0.5, correlation_id=correlation_id, user_id="test")
        
        # Verify metrics were recorded with correlation ID
        metrics_collector = get_metrics_collector()
        
        # Check that metrics contain the correlation ID
        for metric_name in ["test_op_1_attempts", "test_op_2_attempts", "test_op_3_attempts"]:
            if metric_name in metrics_collector._metrics:
                metric_points = metrics_collector._metrics[metric_name]
                if metric_points:
                    assert metric_points[-1].correlation_id == correlation_id
    
    def test_performance_monitoring_and_bottleneck_detection(self):
        """Test performance monitoring and bottleneck detection."""
        
        # Simulate slow operations
        slow_operations = [
            ("slow_condensation", 2.0),
            ("slow_storage", 1.5),
            ("slow_retrieval", 1.0)
        ]
        
        for op_name, duration in slow_operations:
            record_memory_metric(op_name, True, duration, user_id="test-user")
        
        # Simulate fast operations
        fast_operations = [
            ("fast_condensation", 0.1),
            ("fast_storage", 0.05),
            ("fast_retrieval", 0.02)
        ]
        
        for op_name, duration in fast_operations:
            record_memory_metric(op_name, True, duration, user_id="test-user")
        
        # Get performance dashboard
        dashboard = get_monitoring_dashboard()
        performance_data = dashboard.get_performance_dashboard(5)
        
        # Verify performance data structure
        assert "performance_by_operation" in performance_data
        assert "performance_trends" in performance_data
        assert "bottlenecks" in performance_data
        assert "recommendations" in performance_data
        
        # Verify bottleneck detection
        bottlenecks = performance_data["bottlenecks"]
        assert isinstance(bottlenecks, list)
    
    def test_alert_escalation_and_suppression(self):
        """Test alert escalation and suppression functionality."""
        
        alerting_system = get_alerting_system()
        
        # Find a critical alert rule
        critical_rule_id = None
        for rule_id, rule in alerting_system._alert_rules.items():
            if rule.severity == AlertSeverity.CRITICAL:
                critical_rule_id = rule_id
                break
        
        assert critical_rule_id is not None
        
        # Test alert suppression
        result = alerting_system.suppress_alert(critical_rule_id, 30)  # 30 minutes
        assert result is True
        
        # Verify alert is suppressed
        rule = alerting_system._alert_rules[critical_rule_id]
        assert rule.state.name == "SUPPRESSED"
        
        # Test alert acknowledgment (first set to active)
        rule.state = rule.state.__class__.ACTIVE
        result = alerting_system.acknowledge_alert(critical_rule_id)
        assert result is True
        assert rule.state.name == "ACKNOWLEDGED"
    
    def test_metrics_cleanup_and_maintenance(self):
        """Test metrics cleanup and maintenance functionality."""
        
        metrics_collector = get_metrics_collector()
        
        # Record some metrics
        for i in range(10):
            metrics_collector.record_metric(f"test_metric_{i}", i * 10)
        
        # Verify metrics were recorded
        initial_metric_count = len(metrics_collector._metrics)
        assert initial_metric_count >= 10
        
        # Perform cleanup (this should not remove recent metrics)
        cleared_count = metrics_collector.clear_old_metrics(24)  # 24 hours
        
        # Verify recent metrics were not cleared
        current_metric_count = len(metrics_collector._metrics)
        assert current_metric_count == initial_metric_count  # No recent metrics should be cleared
        
        # Test export functionality
        json_export = metrics_collector.export_metrics("json")
        assert isinstance(json_export, str)
        assert "metrics" in json_export
        
        prometheus_export = metrics_collector.export_metrics("prometheus")
        assert isinstance(prometheus_export, str)
        assert "# HELP" in prometheus_export
        assert "# TYPE" in prometheus_export


if __name__ == "__main__":
    pytest.main([__file__, "-v"])