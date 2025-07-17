"""
Integration module for comprehensive memory operation monitoring and observability.
Provides unified interface for metrics, alerting, and monitoring dashboard.
"""

import time
import threading
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

from .logging_utils import get_logger
from .metrics_collector import get_metrics_collector, MetricType, AlertSeverity
from .alerting_system import get_alerting_system, start_alerting, stop_alerting
from .monitoring_dashboard import get_monitoring_dashboard, get_system_health
from .memory_monitoring import create_memory_monitor
from .error_context import create_correlation_id, ErrorSeverity


logger = get_logger(__name__)


class MemoryMonitoringIntegration:
    """
    Unified integration for memory operation monitoring and observability.
    
    Features:
    - Centralized monitoring initialization
    - Automated alert evaluation
    - Health check scheduling
    - Metrics cleanup and maintenance
    - Dashboard data export
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_collector = get_metrics_collector()
        self.alerting_system = get_alerting_system()
        self.monitoring_dashboard = get_monitoring_dashboard()
        
        # Background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitoring_interval = 30  # seconds
        
        # Health check scheduling
        self._health_check_interval = 60  # seconds
        self._last_health_check = 0
        
        # Metrics cleanup scheduling
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = 0
        
        self._initialized = False
    
    def initialize(self, start_background_monitoring: bool = True) -> None:
        """
        Initialize the monitoring integration.
        
        Args:
            start_background_monitoring: Whether to start background monitoring
        """
        if self._initialized:
            self.logger.warning("Monitoring integration already initialized")
            return
        
        try:
            # Start alerting system
            start_alerting()
            
            # Setup custom alert rules if needed
            self._setup_custom_alerts()
            
            # Start background monitoring if requested
            if start_background_monitoring:
                self.start_background_monitoring()
            
            self._initialized = True
            
            self.logger.info(
                "Memory monitoring integration initialized successfully",
                background_monitoring=start_background_monitoring,
                alert_rules=len(self.alerting_system._alert_rules),
                monitoring_interval=self._monitoring_interval
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize monitoring integration",
                error=str(e)
            )
            raise
    
    def shutdown(self) -> None:
        """Shutdown the monitoring integration."""
        if not self._initialized:
            return
        
        try:
            # Stop background monitoring
            self.stop_background_monitoring()
            
            # Stop alerting system
            stop_alerting()
            
            # Perform final cleanup
            self._perform_cleanup()
            
            self._initialized = False
            
            self.logger.info("Memory monitoring integration shutdown completed")
            
        except Exception as e:
            self.logger.error(
                "Error during monitoring integration shutdown",
                error=str(e)
            )
    
    def start_background_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Background monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            daemon=True,
            name="MemoryMonitoring"
        )
        self._monitoring_thread.start()
        
        self.logger.info("Background monitoring started")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=10.0)
        
        if self._monitoring_thread.is_alive():
            self.logger.warning("Background monitoring thread did not stop gracefully")
        else:
            self.logger.info("Background monitoring stopped")
    
    def record_memory_operation(
        self,
        operation_name: str,
        success: bool,
        duration_seconds: Optional[float] = None,
        **context: Any
    ) -> None:
        """
        Record a memory operation for monitoring.
        
        Args:
            operation_name: Name of the operation
            success: Whether the operation was successful
            duration_seconds: Operation duration in seconds
            **context: Additional context information
        """
        correlation_id = context.get("correlation_id") or create_correlation_id()
        
        # Record basic metrics
        self.metrics_collector.increment_counter(
            f"{operation_name}_attempts",
            tags=self._extract_tags(context),
            correlation_id=correlation_id
        )
        
        if success:
            self.metrics_collector.increment_counter(
                f"{operation_name}_successes",
                tags=self._extract_tags(context),
                correlation_id=correlation_id
            )
        else:
            self.metrics_collector.increment_counter(
                f"{operation_name}_failures",
                tags=self._extract_tags(context),
                correlation_id=correlation_id
            )
        
        # Record duration if provided
        if duration_seconds is not None:
            self.metrics_collector.record_histogram(
                f"{operation_name}_duration",
                duration_seconds,
                tags=self._extract_tags(context),
                correlation_id=correlation_id
            )
        
        # Log the operation (avoid duplicate correlation_id)
        log_context = {k: v for k, v in context.items() if k != "correlation_id"}
        self.logger.info(
            f"Memory operation recorded: {operation_name}",
            operation_name=operation_name,
            success=success,
            duration_seconds=duration_seconds,
            correlation_id=correlation_id,
            **log_context
        )
    
    @contextmanager
    def monitor_operation(
        self,
        operation_name: str,
        correlation_id: Optional[str] = None,
        **context: Any
    ):
        """
        Context manager for monitoring memory operations.
        
        Args:
            operation_name: Name of the operation
            correlation_id: Optional correlation ID
            **context: Additional context information
            
        Usage:
            with monitor.monitor_operation("memory_condensation", user_id="123"):
                # Perform memory operation
                pass
        """
        correlation_id = correlation_id or create_correlation_id()
        start_time = time.time()
        success = True
        
        try:
            self.logger.debug(
                f"Starting monitored operation: {operation_name}",
                operation_name=operation_name,
                correlation_id=correlation_id,
                **context
            )
            
            yield {
                "correlation_id": correlation_id,
                "operation_name": operation_name,
                "start_time": start_time
            }
            
        except Exception as e:
            success = False
            
            # Record error metrics
            self.metrics_collector.increment_counter(
                "memory_operation_errors",
                tags={
                    "operation": operation_name,
                    "error_type": type(e).__name__,
                    **self._extract_tags(context)
                },
                correlation_id=correlation_id
            )
            
            self.logger.error(
                f"Error in monitored operation: {operation_name}",
                operation_name=operation_name,
                error=str(e),
                correlation_id=correlation_id,
                **context
            )
            
            raise
            
        finally:
            duration = time.time() - start_time
            
            # Record the operation
            self.record_memory_operation(
                operation_name,
                success,
                duration,
                correlation_id=correlation_id,
                **context
            )
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        health_status = get_system_health()
        
        return {
            "timestamp": time.time(),
            "system_health": health_status.to_dict(),
            "metrics_summary": self.monitoring_dashboard.get_metrics_summary(60),
            "active_alerts": self.alerting_system.get_active_alerts(),
            "recent_errors": self.monitoring_dashboard.get_error_dashboard(1)["recent_errors"][:10],
            "recommendations": health_status.recommendations
        }
    
    def export_monitoring_data(
        self,
        format_type: str = "json",
        include_detailed_metrics: bool = False
    ) -> str:
        """
        Export comprehensive monitoring data.
        
        Args:
            format_type: Export format ("json", "prometheus")
            include_detailed_metrics: Whether to include detailed metrics
            
        Returns:
            Formatted monitoring data
        """
        if include_detailed_metrics:
            return self.monitoring_dashboard.export_dashboard_data(format_type)
        else:
            # Export basic health and alert data
            basic_data = {
                "timestamp": time.time(),
                "system_health": get_system_health().to_dict(),
                "active_alerts": self.alerting_system.get_active_alerts(),
                "alert_statistics": self.alerting_system.get_alert_statistics(1)
            }
            
            if format_type.lower() == "json":
                import json
                return json.dumps(basic_data, indent=2, default=str)
            else:
                # Basic Prometheus export
                lines = []
                health = basic_data["system_health"]
                
                lines.append(f"# HELP memory_system_health System health status")
                lines.append(f"# TYPE memory_system_health gauge")
                health_value = 1 if health["overall_status"] == "healthy" else 0
                lines.append(f"memory_system_health {health_value}")
                
                lines.append(f"# HELP memory_active_alerts Active alerts count")
                lines.append(f"# TYPE memory_active_alerts gauge")
                lines.append(f"memory_active_alerts {health['active_alerts']}")
                
                return "\n".join(lines)
    
    def trigger_health_check(self) -> Dict[str, Any]:
        """
        Trigger an immediate health check.
        
        Returns:
            Health check results
        """
        self.logger.info("Triggering manual health check")
        
        # Force refresh of health status
        health_status = self.monitoring_dashboard.get_system_health(force_refresh=True)
        
        # Evaluate alerts
        triggered_alerts = self.alerting_system.evaluate_alerts()
        
        # Perform cleanup if needed
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._perform_cleanup()
        
        return {
            "health_status": health_status.to_dict(),
            "triggered_alerts": len(triggered_alerts),
            "cleanup_performed": current_time - self._last_cleanup > self._cleanup_interval
        }
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop."""
        self.logger.info("Background monitoring loop started")
        
        while not self._stop_monitoring.is_set():
            try:
                current_time = time.time()
                
                # Evaluate alerts
                triggered_alerts = self.alerting_system.evaluate_alerts()
                if triggered_alerts:
                    self.logger.info(
                        f"Alert evaluation completed",
                        triggered_alerts=len(triggered_alerts)
                    )
                
                # Periodic health check
                if current_time - self._last_health_check > self._health_check_interval:
                    health_status = self.monitoring_dashboard.get_system_health()
                    self._last_health_check = current_time
                    
                    if health_status.overall_status != "healthy":
                        self.logger.warning(
                            f"System health check: {health_status.overall_status}",
                            active_alerts=health_status.active_alerts,
                            error_rate=health_status.error_rate_5min,
                            success_rate=health_status.success_rate_5min
                        )
                
                # Periodic cleanup
                if current_time - self._last_cleanup > self._cleanup_interval:
                    self._perform_cleanup()
                
                # Wait for next iteration
                self._stop_monitoring.wait(self._monitoring_interval)
                
            except Exception as e:
                self.logger.error(
                    "Error in background monitoring loop",
                    error=str(e)
                )
                # Continue monitoring despite errors
                self._stop_monitoring.wait(self._monitoring_interval)
        
        self.logger.info("Background monitoring loop stopped")
    
    def _perform_cleanup(self) -> None:
        """Perform periodic cleanup of old metrics and data."""
        try:
            # Clean old metrics (keep 24 hours)
            cleared_metrics = self.metrics_collector.clear_old_metrics(24)
            
            self._last_cleanup = time.time()
            
            if cleared_metrics > 0:
                self.logger.info(
                    "Periodic cleanup completed",
                    cleared_metrics=cleared_metrics
                )
            
        except Exception as e:
            self.logger.error(
                "Error during periodic cleanup",
                error=str(e)
            )
    
    def _setup_custom_alerts(self) -> None:
        """Setup custom alert rules specific to the application."""
        # Add any custom alert rules here
        # The default rules are already set up in the alerting system
        pass
    
    def _extract_tags(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Extract tags from context for metrics."""
        tags = {}
        
        # Extract common tags
        for key in ["user_id", "session_id", "component", "operation_type"]:
            if key in context:
                tags[key] = str(context[key])
        
        return tags


# Global integration instance
_monitoring_integration = MemoryMonitoringIntegration()


def get_monitoring_integration() -> MemoryMonitoringIntegration:
    """Get the global monitoring integration instance."""
    return _monitoring_integration


def initialize_monitoring(start_background: bool = True) -> None:
    """Initialize the monitoring integration."""
    _monitoring_integration.initialize(start_background)


def shutdown_monitoring() -> None:
    """Shutdown the monitoring integration."""
    _monitoring_integration.shutdown()


@contextmanager
def monitor_memory_operation(
    operation_name: str,
    correlation_id: Optional[str] = None,
    **context: Any
):
    """Convenience context manager for monitoring memory operations."""
    with _monitoring_integration.monitor_operation(operation_name, correlation_id, **context) as op_context:
        yield op_context


def record_memory_metric(
    operation_name: str,
    success: bool,
    duration_seconds: Optional[float] = None,
    **context: Any
) -> None:
    """Convenience function to record memory operation metrics."""
    _monitoring_integration.record_memory_operation(operation_name, success, duration_seconds, **context)


def get_health_summary() -> Dict[str, Any]:
    """Get a summary of system health."""
    return _monitoring_integration.get_health_report()


def export_metrics(format_type: str = "json") -> str:
    """Export monitoring metrics."""
    return _monitoring_integration.export_monitoring_data(format_type)