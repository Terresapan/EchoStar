"""
Comprehensive metrics collection system for memory operations.
Provides structured metrics collection, aggregation, and alerting capabilities.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from threading import Lock

from .logging_utils import get_logger
from .error_context import ErrorSeverity, OperationType


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "correlation_id": self.correlation_id
        }


@dataclass
class Alert:
    """Alert definition and state."""
    
    alert_id: str
    name: str
    condition: str
    severity: AlertSeverity
    threshold: Union[int, float]
    window_minutes: int = 5
    triggered: bool = False
    trigger_time: Optional[float] = None
    trigger_count: int = 0
    last_evaluation: Optional[float] = None
    message_template: str = "Alert {name} triggered: {condition}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MemoryMetricsCollector:
    """
    Comprehensive metrics collector for memory operations.
    
    Features:
    - Real-time metrics collection
    - Metric aggregation and windowing
    - Alert evaluation and triggering
    - Performance monitoring
    - Export capabilities
    """
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.logger = get_logger(__name__)
        self.max_points_per_metric = max_points_per_metric
        self._lock = Lock()
        
        # Metric storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Alert system
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._operation_timers: Dict[str, float] = {}
        
        # Initialize default alerts
        self._setup_default_alerts()
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Record a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for categorization
            correlation_id: Optional correlation ID for tracking
        """
        with self._lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                metric_type=metric_type,
                tags=tags or {},
                correlation_id=correlation_id
            )
            
            self._metrics[name].append(metric_point)
            
            # Update metadata
            if name not in self._metric_metadata:
                self._metric_metadata[name] = {
                    "type": metric_type.value,
                    "first_seen": metric_point.timestamp,
                    "count": 0
                }
            
            self._metric_metadata[name]["count"] += 1
            self._metric_metadata[name]["last_seen"] = metric_point.timestamp
            
            # Log metric for debugging
            self.logger.debug(
                f"Recorded metric: {name}",
                metric_name=name,
                value=value,
                type=metric_type.value,
                tags=tags,
                correlation_id=correlation_id
            )
    
    def increment_counter(
        self,
        name: str,
        value: Union[int, float] = 1,
        tags: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags, correlation_id)
    
    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags, correlation_id)
    
    def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Record a histogram metric."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags, correlation_id)
    
    def start_timer(self, name: str, correlation_id: Optional[str] = None) -> str:
        """
        Start a timer for an operation.
        
        Args:
            name: Timer name
            correlation_id: Optional correlation ID
            
        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{name}_{correlation_id or int(time.time() * 1000)}"
        self._operation_timers[timer_id] = time.time()
        
        self.logger.debug(
            f"Started timer: {name}",
            timer_name=name,
            timer_id=timer_id,
            correlation_id=correlation_id
        )
        
        return timer_id
    
    def stop_timer(
        self,
        timer_id: str,
        tags: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[float]:
        """
        Stop a timer and record the duration.
        
        Args:
            timer_id: Timer ID from start_timer
            tags: Optional tags
            correlation_id: Optional correlation ID
            
        Returns:
            Duration in seconds, or None if timer not found
        """
        if timer_id not in self._operation_timers:
            self.logger.warning(f"Timer not found: {timer_id}")
            return None
        
        start_time = self._operation_timers.pop(timer_id)
        duration = time.time() - start_time
        
        # Extract metric name from timer_id (remove correlation suffix)
        base_name = timer_id.rsplit('_', 1)[0]  # Remove the last part (correlation ID)
        metric_name = base_name + "_duration"
        
        self.record_metric(
            metric_name,
            duration,
            MetricType.TIMER,
            tags,
            correlation_id
        )
        
        self.logger.debug(
            f"Stopped timer: {timer_id}",
            timer_id=timer_id,
            duration_seconds=duration,
            correlation_id=correlation_id
        )
        
        return duration
    
    def get_metric_summary(
        self,
        name: str,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a metric within a time window.
        
        Args:
            name: Metric name
            window_minutes: Time window in minutes
            
        Returns:
            Summary statistics
        """
        if name not in self._metrics:
            return {"error": f"Metric {name} not found"}
        
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_points = [
                point for point in self._metrics[name]
                if point.timestamp > cutoff_time
            ]
        
        if not recent_points:
            return {
                "name": name,
                "window_minutes": window_minutes,
                "count": 0,
                "values": []
            }
        
        values = [point.value for point in recent_points]
        
        summary = {
            "name": name,
            "window_minutes": window_minutes,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "first_timestamp": min(point.timestamp for point in recent_points),
            "last_timestamp": max(point.timestamp for point in recent_points)
        }
        
        # Add percentiles for histograms and timers
        if recent_points[0].metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            summary.update({
                "p50": sorted_values[int(count * 0.5)],
                "p90": sorted_values[int(count * 0.9)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0]
            })
        
        return summary
    
    def add_alert(
        self,
        alert_id: str,
        name: str,
        condition: str,
        severity: AlertSeverity,
        threshold: Union[int, float],
        window_minutes: int = 5,
        message_template: Optional[str] = None
    ) -> None:
        """
        Add a new alert definition.
        
        Args:
            alert_id: Unique alert identifier
            name: Human-readable alert name
            condition: Alert condition description
            severity: Alert severity level
            threshold: Threshold value for triggering
            window_minutes: Time window for evaluation
            message_template: Optional custom message template
        """
        alert = Alert(
            alert_id=alert_id,
            name=name,
            condition=condition,
            severity=severity,
            threshold=threshold,
            window_minutes=window_minutes,
            message_template=message_template or f"Alert {name} triggered: {condition}"
        )
        
        self._alerts[alert_id] = alert
        
        self.logger.info(
            f"Added alert: {name}",
            alert_id=alert_id,
            condition=condition,
            severity=severity.value,
            threshold=threshold
        )
    
    def evaluate_alerts(self) -> List[Dict[str, Any]]:
        """
        Evaluate all alerts and return triggered alerts.
        
        Returns:
            List of triggered alert information
        """
        triggered_alerts = []
        current_time = time.time()
        
        for alert_id, alert in self._alerts.items():
            try:
                # Evaluate alert condition
                is_triggered = self._evaluate_alert_condition(alert)
                
                if is_triggered and not alert.triggered:
                    # Alert newly triggered
                    alert.triggered = True
                    alert.trigger_time = current_time
                    alert.trigger_count += 1
                    
                    alert_info = {
                        "alert_id": alert_id,
                        "name": alert.name,
                        "condition": alert.condition,
                        "severity": alert.severity.value,
                        "threshold": alert.threshold,
                        "trigger_time": current_time,
                        "trigger_count": alert.trigger_count,
                        "message": alert.message_template.format(
                            name=alert.name,
                            condition=alert.condition,
                            threshold=alert.threshold
                        )
                    }
                    
                    triggered_alerts.append(alert_info)
                    self._alert_history.append(alert_info)
                    
                    # Log alert
                    self.logger.error(
                        f"ALERT TRIGGERED: {alert.name}",
                        alert_id=alert_id,
                        condition=alert.condition,
                        severity=alert.severity.value,
                        threshold=alert.threshold,
                        trigger_count=alert.trigger_count
                    )
                
                elif not is_triggered and alert.triggered:
                    # Alert resolved
                    alert.triggered = False
                    alert.trigger_time = None
                    
                    self.logger.info(
                        f"ALERT RESOLVED: {alert.name}",
                        alert_id=alert_id,
                        condition=alert.condition
                    )
                
                alert.last_evaluation = current_time
                
            except Exception as e:
                self.logger.error(
                    f"Error evaluating alert {alert_id}",
                    error=str(e),
                    alert_name=alert.name
                )
        
        return triggered_alerts
    
    def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """
        Evaluate a specific alert condition.
        
        Args:
            alert: Alert to evaluate
            
        Returns:
            True if alert should be triggered
        """
        # Parse condition to extract metric name and comparison
        # Format: "metric_name > threshold" or "metric_name < threshold" etc.
        condition_parts = alert.condition.split()
        if len(condition_parts) < 3:
            return False
        
        metric_name = condition_parts[0]
        operator = condition_parts[1]
        
        # Get metric summary
        summary = self.get_metric_summary(metric_name, alert.window_minutes)
        
        if summary.get("count", 0) == 0:
            return False
        
        # Use appropriate value based on operator context
        if "rate" in alert.condition.lower() or "count" in alert.condition.lower():
            current_value = summary.get("sum", 0)
        elif "avg" in alert.condition.lower():
            current_value = summary.get("avg", 0)
        elif "max" in alert.condition.lower():
            current_value = summary.get("max", 0)
        else:
            current_value = summary.get("avg", 0)  # Default to average
        
        # Evaluate condition
        if operator == ">":
            return current_value > alert.threshold
        elif operator == "<":
            return current_value < alert.threshold
        elif operator == ">=":
            return current_value >= alert.threshold
        elif operator == "<=":
            return current_value <= alert.threshold
        elif operator == "==":
            return current_value == alert.threshold
        elif operator == "!=":
            return current_value != alert.threshold
        
        return False
    
    def _setup_default_alerts(self) -> None:
        """Set up default alerts for memory operations."""
        
        # Memory condensation failure rate
        self.add_alert(
            alert_id="memory_condensation_failure_rate",
            name="Memory Condensation Failure Rate",
            condition="memory_condensation_failures > 3",
            severity=AlertSeverity.CRITICAL,
            threshold=3,
            window_minutes=10,
            message_template="Memory condensation failing frequently: {threshold} failures in {window_minutes} minutes"
        )
        
        # Memory storage failure rate
        self.add_alert(
            alert_id="memory_storage_failure_rate",
            name="Memory Storage Failure Rate",
            condition="memory_storage_failures > 5",
            severity=AlertSeverity.WARNING,
            threshold=5,
            window_minutes=15,
            message_template="Memory storage operations failing: {threshold} failures in {window_minutes} minutes"
        )
        
        # Memory condensation duration
        self.add_alert(
            alert_id="memory_condensation_slow",
            name="Memory Condensation Slow",
            condition="memory_condensation_duration_avg > 30",
            severity=AlertSeverity.WARNING,
            threshold=30.0,
            window_minutes=5,
            message_template="Memory condensation taking too long: average {threshold}s in {window_minutes} minutes"
        )
        
        # Memory retrieval failure rate
        self.add_alert(
            alert_id="memory_retrieval_failure_rate",
            name="Memory Retrieval Failure Rate",
            condition="memory_retrieval_failures > 10",
            severity=AlertSeverity.CRITICAL,
            threshold=10,
            window_minutes=5,
            message_template="Memory retrieval failing frequently: {threshold} failures in {window_minutes} minutes"
        )
        
        # Profile storage failure rate
        self.add_alert(
            alert_id="profile_storage_failure_rate",
            name="Profile Storage Failure Rate",
            condition="profile_storage_failures > 5",
            severity=AlertSeverity.WARNING,
            threshold=5,
            window_minutes=10,
            message_template="Profile storage operations failing: {threshold} failures in {window_minutes} minutes"
        )
    
    def get_all_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Summary of all metrics
        """
        summary = {
            "window_minutes": window_minutes,
            "timestamp": time.time(),
            "metrics": {},
            "total_metrics": len(self._metrics),
            "active_alerts": len([a for a in self._alerts.values() if a.triggered])
        }
        
        for metric_name in self._metrics.keys():
            summary["metrics"][metric_name] = self.get_metric_summary(metric_name, window_minutes)
        
        return summary
    
    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export metrics in specified format.
        
        Args:
            format_type: Export format ("json", "prometheus")
            
        Returns:
            Formatted metrics string
        """
        if format_type.lower() == "json":
            return json.dumps(self.get_all_metrics_summary(), indent=2, default=str)
        elif format_type.lower() == "prometheus":
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, points in self._metrics.items():
            if not points:
                continue
            
            latest_point = points[-1]
            
            # Add metric help
            lines.append(f"# HELP {metric_name} Memory operation metric")
            lines.append(f"# TYPE {metric_name} {latest_point.metric_type.value}")
            
            # Add metric value with tags
            tags_str = ""
            if latest_point.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in latest_point.tags.items()]
                tags_str = "{" + ",".join(tag_pairs) + "}"
            
            lines.append(f"{metric_name}{tags_str} {latest_point.value} {int(latest_point.timestamp * 1000)}")
        
        return "\n".join(lines)
    
    def clear_old_metrics(self, max_age_hours: int = 24) -> int:
        """
        Clear metrics older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of metrics cleared
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleared_count = 0
        
        with self._lock:
            # Create a list of metric names to avoid modifying dict during iteration
            metric_names = list(self._metrics.keys())
            
            for metric_name in metric_names:
                points = self._metrics[metric_name]
                original_count = len(points)
                
                # Filter out old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                    cleared_count += 1
                
                # Update metadata if points were removed
                if len(points) < original_count:
                    if points:
                        if metric_name in self._metric_metadata:
                            self._metric_metadata[metric_name]["first_seen"] = points[0].timestamp
                    else:
                        # Remove empty metrics completely
                        if metric_name in self._metric_metadata:
                            del self._metric_metadata[metric_name]
                        del self._metrics[metric_name]
        
        if cleared_count > 0:
            self.logger.info(
                f"Cleared {cleared_count} old metric points",
                max_age_hours=max_age_hours,
                cutoff_time=cutoff_time
            )
        
        return cleared_count


# Global metrics collector instance
_metrics_collector = MemoryMetricsCollector()


def get_metrics_collector() -> MemoryMetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def record_memory_metric(
    name: str,
    value: Union[int, float],
    metric_type: MetricType = MetricType.GAUGE,
    tags: Optional[Dict[str, str]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Convenience function to record a memory metric."""
    _metrics_collector.record_metric(name, value, metric_type, tags, correlation_id)


def increment_memory_counter(
    name: str,
    value: Union[int, float] = 1,
    tags: Optional[Dict[str, str]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Convenience function to increment a memory counter."""
    _metrics_collector.increment_counter(name, value, tags, correlation_id)


def set_memory_gauge(
    name: str,
    value: Union[int, float],
    tags: Optional[Dict[str, str]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Convenience function to set a memory gauge."""
    _metrics_collector.set_gauge(name, value, tags, correlation_id)


def start_memory_timer(name: str, correlation_id: Optional[str] = None) -> str:
    """Convenience function to start a memory timer."""
    return _metrics_collector.start_timer(name, correlation_id)


def stop_memory_timer(
    timer_id: str,
    tags: Optional[Dict[str, str]] = None,
    correlation_id: Optional[str] = None
) -> Optional[float]:
    """Convenience function to stop a memory timer."""
    return _metrics_collector.stop_timer(timer_id, tags, correlation_id)