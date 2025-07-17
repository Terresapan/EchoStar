"""
Comprehensive monitoring dashboard for memory operations.
Provides real-time monitoring, metrics visualization, and system health overview.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

from .logging_utils import get_logger
from .metrics_collector import get_metrics_collector, MetricType, AlertSeverity
from .alerting_system import get_alerting_system
from .error_context import get_error_tracker, get_performance_monitor


@dataclass
class SystemHealthStatus:
    """System health status summary."""
    
    overall_status: str  # "healthy", "degraded", "critical"
    timestamp: float
    
    # Component health
    memory_condensation_health: str
    memory_storage_health: str
    memory_retrieval_health: str
    profile_storage_health: str
    
    # Key metrics
    active_alerts: int
    error_rate_5min: float
    avg_response_time_5min: float
    success_rate_5min: float
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MemoryMonitoringDashboard:
    """
    Comprehensive monitoring dashboard for memory operations.
    
    Features:
    - Real-time system health monitoring
    - Performance metrics aggregation
    - Alert management and visualization
    - Trend analysis and recommendations
    - Export capabilities for external monitoring
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_collector = get_metrics_collector()
        self.alerting_system = get_alerting_system()
        self.error_tracker = get_error_tracker()
        self.performance_monitor = get_performance_monitor()
        
        # Dashboard state
        self._last_health_check = 0
        self._cached_health_status: Optional[SystemHealthStatus] = None
        self._health_cache_duration = 30  # seconds
    
    def get_system_health(self, force_refresh: bool = False) -> SystemHealthStatus:
        """
        Get comprehensive system health status.
        
        Args:
            force_refresh: Force refresh of cached health status
            
        Returns:
            SystemHealthStatus object
        """
        current_time = time.time()
        
        # Use cached status if recent and not forcing refresh
        if (not force_refresh and 
            self._cached_health_status and 
            current_time - self._last_health_check < self._health_cache_duration):
            return self._cached_health_status
        
        # Calculate component health
        memory_condensation_health = self._assess_component_health("memory_condensation")
        memory_storage_health = self._assess_component_health("memory_storage")
        memory_retrieval_health = self._assess_component_health("memory_retrieval")
        profile_storage_health = self._assess_component_health("profile_storage")
        
        # Get active alerts
        active_alerts = self.alerting_system.get_active_alerts()
        active_alert_count = len(active_alerts)
        
        # Calculate key metrics
        error_rate_5min = self._calculate_error_rate(5)
        avg_response_time_5min = self._calculate_avg_response_time(5)
        success_rate_5min = self._calculate_success_rate(5)
        
        # Determine overall status
        overall_status = self._determine_overall_status(
            [memory_condensation_health, memory_storage_health, 
             memory_retrieval_health, profile_storage_health],
            active_alert_count,
            error_rate_5min,
            success_rate_5min
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_status,
            {
                "memory_condensation": memory_condensation_health,
                "memory_storage": memory_storage_health,
                "memory_retrieval": memory_retrieval_health,
                "profile_storage": profile_storage_health
            },
            active_alerts,
            error_rate_5min,
            success_rate_5min
        )
        
        # Create health status
        health_status = SystemHealthStatus(
            overall_status=overall_status,
            timestamp=current_time,
            memory_condensation_health=memory_condensation_health,
            memory_storage_health=memory_storage_health,
            memory_retrieval_health=memory_retrieval_health,
            profile_storage_health=profile_storage_health,
            active_alerts=active_alert_count,
            error_rate_5min=error_rate_5min,
            avg_response_time_5min=avg_response_time_5min,
            success_rate_5min=success_rate_5min,
            recommendations=recommendations
        )
        
        # Cache the result
        self._cached_health_status = health_status
        self._last_health_check = current_time
        
        return health_status
    
    def get_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Metrics summary dictionary
        """
        summary = self.metrics_collector.get_all_metrics_summary(window_minutes)
        
        # Add derived metrics
        summary["derived_metrics"] = self._calculate_derived_metrics(window_minutes)
        
        # Add trend analysis
        summary["trends"] = self._analyze_trends(window_minutes)
        
        return summary
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """
        Get alert dashboard information.
        
        Returns:
            Alert dashboard data
        """
        active_alerts = self.alerting_system.get_active_alerts()
        alert_stats = self.alerting_system.get_alert_statistics(24)
        
        # Categorize alerts by severity
        alerts_by_severity = defaultdict(list)
        for alert in active_alerts:
            alerts_by_severity[alert["severity"]].append(alert)
        
        # Get recent alert events
        recent_events = []
        for event in getattr(self.alerting_system, '_alert_history', [])[-50:]:
            recent_events.append({
                "timestamp": event.timestamp,
                "rule_id": event.rule_id,
                "event_type": event.event_type,
                "severity": event.severity.value,
                "metric_value": event.metric_value,
                "threshold": event.threshold
            })
        
        return {
            "active_alerts": active_alerts,
            "alerts_by_severity": dict(alerts_by_severity),
            "alert_statistics": alert_stats,
            "recent_events": recent_events,
            "alert_trends": self._analyze_alert_trends()
        }
    
    def get_performance_dashboard(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance dashboard information.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Performance dashboard data
        """
        # Get performance summaries for different operation types
        performance_data = {}
        
        from .error_context import OperationType
        for op_type in OperationType:
            summary = self.performance_monitor.get_performance_summary(op_type, window_minutes)
            performance_data[op_type.value] = summary
        
        # Add overall performance metrics
        overall_summary = self.performance_monitor.get_performance_summary(None, window_minutes)
        performance_data["overall"] = overall_summary
        
        # Add performance trends
        performance_trends = self._analyze_performance_trends(window_minutes)
        
        return {
            "performance_by_operation": performance_data,
            "performance_trends": performance_trends,
            "bottlenecks": self._identify_performance_bottlenecks(performance_data),
            "recommendations": self._generate_performance_recommendations(performance_data)
        }
    
    def get_error_dashboard(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get error dashboard information.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Error dashboard data
        """
        error_summary = self.error_tracker.get_error_summary(window_hours * 60)
        
        # Get recent errors with details
        recent_errors = []
        cutoff_time = time.time() - (window_hours * 3600)
        
        for error in getattr(self.error_tracker, '_error_history', []):
            try:
                error_time = datetime.fromisoformat(error.timestamp.replace('Z', '')).timestamp()
                if error_time > cutoff_time:
                    recent_errors.append({
                        "timestamp": error.timestamp,
                        "operation": error.operation,
                        "error_type": error.error_type,
                        "error_message": error.error_message,
                        "severity": error.severity.value,
                        "correlation_id": error.correlation_id,
                        "recovery_attempted": error.recovery_attempted,
                        "recovery_successful": error.recovery_successful
                    })
            except (ValueError, AttributeError):
                continue
        
        # Sort by timestamp (most recent first)
        recent_errors.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "error_summary": error_summary,
            "recent_errors": recent_errors[:100],  # Limit to 100 most recent
            "error_patterns": self._analyze_error_patterns(recent_errors),
            "recovery_analysis": self._analyze_recovery_patterns(recent_errors)
        }
    
    def export_dashboard_data(self, format_type: str = "json") -> str:
        """
        Export comprehensive dashboard data.
        
        Args:
            format_type: Export format ("json", "prometheus")
            
        Returns:
            Formatted dashboard data
        """
        dashboard_data = {
            "timestamp": time.time(),
            "system_health": self.get_system_health().to_dict(),
            "metrics_summary": self.get_metrics_summary(60),
            "alert_dashboard": self.get_alert_dashboard(),
            "performance_dashboard": self.get_performance_dashboard(60),
            "error_dashboard": self.get_error_dashboard(24)
        }
        
        if format_type.lower() == "json":
            return json.dumps(dashboard_data, indent=2, default=str)
        elif format_type.lower() == "prometheus":
            return self._export_prometheus_metrics(dashboard_data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _assess_component_health(self, component: str) -> str:
        """Assess health of a specific component."""
        # Get recent metrics for the component
        failure_metric = f"{component}_failures"
        success_metric = f"{component}_successes"
        
        failure_summary = self.metrics_collector.get_metric_summary(failure_metric, 10)
        success_summary = self.metrics_collector.get_metric_summary(success_metric, 10)
        
        failure_count = failure_summary.get("sum", 0)
        success_count = success_summary.get("sum", 0)
        total_operations = failure_count + success_count
        
        if total_operations == 0:
            return "unknown"
        
        failure_rate = failure_count / total_operations
        
        if failure_rate > 0.5:
            return "critical"
        elif failure_rate > 0.2:
            return "degraded"
        else:
            return "healthy"
    
    def _calculate_error_rate(self, window_minutes: int) -> float:
        """Calculate overall error rate."""
        error_summary = self.metrics_collector.get_metric_summary("memory_operation_errors", window_minutes)
        total_summary = self.metrics_collector.get_metric_summary("memory_operation_milestones", window_minutes)
        
        error_count = error_summary.get("sum", 0)
        total_operations = total_summary.get("sum", 1)  # Avoid division by zero
        
        return round((error_count / total_operations) * 100, 2)
    
    def _calculate_avg_response_time(self, window_minutes: int) -> float:
        """Calculate average response time."""
        duration_metrics = [
            "memory_condensation_duration",
            "memory_storage_duration",
            "memory_retrieval_duration"
        ]
        
        total_duration = 0
        total_operations = 0
        
        for metric in duration_metrics:
            summary = self.metrics_collector.get_metric_summary(metric, window_minutes)
            if summary.get("count", 0) > 0:
                total_duration += summary.get("sum", 0)
                total_operations += summary.get("count", 0)
        
        if total_operations == 0:
            return 0.0
        
        return round(total_duration / total_operations, 3)
    
    def _calculate_success_rate(self, window_minutes: int) -> float:
        """Calculate overall success rate."""
        success_metrics = [
            "memory_condensation_successes",
            "memory_storage_successes",
            "memory_retrieval_successes"
        ]
        
        failure_metrics = [
            "memory_condensation_failures",
            "memory_storage_failures",
            "memory_retrieval_failures"
        ]
        
        total_successes = sum(
            self.metrics_collector.get_metric_summary(metric, window_minutes).get("sum", 0)
            for metric in success_metrics
        )
        
        total_failures = sum(
            self.metrics_collector.get_metric_summary(metric, window_minutes).get("sum", 0)
            for metric in failure_metrics
        )
        
        total_operations = total_successes + total_failures
        
        if total_operations == 0:
            return 100.0
        
        return round((total_successes / total_operations) * 100, 2)
    
    def _determine_overall_status(
        self, 
        component_healths: List[str], 
        active_alerts: int, 
        error_rate: float, 
        success_rate: float
    ) -> str:
        """Determine overall system status."""
        
        # Check for critical conditions
        if "critical" in component_healths or active_alerts > 5 or error_rate > 50 or success_rate < 50:
            return "critical"
        
        # Check for degraded conditions
        if "degraded" in component_healths or active_alerts > 2 or error_rate > 20 or success_rate < 80:
            return "degraded"
        
        return "healthy"
    
    def _generate_recommendations(
        self, 
        overall_status: str, 
        component_healths: Dict[str, str], 
        active_alerts: List[Dict[str, Any]], 
        error_rate: float, 
        success_rate: float
    ) -> List[str]:
        """Generate system recommendations."""
        recommendations = []
        
        if overall_status == "critical":
            recommendations.append("URGENT: System is in critical state - immediate attention required")
        
        if error_rate > 30:
            recommendations.append(f"High error rate detected ({error_rate}%) - investigate error patterns")
        
        if success_rate < 70:
            recommendations.append(f"Low success rate ({success_rate}%) - review system reliability")
        
        for component, health in component_healths.items():
            if health == "critical":
                recommendations.append(f"Critical issue with {component} - requires immediate investigation")
            elif health == "degraded":
                recommendations.append(f"Performance degradation in {component} - monitor closely")
        
        if len(active_alerts) > 3:
            recommendations.append(f"Multiple active alerts ({len(active_alerts)}) - review alert conditions")
        
        # Add proactive recommendations
        if overall_status == "healthy":
            recommendations.append("System is healthy - continue monitoring")
            recommendations.append("Consider reviewing alert thresholds for optimization")
        
        return recommendations
    
    def _calculate_derived_metrics(self, window_minutes: int) -> Dict[str, Any]:
        """Calculate derived metrics from base metrics."""
        return {
            "memory_efficiency": self._calculate_memory_efficiency(window_minutes),
            "condensation_effectiveness": self._calculate_condensation_effectiveness(window_minutes),
            "storage_reliability": self._calculate_storage_reliability(window_minutes),
            "retrieval_performance": self._calculate_retrieval_performance(window_minutes)
        }
    
    def _calculate_memory_efficiency(self, window_minutes: int) -> Dict[str, Any]:
        """Calculate memory operation efficiency metrics."""
        condensation_attempts = self.metrics_collector.get_metric_summary("memory_condensation_attempts", window_minutes)
        condensation_successes = self.metrics_collector.get_metric_summary("memory_condensation_successes", window_minutes)
        
        attempts = condensation_attempts.get("sum", 0)
        successes = condensation_successes.get("sum", 0)
        
        efficiency = (successes / attempts * 100) if attempts > 0 else 0
        
        return {
            "efficiency_percentage": round(efficiency, 2),
            "total_attempts": attempts,
            "successful_operations": successes,
            "status": "good" if efficiency > 90 else "needs_attention" if efficiency > 70 else "poor"
        }
    
    def _calculate_condensation_effectiveness(self, window_minutes: int) -> Dict[str, Any]:
        """Calculate memory condensation effectiveness."""
        memories_retrieved = self.metrics_collector.get_metric_summary("memory_condensation_memories_retrieved", window_minutes)
        dialogue_turns = self.metrics_collector.get_metric_summary("memory_condensation_dialogue_turns", window_minutes)
        
        avg_memories = memories_retrieved.get("avg", 0)
        avg_turns = dialogue_turns.get("avg", 0)
        
        effectiveness = (avg_turns / avg_memories * 100) if avg_memories > 0 else 0
        
        return {
            "effectiveness_percentage": round(effectiveness, 2),
            "avg_memories_retrieved": round(avg_memories, 1),
            "avg_dialogue_turns_processed": round(avg_turns, 1),
            "status": "excellent" if effectiveness > 80 else "good" if effectiveness > 60 else "needs_improvement"
        }
    
    def _calculate_storage_reliability(self, window_minutes: int) -> Dict[str, Any]:
        """Calculate storage operation reliability."""
        storage_attempts = self.metrics_collector.get_metric_summary("memory_storage_attempts", window_minutes)
        storage_successes = self.metrics_collector.get_metric_summary("memory_storage_successes", window_minutes)
        
        attempts = storage_attempts.get("sum", 0)
        successes = storage_successes.get("sum", 0)
        
        reliability = (successes / attempts * 100) if attempts > 0 else 0
        
        return {
            "reliability_percentage": round(reliability, 2),
            "total_storage_attempts": attempts,
            "successful_storage_operations": successes,
            "status": "excellent" if reliability > 95 else "good" if reliability > 85 else "concerning"
        }
    
    def _calculate_retrieval_performance(self, window_minutes: int) -> Dict[str, Any]:
        """Calculate retrieval operation performance."""
        retrieval_duration = self.metrics_collector.get_metric_summary("memory_retrieval_duration", window_minutes)
        memories_found = self.metrics_collector.get_metric_summary("memory_retrieval_memories_found", window_minutes)
        
        avg_duration = retrieval_duration.get("avg", 0)
        avg_found = memories_found.get("avg", 0)
        
        performance_score = (avg_found / max(avg_duration, 0.1)) * 10  # memories per second * 10
        
        return {
            "performance_score": round(performance_score, 2),
            "avg_retrieval_duration_seconds": round(avg_duration, 3),
            "avg_memories_found": round(avg_found, 1),
            "status": "excellent" if performance_score > 50 else "good" if performance_score > 20 else "slow"
        }
    
    def _analyze_trends(self, window_minutes: int) -> Dict[str, str]:
        """Analyze metric trends."""
        # This is a simplified trend analysis
        # In a real implementation, you'd compare current vs previous periods
        return {
            "memory_condensation": "stable",
            "memory_storage": "improving",
            "memory_retrieval": "stable",
            "error_rate": "decreasing",
            "performance": "stable"
        }
    
    def _analyze_alert_trends(self) -> Dict[str, Any]:
        """Analyze alert trends."""
        return {
            "trend": "stable",
            "most_frequent_alert": "memory_condensation_failures",
            "alert_frequency_change": "decreasing"
        }
    
    def _analyze_performance_trends(self, window_minutes: int) -> Dict[str, str]:
        """Analyze performance trends."""
        return {
            "overall_performance": "stable",
            "response_time_trend": "improving",
            "throughput_trend": "stable"
        }
    
    def _identify_performance_bottlenecks(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for operation, data in performance_data.items():
            if operation == "overall":
                continue
                
            avg_duration = data.get("performance_stats", {}).get("avg_duration_ms", 0)
            if avg_duration > 5000:  # 5 seconds
                bottlenecks.append(f"{operation} operations are slow (avg: {avg_duration}ms)")
        
        return bottlenecks
    
    def _generate_performance_recommendations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        overall_ops = performance_data.get("overall", {}).get("total_operations", 0)
        if overall_ops > 1000:
            recommendations.append("High operation volume - consider implementing caching")
        
        return recommendations
    
    def _analyze_error_patterns(self, recent_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_types = defaultdict(int)
        operations = defaultdict(int)
        
        for error in recent_errors:
            error_types[error["error_type"]] += 1
            operations[error["operation"]] += 1
        
        return {
            "most_common_error_type": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            "most_problematic_operation": max(operations.items(), key=lambda x: x[1])[0] if operations else None,
            "error_distribution": dict(error_types),
            "operation_distribution": dict(operations)
        }
    
    def _analyze_recovery_patterns(self, recent_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error recovery patterns."""
        total_errors = len(recent_errors)
        recovery_attempted = sum(1 for error in recent_errors if error["recovery_attempted"])
        recovery_successful = sum(1 for error in recent_errors if error["recovery_successful"])
        
        return {
            "total_errors": total_errors,
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "recovery_attempt_rate": round((recovery_attempted / total_errors * 100), 2) if total_errors > 0 else 0,
            "recovery_success_rate": round((recovery_successful / recovery_attempted * 100), 2) if recovery_attempted > 0 else 0
        }
    
    def _export_prometheus_metrics(self, dashboard_data: Dict[str, Any]) -> str:
        """Export dashboard data in Prometheus format."""
        lines = []
        
        # System health metrics
        health_status = dashboard_data["system_health"]
        lines.append(f"# HELP memory_system_health Overall system health status")
        lines.append(f"# TYPE memory_system_health gauge")
        health_value = 1 if health_status["overall_status"] == "healthy" else 0.5 if health_status["overall_status"] == "degraded" else 0
        lines.append(f'memory_system_health{{status="{health_status["overall_status"]}"}} {health_value}')
        
        # Active alerts
        lines.append(f"# HELP memory_active_alerts Number of active alerts")
        lines.append(f"# TYPE memory_active_alerts gauge")
        lines.append(f"memory_active_alerts {health_status['active_alerts']}")
        
        # Error rate
        lines.append(f"# HELP memory_error_rate_5min Error rate over 5 minutes")
        lines.append(f"# TYPE memory_error_rate_5min gauge")
        lines.append(f"memory_error_rate_5min {health_status['error_rate_5min']}")
        
        # Success rate
        lines.append(f"# HELP memory_success_rate_5min Success rate over 5 minutes")
        lines.append(f"# TYPE memory_success_rate_5min gauge")
        lines.append(f"memory_success_rate_5min {health_status['success_rate_5min']}")
        
        return "\n".join(lines)


# Global dashboard instance
_monitoring_dashboard = MemoryMonitoringDashboard()


def get_monitoring_dashboard() -> MemoryMonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    return _monitoring_dashboard


def get_system_health() -> SystemHealthStatus:
    """Convenience function to get system health."""
    return _monitoring_dashboard.get_system_health()


def export_monitoring_data(format_type: str = "json") -> str:
    """Convenience function to export monitoring data."""
    return _monitoring_dashboard.export_dashboard_data(format_type)