"""
Advanced alerting system for critical memory failures and performance issues.
Provides real-time monitoring, alert escalation, and notification capabilities.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import Thread, Event
from queue import Queue, Empty

from .logging_utils import get_logger
from .metrics_collector import get_metrics_collector, AlertSeverity
from .error_context import ErrorSeverity as ErrorSeverityLevel


class AlertChannel(Enum):
    """Alert notification channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class AlertState(Enum):
    """Alert states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Advanced alert rule definition."""
    
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str
    threshold: Union[int, float]
    severity: AlertSeverity
    window_minutes: int = 5
    evaluation_interval_seconds: int = 30
    
    # Escalation settings
    escalation_threshold: int = 3  # Number of consecutive triggers before escalation
    escalation_severity: Optional[AlertSeverity] = None
    
    # Notification settings
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    notification_template: Optional[str] = None
    
    # Suppression settings
    suppression_window_minutes: int = 60  # Suppress duplicate alerts
    max_alerts_per_hour: int = 10
    
    # State tracking
    state: AlertState = AlertState.RESOLVED
    trigger_count: int = 0
    last_triggered: Optional[float] = None
    last_resolved: Optional[float] = None
    last_notification: Optional[float] = None
    consecutive_triggers: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class AlertEvent:
    """Alert event for processing."""
    
    event_id: str
    rule_id: str
    event_type: str  # "trigger", "resolve", "escalate"
    timestamp: float
    metric_value: Union[int, float]
    threshold: Union[int, float]
    severity: AlertSeverity
    correlation_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AlertingSystem:
    """
    Advanced alerting system for memory operations.
    
    Features:
    - Real-time alert evaluation
    - Alert escalation and suppression
    - Multiple notification channels
    - Alert correlation and grouping
    - Performance monitoring
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_collector = get_metrics_collector()
        
        # Alert management
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alert_history: List[AlertEvent] = []
        self._notification_handlers: Dict[AlertChannel, Callable] = {}
        
        # Processing queue and thread
        self._alert_queue: Queue = Queue()
        self._processing_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._running = False
        
        # Performance tracking
        self._evaluation_metrics = {
            "total_evaluations": 0,
            "total_triggers": 0,
            "total_notifications": 0,
            "last_evaluation_time": 0
        }
        
        # Setup default handlers and rules
        self._setup_notification_handlers()
        self._setup_default_alert_rules()
    
    def start(self) -> None:
        """Start the alerting system."""
        if self._running:
            self.logger.warning("Alerting system already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._processing_thread = Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        self.logger.info("Alerting system started")
    
    def stop(self) -> None:
        """Stop the alerting system."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        
        self.logger.info("Alerting system stopped")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        Add a new alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self._alert_rules[rule.rule_id] = rule
        
        self.logger.info(
            f"Added alert rule: {rule.name}",
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            condition=rule.condition,
            severity=rule.severity.value
        )
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        if rule_id in self._alert_rules:
            rule = self._alert_rules.pop(rule_id)
            self.logger.info(f"Removed alert rule: {rule.name}", rule_id=rule_id)
            return True
        return False
    
    def acknowledge_alert(self, rule_id: str, correlation_id: Optional[str] = None) -> bool:
        """
        Acknowledge an active alert.
        
        Args:
            rule_id: ID of alert rule
            correlation_id: Optional correlation ID
            
        Returns:
            True if alert was acknowledged
        """
        if rule_id not in self._alert_rules:
            return False
        
        rule = self._alert_rules[rule_id]
        if rule.state == AlertState.ACTIVE:
            rule.state = AlertState.ACKNOWLEDGED
            
            self.logger.info(
                f"Alert acknowledged: {rule.name}",
                rule_id=rule_id,
                correlation_id=correlation_id
            )
            
            return True
        
        return False
    
    def suppress_alert(self, rule_id: str, duration_minutes: int = 60) -> bool:
        """
        Suppress an alert for a specified duration.
        
        Args:
            rule_id: ID of alert rule
            duration_minutes: Suppression duration in minutes
            
        Returns:
            True if alert was suppressed
        """
        if rule_id not in self._alert_rules:
            return False
        
        rule = self._alert_rules[rule_id]
        rule.state = AlertState.SUPPRESSED
        rule.suppression_window_minutes = duration_minutes
        
        self.logger.info(
            f"Alert suppressed: {rule.name}",
            rule_id=rule_id,
            duration_minutes=duration_minutes
        )
        
        return True
    
    def evaluate_alerts(self) -> List[AlertEvent]:
        """
        Evaluate all alert rules and return triggered events.
        
        Returns:
            List of alert events
        """
        events = []
        current_time = time.time()
        
        self._evaluation_metrics["total_evaluations"] += 1
        self._evaluation_metrics["last_evaluation_time"] = current_time
        
        for rule_id, rule in self._alert_rules.items():
            try:
                # Skip suppressed alerts
                if rule.state == AlertState.SUPPRESSED:
                    # Check if suppression period has expired
                    if (rule.last_triggered and 
                        current_time - rule.last_triggered > rule.suppression_window_minutes * 60):
                        rule.state = AlertState.RESOLVED
                    else:
                        continue
                
                # Get metric summary
                summary = self.metrics_collector.get_metric_summary(
                    rule.metric_name, 
                    rule.window_minutes
                )
                
                if summary.get("count", 0) == 0:
                    continue
                
                # Determine current metric value
                current_value = self._extract_metric_value(summary, rule.condition)
                
                # Evaluate condition
                is_triggered = self._evaluate_condition(current_value, rule.condition, rule.threshold)
                
                if is_triggered:
                    # Check rate limiting
                    if self._is_rate_limited(rule, current_time):
                        continue
                    
                    # Handle alert trigger
                    event = self._handle_alert_trigger(rule, current_value, current_time)
                    if event:
                        events.append(event)
                        self._alert_queue.put(event)
                
                elif rule.state == AlertState.ACTIVE:
                    # Handle alert resolution
                    event = self._handle_alert_resolution(rule, current_value, current_time)
                    if event:
                        events.append(event)
                        self._alert_queue.put(event)
                
            except Exception as e:
                self.logger.error(
                    f"Error evaluating alert rule {rule_id}",
                    error=str(e),
                    rule_name=rule.name
                )
        
        return events
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all currently active alerts."""
        active_alerts = []
        
        for rule in self._alert_rules.values():
            if rule.state == AlertState.ACTIVE:
                active_alerts.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "severity": rule.severity.value,
                    "metric_name": rule.metric_name,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered,
                    "consecutive_triggers": rule.consecutive_triggers
                })
        
        return active_alerts
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alerting system statistics.
        
        Args:
            hours: Time window in hours
            
        Returns:
            Statistics dictionary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        recent_events = [
            event for event in self._alert_history
            if event.timestamp > cutoff_time
        ]
        
        stats = {
            "time_window_hours": hours,
            "total_rules": len(self._alert_rules),
            "active_alerts": len([r for r in self._alert_rules.values() if r.state == AlertState.ACTIVE]),
            "recent_events": len(recent_events),
            "events_by_type": {},
            "events_by_severity": {},
            "most_frequent_alerts": {},
            "evaluation_metrics": self._evaluation_metrics.copy()
        }
        
        # Analyze recent events
        for event in recent_events:
            # Count by type
            event_type = event.event_type
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
            
            # Count by severity
            severity = event.severity.value
            stats["events_by_severity"][severity] = stats["events_by_severity"].get(severity, 0) + 1
            
            # Count by rule
            rule_id = event.rule_id
            stats["most_frequent_alerts"][rule_id] = stats["most_frequent_alerts"].get(rule_id, 0) + 1
        
        return stats
    
    def _processing_loop(self) -> None:
        """Main processing loop for alert events."""
        self.logger.info("Alert processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get event from queue with timeout
                try:
                    event = self._alert_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process the event
                self._process_alert_event(event)
                
                # Mark task as done
                self._alert_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in alert processing loop", error=str(e))
        
        self.logger.info("Alert processing loop stopped")
    
    def _process_alert_event(self, event: AlertEvent) -> None:
        """
        Process an individual alert event.
        
        Args:
            event: Alert event to process
        """
        rule = self._alert_rules.get(event.rule_id)
        if not rule:
            return
        
        # Add to history
        self._alert_history.append(event)
        
        # Limit history size
        if len(self._alert_history) > 10000:
            self._alert_history = self._alert_history[-5000:]
        
        # Send notifications
        self._send_notifications(rule, event)
        
        # Update metrics
        self._evaluation_metrics["total_notifications"] += 1
        
        self.logger.info(
            f"Processed alert event: {event.event_type}",
            rule_id=event.rule_id,
            rule_name=rule.name,
            severity=event.severity.value,
            correlation_id=event.correlation_id
        )
    
    def _send_notifications(self, rule: AlertRule, event: AlertEvent) -> None:
        """
        Send notifications for an alert event.
        
        Args:
            rule: Alert rule
            event: Alert event
        """
        for channel in rule.channels:
            try:
                handler = self._notification_handlers.get(channel)
                if handler:
                    handler(rule, event)
                else:
                    self.logger.warning(f"No handler for notification channel: {channel.value}")
            except Exception as e:
                self.logger.error(
                    f"Error sending notification via {channel.value}",
                    error=str(e),
                    rule_id=rule.rule_id
                )
    
    def _handle_alert_trigger(
        self, 
        rule: AlertRule, 
        current_value: Union[int, float], 
        current_time: float
    ) -> Optional[AlertEvent]:
        """Handle alert trigger logic."""
        
        if rule.state != AlertState.ACTIVE:
            # New alert trigger
            rule.state = AlertState.ACTIVE
            rule.trigger_count += 1
            rule.consecutive_triggers = 1
            rule.last_triggered = current_time
            
            event_type = "trigger"
            severity = rule.severity
            
        else:
            # Existing alert, increment consecutive triggers
            rule.consecutive_triggers += 1
            
            # Check for escalation
            if (rule.escalation_threshold and 
                rule.consecutive_triggers >= rule.escalation_threshold and
                rule.escalation_severity):
                
                event_type = "escalate"
                severity = rule.escalation_severity
                
                self.logger.warning(
                    f"Alert escalated: {rule.name}",
                    rule_id=rule.rule_id,
                    consecutive_triggers=rule.consecutive_triggers,
                    escalation_threshold=rule.escalation_threshold
                )
            else:
                # Don't create duplicate events for ongoing alerts
                return None
        
        # Create alert event
        event = AlertEvent(
            event_id=f"{rule.rule_id}_{int(current_time)}",
            rule_id=rule.rule_id,
            event_type=event_type,
            timestamp=current_time,
            metric_value=current_value,
            threshold=rule.threshold,
            severity=severity,
            context={
                "rule_name": rule.name,
                "metric_name": rule.metric_name,
                "condition": rule.condition,
                "consecutive_triggers": rule.consecutive_triggers
            }
        )
        
        self._evaluation_metrics["total_triggers"] += 1
        
        return event
    
    def _handle_alert_resolution(
        self, 
        rule: AlertRule, 
        current_value: Union[int, float], 
        current_time: float
    ) -> Optional[AlertEvent]:
        """Handle alert resolution logic."""
        
        rule.state = AlertState.RESOLVED
        rule.consecutive_triggers = 0
        rule.last_resolved = current_time
        
        event = AlertEvent(
            event_id=f"{rule.rule_id}_resolve_{int(current_time)}",
            rule_id=rule.rule_id,
            event_type="resolve",
            timestamp=current_time,
            metric_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            context={
                "rule_name": rule.name,
                "metric_name": rule.metric_name,
                "condition": rule.condition,
                "resolution_time": current_time - (rule.last_triggered or current_time)
            }
        )
        
        return event
    
    def _is_rate_limited(self, rule: AlertRule, current_time: float) -> bool:
        """Check if alert is rate limited."""
        
        # Check suppression window
        if (rule.last_notification and 
            current_time - rule.last_notification < rule.suppression_window_minutes * 60):
            return True
        
        # Check max alerts per hour
        hour_ago = current_time - 3600
        recent_triggers = len([
            event for event in self._alert_history
            if (event.rule_id == rule.rule_id and 
                event.event_type == "trigger" and
                event.timestamp > hour_ago)
        ])
        
        if recent_triggers >= rule.max_alerts_per_hour:
            return True
        
        return False
    
    def _extract_metric_value(self, summary: Dict[str, Any], condition: str) -> Union[int, float]:
        """Extract the appropriate metric value based on condition."""
        
        if "rate" in condition.lower() or "count" in condition.lower():
            return summary.get("sum", 0)
        elif "avg" in condition.lower():
            return summary.get("avg", 0)
        elif "max" in condition.lower():
            return summary.get("max", 0)
        elif "min" in condition.lower():
            return summary.get("min", 0)
        else:
            return summary.get("avg", 0)  # Default to average
    
    def _evaluate_condition(
        self, 
        current_value: Union[int, float], 
        condition: str, 
        threshold: Union[int, float]
    ) -> bool:
        """Evaluate alert condition."""
        
        if ">" in condition:
            return current_value > threshold
        elif "<" in condition:
            return current_value < threshold
        elif ">=" in condition:
            return current_value >= threshold
        elif "<=" in condition:
            return current_value <= threshold
        elif "==" in condition:
            return current_value == threshold
        elif "!=" in condition:
            return current_value != threshold
        
        return False
    
    def _setup_notification_handlers(self) -> None:
        """Set up notification handlers for different channels."""
        
        def log_handler(rule: AlertRule, event: AlertEvent) -> None:
            """Log notification handler."""
            message = rule.notification_template or f"Alert {event.event_type}: {rule.name}"
            
            if event.severity == AlertSeverity.CRITICAL:
                self.logger.critical(
                    message,
                    rule_id=rule.rule_id,
                    event_type=event.event_type,
                    metric_value=event.metric_value,
                    threshold=event.threshold,
                    correlation_id=event.correlation_id
                )
            elif event.severity == AlertSeverity.WARNING:
                self.logger.warning(
                    message,
                    rule_id=rule.rule_id,
                    event_type=event.event_type,
                    metric_value=event.metric_value,
                    threshold=event.threshold,
                    correlation_id=event.correlation_id
                )
            else:
                self.logger.info(
                    message,
                    rule_id=rule.rule_id,
                    event_type=event.event_type,
                    metric_value=event.metric_value,
                    threshold=event.threshold,
                    correlation_id=event.correlation_id
                )
        
        def console_handler(rule: AlertRule, event: AlertEvent) -> None:
            """Console notification handler."""
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            message = f"[{timestamp}] ALERT {event.event_type.upper()}: {rule.name} - {rule.condition} (value: {event.metric_value}, threshold: {event.threshold})"
            print(f"\033[91m{message}\033[0m")  # Red color for alerts
        
        self._notification_handlers[AlertChannel.LOG] = log_handler
        self._notification_handlers[AlertChannel.CONSOLE] = console_handler
    
    def _setup_default_alert_rules(self) -> None:
        """Set up default alert rules for memory operations."""
        
        # Critical memory condensation failures
        self.add_alert_rule(AlertRule(
            rule_id="memory_condensation_critical_failures",
            name="Critical Memory Condensation Failures",
            description="Memory condensation is failing at a critical rate",
            metric_name="memory_condensation_failures",
            condition="memory_condensation_failures count > 5",
            threshold=5,
            severity=AlertSeverity.CRITICAL,
            window_minutes=10,
            escalation_threshold=2,
            escalation_severity=AlertSeverity.EMERGENCY,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            notification_template="CRITICAL: Memory condensation failing - {metric_value} failures in {window_minutes} minutes"
        ))
        
        # Memory storage failures
        self.add_alert_rule(AlertRule(
            rule_id="memory_storage_failures",
            name="Memory Storage Failures",
            description="Memory storage operations are failing frequently",
            metric_name="memory_storage_failures",
            condition="memory_storage_failures > 10",
            threshold=10,
            severity=AlertSeverity.WARNING,
            window_minutes=15,
            channels=[AlertChannel.LOG],
            notification_template="WARNING: Memory storage failing - {metric_value} failures in {window_minutes} minutes"
        ))
        
        # Slow memory condensation
        self.add_alert_rule(AlertRule(
            rule_id="memory_condensation_slow_performance",
            name="Slow Memory Condensation Performance",
            description="Memory condensation operations are taking too long",
            metric_name="memory_condensation_duration",
            condition="memory_condensation_duration avg > 30",
            threshold=30.0,
            severity=AlertSeverity.WARNING,
            window_minutes=5,
            channels=[AlertChannel.LOG],
            notification_template="WARNING: Memory condensation slow - average {metric_value}s duration"
        ))
        
        # Memory retrieval failures
        self.add_alert_rule(AlertRule(
            rule_id="memory_retrieval_failures",
            name="Memory Retrieval Failures",
            description="Memory retrieval operations are failing",
            metric_name="memory_retrieval_failures",
            condition="memory_retrieval_failures > 15",
            threshold=15,
            severity=AlertSeverity.CRITICAL,
            window_minutes=5,
            escalation_threshold=3,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            notification_template="CRITICAL: Memory retrieval failing - {metric_value} failures in {window_minutes} minutes"
        ))
        
        # Profile storage failures
        self.add_alert_rule(AlertRule(
            rule_id="profile_storage_failures",
            name="Profile Storage Failures",
            description="Profile storage operations are failing",
            metric_name="profile_storage_failures",
            condition="profile_storage_failures > 8",
            threshold=8,
            severity=AlertSeverity.WARNING,
            window_minutes=10,
            channels=[AlertChannel.LOG],
            notification_template="WARNING: Profile storage failing - {metric_value} failures in {window_minutes} minutes"
        ))


# Global alerting system instance
_alerting_system = AlertingSystem()


def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system instance."""
    return _alerting_system


def start_alerting() -> None:
    """Start the global alerting system."""
    _alerting_system.start()


def stop_alerting() -> None:
    """Stop the global alerting system."""
    _alerting_system.stop()


def add_memory_alert(
    rule_id: str,
    name: str,
    metric_name: str,
    condition: str,
    threshold: Union[int, float],
    severity: AlertSeverity = AlertSeverity.WARNING,
    **kwargs
) -> None:
    """Convenience function to add a memory-related alert rule."""
    
    rule = AlertRule(
        rule_id=rule_id,
        name=name,
        description=f"Memory alert for {metric_name}",
        metric_name=metric_name,
        condition=condition,
        threshold=threshold,
        severity=severity,
        **kwargs
    )
    
    _alerting_system.add_alert_rule(rule)