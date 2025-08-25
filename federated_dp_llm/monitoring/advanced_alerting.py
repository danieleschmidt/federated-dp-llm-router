"""
Advanced Alerting and Monitoring System for Federated DP-LLM Router

Implements intelligent alerting, anomaly detection, and proactive monitoring
for healthcare federated learning with privacy-aware metrics collection.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()          # Informational alerts
    WARNING = auto()       # Warning conditions
    CRITICAL = auto()      # Critical issues requiring attention
    EMERGENCY = auto()     # Emergency situations requiring immediate action


class AlertCategory(Enum):
    """Categories of alerts."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    PRIVACY = "privacy"
    SYSTEM_HEALTH = "system_health"
    COMPLIANCE = "compliance"
    CAPACITY = "capacity"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    rule_id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str  # Python expression to evaluate
    threshold_value: float
    evaluation_window: int  # seconds
    cooldown_period: int = 300  # 5 minutes default cooldown
    max_frequency: int = 10  # maximum alerts per hour
    enabled: bool = True
    recipients: List[str] = field(default_factory=list)
    custom_message: Optional[str] = None


@dataclass
class Alert:
    """Represents an active alert."""
    alert_id: str
    rule_id: str
    timestamp: float
    severity: AlertSeverity
    category: AlertCategory
    message: str
    current_value: float
    threshold_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class MetricData:
    """Represents a metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedAlertingSystem:
    """Advanced alerting and monitoring system."""
    
    def __init__(self, max_alerts: int = 1000):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=max_alerts)
        self.metrics_buffer = deque(maxlen=10000)
        self.metric_aggregates = defaultdict(list)
        self.alert_counts = defaultdict(int)
        self.notification_handlers = {}
        self.anomaly_detectors = {}
        self.silence_rules = {}
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default alert rules for federated system."""
        default_rules = [
            # Performance Alerts
            AlertRule(
                rule_id="high_response_time",
                name="High Response Time",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                condition="avg_response_time > threshold_value",
                threshold_value=500.0,  # 500ms
                evaluation_window=300,  # 5 minutes
                recipients=["ops@hospital.com"]
            ),
            
            AlertRule(
                rule_id="low_throughput",
                name="Low Request Throughput",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                condition="requests_per_second < threshold_value",
                threshold_value=10.0,
                evaluation_window=600,  # 10 minutes
                recipients=["ops@hospital.com"]
            ),
            
            # Privacy Budget Alerts
            AlertRule(
                rule_id="privacy_budget_exhaustion",
                name="Privacy Budget Near Exhaustion",
                category=AlertCategory.PRIVACY,
                severity=AlertSeverity.CRITICAL,
                condition="privacy_budget_remaining < threshold_value",
                threshold_value=0.1,  # 10% remaining
                evaluation_window=60,
                recipients=["privacy-officer@hospital.com", "ops@hospital.com"]
            ),
            
            AlertRule(
                rule_id="privacy_budget_violation",
                name="Privacy Budget Violation Detected",
                category=AlertCategory.PRIVACY,
                severity=AlertSeverity.EMERGENCY,
                condition="privacy_violations > threshold_value",
                threshold_value=0,
                evaluation_window=60,
                recipients=["privacy-officer@hospital.com", "security@hospital.com"]
            ),
            
            # Security Alerts
            AlertRule(
                rule_id="high_failed_auth",
                name="High Failed Authentication Rate",
                category=AlertCategory.SECURITY,
                severity=AlertSeverity.CRITICAL,
                condition="failed_auth_rate > threshold_value",
                threshold_value=10.0,  # 10 failures per minute
                evaluation_window=60,
                recipients=["security@hospital.com"]
            ),
            
            AlertRule(
                rule_id="suspicious_ip_activity",
                name="Suspicious IP Activity",
                category=AlertCategory.SECURITY,
                severity=AlertSeverity.WARNING,
                condition="blocked_ips_count > threshold_value",
                threshold_value=5,
                evaluation_window=300,
                recipients=["security@hospital.com"]
            ),
            
            # System Health Alerts
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                category=AlertCategory.ERROR_RATE,
                severity=AlertSeverity.CRITICAL,
                condition="error_rate > threshold_value",
                threshold_value=0.05,  # 5% error rate
                evaluation_window=300,
                recipients=["ops@hospital.com", "dev@hospital.com"]
            ),
            
            AlertRule(
                rule_id="node_down",
                name="Federated Node Unavailable",
                category=AlertCategory.AVAILABILITY,
                severity=AlertSeverity.CRITICAL,
                condition="available_nodes < threshold_value",
                threshold_value=2,  # Less than 2 nodes available
                evaluation_window=120,
                recipients=["ops@hospital.com"]
            ),
            
            # Capacity Alerts
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                category=AlertCategory.CAPACITY,
                severity=AlertSeverity.WARNING,
                condition="cpu_usage > threshold_value",
                threshold_value=0.8,  # 80% CPU usage
                evaluation_window=600,
                recipients=["ops@hospital.com"]
            ),
            
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                category=AlertCategory.CAPACITY,
                severity=AlertSeverity.WARNING,
                condition="memory_usage > threshold_value",
                threshold_value=0.85,  # 85% memory usage
                evaluation_window=300,
                recipients=["ops@hospital.com"]
            ),
            
            # Compliance Alerts
            AlertRule(
                rule_id="compliance_violation",
                name="HIPAA Compliance Violation",
                category=AlertCategory.COMPLIANCE,
                severity=AlertSeverity.EMERGENCY,
                condition="compliance_violations > threshold_value",
                threshold_value=0,
                evaluation_window=60,
                recipients=["compliance@hospital.com", "privacy-officer@hospital.com"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def add_metric(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Add a metric data point."""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics_buffer.append(metric)
        self.metric_aggregates[name].append((metric.timestamp, metric.value))
        
        # Keep only recent data points (last hour)
        cutoff_time = time.time() - 3600
        self.metric_aggregates[name] = [
            (ts, val) for ts, val in self.metric_aggregates[name] 
            if ts > cutoff_time
        ]
    
    async def evaluate_alerts(self):
        """Evaluate all alert rules against current metrics."""
        current_time = time.time()
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
                
            # Check if rule is in cooldown
            if self._is_in_cooldown(rule, current_time):
                continue
                
            # Check frequency limits
            if self._exceeds_frequency_limit(rule, current_time):
                continue
                
            # Evaluate rule condition
            try:
                triggered = await self._evaluate_rule(rule, current_time)
                if triggered:
                    alert = await self._create_alert(rule, current_time)
                    triggered_alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.rule_id}: {str(e)}")
        
        # Process triggered alerts
        for alert in triggered_alerts:
            await self._process_alert(alert)
        
        return triggered_alerts
    
    async def _evaluate_rule(self, rule: AlertRule, current_time: float) -> bool:
        """Evaluate a specific alert rule."""
        evaluation_window = rule.evaluation_window
        cutoff_time = current_time - evaluation_window
        
        # Get metrics for evaluation window
        window_metrics = {}
        
        for metric_name, data_points in self.metric_aggregates.items():
            recent_points = [(ts, val) for ts, val in data_points if ts > cutoff_time]
            
            if recent_points:
                values = [val for _, val in recent_points]
                window_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
                window_metrics[f"max_{metric_name}"] = max(values)
                window_metrics[f"min_{metric_name}"] = min(values)
                window_metrics[f"count_{metric_name}"] = len(values)
                window_metrics[metric_name] = values[-1]  # Latest value
            else:
                window_metrics[f"avg_{metric_name}"] = 0
                window_metrics[f"max_{metric_name}"] = 0
                window_metrics[f"min_{metric_name}"] = 0
                window_metrics[f"count_{metric_name}"] = 0
                window_metrics[metric_name] = 0
        
        # Add threshold value to evaluation context
        eval_context = {
            **window_metrics,
            "threshold_value": rule.threshold_value
        }
        
        # Evaluate condition
        try:
            return eval(rule.condition, {"__builtins__": {}}, eval_context)
        except Exception as e:
            logger.warning(f"Failed to evaluate rule {rule.rule_id}: {str(e)}")
            return False
    
    def _is_in_cooldown(self, rule: AlertRule, current_time: float) -> bool:
        """Check if rule is in cooldown period."""
        if rule.rule_id in self.active_alerts:
            last_alert_time = self.active_alerts[rule.rule_id].timestamp
            return (current_time - last_alert_time) < rule.cooldown_period
        return False
    
    def _exceeds_frequency_limit(self, rule: AlertRule, current_time: float) -> bool:
        """Check if rule exceeds maximum frequency."""
        hour_ago = current_time - 3600
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.rule_id == rule.rule_id and alert.timestamp > hour_ago
        ]
        return len(recent_alerts) >= rule.max_frequency
    
    async def _create_alert(self, rule: AlertRule, current_time: float) -> Alert:
        """Create an alert from a triggered rule."""
        alert_id = f"{rule.rule_id}_{int(current_time)}"
        
        # Get current value for the alert
        current_value = await self._get_current_metric_value(rule)
        
        # Generate alert message
        message = rule.custom_message or f"{rule.name}: Current value {current_value:.2f} exceeds threshold {rule.threshold_value:.2f}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=current_time,
            severity=rule.severity,
            category=rule.category,
            message=message,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            metadata={
                "rule_name": rule.name,
                "evaluation_window": rule.evaluation_window,
                "recipients": rule.recipients
            }
        )
        
        return alert
    
    async def _get_current_metric_value(self, rule: AlertRule) -> float:
        """Extract current metric value for the alert rule."""
        # Parse the condition to find metric name
        condition_parts = rule.condition.split()
        metric_name = condition_parts[0] if condition_parts else "unknown"
        
        if metric_name in self.metric_aggregates:
            recent_points = self.metric_aggregates[metric_name]
            if recent_points:
                return recent_points[-1][1]  # Latest value
        
        return 0.0
    
    async def _process_alert(self, alert: Alert):
        """Process a triggered alert."""
        # Add to active alerts
        self.active_alerts[alert.rule_id] = alert
        self.alert_history.append(alert)
        
        # Increment alert count
        self.alert_counts[alert.rule_id] += 1
        
        # Log alert
        logger.warning(f"ALERT [{alert.severity.name}] {alert.category.value}: {alert.message}")
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Trigger custom handlers
        if alert.category in self.notification_handlers:
            try:
                await self.notification_handlers[alert.category](alert)
            except Exception as e:
                logger.error(f"Error in custom alert handler: {str(e)}")
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications to recipients."""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule or not rule.recipients:
            return
        
        # Email notification (mock implementation)
        await self._send_email_notification(alert, rule.recipients)
        
        # Could add other notification methods (Slack, SMS, etc.)
    
    async def _send_email_notification(self, alert: Alert, recipients: List[str]):
        """Send email notification for alert (mock implementation)."""
        try:
            subject = f"[{alert.severity.name}] {alert.category.value.title()} Alert: {alert.rule_id}"
            
            body = f"""
            Alert Details:
            - Alert ID: {alert.alert_id}
            - Severity: {alert.severity.name}
            - Category: {alert.category.value}
            - Message: {alert.message}
            - Current Value: {alert.current_value:.2f}
            - Threshold: {alert.threshold_value:.2f}
            - Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
            
            Please investigate this issue promptly.
            
            Federated DP-LLM Monitoring System
            """
            
            logger.info(f"Email notification sent to {recipients}: {subject}")
            # In production, implement actual email sending logic here
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an active alert."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.metadata["acknowledged_by"] = user
                alert.metadata["acknowledged_at"] = time.time()
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        return False
    
    def resolve_alert(self, rule_id: str, user: str) -> bool:
        """Resolve an active alert."""
        if rule_id in self.active_alerts:
            alert = self.active_alerts[rule_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            alert.metadata["resolved_by"] = user
            
            # Remove from active alerts
            del self.active_alerts[rule_id]
            
            logger.info(f"Alert {alert.alert_id} resolved by {user}")
            return True
        return False
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def disable_alert_rule(self, rule_id: str):
        """Disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")
    
    def enable_alert_rule(self, rule_id: str):
        """Enable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
    
    def register_notification_handler(self, category: AlertCategory, handler: Callable):
        """Register custom notification handler for alert category."""
        self.notification_handlers[category] = handler
        logger.info(f"Registered notification handler for {category.value}")
    
    def get_alert_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get summary of alerts in the specified time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for alert in recent_alerts:
            severity_counts[alert.severity.name] += 1
            category_counts[alert.category.value] += 1
        
        return {
            "time_window_hours": time_window / 3600,
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "severity_breakdown": dict(severity_counts),
            "category_breakdown": dict(category_counts),
            "alert_rules_total": len(self.alert_rules),
            "alert_rules_enabled": sum(1 for rule in self.alert_rules.values() if rule.enabled)
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.name,
                "category": alert.category.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "acknowledged": alert.acknowledged
            }
            for alert in self.active_alerts.values()
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check and return status."""
        current_time = time.time()
        
        # Check recent metrics
        recent_metrics = [
            metric for metric in self.metrics_buffer
            if current_time - metric.timestamp < 300  # Last 5 minutes
        ]
        
        # Check alert system health
        system_health = {
            "status": "healthy",
            "timestamp": current_time,
            "metrics_collected": len(recent_metrics),
            "active_alerts": len(self.active_alerts),
            "alert_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "recent_evaluations": len(self.alert_history) if self.alert_history else 0
        }
        
        # Check for critical issues
        critical_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        ]
        
        if critical_alerts:
            system_health["status"] = "degraded"
            system_health["critical_alerts_count"] = len(critical_alerts)
        
        if len(recent_metrics) == 0:
            system_health["status"] = "unhealthy"
            system_health["issue"] = "No recent metrics collected"
        
        return system_health


# Global alerting system instance
alerting_system = AdvancedAlertingSystem()


# Convenience functions for common metrics
def record_response_time(value: float, endpoint: str = ""):
    """Record response time metric."""
    alerting_system.add_metric("response_time", value, {"endpoint": endpoint})


def record_error_rate(value: float, error_type: str = ""):
    """Record error rate metric."""
    alerting_system.add_metric("error_rate", value, {"error_type": error_type})


def record_privacy_budget(remaining: float, total: float, user_id: str = ""):
    """Record privacy budget metrics."""
    alerting_system.add_metric("privacy_budget_remaining", remaining, {"user_id": user_id})
    alerting_system.add_metric("privacy_budget_total", total, {"user_id": user_id})


def record_system_metrics(cpu_usage: float, memory_usage: float, available_nodes: int):
    """Record system health metrics."""
    alerting_system.add_metric("cpu_usage", cpu_usage)
    alerting_system.add_metric("memory_usage", memory_usage)
    alerting_system.add_metric("available_nodes", available_nodes)


# Background task for alert evaluation
async def alert_evaluation_loop():
    """Background loop for evaluating alerts."""
    while True:
        try:
            await alerting_system.evaluate_alerts()
            await asyncio.sleep(30)  # Evaluate every 30 seconds
        except Exception as e:
            logger.error(f"Error in alert evaluation loop: {str(e)}")
            await asyncio.sleep(60)  # Wait longer on error