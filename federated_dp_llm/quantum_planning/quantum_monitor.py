"""
Quantum Planning System Monitor

Real-time monitoring, alerting, and health checking for quantum-inspired
task planning components with healthcare-grade observability.
"""

import asyncio
import time
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class QuantumMetric:
    """Quantum system metric with metadata."""
    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class QuantumAlert:
    """Quantum system alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    component: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass  
class ComponentHealthCheck:
    """Health check result for quantum component."""
    component_name: str
    status: HealthStatus
    last_check_time: float
    response_time: float
    error_rate: float
    availability: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class QuantumMonitor:
    """
    Comprehensive monitoring system for quantum planning components.
    
    Provides:
    - Real-time metric collection and aggregation
    - Health monitoring and alerting
    - Performance tracking and optimization insights
    - Anomaly detection for quantum states
    - Healthcare compliance monitoring
    """
    
    def __init__(self, 
                 collection_interval: float = 10.0,
                 alert_retention_hours: int = 24,
                 metric_retention_hours: int = 168):  # 1 week
        
        self.collection_interval = collection_interval
        self.alert_retention_seconds = alert_retention_hours * 3600
        self.metric_retention_seconds = metric_retention_hours * 3600
        
        # Core monitoring state
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[QuantumAlert] = []
        self.component_health: Dict[str, ComponentHealthCheck] = {}
        
        # Monitoring configuration
        self.alert_thresholds: Dict[str, Dict[str, float]] = {
            "quantum_planner": {
                "decoherence_rate_warning": 0.1,
                "decoherence_rate_critical": 0.3,
                "planning_time_warning": 5.0,
                "planning_time_critical": 15.0,
                "task_success_rate_warning": 0.8,
                "task_success_rate_critical": 0.6
            },
            "superposition_scheduler": {
                "coherence_loss_warning": 0.2,
                "coherence_loss_critical": 0.5,
                "measurement_accuracy_warning": 0.7,
                "measurement_accuracy_critical": 0.5,
                "superposition_time_warning": 300.0,
                "superposition_time_critical": 600.0
            },
            "entanglement_optimizer": {
                "entanglement_strength_warning": 0.3,
                "entanglement_strength_critical": 0.1,
                "correlation_violation_warning": 0.1,
                "correlation_violation_critical": 0.3,
                "bell_inequality_violation_warning": 2.5,
                "bell_inequality_violation_critical": 3.0
            },
            "interference_balancer": {
                "load_imbalance_warning": 0.3,
                "load_imbalance_critical": 0.6,
                "interference_strength_warning": 0.2,
                "interference_strength_critical": 0.1,
                "phase_coherence_warning": 0.5,
                "phase_coherence_critical": 0.2
            }
        }
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {
            "average_planning_time": 2.0,
            "average_measurement_time": 0.5,
            "expected_success_rate": 0.95,
            "target_coherence": 0.8,
            "optimal_entanglement_strength": 0.7
        }
        
        # Anomaly detection
        self.anomaly_detectors: Dict[str, Any] = {}
        self.anomaly_history: List[Dict[str, Any]] = []
        
        # Control flags
        self._monitoring_active = False
        self._monitor_task = None
        self.alert_callbacks: List[Callable] = []
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        self._alerts_lock = threading.Lock()
    
    async def start_monitoring(self):
        """Start continuous monitoring of quantum components."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Quantum monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring system."""
        self._monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Quantum monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect metrics from all registered components
                await self._collect_system_metrics()
                
                # Check health of all components
                await self._check_component_health()
                
                # Evaluate alert conditions
                await self._evaluate_alerts()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Run anomaly detection
                await self._detect_anomalies()
                
                # Wait for next collection cycle
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(self.collection_interval)
    
    def register_component(self, 
                          component_name: str,
                          component_instance: Any,
                          health_check_method: str = "get_health_status"):
        """Register component for monitoring."""
        
        # Initialize health check
        self.component_health[component_name] = ComponentHealthCheck(
            component_name=component_name,
            status=HealthStatus.UNKNOWN,
            last_check_time=time.time(),
            response_time=0.0,
            error_rate=0.0,
            availability=1.0
        )
        
        # Store component reference for metric collection
        setattr(self, f"_{component_name}_instance", component_instance)
        setattr(self, f"_{component_name}_health_method", health_check_method)
        
        logger.info(f"Registered component for monitoring: {component_name}")
    
    async def record_metric(self, 
                           metric_name: str,
                           value: float,
                           component: str = "system",
                           unit: str = "",
                           tags: Dict[str, str] = None):
        """Record a quantum system metric."""
        
        if tags is None:
            tags = {}
        
        metric = QuantumMetric(
            name=metric_name,
            value=value,
            unit=unit,
            tags={"component": component, **tags}
        )
        
        metric_key = f"{component}.{metric_name}"
        
        with self._metrics_lock:
            self.metrics[metric_key].append(metric)
        
        # Check for immediate alerts
        await self._check_metric_alerts(metric_key, value, component)
    
    async def _collect_system_metrics(self):
        """Collect metrics from all registered components."""
        
        for component_name in self.component_health.keys():
            try:
                component_instance = getattr(self, f"_{component_name}_instance", None)
                if not component_instance:
                    continue
                
                # Collect component-specific metrics
                await self._collect_component_metrics(component_name, component_instance)
                
            except Exception as e:
                logger.error(f"Failed to collect metrics from {component_name}: {str(e)}")
    
    async def _collect_component_metrics(self, component_name: str, component_instance: Any):
        """Collect metrics from specific component."""
        
        current_time = time.time()
        
        if component_name == "quantum_planner":
            await self._collect_quantum_planner_metrics(component_instance, current_time)
        elif component_name == "superposition_scheduler":
            await self._collect_superposition_metrics(component_instance, current_time)
        elif component_name == "entanglement_optimizer":
            await self._collect_entanglement_metrics(component_instance, current_time)
        elif component_name == "interference_balancer":
            await self._collect_interference_metrics(component_instance, current_time)
    
    async def _collect_quantum_planner_metrics(self, planner: Any, timestamp: float):
        """Collect quantum planner specific metrics."""
        
        try:
            stats = planner.get_quantum_statistics()
            
            await self.record_metric("active_tasks", stats.get("active_tasks", 0), "quantum_planner")
            await self.record_metric("superposition_tasks", stats.get("superposition_tasks", 0), "quantum_planner")
            await self.record_metric("entangled_tasks", stats.get("entangled_tasks", 0), "quantum_planner")
            await self.record_metric("collapsed_tasks", stats.get("collapsed_tasks", 0), "quantum_planner")
            await self.record_metric("decoherent_tasks", stats.get("decoherent_tasks", 0), "quantum_planner")
            
            await self.record_metric("average_planning_time", 
                                   stats.get("average_planning_time", 0), 
                                   "quantum_planner", "seconds")
            
            await self.record_metric("average_node_coherence",
                                   stats.get("average_node_coherence", 0),
                                   "quantum_planner")
            
            # Calculate derived metrics
            total_tasks = stats.get("active_tasks", 0) + stats.get("decoherent_tasks", 0)
            if total_tasks > 0:
                success_rate = stats.get("active_tasks", 0) / total_tasks
                await self.record_metric("task_success_rate", success_rate, "quantum_planner")
            
        except Exception as e:
            logger.error(f"Failed to collect quantum planner metrics: {str(e)}")
    
    async def _collect_superposition_metrics(self, scheduler: Any, timestamp: float):
        """Collect superposition scheduler metrics."""
        
        try:
            status = scheduler.get_superposition_status()
            
            await self.record_metric("active_superpositions", 
                                   status.get("active_superpositions", 0), 
                                   "superposition_scheduler")
            
            await self.record_metric("total_amplitude", 
                                   status.get("total_amplitude", 0), 
                                   "superposition_scheduler")
            
            metrics = status.get("scheduler_metrics", {})
            await self.record_metric("average_superposition_time",
                                   metrics.get("average_superposition_time", 0),
                                   "superposition_scheduler", "seconds")
            
            await self.record_metric("measurement_accuracy",
                                   metrics.get("measurement_accuracy", 0),
                                   "superposition_scheduler")
            
            await self.record_metric("interference_events",
                                   metrics.get("interference_events", 0),
                                   "superposition_scheduler")
            
        except Exception as e:
            logger.error(f"Failed to collect superposition metrics: {str(e)}")
    
    async def _collect_entanglement_metrics(self, optimizer: Any, timestamp: float):
        """Collect entanglement optimizer metrics."""
        
        try:
            stats = optimizer.get_entanglement_statistics()
            
            await self.record_metric("active_entanglements",
                                   stats.get("active_entanglements", 0),
                                   "entanglement_optimizer")
            
            await self.record_metric("global_entanglement_strength",
                                   stats.get("global_entanglement_strength", 0),
                                   "entanglement_optimizer")
            
            await self.record_metric("bell_inequality_violation_rate",
                                   stats.get("bell_inequality_violation_rate", 0),
                                   "entanglement_optimizer")
            
            correlation_stats = stats.get("recent_correlation_distribution", {})
            await self.record_metric("correlation_mean",
                                   correlation_stats.get("mean", 0),
                                   "entanglement_optimizer")
            
            await self.record_metric("correlation_std",
                                   correlation_stats.get("std", 0),
                                   "entanglement_optimizer")
            
        except Exception as e:
            logger.error(f"Failed to collect entanglement metrics: {str(e)}")
    
    async def _collect_interference_metrics(self, balancer: Any, timestamp: float):
        """Collect interference balancer metrics."""
        
        try:
            stats = balancer.get_interference_statistics()
            
            await self.record_metric("active_interferences",
                                   stats.get("active_interferences", 0),
                                   "interference_balancer")
            
            metrics = stats.get("balancer_metrics", {})
            await self.record_metric("load_balancing_efficiency",
                                   metrics.get("load_balancing_efficiency", 0),
                                   "interference_balancer")
            
            await self.record_metric("average_coherence",
                                   metrics.get("average_coherence", 0),
                                   "interference_balancer")
            
            await self.record_metric("phase_lock_stability",
                                   metrics.get("phase_lock_stability", 0),
                                   "interference_balancer")
            
            await self.record_metric("wave_interference_gain",
                                   metrics.get("wave_interference_gain", 0),
                                   "interference_balancer")
            
        except Exception as e:
            logger.error(f"Failed to collect interference metrics: {str(e)}")
    
    async def _check_component_health(self):
        """Check health of all registered components."""
        
        for component_name in self.component_health.keys():
            try:
                health_check = await self._perform_component_health_check(component_name)
                self.component_health[component_name] = health_check
                
                # Generate alerts for unhealthy components
                if health_check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    await self._create_health_alert(component_name, health_check)
                    
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {str(e)}")
                
                # Mark component as unknown
                self.component_health[component_name].status = HealthStatus.UNKNOWN
                self.component_health[component_name].issues.append(f"Health check error: {str(e)}")
    
    async def _perform_component_health_check(self, component_name: str) -> ComponentHealthCheck:
        """Perform health check on specific component."""
        
        start_time = time.time()
        component_instance = getattr(self, f"_{component_name}_instance", None)
        
        if not component_instance:
            return ComponentHealthCheck(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                last_check_time=start_time,
                response_time=0.0,
                error_rate=1.0,
                availability=0.0,
                issues=["Component instance not found"]
            )
        
        try:
            # Get component statistics for health assessment
            if hasattr(component_instance, 'get_quantum_statistics'):
                stats = component_instance.get_quantum_statistics()
            elif hasattr(component_instance, 'get_superposition_status'):
                stats = component_instance.get_superposition_status()
            elif hasattr(component_instance, 'get_entanglement_statistics'):
                stats = component_instance.get_entanglement_statistics()
            elif hasattr(component_instance, 'get_interference_statistics'):
                stats = component_instance.get_interference_statistics()
            else:
                stats = {}
            
            response_time = time.time() - start_time
            
            # Assess health based on component-specific metrics
            status, issues = self._assess_component_health(component_name, stats)
            
            return ComponentHealthCheck(
                component_name=component_name,
                status=status,
                last_check_time=start_time,
                response_time=response_time,
                error_rate=0.0,  # Would be calculated from error history
                availability=1.0 if status != HealthStatus.CRITICAL else 0.0,
                custom_metrics=self._extract_health_metrics(stats),
                issues=issues
            )
            
        except Exception as e:
            return ComponentHealthCheck(
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                last_check_time=start_time,
                response_time=time.time() - start_time,
                error_rate=1.0,
                availability=0.0,
                issues=[f"Health check exception: {str(e)}"]
            )
    
    def _assess_component_health(self, component_name: str, stats: Dict[str, Any]) -> Tuple[HealthStatus, List[str]]:
        """Assess component health based on statistics."""
        
        issues = []
        
        if component_name == "quantum_planner":
            coherence = stats.get("average_node_coherence", 1.0)
            decoherent_tasks = stats.get("decoherent_tasks", 0)
            total_tasks = stats.get("active_tasks", 0) + decoherent_tasks
            
            if coherence < 0.3:
                issues.append(f"Low node coherence: {coherence:.3f}")
            
            if total_tasks > 0 and decoherent_tasks / total_tasks > 0.5:
                issues.append(f"High decoherence rate: {decoherent_tasks}/{total_tasks}")
        
        elif component_name == "superposition_scheduler":
            metrics = stats.get("scheduler_metrics", {})
            measurement_accuracy = metrics.get("measurement_accuracy", 1.0)
            
            if measurement_accuracy < 0.5:
                issues.append(f"Low measurement accuracy: {measurement_accuracy:.3f}")
        
        elif component_name == "entanglement_optimizer":
            entanglement_strength = stats.get("global_entanglement_strength", 0.0)
            
            if entanglement_strength < 0.2:
                issues.append(f"Low entanglement strength: {entanglement_strength:.3f}")
        
        elif component_name == "interference_balancer":
            metrics = stats.get("balancer_metrics", {})
            coherence = metrics.get("average_coherence", 1.0)
            
            if coherence < 0.3:
                issues.append(f"Low interference coherence: {coherence:.3f}")
        
        # Determine overall status
        if len(issues) == 0:
            return HealthStatus.HEALTHY, issues
        elif len(issues) <= 2:
            return HealthStatus.DEGRADED, issues
        elif len(issues) <= 4:
            return HealthStatus.UNHEALTHY, issues
        else:
            return HealthStatus.CRITICAL, issues
    
    def _extract_health_metrics(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant health metrics from component statistics."""
        
        health_metrics = {}
        
        # Extract numeric values for health monitoring
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                health_metrics[key] = float(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        health_metrics[f"{key}.{nested_key}"] = float(nested_value)
        
        return health_metrics
    
    async def _check_metric_alerts(self, metric_key: str, value: float, component: str):
        """Check if metric value triggers any alerts."""
        
        thresholds = self.alert_thresholds.get(component, {})
        metric_name = metric_key.split('.')[-1]  # Extract metric name
        
        # Check warning threshold
        warning_key = f"{metric_name}_warning"
        if warning_key in thresholds:
            threshold = thresholds[warning_key]
            
            # Different comparison logic based on metric type
            should_alert = False
            if "rate" in metric_name or "accuracy" in metric_name or "coherence" in metric_name:
                # Lower is worse for rates/accuracy/coherence
                should_alert = value < threshold
            else:
                # Higher is worse for times/violations
                should_alert = value > threshold
            
            if should_alert:
                await self._create_alert(
                    severity=AlertSeverity.WARNING,
                    message=f"{component} {metric_name} threshold exceeded",
                    component=component,
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=threshold
                )
        
        # Check critical threshold
        critical_key = f"{metric_name}_critical"
        if critical_key in thresholds:
            threshold = thresholds[critical_key]
            
            should_alert = False
            if "rate" in metric_name or "accuracy" in metric_name or "coherence" in metric_name:
                should_alert = value < threshold
            else:
                should_alert = value > threshold
            
            if should_alert:
                await self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"{component} {metric_name} critical threshold exceeded",
                    component=component,
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=threshold
                )
    
    async def _create_alert(self, 
                           severity: AlertSeverity,
                           message: str,
                           component: str,
                           metric_name: str = None,
                           metric_value: float = None,
                           threshold: float = None):
        """Create new alert."""
        
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"
        
        alert = QuantumAlert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            component=component,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        with self._alerts_lock:
            self.alerts.append(alert)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {str(e)}")
        
        logger.warning(f"Alert created: {alert.message}")
    
    async def _create_health_alert(self, component_name: str, health_check: ComponentHealthCheck):
        """Create alert for unhealthy component."""
        
        severity = AlertSeverity.WARNING if health_check.status == HealthStatus.UNHEALTHY else AlertSeverity.CRITICAL
        
        await self._create_alert(
            severity=severity,
            message=f"Component {component_name} health status: {health_check.status.value}",
            component=component_name
        )
    
    async def _evaluate_alerts(self):
        """Evaluate existing alerts for resolution."""
        
        with self._alerts_lock:
            for alert in self.alerts:
                if alert.resolved:
                    continue
                
                # Check if alert condition still exists
                if await self._is_alert_resolved(alert):
                    alert.resolved = True
                    alert.resolution_time = time.time()
                    logger.info(f"Alert resolved: {alert.alert_id}")
    
    async def _is_alert_resolved(self, alert: QuantumAlert) -> bool:
        """Check if alert condition has been resolved."""
        
        if not alert.metric_name or not alert.threshold:
            return False
        
        # Get recent metric value
        metric_key = f"{alert.component}.{alert.metric_name}"
        recent_metrics = self.metrics.get(metric_key, deque())
        
        if not recent_metrics:
            return False
        
        recent_value = recent_metrics[-1].value
        
        # Check if value is now within acceptable range
        if "rate" in alert.metric_name or "accuracy" in alert.metric_name or "coherence" in alert.metric_name:
            return recent_value >= alert.threshold
        else:
            return recent_value <= alert.threshold
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        
        current_time = time.time()
        
        # Clean up old metrics
        with self._metrics_lock:
            for metric_key, metric_deque in self.metrics.items():
                # Remove metrics older than retention period
                while metric_deque and (current_time - metric_deque[0].timestamp) > self.metric_retention_seconds:
                    metric_deque.popleft()
        
        # Clean up old alerts
        with self._alerts_lock:
            self.alerts = [alert for alert in self.alerts 
                          if (current_time - alert.timestamp) < self.alert_retention_seconds]
    
    async def _detect_anomalies(self):
        """Detect anomalies in quantum system behavior."""
        
        # Simple anomaly detection based on statistical thresholds
        for metric_key, metric_deque in self.metrics.items():
            if len(metric_deque) < 10:  # Need minimum data points
                continue
            
            try:
                values = np.array([m.value for m in metric_deque])
                mean = np.mean(values)
                std = np.std(values)
                
                # Detect outliers (values beyond 3 standard deviations)
                recent_value = values[-1]
                if std > 0 and abs(recent_value - mean) > 3 * std:
                    self.anomaly_history.append({
                        "timestamp": time.time(),
                        "metric": metric_key,
                        "value": recent_value,
                        "mean": mean,
                        "std": std,
                        "z_score": abs(recent_value - mean) / std
                    })
                    
                    # Create anomaly alert
                    component = metric_key.split('.')[0]
                    metric_name = metric_key.split('.')[-1]
                    
                    await self._create_alert(
                        severity=AlertSeverity.WARNING,
                        message=f"Anomaly detected in {metric_name}",
                        component=component,
                        metric_name=metric_name,
                        metric_value=recent_value
                    )
            
            except Exception as e:
                logger.error(f"Anomaly detection failed for {metric_key}: {str(e)}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function to be called when alerts are created."""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        
        current_time = time.time()
        
        # Component health summary
        health_summary = {}
        for component, health in self.component_health.items():
            health_summary[component] = {
                "status": health.status.value,
                "response_time": health.response_time,
                "availability": health.availability,
                "issues": health.issues,
                "last_check": current_time - health.last_check_time
            }
        
        # Active alerts summary
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        alerts_by_severity = {severity.value: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            alerts_by_severity[alert.severity.value] += 1
        
        # Recent metrics summary
        recent_metrics = {}
        for metric_key, metric_deque in self.metrics.items():
            if metric_deque:
                recent_metric = metric_deque[-1]
                recent_metrics[metric_key] = {
                    "value": recent_metric.value,
                    "unit": recent_metric.unit,
                    "timestamp": recent_metric.timestamp,
                    "age_seconds": current_time - recent_metric.timestamp
                }
        
        return {
            "monitoring_status": "active" if self._monitoring_active else "inactive",
            "collection_interval": self.collection_interval,
            "component_health": health_summary,
            "active_alerts": len(active_alerts),
            "alerts_by_severity": alerts_by_severity,
            "recent_metrics": recent_metrics,
            "anomaly_count": len(self.anomaly_history),
            "monitored_components": list(self.component_health.keys()),
            "uptime": current_time - getattr(self, '_start_time', current_time)
        }
    
    def get_metric_history(self, 
                          metric_key: str, 
                          hours_back: int = 1) -> List[Dict[str, Any]]:
        """Get historical data for specific metric."""
        
        cutoff_time = time.time() - (hours_back * 3600)
        metric_deque = self.metrics.get(metric_key, deque())
        
        history = []
        for metric in metric_deque:
            if metric.timestamp >= cutoff_time:
                history.append({
                    "timestamp": metric.timestamp,
                    "value": metric.value,
                    "unit": metric.unit,
                    "tags": metric.tags
                })
        
        return history
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        
        if format_type == "json":
            export_data = {
                "export_timestamp": time.time(),
                "metrics": {},
                "component_health": self.component_health,
                "alerts": self.alerts,
                "anomalies": self.anomaly_history
            }
            
            # Convert metrics to serializable format
            for metric_key, metric_deque in self.metrics.items():
                export_data["metrics"][metric_key] = [
                    {
                        "timestamp": m.timestamp,
                        "value": m.value,
                        "unit": m.unit,
                        "tags": m.tags
                    } for m in metric_deque
                ]
            
            return json.dumps(export_data, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")