"""
Metrics Collection and Privacy Dashboard

Implements comprehensive metrics collection, privacy tracking, and monitoring
dashboards with Prometheus integration.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import asyncio
import logging

# Conditional prometheus import
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None
    Histogram = None
    Gauge = None
    Info = None
    start_http_server = None
    PROMETHEUS_AVAILABLE = False


@dataclass
class PrivacyMetric:
    """Privacy budget metric."""
    user_id: str
    department: str
    epsilon_spent: float
    delta_spent: float
    query_type: str
    timestamp: float
    node_id: str


@dataclass
class PerformanceMetric:
    """Performance metric."""
    metric_name: str
    value: float
    labels: Dict[str, str]
    timestamp: float


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus
        self.privacy_metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.performance_metrics: deque = deque(maxlen=10000)
        self.system_stats: Dict[str, Any] = {}
        
        # Thread-safe locks
        self._privacy_lock = threading.Lock()
        self._performance_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        # Prometheus metrics
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        elif enable_prometheus and not PROMETHEUS_AVAILABLE:
            logging.getLogger(__name__).warning("Prometheus metrics requested but prometheus_client not available.")
            
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # Privacy metrics
        self.privacy_budget_spent = Counter(
            'privacy_budget_spent_total',
            'Total privacy budget spent',
            ['user_id', 'department', 'query_type', 'node_id']
        )
        
        self.privacy_queries_total = Counter(
            'privacy_queries_total',
            'Total privacy-preserving queries',
            ['department', 'query_type', 'status']
        )
        
        # Performance metrics
        self.inference_latency = Histogram(
            'inference_latency_seconds',
            'Inference request latency',
            ['model_name', 'node_id', 'consensus'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, float('inf'))
        )
        
        self.active_requests = Gauge(
            'active_requests',
            'Number of active inference requests'
        )
        
        self.node_health = Gauge(
            'node_health_status',
            'Node health status (1=healthy, 0=unhealthy)',
            ['node_id', 'department']
        )
        
        # System metrics
        self.privacy_budget_remaining = Gauge(
            'privacy_budget_remaining',
            'Remaining privacy budget',
            ['department']
        )
        
        self.federated_training_rounds = Counter(
            'federated_training_rounds_total',
            'Total federated training rounds completed',
            ['model_name', 'status']
        )
        
        # System info
        self.system_info = Info(
            'federated_dp_llm_info',
            'System information'
        )
        
        self.system_info.info({
            'version': '0.1.0',
            'component': 'federated_dp_llm_router'
        })
    
    def record_privacy_metric(self, metric: PrivacyMetric):
        """Record a privacy metric."""
        with self._privacy_lock:
            self.privacy_metrics.append(metric)
        
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            self.privacy_budget_spent.labels(
                user_id=metric.user_id,
                department=metric.department,
                query_type=metric.query_type,
                node_id=metric.node_id
            ).inc(metric.epsilon_spent)
            
            self.privacy_queries_total.labels(
                department=metric.department,
                query_type=metric.query_type,
                status='success'
            ).inc()
        
        self.logger.info(f"Privacy metric recorded: {metric.user_id} spent {metric.epsilon_spent} epsilon")
    
    def record_performance_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self._performance_lock:
            self.performance_metrics.append(metric)
        
        if self.enable_prometheus and PROMETHEUS_AVAILABLE and metric.metric_name == "inference_latency":
            self.inference_latency.labels(
                model_name=metric.labels.get('model_name', 'unknown'),
                node_id=metric.labels.get('node_id', 'unknown'),
                consensus=metric.labels.get('consensus', 'false')
            ).observe(metric.value)
    
    def update_system_stats(self, stats: Dict[str, Any]):
        """Update system statistics."""
        with self._stats_lock:
            self.system_stats.update(stats)
            self.system_stats['last_updated'] = time.time()
        
        if self.enable_prometheus:
            # Update Prometheus gauges
            if 'active_requests' in stats:
                self.active_requests.set(stats['active_requests'])
            
            if 'node_health' in stats:
                for node_id, health_info in stats['node_health'].items():
                    is_healthy = 1 if health_info.get('healthy', False) else 0
                    department = health_info.get('department', 'unknown')
                    self.node_health.labels(node_id=node_id, department=department).set(is_healthy)
            
            if 'privacy_budgets' in stats:
                for dept, budget_info in stats['privacy_budgets'].items():
                    remaining = budget_info.get('remaining', 0)
                    self.privacy_budget_remaining.labels(department=dept).set(remaining)
    
    def get_privacy_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get privacy metrics summary for time window (seconds)."""
        cutoff_time = time.time() - time_window
        
        with self._privacy_lock:
            recent_metrics = [
                m for m in self.privacy_metrics
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {
                'total_queries': 0,
                'total_epsilon_spent': 0.0,
                'by_department': {},
                'by_query_type': {},
                'average_epsilon_per_query': 0.0
            }
        
        # Aggregate metrics
        total_epsilon = sum(m.epsilon_spent for m in recent_metrics)
        dept_epsilon = defaultdict(float)
        query_type_epsilon = defaultdict(float)
        
        for metric in recent_metrics:
            dept_epsilon[metric.department] += metric.epsilon_spent
            query_type_epsilon[metric.query_type] += metric.epsilon_spent
        
        return {
            'total_queries': len(recent_metrics),
            'total_epsilon_spent': total_epsilon,
            'by_department': dict(dept_epsilon),
            'by_query_type': dict(query_type_epsilon),
            'average_epsilon_per_query': total_epsilon / len(recent_metrics)
        }
    
    def get_performance_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get performance metrics summary."""
        cutoff_time = time.time() - time_window
        
        with self._performance_lock:
            recent_metrics = [
                m for m in self.performance_metrics
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {'total_metrics': 0}
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.metric_name].append(metric.value)
        
        summary = {'total_metrics': len(recent_metrics)}
        
        for metric_name, values in metrics_by_name.items():
            summary[metric_name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'p95': sorted(values)[int(0.95 * len(values))] if values else 0
            }
        
        return summary
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        privacy_summary = self.get_privacy_summary(86400)  # 24 hours
        performance_summary = self.get_performance_summary(86400)
        
        with self._stats_lock:
            system_stats = self.system_stats.copy()
        
        export_data = {
            'timestamp': time.time(),
            'privacy_summary': privacy_summary,
            'performance_summary': performance_summary,
            'system_stats': system_stats
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def start_prometheus_server(self, port: int = 8090):
        """Start Prometheus metrics server."""
        if not self.enable_prometheus:
            raise RuntimeError("Prometheus not enabled")
        
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError("prometheus_client not available. Install prometheus_client to use metrics server.")
        
        start_http_server(port)
        self.logger.info(f"Prometheus metrics server started on port {port}")


class PrivacyDashboard:
    """Privacy-focused monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 prometheus_url: str = None, grafana_url: str = None):
        self.metrics_collector = metrics_collector
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_privacy_consumption': 0.5,  # 50% of budget in 5 minutes
            'budget_depletion_warning': 0.8,  # 80% budget used
            'budget_depletion_critical': 0.95,  # 95% budget used
            'unusual_query_pattern': 10  # More than 10 queries per minute from single user
        }
        
        # Alert history
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def record_query(self, user_id: str, epsilon_spent: float, 
                    query_type: str, node: str, department: str = "general"):
        """Record a privacy query for dashboard tracking."""
        metric = PrivacyMetric(
            user_id=user_id,
            department=department,
            epsilon_spent=epsilon_spent,
            delta_spent=0.0,  # Simplified
            query_type=query_type,
            timestamp=time.time(),
            node_id=node
        )
        
        self.metrics_collector.record_privacy_metric(metric)
        
        # Check for alerts
        self._check_privacy_alerts(user_id, department, epsilon_spent)
    
    def _check_privacy_alerts(self, user_id: str, department: str, epsilon_spent: float):
        """Check for privacy-related alerts."""
        current_time = time.time()
        
        # Check for high consumption rate
        recent_consumption = self._get_recent_consumption(user_id, 300)  # 5 minutes
        if recent_consumption > self.alert_thresholds['high_privacy_consumption']:
            self._trigger_alert(
                alert_type="high_privacy_consumption",
                severity="warning",
                message=f"High privacy consumption detected for user {user_id}",
                details={
                    "user_id": user_id,
                    "department": department,
                    "consumption_rate": recent_consumption,
                    "time_window": "5_minutes"
                }
            )
        
        # Check for unusual query patterns
        recent_queries = self._get_recent_query_count(user_id, 60)  # 1 minute
        if recent_queries > self.alert_thresholds['unusual_query_pattern']:
            self._trigger_alert(
                alert_type="unusual_query_pattern",
                severity="warning",
                message=f"Unusual query pattern detected for user {user_id}",
                details={
                    "user_id": user_id,
                    "query_count": recent_queries,
                    "time_window": "1_minute"
                }
            )
    
    def _get_recent_consumption(self, user_id: str, time_window: int) -> float:
        """Get recent privacy consumption for user."""
        cutoff_time = time.time() - time_window
        
        with self.metrics_collector._privacy_lock:
            recent_metrics = [
                m for m in self.metrics_collector.privacy_metrics
                if m.user_id == user_id and m.timestamp >= cutoff_time
            ]
        
        return sum(m.epsilon_spent for m in recent_metrics)
    
    def _get_recent_query_count(self, user_id: str, time_window: int) -> int:
        """Get recent query count for user."""
        cutoff_time = time.time() - time_window
        
        with self.metrics_collector._privacy_lock:
            recent_metrics = [
                m for m in self.metrics_collector.privacy_metrics
                if m.user_id == user_id and m.timestamp >= cutoff_time
            ]
        
        return len(recent_metrics)
    
    def _trigger_alert(self, alert_type: str, severity: str, message: str, details: Dict[str, Any]):
        """Trigger a privacy alert."""
        alert_id = f"{alert_type}_{details.get('user_id', 'unknown')}_{int(time.time())}"
        
        alert = {
            'alert_id': alert_id,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'details': details,
            'timestamp': time.time(),
            'status': 'active'
        }
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert.copy())
        
        self.logger.warning(f"Privacy alert triggered: {message}")
        
        # Keep alert history manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def setup_alerts(self, alert_configs: List[Dict[str, Any]]):
        """Setup custom alert configurations."""
        for config in alert_configs:
            alert_name = config.get('name')
            condition = config.get('condition')
            severity = config.get('severity', 'warning')
            
            if alert_name in self.alert_thresholds:
                # Update existing threshold
                if 'threshold' in config:
                    self.alert_thresholds[alert_name] = config['threshold']
                
                self.logger.info(f"Updated alert threshold for {alert_name}")
            else:
                self.logger.warning(f"Unknown alert type: {alert_name}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        privacy_summary = self.metrics_collector.get_privacy_summary(3600)  # 1 hour
        performance_summary = self.metrics_collector.get_performance_summary(3600)
        
        # Active alerts
        active_alerts_count = len(self.active_alerts)
        critical_alerts = [
            alert for alert in self.active_alerts.values()
            if alert['severity'] == 'critical'
        ]
        
        # Recent trends
        hourly_consumption = self._get_hourly_consumption_trend()
        top_consumers = self._get_top_privacy_consumers()
        
        return {
            'timestamp': time.time(),
            'privacy_overview': privacy_summary,
            'performance_overview': performance_summary,
            'alerts': {
                'active_count': active_alerts_count,
                'critical_alerts': critical_alerts,
                'recent_alerts': self.alert_history[-10:]  # Last 10 alerts
            },
            'trends': {
                'hourly_consumption': hourly_consumption,
                'top_consumers': top_consumers
            },
            'system_health': self._get_system_health_summary()
        }
    
    def _get_hourly_consumption_trend(self) -> List[Dict[str, Any]]:
        """Get hourly privacy consumption trend."""
        current_time = time.time()
        hourly_data = []
        
        for i in range(24):  # Last 24 hours
            hour_start = current_time - (i + 1) * 3600
            hour_end = current_time - i * 3600
            
            with self.metrics_collector._privacy_lock:
                hour_metrics = [
                    m for m in self.metrics_collector.privacy_metrics
                    if hour_start <= m.timestamp < hour_end
                ]
            
            total_epsilon = sum(m.epsilon_spent for m in hour_metrics)
            
            hourly_data.append({
                'hour': hour_start,
                'queries': len(hour_metrics),
                'epsilon_spent': total_epsilon
            })
        
        return list(reversed(hourly_data))  # Chronological order
    
    def _get_top_privacy_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top privacy budget consumers."""
        cutoff_time = time.time() - 86400  # 24 hours
        
        with self.metrics_collector._privacy_lock:
            recent_metrics = [
                m for m in self.metrics_collector.privacy_metrics
                if m.timestamp >= cutoff_time
            ]
        
        # Aggregate by user
        user_consumption = defaultdict(lambda: {'epsilon': 0.0, 'queries': 0, 'department': 'unknown'})
        
        for metric in recent_metrics:
            user_consumption[metric.user_id]['epsilon'] += metric.epsilon_spent
            user_consumption[metric.user_id]['queries'] += 1
            user_consumption[metric.user_id]['department'] = metric.department
        
        # Sort by consumption
        top_users = sorted(
            user_consumption.items(),
            key=lambda x: x[1]['epsilon'],
            reverse=True
        )[:limit]
        
        return [
            {
                'user_id': user_id,
                'epsilon_spent': data['epsilon'],
                'query_count': data['queries'],
                'department': data['department']
            }
            for user_id, data in top_users
        ]
    
    def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        with self.metrics_collector._stats_lock:
            system_stats = self.metrics_collector.system_stats.copy()
        
        # Calculate health score
        health_score = 100.0
        
        # Deduct for unhealthy nodes
        node_health = system_stats.get('node_health', {})
        total_nodes = len(node_health)
        if total_nodes > 0:
            healthy_nodes = sum(1 for h in node_health.values() if h.get('healthy', False))
            node_health_ratio = healthy_nodes / total_nodes
            health_score *= node_health_ratio
        
        # Deduct for high active alerts
        if len(self.active_alerts) > 5:
            health_score *= 0.8
        
        # Deduct for critical alerts
        critical_alerts = [a for a in self.active_alerts.values() if a['severity'] == 'critical']
        if critical_alerts:
            health_score *= 0.5
        
        return {
            'overall_health_score': health_score,
            'total_nodes': total_nodes,
            'healthy_nodes': sum(1 for h in node_health.values() if h.get('healthy', False)),
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len(critical_alerts),
            'last_updated': system_stats.get('last_updated', 0),
            'uptime': time.time() - system_stats.get('start_time', time.time())
        }
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = None) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['status'] = 'resolved'
            self.active_alerts[alert_id]['resolved_at'] = time.time()
            if resolution_notes:
                self.active_alerts[alert_id]['resolution_notes'] = resolution_notes
            
            # Move to history and remove from active
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved")
            return True
        
        return False
    
    def record_privacy_spend(self, user_id: str, epsilon_spent: float, remaining_budget: float):
        """Record privacy budget spending metrics."""
        try:
            self.privacy_budget_spent.labels(user_id=user_id).inc(epsilon_spent)
            self.privacy_budget_remaining.labels(user_id=user_id).set(remaining_budget)
        except AttributeError:
            # Create metrics if they don't exist
            self.privacy_budget_spent = Counter('privacy_budget_spent_total', 'Total privacy budget spent', ['user_id'])
            self.privacy_budget_remaining = Gauge('privacy_budget_remaining', 'Remaining privacy budget', ['user_id'])
            self.privacy_budget_spent.labels(user_id=user_id).inc(epsilon_spent)
            self.privacy_budget_remaining.labels(user_id=user_id).set(remaining_budget)
    
    def trigger_privacy_alert(self, user_id: str, current_budget: float, alert_type: str):
        """Trigger privacy-related alert."""
        alert_id = f"privacy_{user_id}_{int(time.time())}"
        self.trigger_alert(
            alert_id=alert_id,
            severity="warning",
            component="privacy_accountant",
            message=f"Privacy budget low for user {user_id}: {current_budget:.3f}",
            details={"user_id": user_id, "current_budget": current_budget, "alert_type": alert_type}
        )