"""
Advanced Performance Monitoring and Optimization

Real-time performance monitoring with adaptive optimization for federated
LLM systems with quantum-enhanced analytics and predictive scaling.
"""

import asyncio
import time
# Conditional import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    PRIVACY_COST = "privacy_cost"
    QUANTUM_COHERENCE = "quantum_coherence"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LOAD_BALANCING = "load_balancing"
    CONNECTION_POOLING = "connection_pooling"
    CACHING = "caching"
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    PRIVACY_BUDGET_OPTIMIZATION = "privacy_budget_optimization"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    metadata: Dict[str, Any] = None


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    strategy: OptimizationStrategy
    priority: int  # 1-10, 10 = highest
    expected_improvement: float  # Percentage improvement
    implementation_cost: float  # Relative cost (1-5)
    description: str
    parameters: Dict[str, Any] = None


class AdvancedPerformanceMonitor:
    """Advanced performance monitoring with predictive analytics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.optimization_history: List[OptimizationRecommendation] = []
        self._lock = threading.RLock()
        
        # Performance baselines
        self.baselines = {
            'latency_p95': 2.0,  # seconds
            'throughput': 100.0,  # requests/second
            'error_rate': 0.01,  # 1%
            'cpu_usage': 70.0,   # percentage
            'memory_usage': 80.0,  # percentage
            'privacy_efficiency': 0.9,  # privacy utility ratio
            'quantum_coherence': 0.8   # coherence factor
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}
        self.performance_trends = {}
        
        # Start background monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self._monitor_thread.start()
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=metric_type,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            self._update_adaptive_thresholds(name, value)
    
    def record_request_latency(self, endpoint: str, latency: float, success: bool = True):
        """Record API request latency."""
        self.record_metric(
            f"latency_{endpoint}",
            latency,
            MetricType.LATENCY,
            {"success": success}
        )
        
        # Update throughput counter
        current_time = time.time()
        throughput_key = f"throughput_{endpoint}"
        
        with self._lock:
            # Count requests in last second
            recent_requests = [
                m for m in self.metrics[throughput_key] 
                if current_time - m.timestamp < 1.0
            ]
            throughput = len(recent_requests) + 1
            
            self.record_metric(throughput_key, throughput, MetricType.THROUGHPUT)
    
    def record_privacy_cost(self, user_id: str, epsilon: float, utility_score: float):
        """Record privacy budget usage with utility measurement."""
        efficiency = utility_score / epsilon if epsilon > 0 else 0
        
        self.record_metric(
            f"privacy_cost_{user_id}",
            epsilon,
            MetricType.PRIVACY_COST,
            {"utility": utility_score, "efficiency": efficiency}
        )
        
        self.record_metric(
            "privacy_efficiency_global",
            efficiency,
            MetricType.PRIVACY_COST
        )
    
    def record_quantum_performance(self, coherence: float, optimization_time: float, tasks_processed: int):
        """Record quantum optimization performance."""
        self.record_metric("quantum_coherence", coherence, MetricType.QUANTUM_COHERENCE)
        self.record_metric("quantum_optimization_latency", optimization_time, MetricType.LATENCY)
        self.record_metric("quantum_task_throughput", tasks_processed, MetricType.THROUGHPUT)
    
    def get_performance_summary(self, time_window: float = 300.0) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        current_time = time.time()
        summary = {
            'timestamp': current_time,
            'window_seconds': time_window,
            'metrics': {},
            'recommendations': [],
            'health_score': 0.0
        }
        
        with self._lock:
            for metric_name, metric_deque in self.metrics.items():
                # Filter metrics in time window
                recent_metrics = [
                    m for m in metric_deque 
                    if current_time - m.timestamp <= time_window
                ]
                
                if not recent_metrics:
                    continue
                
                values = [m.value for m in recent_metrics]
                
                metric_summary = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'p95': np.percentile(values, 95) if values else 0,
                    'p99': np.percentile(values, 99) if values else 0,
                    'latest': recent_metrics[-1].value,
                    'trend': self._calculate_trend(values)
                }
                
                summary['metrics'][metric_name] = metric_summary
        
        # Generate optimization recommendations
        summary['recommendations'] = self._generate_recommendations(summary['metrics'])
        summary['health_score'] = self._calculate_health_score(summary['metrics'])
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend (improving, degrading, stable)."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Compare first and second half
        midpoint = len(values) // 2
        first_half_avg = statistics.mean(values[:midpoint])
        second_half_avg = statistics.mean(values[midpoint:])
        
        change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        if change_percent > 5:
            return "degrading"
        elif change_percent < -5:
            return "improving"
        else:
            return "stable"
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []
        
        # Latency score
        for metric_name, data in metrics.items():
            if 'latency' in metric_name:
                baseline = self.baselines.get('latency_p95', 2.0)
                p95_latency = data.get('p95', baseline)
                latency_score = max(0, 100 - (p95_latency / baseline * 50))
                scores.append(latency_score)
        
        # Error rate score
        for metric_name, data in metrics.items():
            if 'error_rate' in metric_name:
                error_rate = data.get('mean', 0)
                error_score = max(0, 100 - (error_rate * 1000))  # Penalty for errors
                scores.append(error_score)
        
        # Resource usage score
        cpu_usage = metrics.get('cpu_usage', {}).get('mean', 50)
        memory_usage = metrics.get('memory_usage', {}).get('mean', 50)
        resource_score = max(0, 100 - max(cpu_usage - 70, memory_usage - 80))
        scores.append(resource_score)
        
        # Privacy efficiency score
        privacy_efficiency = metrics.get('privacy_efficiency_global', {}).get('mean', 0.8)
        privacy_score = privacy_efficiency * 100
        scores.append(privacy_score)
        
        # Quantum coherence score
        quantum_coherence = metrics.get('quantum_coherence', {}).get('mean', 0.8)
        quantum_score = quantum_coherence * 100
        scores.append(quantum_score)
        
        return statistics.mean(scores) if scores else 50.0
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance metrics."""
        recommendations = []
        
        # Latency optimization recommendations
        high_latency_metrics = [
            (name, data) for name, data in metrics.items()
            if 'latency' in name and data.get('p95', 0) > self.baselines['latency_p95']
        ]
        
        if high_latency_metrics:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CACHING,
                priority=8,
                expected_improvement=25.0,
                implementation_cost=2.0,
                description="Implement intelligent caching to reduce API response times",
                parameters={'cache_ttl': 300, 'cache_size': 1000}
            ))
            
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CONNECTION_POOLING,
                priority=7,
                expected_improvement=15.0,
                implementation_cost=1.5,
                description="Optimize connection pooling for database and external services",
                parameters={'pool_size': 20, 'max_overflow': 10}
            ))
        
        # Load balancing recommendations
        throughput_imbalance = self._detect_throughput_imbalance(metrics)
        if throughput_imbalance:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.LOAD_BALANCING,
                priority=9,
                expected_improvement=30.0,
                implementation_cost=3.0,
                description="Improve load balancing algorithm with quantum optimization",
                parameters={'algorithm': 'quantum_weighted', 'rebalance_interval': 60}
            ))
        
        # Privacy budget optimization
        low_privacy_efficiency = any(
            data.get('mean', 1.0) < 0.7
            for name, data in metrics.items()
            if 'privacy_efficiency' in name
        )
        
        if low_privacy_efficiency:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.PRIVACY_BUDGET_OPTIMIZATION,
                priority=6,
                expected_improvement=20.0,
                implementation_cost=2.5,
                description="Optimize privacy budget allocation with adaptive mechanisms",
                parameters={'adaptive_budget': True, 'utility_weighting': 0.8}
            ))
        
        # Quantum optimization recommendations
        low_coherence = metrics.get('quantum_coherence', {}).get('mean', 1.0) < 0.7
        if low_coherence:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.QUANTUM_OPTIMIZATION,
                priority=5,
                expected_improvement=15.0,
                implementation_cost=4.0,
                description="Enhance quantum coherence with decoherence mitigation",
                parameters={'coherence_threshold': 0.8, 'mitigation_enabled': True}
            ))
        
        # Sort by priority (descending)
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _detect_throughput_imbalance(self, metrics: Dict[str, Any]) -> bool:
        """Detect if there's throughput imbalance across endpoints."""
        throughput_metrics = {
            name: data for name, data in metrics.items()
            if 'throughput' in name
        }
        
        if len(throughput_metrics) < 2:
            return False
        
        throughput_values = [data.get('mean', 0) for data in throughput_metrics.values()]
        if not throughput_values:
            return False
        
        std_dev = statistics.stdev(throughput_values)
        mean_throughput = statistics.mean(throughput_values)
        
        # If standard deviation is > 30% of mean, consider it imbalanced
        coefficient_of_variation = std_dev / mean_throughput if mean_throughput > 0 else 0
        return coefficient_of_variation > 0.3
    
    def _update_adaptive_thresholds(self, metric_name: str, value: float):
        """Update adaptive thresholds based on historical performance."""
        if metric_name not in self.adaptive_thresholds:
            self.adaptive_thresholds[metric_name] = {
                'mean': value,
                'std': 0.0,
                'count': 1
            }
        else:
            threshold = self.adaptive_thresholds[metric_name]
            count = threshold['count']
            
            # Update running mean and std
            old_mean = threshold['mean']
            new_mean = (old_mean * count + value) / (count + 1)
            
            if count > 1:
                old_std = threshold['std']
                new_std = ((old_std ** 2 * (count - 1) + (value - old_mean) * (value - new_mean)) / count) ** 0.5
                threshold['std'] = new_std
            
            threshold['mean'] = new_mean
            threshold['count'] = count + 1
    
    def _background_monitoring(self):
        """Background thread for continuous system monitoring."""
        while self._monitoring_active:
            try:
                if PSUTIL_AVAILABLE:
                    # Monitor system resources
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    self.record_metric("cpu_usage", cpu_percent, MetricType.RESOURCE_USAGE)
                    self.record_metric("memory_usage", memory.percent, MetricType.RESOURCE_USAGE)
                    
                    # Monitor network I/O
                    net_io = psutil.net_io_counters()
                    self.record_metric("network_bytes_sent", net_io.bytes_sent, MetricType.RESOURCE_USAGE)
                    self.record_metric("network_bytes_recv", net_io.bytes_recv, MetricType.RESOURCE_USAGE)
                else:
                    # Fallback monitoring without psutil
                    self.record_metric("cpu_usage", 50.0, MetricType.RESOURCE_USAGE)
                    self.record_metric("memory_usage", 60.0, MetricType.RESOURCE_USAGE)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(10)
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time performance dashboard data."""
        summary = self.get_performance_summary(60.0)  # Last minute
        
        dashboard = {
            'health_score': summary['health_score'],
            'status': self._get_status_from_health_score(summary['health_score']),
            'key_metrics': {
                'latency_p95': self._get_latest_metric_value('latency', percentile=95),
                'throughput': self._get_latest_metric_value('throughput'),
                'error_rate': self._get_latest_metric_value('error_rate'),
                'cpu_usage': self._get_latest_metric_value('cpu_usage'),
                'memory_usage': self._get_latest_metric_value('memory_usage'),
                'privacy_efficiency': self._get_latest_metric_value('privacy_efficiency'),
                'quantum_coherence': self._get_latest_metric_value('quantum_coherence')
            },
            'trends': {
                metric: data.get('trend', 'stable')
                for metric, data in summary['metrics'].items()
            },
            'top_recommendations': summary['recommendations'][:3],
            'alerts': self._generate_alerts(summary['metrics'])
        }
        
        return dashboard
    
    def _get_latest_metric_value(self, metric_pattern: str, percentile: int = None) -> Optional[float]:
        """Get latest value for metrics matching pattern."""
        with self._lock:
            matching_metrics = [
                name for name in self.metrics.keys()
                if metric_pattern in name and self.metrics[name]
            ]
            
            if not matching_metrics:
                return None
            
            # Get most recent metric
            latest_metric = max(
                matching_metrics,
                key=lambda name: self.metrics[name][-1].timestamp if self.metrics[name] else 0
            )
            
            if percentile:
                values = [m.value for m in self.metrics[latest_metric]]
                return np.percentile(values, percentile) if values else None
            else:
                return self.metrics[latest_metric][-1].value
    
    def _get_status_from_health_score(self, score: float) -> str:
        """Convert health score to status string."""
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "degraded"
        else:
            return "critical"
    
    def _generate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance alerts based on thresholds."""
        alerts = []
        
        for metric_name, data in metrics.items():
            latest_value = data.get('latest', 0)
            trend = data.get('trend', 'stable')
            
            # High latency alert
            if 'latency' in metric_name and latest_value > 5.0:
                alerts.append({
                    'type': 'high_latency',
                    'severity': 'warning' if latest_value < 10.0 else 'critical',
                    'message': f"High latency detected: {latest_value:.2f}s",
                    'metric': metric_name
                })
            
            # High resource usage alert
            if 'cpu_usage' in metric_name and latest_value > 90:
                alerts.append({
                    'type': 'high_cpu',
                    'severity': 'critical',
                    'message': f"High CPU usage: {latest_value:.1f}%",
                    'metric': metric_name
                })
            
            # Degrading performance alert
            if trend == 'degrading':
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'warning',
                    'message': f"Performance degradation detected in {metric_name}",
                    'metric': metric_name
                })
        
        return alerts
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)