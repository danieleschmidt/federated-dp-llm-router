"""
Self-Healing System

Automatically detects, diagnoses, and recovers from system failures
and performance degradations using adaptive recovery strategies.
"""

import asyncio
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from collections import deque, defaultdict
import logging
from pathlib import Path

from ..quantum_planning.numpy_fallback import get_numpy_backend
from ..monitoring.metrics import MetricsCollector
from ..monitoring.health_check import HealthChecker
from ..resilience.circuit_breaker import CircuitBreaker
from ..core.error_handling import ErrorHandler

HAS_NUMPY, np = get_numpy_backend()
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class RecoveryActionType(Enum):
    """Types of recovery actions available."""
    RESTART_COMPONENT = "restart_component"
    SCALE_UP_RESOURCES = "scale_up_resources"
    FAILOVER_NODE = "failover_node"
    REDUCE_LOAD = "reduce_load"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTIONS = "reset_connections"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    GRADUAL_RECOVERY = "gradual_recovery"
    ROLLBACK_CONFIG = "rollback_config"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    disk_utilization: float
    network_latency: float
    error_rate: float
    response_time_p95: float
    throughput: float
    active_connections: int
    queue_size: int
    privacy_budget_remaining: float
    node_availability: Dict[str, bool]
    component_health: Dict[str, HealthStatus]
    
    def overall_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        scores = []
        
        # Resource utilization (optimal around 70%)
        cpu_score = 1.0 - abs(self.cpu_utilization - 0.7)
        memory_score = 1.0 - abs(self.memory_utilization - 0.7) 
        disk_score = 1.0 - min(self.disk_utilization, 0.9)  # Lower disk usage is better
        scores.extend([cpu_score, memory_score, disk_score])
        
        # Performance metrics
        error_score = 1.0 - self.error_rate
        latency_score = max(0.0, 1.0 - (self.network_latency / 1000.0))  # Normalize to 1s
        response_score = max(0.0, 1.0 - (self.response_time_p95 / 5000.0))  # Normalize to 5s
        throughput_score = min(1.0, self.throughput / 1000.0)  # Normalize to 1000 RPS
        scores.extend([error_score, latency_score, response_score, throughput_score])
        
        # System capacity
        queue_score = max(0.0, 1.0 - (self.queue_size / 10000.0))  # Normalize to 10k items
        privacy_score = self.privacy_budget_remaining
        scores.extend([queue_score, privacy_score])
        
        # Node availability
        if self.node_availability:
            availability_score = sum(self.node_availability.values()) / len(self.node_availability)
            scores.append(availability_score)
        
        return sum(scores) / len(scores) if scores else 0.0


@dataclass  
class RecoveryAction:
    """Recovery action with metadata."""
    action_type: RecoveryActionType
    target_component: str
    parameters: Dict[str, Any]
    priority: int  # 1 = highest priority
    estimated_recovery_time: float  # seconds
    risk_level: str  # "low", "medium", "high"
    prerequisites: List[str]  # Required conditions
    
    def can_execute(self, current_state: Dict[str, Any]) -> bool:
        """Check if action can be executed given current state."""
        for prereq in self.prerequisites:
            if prereq not in current_state or not current_state[prereq]:
                return False
        return True


@dataclass
class IncidentReport:
    """Incident detection and recovery report."""
    incident_id: str
    timestamp: float
    detected_issues: List[str]
    root_cause_analysis: str
    recovery_actions_taken: List[RecoveryAction]
    recovery_time_seconds: float
    success: bool
    lessons_learned: str
    prevented_future_incidents: bool


class SelfHealingSystem:
    """Advanced self-healing system with predictive recovery."""
    
    def __init__(self, 
                 health_check_interval: float = 30.0,
                 critical_threshold: float = 0.3,
                 degraded_threshold: float = 0.6,
                 recovery_timeout: float = 300.0,
                 max_concurrent_recoveries: int = 3):
        
        self.health_check_interval = health_check_interval
        self.critical_threshold = critical_threshold
        self.degraded_threshold = degraded_threshold
        self.recovery_timeout = recovery_timeout
        self.max_concurrent_recoveries = max_concurrent_recoveries
        
        # Health monitoring
        self.current_health: Optional[HealthMetrics] = None
        self.health_history: deque = deque(maxlen=1000)
        self.health_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Recovery management
        self.available_actions: List[RecoveryAction] = []
        self.active_recoveries: Dict[str, RecoveryAction] = {}
        self.recovery_history: List[IncidentReport] = []
        self.learned_patterns: Dict[str, List[RecoveryAction]] = {}
        
        # Monitoring components
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery state
        self.system_status: HealthStatus = HealthStatus.HEALTHY
        self.last_health_check: float = 0.0
        self.monitoring_active: bool = False
        self.recovery_in_progress: bool = False
        
        # Initialize recovery actions
        self._initialize_recovery_actions()
        
        logger.info("Self-healing system initialized")
    
    def _initialize_recovery_actions(self):
        """Initialize available recovery actions."""
        self.available_actions = [
            RecoveryAction(
                action_type=RecoveryActionType.RESTART_COMPONENT,
                target_component="routing_service",
                parameters={"graceful": True, "timeout": 30},
                priority=2,
                estimated_recovery_time=45.0,
                risk_level="medium",
                prerequisites=["component_responsive"]
            ),
            RecoveryAction(
                action_type=RecoveryActionType.SCALE_UP_RESOURCES,
                target_component="worker_nodes", 
                parameters={"scale_factor": 1.5, "max_instances": 10},
                priority=3,
                estimated_recovery_time=120.0,
                risk_level="low",
                prerequisites=["resources_available"]
            ),
            RecoveryAction(
                action_type=RecoveryActionType.FAILOVER_NODE,
                target_component="primary_node",
                parameters={"backup_node": "secondary", "sync_data": True},
                priority=1,
                estimated_recovery_time=60.0,
                risk_level="high",
                prerequisites=["backup_available", "data_consistent"]
            ),
            RecoveryAction(
                action_type=RecoveryActionType.REDUCE_LOAD,
                target_component="load_balancer",
                parameters={"reduction_factor": 0.5, "queue_limit": 1000},
                priority=4,
                estimated_recovery_time=10.0,
                risk_level="low",
                prerequisites=[]
            ),
            RecoveryAction(
                action_type=RecoveryActionType.CLEAR_CACHE,
                target_component="cache_service",
                parameters={"preserve_critical": True, "gradual": True},
                priority=5,
                estimated_recovery_time=30.0,
                risk_level="low",
                prerequisites=[]
            ),
            RecoveryAction(
                action_type=RecoveryActionType.RESET_CONNECTIONS,
                target_component="connection_pool",
                parameters={"graceful_close": True, "reconnect_delay": 5},
                priority=3,
                estimated_recovery_time=20.0,
                risk_level="medium",
                prerequisites=["connections_stable"]
            ),
            RecoveryAction(
                action_type=RecoveryActionType.ROLLBACK_CONFIG,
                target_component="system_config",
                parameters={"versions_back": 1, "verify_health": True},
                priority=2,
                estimated_recovery_time=90.0,
                risk_level="medium",
                prerequisites=["backup_config_available"]
            )
        ]
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.monitoring_active = True
        logger.info("Starting self-healing monitoring")
        
        while self.monitoring_active:
            try:
                await self._perform_health_check()
                await self._analyze_health_trends()
                await self._execute_proactive_measures()
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        logger.info("Stopped self-healing monitoring")
    
    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            # Collect system metrics
            health_metrics = await self._collect_health_metrics()
            
            self.current_health = health_metrics
            self.health_history.append(health_metrics)
            self.last_health_check = time.time()
            
            # Update health trends
            self._update_health_trends(health_metrics)
            
            # Determine system status
            overall_score = health_metrics.overall_health_score()
            
            if overall_score < self.critical_threshold:
                new_status = HealthStatus.CRITICAL
            elif overall_score < self.degraded_threshold:
                new_status = HealthStatus.DEGRADED
            else:
                new_status = HealthStatus.HEALTHY
            
            # Trigger recovery if status deteriorated
            if new_status != self.system_status and new_status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                await self._trigger_recovery(health_metrics, new_status)
            
            self.system_status = new_status
            
            # Record metrics
            self.metrics_collector.record_metric("system_health_score", overall_score)
            self.metrics_collector.record_metric("health_check_completed", 1)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.system_status = HealthStatus.FAILED
    
    async def _collect_health_metrics(self) -> HealthMetrics:
        """Collect comprehensive health metrics."""
        # In a real system, these would be collected from actual monitors
        # For demonstration, we'll simulate realistic metrics
        
        current_time = time.time()
        
        # Simulate some variability and potential issues
        base_cpu = 0.6 + 0.2 * (hash(str(current_time)) % 1000) / 1000
        base_memory = 0.7 + 0.15 * (hash(str(current_time + 1)) % 1000) / 1000
        base_error_rate = 0.02 + 0.03 * (hash(str(current_time + 2)) % 1000) / 1000
        
        # Check for degradation patterns
        if len(self.health_history) > 5:
            recent_scores = [h.overall_health_score() for h in list(self.health_history)[-5:]]
            if all(score < 0.7 for score in recent_scores):
                # System showing consistent degradation
                base_error_rate *= 2
                base_cpu = min(0.95, base_cpu * 1.3)
        
        node_availability = {
            f"node_{i}": (hash(f"node_{i}_{current_time}") % 100) > 5
            for i in range(4)
        }
        
        component_health = {
            "routing_service": HealthStatus.HEALTHY,
            "privacy_accountant": HealthStatus.HEALTHY,
            "load_balancer": HealthStatus.HEALTHY,
            "cache_service": HealthStatus.HEALTHY
        }
        
        # Simulate component issues based on overall health
        if base_error_rate > 0.04:
            component_health["routing_service"] = HealthStatus.DEGRADED
        if base_cpu > 0.85:
            component_health["load_balancer"] = HealthStatus.CRITICAL
        
        return HealthMetrics(
            timestamp=current_time,
            cpu_utilization=base_cpu,
            memory_utilization=base_memory,
            disk_utilization=0.4,
            network_latency=50 + 100 * base_error_rate,
            error_rate=base_error_rate,
            response_time_p95=200 + 500 * base_error_rate,
            throughput=max(50, 500 - 1000 * base_error_rate),
            active_connections=100 + int(50 * base_cpu),
            queue_size=int(1000 * base_error_rate),
            privacy_budget_remaining=max(0.1, 1.0 - base_error_rate * 5),
            node_availability=node_availability,
            component_health=component_health
        )
    
    def _update_health_trends(self, health_metrics: HealthMetrics):
        """Update health trend analysis."""
        trend_window = 20
        
        # Track key metrics over time
        metrics_to_track = [
            'cpu_utilization', 'memory_utilization', 'error_rate',
            'response_time_p95', 'throughput'
        ]
        
        for metric_name in metrics_to_track:
            if hasattr(health_metrics, metric_name):
                value = getattr(health_metrics, metric_name)
                trend_list = self.health_trends[metric_name]
                trend_list.append(value)
                
                # Keep only recent values
                if len(trend_list) > trend_window:
                    trend_list.pop(0)
    
    async def _analyze_health_trends(self):
        """Analyze health trends for predictive interventions."""
        if len(self.health_history) < 10:
            return
        
        predictions = {}
        
        # Analyze each metric trend
        for metric_name, trend_values in self.health_trends.items():
            if len(trend_values) >= 5:
                prediction = self._predict_metric_trend(metric_name, trend_values)
                predictions[metric_name] = prediction
        
        # Identify concerning trends
        concerning_trends = []
        
        for metric_name, prediction in predictions.items():
            if metric_name == 'error_rate' and prediction > 0.1:
                concerning_trends.append(f"Error rate trending upward: {prediction:.3f}")
            elif metric_name == 'response_time_p95' and prediction > 1000:
                concerning_trends.append(f"Response time degrading: {prediction:.1f}ms")
            elif metric_name in ['cpu_utilization', 'memory_utilization'] and prediction > 0.9:
                concerning_trends.append(f"{metric_name} trending high: {prediction:.2f}")
        
        if concerning_trends:
            logger.warning(f"Concerning trends detected: {concerning_trends}")
            await self._schedule_proactive_recovery(concerning_trends)
    
    def _predict_metric_trend(self, metric_name: str, values: List[float]) -> float:
        """Simple trend prediction using linear regression."""
        if len(values) < 3:
            return values[-1] if values else 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        if HAS_NUMPY:
            # Use numpy for more accurate calculation
            x_mean = np.mean(x_values)
            y_mean = np.mean(values)
            
            numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            if denominator != 0:
                slope = numerator / denominator
                # Predict 3 steps ahead
                predicted = y_mean + slope * (n + 2)
            else:
                predicted = values[-1]
        else:
            # Simple trend without numpy
            if len(values) >= 2:
                recent_change = values[-1] - values[-2]
                predicted = values[-1] + recent_change * 2  # Extrapolate
            else:
                predicted = values[-1]
        
        return max(0.0, predicted)
    
    async def _execute_proactive_measures(self):
        """Execute proactive measures before issues become critical."""
        if not self.current_health:
            return
        
        health_score = self.current_health.overall_health_score()
        
        # Proactive measures based on early warning signs
        if 0.6 < health_score < 0.8:
            # System showing early signs of stress
            await self._execute_preventive_actions()
        
        # Check circuit breakers and reset if appropriate
        for component, breaker in self.circuit_breakers.items():
            if breaker.state == "open" and time.time() - breaker.last_failure > 300:  # 5 minutes
                logger.info(f"Attempting to reset circuit breaker for {component}")
                breaker.reset()
    
    async def _execute_preventive_actions(self):
        """Execute low-risk preventive actions."""
        preventive_actions = [
            ("clear_cache", "Clearing non-critical caches"),
            ("optimize_connections", "Optimizing connection pools"),
            ("garbage_collection", "Triggering garbage collection")
        ]
        
        for action_name, description in preventive_actions:
            try:
                logger.info(f"Executing preventive action: {description}")
                await self._execute_recovery_action_by_name(action_name)
                
                # Small delay between actions
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.warning(f"Preventive action {action_name} failed: {e}")
    
    async def _trigger_recovery(self, health_metrics: HealthMetrics, status: HealthStatus):
        """Trigger recovery procedures based on health status."""
        if self.recovery_in_progress and len(self.active_recoveries) >= self.max_concurrent_recoveries:
            logger.warning("Recovery already in progress, skipping new recovery")
            return
        
        logger.warning(f"System health critical, triggering recovery. Status: {status.value}")
        
        incident_id = f"incident_{int(time.time())}"
        recovery_start = time.time()
        
        # Analyze issues and determine root cause
        issues = self._diagnose_issues(health_metrics)
        root_cause = self._determine_root_cause(issues, health_metrics)
        
        # Select recovery actions
        recovery_actions = self._select_recovery_actions(issues, root_cause, status)
        
        logger.info(f"Executing {len(recovery_actions)} recovery actions for {incident_id}")
        
        # Execute recovery actions
        executed_actions = []
        recovery_success = True
        
        for action in recovery_actions[:3]:  # Limit concurrent actions
            try:
                success = await self._execute_recovery_action(action)
                executed_actions.append(action)
                
                if not success:
                    recovery_success = False
                    logger.error(f"Recovery action failed: {action.action_type.value}")
                
            except Exception as e:
                logger.error(f"Recovery action execution failed: {e}")
                recovery_success = False
        
        recovery_time = time.time() - recovery_start
        
        # Generate incident report
        incident_report = IncidentReport(
            incident_id=incident_id,
            timestamp=recovery_start,
            detected_issues=issues,
            root_cause_analysis=root_cause,
            recovery_actions_taken=executed_actions,
            recovery_time_seconds=recovery_time,
            success=recovery_success,
            lessons_learned=self._generate_lessons_learned(issues, executed_actions, recovery_success),
            prevented_future_incidents=False  # Will be determined over time
        )
        
        self.recovery_history.append(incident_report)
        
        # Learn from this recovery
        self._learn_from_recovery(incident_report)
        
        if recovery_success:
            self.system_status = HealthStatus.RECOVERING
            logger.info(f"Recovery completed successfully in {recovery_time:.1f} seconds")
        else:
            logger.error(f"Recovery partially failed after {recovery_time:.1f} seconds")
    
    def _diagnose_issues(self, health_metrics: HealthMetrics) -> List[str]:
        """Diagnose specific issues from health metrics."""
        issues = []
        
        if health_metrics.cpu_utilization > 0.9:
            issues.append("high_cpu_utilization")
        if health_metrics.memory_utilization > 0.9:
            issues.append("high_memory_utilization")
        if health_metrics.error_rate > 0.05:
            issues.append("high_error_rate")
        if health_metrics.response_time_p95 > 2000:
            issues.append("slow_response_times")
        if health_metrics.queue_size > 5000:
            issues.append("queue_backlog")
        if health_metrics.privacy_budget_remaining < 0.1:
            issues.append("privacy_budget_exhausted")
        
        # Check node availability
        unavailable_nodes = [node for node, available in health_metrics.node_availability.items() 
                           if not available]
        if unavailable_nodes:
            issues.append(f"nodes_unavailable:{','.join(unavailable_nodes)}")
        
        # Check component health
        unhealthy_components = [comp for comp, status in health_metrics.component_health.items()
                              if status != HealthStatus.HEALTHY]
        if unhealthy_components:
            issues.append(f"components_unhealthy:{','.join(unhealthy_components)}")
        
        return issues
    
    def _determine_root_cause(self, issues: List[str], health_metrics: HealthMetrics) -> str:
        """Determine root cause from diagnosed issues."""
        # Simple rule-based root cause analysis
        if "high_cpu_utilization" in issues and "slow_response_times" in issues:
            return "resource_exhaustion_cpu_bottleneck"
        elif "high_memory_utilization" in issues and "queue_backlog" in issues:
            return "resource_exhaustion_memory_bottleneck" 
        elif "high_error_rate" in issues and any("components_unhealthy" in issue for issue in issues):
            return "component_failure_cascade"
        elif "privacy_budget_exhausted" in issues:
            return "privacy_budget_management_issue"
        elif any("nodes_unavailable" in issue for issue in issues):
            return "infrastructure_failure"
        elif "slow_response_times" in issues and "queue_backlog" in issues:
            return "load_balancing_inefficiency"
        else:
            return "general_performance_degradation"
    
    def _select_recovery_actions(self, issues: List[str], root_cause: str, status: HealthStatus) -> List[RecoveryAction]:
        """Select appropriate recovery actions based on diagnosis."""
        # Check if we've learned patterns for this root cause
        if root_cause in self.learned_patterns:
            logger.info(f"Using learned recovery pattern for {root_cause}")
            return self.learned_patterns[root_cause]
        
        selected_actions = []
        
        # Rule-based action selection
        if "resource_exhaustion" in root_cause:
            selected_actions.extend([
                action for action in self.available_actions 
                if action.action_type in [RecoveryActionType.SCALE_UP_RESOURCES, RecoveryActionType.REDUCE_LOAD]
            ])
        
        if "component_failure" in root_cause:
            selected_actions.extend([
                action for action in self.available_actions
                if action.action_type in [RecoveryActionType.RESTART_COMPONENT, RecoveryActionType.FAILOVER_NODE]
            ])
        
        if "infrastructure_failure" in root_cause:
            selected_actions.extend([
                action for action in self.available_actions
                if action.action_type == RecoveryActionType.FAILOVER_NODE
            ])
        
        if "load_balancing" in root_cause:
            selected_actions.extend([
                action for action in self.available_actions
                if action.action_type in [RecoveryActionType.REDUCE_LOAD, RecoveryActionType.RESET_CONNECTIONS]
            ])
        
        # Always include low-risk actions
        selected_actions.extend([
            action for action in self.available_actions
            if action.risk_level == "low" and action not in selected_actions
        ])
        
        # Sort by priority and risk
        selected_actions.sort(key=lambda a: (a.priority, a.risk_level == "high"))
        
        return selected_actions
    
    async def _execute_recovery_action(self, action: RecoveryAction) -> bool:
        """Execute a specific recovery action."""
        logger.info(f"Executing recovery action: {action.action_type.value} on {action.target_component}")
        
        # Check prerequisites
        current_state = await self._get_current_system_state()
        if not action.can_execute(current_state):
            logger.warning(f"Recovery action prerequisites not met: {action.prerequisites}")
            return False
        
        # Add to active recoveries
        action_key = f"{action.action_type.value}_{action.target_component}"
        self.active_recoveries[action_key] = action
        
        try:
            # Execute the action (simulate execution)
            await self._simulate_recovery_action(action)
            
            # Remove from active recoveries
            del self.active_recoveries[action_key]
            
            logger.info(f"Recovery action completed: {action.action_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Recovery action failed: {e}")
            if action_key in self.active_recoveries:
                del self.active_recoveries[action_key]
            return False
    
    async def _simulate_recovery_action(self, action: RecoveryAction):
        """Simulate recovery action execution."""
        # In a real system, this would execute actual recovery procedures
        execution_time = min(action.estimated_recovery_time, 30)  # Cap at 30s for simulation
        
        if action.action_type == RecoveryActionType.RESTART_COMPONENT:
            logger.info(f"Restarting component: {action.target_component}")
            await asyncio.sleep(execution_time / 10)  # Simulated restart time
            
        elif action.action_type == RecoveryActionType.SCALE_UP_RESOURCES:
            logger.info(f"Scaling up resources for: {action.target_component}")
            await asyncio.sleep(execution_time / 10)
            
        elif action.action_type == RecoveryActionType.REDUCE_LOAD:
            logger.info(f"Reducing load on: {action.target_component}")
            await asyncio.sleep(1)  # Quick action
            
        elif action.action_type == RecoveryActionType.CLEAR_CACHE:
            logger.info(f"Clearing cache for: {action.target_component}")
            await asyncio.sleep(2)  # Quick action
            
        else:
            # Generic action simulation
            await asyncio.sleep(execution_time / 20)
    
    async def _execute_recovery_action_by_name(self, action_name: str):
        """Execute recovery action by name (for preventive measures)."""
        # Simple action execution for preventive measures
        await asyncio.sleep(0.1)  # Simulate execution
        logger.debug(f"Executed preventive action: {action_name}")
    
    async def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state for prerequisite checking."""
        return {
            "component_responsive": True,
            "resources_available": True,
            "backup_available": True,
            "data_consistent": True,
            "connections_stable": True,
            "backup_config_available": True
        }
    
    async def _schedule_proactive_recovery(self, concerning_trends: List[str]):
        """Schedule proactive recovery actions."""
        logger.info(f"Scheduling proactive recovery for trends: {concerning_trends}")
        
        # Select low-risk proactive actions
        proactive_actions = [
            action for action in self.available_actions
            if action.risk_level == "low" and action.priority >= 3
        ]
        
        for action in proactive_actions[:2]:  # Limit proactive actions
            try:
                await self._execute_recovery_action(action)
                await asyncio.sleep(5)  # Spacing between actions
            except Exception as e:
                logger.warning(f"Proactive recovery action failed: {e}")
    
    def _learn_from_recovery(self, incident_report: IncidentReport):
        """Learn from recovery experience to improve future responses."""
        root_cause = incident_report.root_cause_analysis
        
        if incident_report.success:
            # Store successful recovery pattern
            if root_cause not in self.learned_patterns:
                self.learned_patterns[root_cause] = []
            
            # Add successful actions to learned patterns
            successful_actions = incident_report.recovery_actions_taken
            self.learned_patterns[root_cause] = successful_actions
            
            logger.info(f"Learned successful recovery pattern for: {root_cause}")
        else:
            # Learn what didn't work
            logger.info(f"Recording failed recovery pattern for: {root_cause}")
    
    def _generate_lessons_learned(self, issues: List[str], actions: List[RecoveryAction], success: bool) -> str:
        """Generate lessons learned from recovery experience."""
        if success:
            return f"Successfully recovered from {', '.join(issues)} using {len(actions)} recovery actions. " \
                   f"Key actions: {', '.join([a.action_type.value for a in actions[:2]])}."
        else:
            return f"Recovery partially failed for {', '.join(issues)}. " \
                   f"Consider alternative actions or investigate root cause further."
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health and recovery summary."""
        summary = {
            "current_status": self.system_status.value,
            "monitoring_active": self.monitoring_active,
            "recovery_in_progress": len(self.active_recoveries) > 0,
            "active_recoveries": len(self.active_recoveries),
            "last_health_check": self.last_health_check,
            "total_incidents": len(self.recovery_history),
            "successful_recoveries": sum(1 for r in self.recovery_history if r.success),
            "learned_patterns": len(self.learned_patterns),
            "available_actions": len(self.available_actions)
        }
        
        if self.current_health:
            summary.update({
                "current_health_score": self.current_health.overall_health_score(),
                "cpu_utilization": self.current_health.cpu_utilization,
                "memory_utilization": self.current_health.memory_utilization,
                "error_rate": self.current_health.error_rate,
                "response_time_p95": self.current_health.response_time_p95
            })
        
        # Recent recovery statistics
        if self.recovery_history:
            recent_recoveries = self.recovery_history[-10:]
            avg_recovery_time = sum(r.recovery_time_seconds for r in recent_recoveries) / len(recent_recoveries)
            summary["avg_recovery_time_seconds"] = avg_recovery_time
        
        return summary
    
    async def force_health_check(self) -> HealthMetrics:
        """Force an immediate health check."""
        await self._perform_health_check()
        return self.current_health
    
    async def simulate_incident(self, incident_type: str = "high_load"):
        """Simulate an incident for testing purposes."""
        logger.info(f"Simulating incident: {incident_type}")
        
        # Create simulated health metrics showing problems
        if incident_type == "high_load":
            simulated_health = HealthMetrics(
                timestamp=time.time(),
                cpu_utilization=0.95,
                memory_utilization=0.88,
                disk_utilization=0.6,
                network_latency=500,
                error_rate=0.15,
                response_time_p95=3000,
                throughput=50,
                active_connections=500,
                queue_size=8000,
                privacy_budget_remaining=0.3,
                node_availability={f"node_{i}": True for i in range(4)},
                component_health={
                    "routing_service": HealthStatus.CRITICAL,
                    "privacy_accountant": HealthStatus.DEGRADED,
                    "load_balancer": HealthStatus.CRITICAL,
                    "cache_service": HealthStatus.HEALTHY
                }
            )
        else:
            # Generic incident
            simulated_health = HealthMetrics(
                timestamp=time.time(),
                cpu_utilization=0.85,
                memory_utilization=0.92,
                disk_utilization=0.7,
                network_latency=300,
                error_rate=0.08,
                response_time_p95=2000,
                throughput=100,
                active_connections=200,
                queue_size=5000,
                privacy_budget_remaining=0.2,
                node_availability={f"node_{i}": i < 3 for i in range(4)},  # One node down
                component_health={comp: HealthStatus.DEGRADED for comp in ["routing_service", "privacy_accountant", "load_balancer", "cache_service"]}
            )
        
        # Trigger recovery with simulated metrics
        await self._trigger_recovery(simulated_health, HealthStatus.CRITICAL)