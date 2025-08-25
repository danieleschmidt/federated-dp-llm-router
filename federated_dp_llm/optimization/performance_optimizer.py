"""
Performance Optimization Engine

Implements adaptive performance optimization with ML-based load prediction,
resource allocation, and auto-scaling for federated learning systems.
"""

import asyncio
import time
try:
    from ..quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
except ImportError:
    # For files outside quantum_planning module
    from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import json


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    RESOURCE_EFFICIENT = "resource_efficient"
    PRIVACY_OPTIMIZED = "privacy_optimized"
    BALANCED = "balanced"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    storage_io: float
    active_requests: int
    queue_length: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML processing."""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "network_io": self.network_io,
            "storage_io": self.storage_io,
            "active_requests": float(self.active_requests),
            "queue_length": float(self.queue_length)
        }


@dataclass
class PerformanceTarget:
    """Performance optimization targets."""
    max_latency: float = 5.0  # seconds
    min_throughput: float = 100.0  # requests per second
    max_cpu_usage: float = 0.8  # 80%
    max_memory_usage: float = 0.85  # 85%
    max_gpu_usage: float = 0.9  # 90%
    max_queue_length: int = 50
    target_availability: float = 0.99  # 99%


@dataclass
class OptimizationAction:
    """Represents an optimization action."""
    action_type: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    confidence: float
    priority: int = 1
    timestamp: float = field(default_factory=time.time)


class LoadPredictor:
    """ML-based load prediction system."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.load_patterns = {}
        
        # Simple moving averages for prediction
        self.short_term_window = 10  # 10 minutes
        self.medium_term_window = 60  # 1 hour
        self.long_term_window = 240  # 4 hours
        
        self.logger = logging.getLogger(__name__)
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics to history."""
        self.metrics_history.append(metrics)
        
        # Update load patterns
        self._update_patterns(metrics)
    
    def _update_patterns(self, metrics: ResourceMetrics):
        """Update load patterns for prediction."""
        current_time = time.time()
        
        # Extract time-based features
        time_struct = time.localtime(current_time)
        hour_of_day = time_struct.tm_hour
        day_of_week = time_struct.tm_wday
        
        # Store patterns
        hour_key = f"hour_{hour_of_day}"
        day_key = f"day_{day_of_week}"
        
        if hour_key not in self.load_patterns:
            self.load_patterns[hour_key] = deque(maxlen=100)
        if day_key not in self.load_patterns:
            self.load_patterns[day_key] = deque(maxlen=100)
        
        self.load_patterns[hour_key].append(metrics.active_requests)
        self.load_patterns[day_key].append(metrics.active_requests)
    
    def predict_load(self, horizon_minutes: int = 30) -> Dict[str, float]:
        """Predict load for the next horizon_minutes."""
        if len(self.metrics_history) < 10:
            return {"predicted_requests": 0.0, "confidence": 0.0}
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-self.short_term_window:]
        
        # Calculate trends
        recent_requests = [m.active_requests for m in recent_metrics]
        recent_cpu = [m.cpu_usage for m in recent_metrics]
        recent_memory = [m.memory_usage for m in recent_metrics]
        
        # Simple trend analysis
        if len(recent_requests) >= 2:
            trend = (recent_requests[-1] - recent_requests[0]) / len(recent_requests)
        else:
            trend = 0.0
        
        # Pattern-based prediction
        current_time = time.time()
        future_time = current_time + (horizon_minutes * 60)
        future_hour = time.localtime(future_time).tm_hour
        
        pattern_key = f"hour_{future_hour}"
        pattern_avg = 0.0
        if pattern_key in self.load_patterns and self.load_patterns[pattern_key]:
            pattern_avg = np.mean(list(self.load_patterns[pattern_key]))
        
        # Combine trend and pattern
        base_load = np.mean(recent_requests) if recent_requests else 0.0
        predicted_load = base_load + (trend * horizon_minutes) + (pattern_avg * 0.3)
        
        # Calculate confidence based on data quality
        confidence = min(len(recent_metrics) / self.short_term_window, 1.0)
        
        return {
            "predicted_requests": max(0.0, predicted_load),
            "predicted_cpu": np.mean(recent_cpu) if recent_cpu else 0.0,
            "predicted_memory": np.mean(recent_memory) if recent_memory else 0.0,
            "trend": trend,
            "confidence": confidence
        }
    
    def detect_anomalies(self, current_metrics: ResourceMetrics) -> List[str]:
        """Detect performance anomalies."""
        anomalies = []
        
        if len(self.metrics_history) < 20:
            return anomalies
        
        # Get baseline metrics
        recent_metrics = list(self.metrics_history)[-20:]
        
        # Calculate baselines
        baseline_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        baseline_memory = np.mean([m.memory_usage for m in recent_metrics])
        baseline_requests = np.mean([m.active_requests for m in recent_metrics])
        
        baseline_cpu_std = np.std([m.cpu_usage for m in recent_metrics])
        baseline_memory_std = np.std([m.memory_usage for m in recent_metrics])
        baseline_requests_std = np.std([m.active_requests for m in recent_metrics])
        
        # Detect anomalies (> 2 standard deviations)
        if abs(current_metrics.cpu_usage - baseline_cpu) > 2 * baseline_cpu_std:
            anomalies.append("cpu_anomaly")
        
        if abs(current_metrics.memory_usage - baseline_memory) > 2 * baseline_memory_std:
            anomalies.append("memory_anomaly")
        
        if abs(current_metrics.active_requests - baseline_requests) > 2 * baseline_requests_std:
            anomalies.append("load_anomaly")
        
        return anomalies


class ResourceOptimizer:
    """Optimizes resource allocation and configuration."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.optimization_history = deque(maxlen=100)
        self.active_optimizations: Dict[str, OptimizationAction] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_resource_usage(self, metrics: ResourceMetrics, 
                             targets: PerformanceTarget) -> List[OptimizationAction]:
        """Analyze resource usage and suggest optimizations."""
        actions = []
        
        # CPU optimization
        if metrics.cpu_usage > targets.max_cpu_usage:
            actions.extend(self._optimize_cpu(metrics, targets))
        
        # Memory optimization
        if metrics.memory_usage > targets.max_memory_usage:
            actions.extend(self._optimize_memory(metrics, targets))
        
        # GPU optimization
        if metrics.gpu_usage > targets.max_gpu_usage:
            actions.extend(self._optimize_gpu(metrics, targets))
        
        # Queue optimization
        if metrics.queue_length > targets.max_queue_length:
            actions.extend(self._optimize_queue(metrics, targets))
        
        return actions
    
    def _optimize_cpu(self, metrics: ResourceMetrics, targets: PerformanceTarget) -> List[OptimizationAction]:
        """Generate CPU optimization actions."""
        actions = []
        
        if self.strategy in [OptimizationStrategy.THROUGHPUT_FOCUSED, OptimizationStrategy.BALANCED]:
            # Suggest scaling out
            action = OptimizationAction(
                action_type="scale_out",
                parameters={
                    "resource_type": "cpu",
                    "scale_factor": 1.5,
                    "reason": "high_cpu_usage"
                },
                expected_impact={"cpu_usage": -0.3, "throughput": 0.4},
                confidence=0.8,
                priority=2
            )
            actions.append(action)
        
        # Suggest connection pool optimization
        if metrics.active_requests > 50:
            action = OptimizationAction(
                action_type="optimize_connection_pool",
                parameters={
                    "increase_pool_size": True,
                    "max_connections": min(100, metrics.active_requests * 2)
                },
                expected_impact={"cpu_usage": -0.1, "latency": -0.2},
                confidence=0.7,
                priority=3
            )
            actions.append(action)
        
        return actions
    
    def _optimize_memory(self, metrics: ResourceMetrics, targets: PerformanceTarget) -> List[OptimizationAction]:
        """Generate memory optimization actions."""
        actions = []
        
        # Suggest cache optimization
        action = OptimizationAction(
            action_type="optimize_cache",
            parameters={
                "reduce_cache_size": True,
                "enable_compression": True,
                "ttl_reduction": 0.5
            },
            expected_impact={"memory_usage": -0.2},
            confidence=0.8,
            priority=2
        )
        actions.append(action)
        
        # Suggest garbage collection optimization
        if metrics.memory_usage > 0.9:
            action = OptimizationAction(
                action_type="trigger_gc",
                parameters={
                    "force_full_gc": True,
                    "optimize_heap": True
                },
                expected_impact={"memory_usage": -0.1},
                confidence=0.6,
                priority=1
            )
            actions.append(action)
        
        return actions
    
    def _optimize_gpu(self, metrics: ResourceMetrics, targets: PerformanceTarget) -> List[OptimizationAction]:
        """Generate GPU optimization actions."""
        actions = []
        
        if self.strategy == OptimizationStrategy.PRIVACY_OPTIMIZED:
            # Optimize for privacy-preserving computations
            action = OptimizationAction(
                action_type="optimize_privacy_computation",
                parameters={
                    "batch_privacy_operations": True,
                    "enable_gpu_privacy_acceleration": True
                },
                expected_impact={"gpu_usage": -0.15, "privacy_computation_speed": 0.3},
                confidence=0.7,
                priority=2
            )
            actions.append(action)
        
        # Model optimization
        action = OptimizationAction(
            action_type="optimize_model_inference",
            parameters={
                "enable_tensor_fusion": True,
                "optimize_batch_size": True,
                "enable_mixed_precision": True
            },
            expected_impact={"gpu_usage": -0.2, "inference_speed": 0.25},
            confidence=0.8,
            priority=2
        )
        actions.append(action)
        
        return actions
    
    def _optimize_queue(self, metrics: ResourceMetrics, targets: PerformanceTarget) -> List[OptimizationAction]:
        """Generate queue optimization actions."""
        actions = []
        
        if self.strategy == OptimizationStrategy.LATENCY_FOCUSED:
            # Prioritize latency-sensitive requests
            action = OptimizationAction(
                action_type="optimize_request_scheduling",
                parameters={
                    "enable_priority_queue": True,
                    "latency_threshold": targets.max_latency * 0.5
                },
                expected_impact={"queue_length": -0.3, "avg_latency": -0.4},
                confidence=0.8,
                priority=1
            )
            actions.append(action)
        
        # Increase processing capacity
        action = OptimizationAction(
            action_type="increase_workers",
            parameters={
                "worker_increase": min(10, metrics.queue_length // 5),
                "temporary": True
            },
            expected_impact={"queue_length": -0.5, "throughput": 0.3},
            confidence=0.9,
            priority=1
        )
        actions.append(action)
        
        return actions


class AdaptiveScaler:
    """Adaptive auto-scaling system."""
    
    def __init__(self, min_instances: int = 2, max_instances: int = 20):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        # Scaling parameters
        self.scale_up_threshold = 0.8  # 80% resource utilization
        self.scale_down_threshold = 0.3  # 30% resource utilization
        self.scale_up_cooldown = 300  # 5 minutes
        self.scale_down_cooldown = 600  # 10 minutes
        
        self.last_scale_up = 0
        self.last_scale_down = 0
        
        self.scaling_history = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
    
    def should_scale(self, metrics: ResourceMetrics, prediction: Dict[str, float]) -> Optional[str]:
        """Determine if scaling is needed."""
        current_time = time.time()
        
        # Calculate resource pressure
        resource_pressure = max(
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.gpu_usage
        )
        
        # Consider predicted load
        predicted_pressure = max(
            prediction.get("predicted_cpu", 0),
            prediction.get("predicted_memory", 0)
        )
        
        # Scale up conditions
        if (resource_pressure > self.scale_up_threshold or 
            predicted_pressure > self.scale_up_threshold):
            
            if (current_time - self.last_scale_up > self.scale_up_cooldown and
                self.current_instances < self.max_instances):
                return "scale_up"
        
        # Scale down conditions
        elif (resource_pressure < self.scale_down_threshold and
              predicted_pressure < self.scale_down_threshold):
            
            if (current_time - self.last_scale_down > self.scale_down_cooldown and
                self.current_instances > self.min_instances):
                return "scale_down"
        
        return None
    
    def execute_scaling(self, action: str, factor: float = 1.0) -> Dict[str, Any]:
        """Execute scaling action."""
        current_time = time.time()
        
        if action == "scale_up":
            new_instances = min(
                self.max_instances,
                int(self.current_instances * (1 + factor))
            )
            if new_instances > self.current_instances:
                self.current_instances = new_instances
                self.last_scale_up = current_time
                
                self.logger.info(f"Scaled up to {self.current_instances} instances")
                
                return {
                    "action": "scale_up",
                    "old_instances": self.current_instances - (new_instances - self.current_instances),
                    "new_instances": self.current_instances,
                    "timestamp": current_time
                }
        
        elif action == "scale_down":
            new_instances = max(
                self.min_instances,
                int(self.current_instances * (1 - factor))
            )
            if new_instances < self.current_instances:
                self.current_instances = new_instances
                self.last_scale_down = current_time
                
                self.logger.info(f"Scaled down to {self.current_instances} instances")
                
                return {
                    "action": "scale_down",
                    "old_instances": self.current_instances + (self.current_instances - new_instances),
                    "new_instances": self.current_instances,
                    "timestamp": current_time
                }
        
        return {}


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = OptimizationStrategy(config.get("strategy", "balanced"))
        self.targets = PerformanceTarget(**config.get("targets", {}))
        
        # Components
        self.load_predictor = LoadPredictor(config.get("history_size", 1000))
        self.resource_optimizer = ResourceOptimizer(self.strategy)
        self.adaptive_scaler = AdaptiveScaler(
            min_instances=config.get("min_instances", 2),
            max_instances=config.get("max_instances", 20)
        )
        
        # State
        self.current_metrics: Optional[ResourceMetrics] = None
        self.optimization_queue = asyncio.Queue()
        self.active_optimizations: Dict[str, OptimizationAction] = {}
        
        # Task management
        self.optimization_task: Optional[asyncio.Task] = None
        self.metrics_collection_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the performance optimizer."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        self.logger.info("Performance optimizer started")
    
    async def stop(self):
        """Stop the performance optimizer."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel tasks
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        if self.metrics_collection_task:
            self.metrics_collection_task.cancel()
            try:
                await self.metrics_collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance optimizer stopped")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.running:
            try:
                # Collect current metrics (would integrate with actual system monitoring)
                metrics = await self._collect_current_metrics()
                
                if metrics:
                    self.current_metrics = metrics
                    self.load_predictor.add_metrics(metrics)
                    
                    # Check for immediate optimization needs
                    await self._check_immediate_optimizations(metrics)
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Background optimization execution loop."""
        while self.running:
            try:
                # Process optimization queue
                try:
                    action = await asyncio.wait_for(
                        self.optimization_queue.get(), timeout=60
                    )
                    await self._execute_optimization(action)
                    
                except asyncio.TimeoutError:
                    # Periodic optimization analysis
                    await self._periodic_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_current_metrics(self) -> Optional[ResourceMetrics]:
        """Collect current system metrics."""
        try:
            # In a real implementation, this would integrate with system monitoring
            # For now, return simulated metrics
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # Simulate other metrics
            gpu_usage = np.random.uniform(0.2, 0.8)  # Would use actual GPU monitoring
            network_io = np.random.uniform(0.1, 0.5)
            storage_io = np.random.uniform(0.1, 0.4)
            active_requests = np.random.randint(10, 100)
            queue_length = max(0, active_requests - 50)
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                network_io=network_io,
                storage_io=storage_io,
                active_requests=active_requests,
                queue_length=queue_length
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return None
    
    async def _check_immediate_optimizations(self, metrics: ResourceMetrics):
        """Check for immediate optimization needs."""
        # Detect anomalies
        anomalies = self.load_predictor.detect_anomalies(metrics)
        
        if anomalies:
            self.logger.warning(f"Performance anomalies detected: {anomalies}")
            
            # Queue immediate optimization actions
            if "cpu_anomaly" in anomalies:
                action = OptimizationAction(
                    action_type="emergency_cpu_optimization",
                    parameters={"reduce_load": True},
                    expected_impact={"cpu_usage": -0.2},
                    confidence=0.7,
                    priority=0  # Highest priority
                )
                await self.optimization_queue.put(action)
        
        # Check scaling needs
        prediction = self.load_predictor.predict_load(horizon_minutes=15)
        scaling_action = self.adaptive_scaler.should_scale(metrics, prediction)
        
        if scaling_action:
            action = OptimizationAction(
                action_type=f"auto_{scaling_action}",
                parameters={"factor": 0.5 if scaling_action == "scale_up" else 0.25},
                expected_impact={"resource_pressure": -0.3 if scaling_action == "scale_up" else 0.1},
                confidence=prediction.get("confidence", 0.5),
                priority=1
            )
            await self.optimization_queue.put(action)
    
    async def _periodic_optimization(self):
        """Perform periodic optimization analysis."""
        if not self.current_metrics:
            return
        
        # Generate optimization actions
        actions = self.resource_optimizer.analyze_resource_usage(
            self.current_metrics, self.targets
        )
        
        # Queue actions by priority
        for action in sorted(actions, key=lambda a: a.priority):
            await self.optimization_queue.put(action)
    
    async def _execute_optimization(self, action: OptimizationAction):
        """Execute an optimization action."""
        try:
            self.logger.info(f"Executing optimization: {action.action_type}")
            
            # Store active optimization
            self.active_optimizations[action.action_type] = action
            
            # Execute based on action type
            if action.action_type.startswith("auto_scale"):
                await self._execute_scaling(action)
            elif action.action_type == "optimize_cache":
                await self._execute_cache_optimization(action)
            elif action.action_type == "optimize_connection_pool":
                await self._execute_connection_pool_optimization(action)
            elif action.action_type == "optimize_model_inference":
                await self._execute_model_optimization(action)
            else:
                self.logger.warning(f"Unknown optimization action: {action.action_type}")
            
            # Mark as completed
            self.active_optimizations.pop(action.action_type, None)
            
        except Exception as e:
            self.logger.error(f"Failed to execute optimization {action.action_type}: {e}")
    
    async def _execute_scaling(self, action: OptimizationAction):
        """Execute scaling action."""
        scaling_type = action.action_type.replace("auto_", "")
        factor = action.parameters.get("factor", 0.5)
        
        result = self.adaptive_scaler.execute_scaling(scaling_type, factor)
        
        if result:
            self.logger.info(f"Scaling executed: {result}")
    
    async def _execute_cache_optimization(self, action: OptimizationAction):
        """Execute cache optimization."""
        # This would integrate with the cache manager
        self.logger.info("Cache optimization executed")
    
    async def _execute_connection_pool_optimization(self, action: OptimizationAction):
        """Execute connection pool optimization."""
        # This would integrate with the connection pool manager
        self.logger.info("Connection pool optimization executed")
    
    async def _execute_model_optimization(self, action: OptimizationAction):
        """Execute model inference optimization."""
        # This would integrate with the model inference system
        self.logger.info("Model optimization executed")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_time = time.time()
        
        # Get predictions
        prediction = self.load_predictor.predict_load() if self.current_metrics else {}
        
        # Scaling status
        scaling_status = {
            "current_instances": self.adaptive_scaler.current_instances,
            "min_instances": self.adaptive_scaler.min_instances,
            "max_instances": self.adaptive_scaler.max_instances,
            "last_scale_up": self.adaptive_scaler.last_scale_up,
            "last_scale_down": self.adaptive_scaler.last_scale_down
        }
        
        return {
            "timestamp": current_time,
            "strategy": self.strategy.value,
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else {},
            "predictions": prediction,
            "scaling_status": scaling_status,
            "active_optimizations": len(self.active_optimizations),
            "targets": {
                "max_latency": self.targets.max_latency,
                "min_throughput": self.targets.min_throughput,
                "max_cpu_usage": self.targets.max_cpu_usage,
                "max_memory_usage": self.targets.max_memory_usage,
                "target_availability": self.targets.target_availability
            },
            "optimization_queue_size": self.optimization_queue.qsize()
        }