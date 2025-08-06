"""
Quantum Performance Optimizer

Advanced performance optimization for quantum-inspired task planning
with auto-scaling, resource pooling, and distributed execution.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import concurrent.futures
import multiprocessing as mp
import threading
import math

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    RESOURCE_EFFICIENT = "resource_efficient"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class ScalingTrigger(Enum):
    """Auto-scaling trigger conditions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    QUANTUM_COHERENCE_LOSS = "quantum_coherence_loss"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    timestamp: float
    
    # Latency metrics (milliseconds)
    avg_planning_time: float = 0.0
    avg_measurement_time: float = 0.0
    avg_entanglement_time: float = 0.0
    avg_interference_time: float = 0.0
    p95_planning_time: float = 0.0
    p99_planning_time: float = 0.0
    
    # Throughput metrics
    tasks_per_second: float = 0.0
    measurements_per_second: float = 0.0
    entanglements_per_second: float = 0.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    quantum_coherence_utilization: float = 0.0
    
    # Quality metrics
    success_rate: float = 0.0
    quantum_fidelity: float = 0.0
    optimization_efficiency: float = 0.0


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int = 1
    max_instances: int = 10
    cooldown_seconds: int = 300
    evaluation_periods: int = 2
    
    # Quantum-specific parameters
    coherence_threshold: float = 0.5
    entanglement_density_threshold: float = 0.8


class QuantumResourcePool:
    """Pool of quantum computational resources."""
    
    def __init__(self, 
                 pool_size: int = mp.cpu_count(),
                 max_workers: int = None):
        self.pool_size = pool_size
        self.max_workers = max_workers or pool_size * 2
        
        # Thread and process pools
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.pool_size)
        
        # Resource tracking
        self.active_tasks: Set[str] = set()
        self.resource_usage: Dict[str, float] = {}
        self.performance_counters: Dict[str, int] = defaultdict(int)
        
        # Async coordination
        self.resource_semaphore = asyncio.Semaphore(self.max_workers)
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.result_cache: Dict[str, Any] = {}
        
        # Pool statistics
        self.pool_stats = {
            "tasks_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time": 0.0,
            "peak_utilization": 0.0
        }
    
    async def submit_quantum_task(self, 
                                task_id: str,
                                task_func: Callable,
                                *args, 
                                use_cache: bool = True,
                                execution_mode: str = "thread") -> Any:
        """Submit quantum task to resource pool."""
        
        # Check cache first
        cache_key = self._generate_cache_key(task_func.__name__, args)
        if use_cache and cache_key in self.result_cache:
            self.pool_stats["cache_hits"] += 1
            return self.result_cache[cache_key]
        
        self.pool_stats["cache_misses"] += 1
        
        # Acquire resource semaphore
        async with self.resource_semaphore:
            start_time = time.time()
            
            try:
                self.active_tasks.add(task_id)
                
                # Execute based on mode
                if execution_mode == "thread":
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.thread_pool, task_func, *args)
                elif execution_mode == "process":
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.process_pool, task_func, *args)
                else:  # direct async execution
                    result = await task_func(*args)
                
                # Update statistics
                execution_time = time.time() - start_time
                self.pool_stats["tasks_executed"] += 1
                self.pool_stats["avg_execution_time"] = (
                    (self.pool_stats["avg_execution_time"] * (self.pool_stats["tasks_executed"] - 1) + 
                     execution_time) / self.pool_stats["tasks_executed"]
                )
                
                # Cache result
                if use_cache:
                    self.result_cache[cache_key] = result
                    
                    # Limit cache size
                    if len(self.result_cache) > 1000:
                        # Remove oldest entries (simple FIFO)
                        oldest_keys = list(self.result_cache.keys())[:100]
                        for key in oldest_keys:
                            del self.result_cache[key]
                
                return result
                
            finally:
                self.active_tasks.discard(task_id)
    
    def _generate_cache_key(self, func_name: str, args: Tuple) -> str:
        """Generate cache key for task."""
        # Simple hash-based cache key
        args_str = str(args)
        return f"{func_name}_{hash(args_str)}"
    
    def get_utilization(self) -> float:
        """Get current resource pool utilization."""
        active_count = len(self.active_tasks)
        utilization = active_count / self.max_workers
        
        # Update peak utilization
        self.pool_stats["peak_utilization"] = max(
            self.pool_stats["peak_utilization"], 
            utilization
        )
        
        return utilization
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        return {
            **self.pool_stats,
            "current_utilization": self.get_utilization(),
            "active_tasks": len(self.active_tasks),
            "cache_size": len(self.result_cache),
            "cache_hit_rate": self.pool_stats["cache_hits"] / max(
                self.pool_stats["cache_hits"] + self.pool_stats["cache_misses"], 1
            )
        }
    
    async def shutdown(self):
        """Shutdown resource pool."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class QuantumPerformanceOptimizer:
    """
    Advanced performance optimizer for quantum planning systems.
    
    Provides:
    - Adaptive performance optimization
    - Auto-scaling based on quantum metrics
    - Resource pooling and caching
    - Concurrent execution optimization
    - Performance monitoring and tuning
    """
    
    def __init__(self,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
                 enable_auto_scaling: bool = True,
                 enable_caching: bool = True):
        
        self.optimization_strategy = optimization_strategy
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_caching = enable_caching
        
        # Resource management
        self.resource_pool = QuantumResourcePool()
        self.scaling_policies: List[ScalingPolicy] = []
        self.active_optimizations: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_targets: Dict[str, float] = {
            "max_planning_time": 5000.0,    # 5 seconds
            "min_throughput": 10.0,         # 10 tasks/second
            "max_memory_usage": 0.8,        # 80%
            "min_success_rate": 0.95,       # 95%
            "min_quantum_fidelity": 0.8     # 80%
        }
        
        # Optimization state
        self.current_instances: Dict[str, int] = defaultdict(lambda: 1)
        self.scaling_history: List[Dict[str, Any]] = []
        self.optimization_enabled = False
        self._optimization_task = None
        
        # Concurrent execution management
        self.concurrent_limits: Dict[str, asyncio.Semaphore] = {
            "planning": asyncio.Semaphore(5),
            "measurement": asyncio.Semaphore(10),
            "entanglement": asyncio.Semaphore(3),
            "interference": asyncio.Semaphore(8)
        }
        
        # Advanced optimization features
        self.adaptive_batching_enabled = True
        self.dynamic_load_balancing_enabled = True
        self.predictive_scaling_enabled = True
        
        # Initialize default scaling policies
        self._initialize_default_scaling_policies()
    
    def _initialize_default_scaling_policies(self):
        """Initialize default auto-scaling policies."""
        
        self.scaling_policies = [
            ScalingPolicy(
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                min_instances=1,
                max_instances=8,
                cooldown_seconds=300
            ),
            ScalingPolicy(
                trigger=ScalingTrigger.QUEUE_DEPTH,
                scale_up_threshold=50,
                scale_down_threshold=5,
                min_instances=1,
                max_instances=12,
                cooldown_seconds=180
            ),
            ScalingPolicy(
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=3000,  # 3 seconds
                scale_down_threshold=500,  # 0.5 seconds
                min_instances=1,
                max_instances=10,
                cooldown_seconds=240
            ),
            ScalingPolicy(
                trigger=ScalingTrigger.QUANTUM_COHERENCE_LOSS,
                scale_up_threshold=0.7,   # Scale up if coherence drops below 70%
                scale_down_threshold=0.9, # Scale down if coherence above 90%
                min_instances=2,
                max_instances=6,
                cooldown_seconds=600,  # Longer cooldown for quantum metrics
                coherence_threshold=0.5
            )
        ]
    
    async def start_optimization(self):
        """Start performance optimization system."""
        if self.optimization_enabled:
            return
        
        self.optimization_enabled = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Quantum performance optimization started")
    
    async def stop_optimization(self):
        """Stop performance optimization system."""
        self.optimization_enabled = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        await self.resource_pool.shutdown()
        logger.info("Quantum performance optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_enabled:
            try:
                # Collect current performance metrics
                metrics = await self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Apply optimization strategy
                await self._apply_optimization_strategy(metrics)
                
                # Check auto-scaling triggers
                if self.enable_auto_scaling:
                    await self._evaluate_scaling_policies(metrics)
                
                # Adaptive optimization adjustments
                if self.optimization_strategy == OptimizationStrategy.ADAPTIVE:
                    await self._adaptive_optimization(metrics)
                
                # Performance tuning
                await self._tune_performance_parameters(metrics)
                
                # Clean up old optimizations
                await self._cleanup_optimizations()
                
                # Wait for next optimization cycle
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Optimization loop error: {str(e)}")
                await asyncio.sleep(30.0)
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        
        # This would integrate with actual quantum components
        # For now, simulate metrics collection
        
        current_time = time.time()
        
        # Simulate metric collection
        metrics = PerformanceMetrics(
            timestamp=current_time,
            avg_planning_time=np.random.normal(2000, 500),  # ms
            avg_measurement_time=np.random.normal(100, 20),
            avg_entanglement_time=np.random.normal(800, 150),
            avg_interference_time=np.random.normal(300, 50),
            p95_planning_time=np.random.normal(4000, 800),
            p99_planning_time=np.random.normal(6000, 1200),
            tasks_per_second=np.random.normal(15, 5),
            measurements_per_second=np.random.normal(50, 10),
            entanglements_per_second=np.random.normal(8, 2),
            cpu_utilization=np.random.uniform(0.3, 0.9),
            memory_utilization=np.random.uniform(0.4, 0.8),
            quantum_coherence_utilization=np.random.uniform(0.6, 0.95),
            success_rate=np.random.uniform(0.9, 1.0),
            quantum_fidelity=np.random.uniform(0.75, 0.98),
            optimization_efficiency=np.random.uniform(0.7, 0.95)
        )
        
        # Add resource pool metrics
        pool_stats = self.resource_pool.get_pool_stats()
        metrics.avg_planning_time *= (1 + pool_stats["current_utilization"] * 0.5)
        
        return metrics
    
    async def _apply_optimization_strategy(self, metrics: PerformanceMetrics):
        """Apply selected optimization strategy."""
        
        if self.optimization_strategy == OptimizationStrategy.LATENCY_FOCUSED:
            await self._optimize_for_latency(metrics)
        elif self.optimization_strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            await self._optimize_for_throughput(metrics)
        elif self.optimization_strategy == OptimizationStrategy.RESOURCE_EFFICIENT:
            await self._optimize_for_resource_efficiency(metrics)
        elif self.optimization_strategy == OptimizationStrategy.BALANCED:
            await self._optimize_balanced(metrics)
        elif self.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            await self._optimize_adaptive(metrics)
    
    async def _optimize_for_latency(self, metrics: PerformanceMetrics):
        """Optimize for minimum latency."""
        
        # Increase concurrency limits if latency is high
        if metrics.avg_planning_time > self.optimization_targets["max_planning_time"]:
            for operation, semaphore in self.concurrent_limits.items():
                if semaphore._value < 20:  # Max concurrent operations
                    # Increase semaphore capacity
                    new_semaphore = asyncio.Semaphore(semaphore._value + 2)
                    self.concurrent_limits[operation] = new_semaphore
                    
                    self.active_optimizations[f"latency_concurrency_{operation}"] = {
                        "type": "concurrency_increase",
                        "operation": operation,
                        "previous_limit": semaphore._value,
                        "new_limit": new_semaphore._value,
                        "timestamp": time.time()
                    }
    
    async def _optimize_for_throughput(self, metrics: PerformanceMetrics):
        """Optimize for maximum throughput."""
        
        # Enable aggressive batching if throughput is low
        if metrics.tasks_per_second < self.optimization_targets["min_throughput"]:
            self.adaptive_batching_enabled = True
            
            self.active_optimizations["throughput_batching"] = {
                "type": "batching_enabled",
                "target_throughput": self.optimization_targets["min_throughput"],
                "current_throughput": metrics.tasks_per_second,
                "timestamp": time.time()
            }
    
    async def _optimize_for_resource_efficiency(self, metrics: PerformanceMetrics):
        """Optimize for resource efficiency."""
        
        # Reduce resource usage if memory utilization is high
        if metrics.memory_utilization > self.optimization_targets["max_memory_usage"]:
            # Reduce cache size
            if len(self.resource_pool.result_cache) > 500:
                # Clear half the cache
                cache_keys = list(self.resource_pool.result_cache.keys())
                for key in cache_keys[:len(cache_keys)//2]:
                    del self.resource_pool.result_cache[key]
                
                self.active_optimizations["memory_cache_reduction"] = {
                    "type": "cache_reduction",
                    "previous_size": len(cache_keys),
                    "new_size": len(self.resource_pool.result_cache),
                    "timestamp": time.time()
                }
    
    async def _optimize_balanced(self, metrics: PerformanceMetrics):
        """Apply balanced optimization across all dimensions."""
        
        # Balance latency and throughput
        latency_score = 1.0 - (metrics.avg_planning_time / self.optimization_targets["max_planning_time"])
        throughput_score = metrics.tasks_per_second / self.optimization_targets["min_throughput"]
        
        if latency_score < 0.8:  # Poor latency
            await self._optimize_for_latency(metrics)
        elif throughput_score < 0.8:  # Poor throughput
            await self._optimize_for_throughput(metrics)
        elif metrics.memory_utilization > 0.85:  # High memory usage
            await self._optimize_for_resource_efficiency(metrics)
    
    async def _optimize_adaptive(self, metrics: PerformanceMetrics):
        """Adaptive optimization based on current conditions."""
        
        # Analyze recent performance trends
        if len(self.performance_history) < 3:
            return
        
        recent_metrics = list(self.performance_history)[-3:]
        
        # Check for performance degradation trends
        latency_trend = [m.avg_planning_time for m in recent_metrics]
        throughput_trend = [m.tasks_per_second for m in recent_metrics]
        
        # Detect increasing latency trend
        if all(latency_trend[i] >= latency_trend[i-1] for i in range(1, len(latency_trend))):
            await self._optimize_for_latency(metrics)
        
        # Detect decreasing throughput trend
        elif all(throughput_trend[i] <= throughput_trend[i-1] for i in range(1, len(throughput_trend))):
            await self._optimize_for_throughput(metrics)
        
        # Otherwise, apply balanced optimization
        else:
            await self._optimize_balanced(metrics)
    
    async def _evaluate_scaling_policies(self, metrics: PerformanceMetrics):
        """Evaluate auto-scaling policies and trigger scaling if needed."""
        
        for policy in self.scaling_policies:
            try:
                should_scale, direction = await self._evaluate_scaling_policy(policy, metrics)
                
                if should_scale:
                    await self._execute_scaling_action(policy, direction, metrics)
                    
            except Exception as e:
                logger.error(f"Scaling policy evaluation failed: {str(e)}")
    
    async def _evaluate_scaling_policy(self, 
                                     policy: ScalingPolicy, 
                                     metrics: PerformanceMetrics) -> Tuple[bool, str]:
        """Evaluate individual scaling policy."""
        
        # Get current metric value based on trigger
        if policy.trigger == ScalingTrigger.CPU_UTILIZATION:
            current_value = metrics.cpu_utilization
        elif policy.trigger == ScalingTrigger.MEMORY_UTILIZATION:
            current_value = metrics.memory_utilization
        elif policy.trigger == ScalingTrigger.RESPONSE_TIME:
            current_value = metrics.avg_planning_time
        elif policy.trigger == ScalingTrigger.THROUGHPUT:
            current_value = metrics.tasks_per_second
        elif policy.trigger == ScalingTrigger.QUANTUM_COHERENCE_LOSS:
            current_value = 1.0 - metrics.quantum_coherence_utilization  # Loss = 1 - utilization
        else:
            # Queue depth would require integration with actual queue
            current_value = 0.0
        
        # Check scaling conditions
        current_instances = self.current_instances[policy.trigger.value]
        
        # Scale up condition
        if (current_value > policy.scale_up_threshold and 
            current_instances < policy.max_instances):
            return True, "up"
        
        # Scale down condition
        elif (current_value < policy.scale_down_threshold and 
              current_instances > policy.min_instances):
            return True, "down"
        
        return False, "none"
    
    async def _execute_scaling_action(self, 
                                    policy: ScalingPolicy, 
                                    direction: str, 
                                    metrics: PerformanceMetrics):
        """Execute scaling action based on policy."""
        
        component = policy.trigger.value
        current_instances = self.current_instances[component]
        
        if direction == "up":
            new_instances = min(current_instances + 1, policy.max_instances)
        else:  # direction == "down"
            new_instances = max(current_instances - 1, policy.min_instances)
        
        if new_instances != current_instances:
            # Update instance count
            self.current_instances[component] = new_instances
            
            # Apply scaling (would integrate with actual infrastructure)
            await self._apply_scaling(component, new_instances, current_instances)
            
            # Record scaling event
            scaling_event = {
                "timestamp": time.time(),
                "component": component,
                "trigger": policy.trigger.value,
                "direction": direction,
                "previous_instances": current_instances,
                "new_instances": new_instances,
                "trigger_value": getattr(metrics, policy.trigger.value, 0.0),
                "threshold": policy.scale_up_threshold if direction == "up" else policy.scale_down_threshold
            }
            
            self.scaling_history.append(scaling_event)
            
            # Limit history size
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-50:]
            
            logger.info(f"Scaling {direction} {component}: {current_instances} -> {new_instances}")
    
    async def _apply_scaling(self, component: str, new_instances: int, previous_instances: int):
        """Apply actual scaling changes to system."""
        
        # Adjust resource pool based on scaling
        if component in ["cpu_utilization", "memory_utilization"]:
            # Adjust thread pool size
            new_pool_size = max(4, new_instances * 2)
            
            # Create new thread pool (simplified - in practice would be more sophisticated)
            old_pool = self.resource_pool.thread_pool
            self.resource_pool.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=new_pool_size
            )
            
            # Shutdown old pool gracefully
            old_pool.shutdown(wait=False)
        
        elif component == "quantum_coherence_loss":
            # Adjust quantum-specific parameters
            for operation, semaphore in self.concurrent_limits.items():
                if operation in ["entanglement", "interference"]:
                    # Adjust quantum operation concurrency
                    adjustment = 1 if new_instances > previous_instances else -1
                    new_limit = max(1, min(20, semaphore._value + adjustment))
                    
                    self.concurrent_limits[operation] = asyncio.Semaphore(new_limit)
    
    async def _adaptive_optimization(self, metrics: PerformanceMetrics):
        """Apply adaptive optimizations based on performance patterns."""
        
        # Analyze performance patterns over time
        if len(self.performance_history) < 10:
            return
        
        recent_metrics = list(self.performance_history)[-10:]
        
        # Detect performance patterns
        patterns = self._detect_performance_patterns(recent_metrics)
        
        for pattern in patterns:
            await self._apply_pattern_optimization(pattern, metrics)
    
    def _detect_performance_patterns(self, metrics_history: List[PerformanceMetrics]) -> List[str]:
        """Detect performance patterns in historical data."""
        
        patterns = []
        
        # Extract time series data
        latencies = [m.avg_planning_time for m in metrics_history]
        throughputs = [m.tasks_per_second for m in metrics_history]
        cpu_utils = [m.cpu_utilization for m in metrics_history]
        
        # Detect increasing latency pattern
        latency_slope = np.polyfit(range(len(latencies)), latencies, 1)[0]
        if latency_slope > 100:  # Increasing by 100ms per measurement
            patterns.append("increasing_latency")
        
        # Detect throughput degradation
        throughput_slope = np.polyfit(range(len(throughputs)), throughputs, 1)[0]
        if throughput_slope < -0.5:  # Decreasing by 0.5 tasks/sec per measurement
            patterns.append("decreasing_throughput")
        
        # Detect high variance (instability)
        latency_cv = np.std(latencies) / np.mean(latencies)
        if latency_cv > 0.3:  # High coefficient of variation
            patterns.append("unstable_performance")
        
        # Detect resource saturation
        if np.mean(cpu_utils) > 0.85:
            patterns.append("resource_saturation")
        
        return patterns
    
    async def _apply_pattern_optimization(self, pattern: str, metrics: PerformanceMetrics):
        """Apply optimization based on detected pattern."""
        
        if pattern == "increasing_latency":
            # Increase concurrency and enable caching
            await self._optimize_for_latency(metrics)
            self.enable_caching = True
            
        elif pattern == "decreasing_throughput":
            # Enable batching and load balancing
            self.adaptive_batching_enabled = True
            self.dynamic_load_balancing_enabled = True
            
        elif pattern == "unstable_performance":
            # Apply stability optimizations
            # Reduce concurrency to stable levels
            for operation, semaphore in self.concurrent_limits.items():
                if semaphore._value > 5:
                    self.concurrent_limits[operation] = asyncio.Semaphore(5)
            
        elif pattern == "resource_saturation":
            # Apply resource optimization
            await self._optimize_for_resource_efficiency(metrics)
    
    async def _tune_performance_parameters(self, metrics: PerformanceMetrics):
        """Fine-tune performance parameters based on current metrics."""
        
        # Dynamic batch size tuning
        if self.adaptive_batching_enabled:
            optimal_batch_size = self._calculate_optimal_batch_size(metrics)
            self.active_optimizations["adaptive_batch_size"] = {
                "type": "batch_size_tuning",
                "optimal_size": optimal_batch_size,
                "timestamp": time.time()
            }
        
        # Dynamic timeout tuning
        optimal_timeout = self._calculate_optimal_timeout(metrics)
        self.active_optimizations["adaptive_timeout"] = {
            "type": "timeout_tuning",
            "optimal_timeout": optimal_timeout,
            "timestamp": time.time()
        }
    
    def _calculate_optimal_batch_size(self, metrics: PerformanceMetrics) -> int:
        """Calculate optimal batch size based on current performance."""
        
        # Base batch size on current throughput and latency
        if metrics.avg_planning_time < 1000:  # Low latency
            return min(20, max(5, int(metrics.tasks_per_second / 2)))
        else:  # High latency
            return max(2, int(metrics.tasks_per_second / 4))
    
    def _calculate_optimal_timeout(self, metrics: PerformanceMetrics) -> float:
        """Calculate optimal timeout based on performance characteristics."""
        
        # Set timeout to 3x P95 latency for reliability
        optimal_timeout = metrics.p95_planning_time * 3
        return min(30000, max(5000, optimal_timeout))  # Between 5-30 seconds
    
    async def _cleanup_optimizations(self):
        """Clean up old optimization records."""
        
        current_time = time.time()
        expired_optimizations = []
        
        for opt_id, optimization in self.active_optimizations.items():
            # Remove optimizations older than 1 hour
            if current_time - optimization["timestamp"] > 3600:
                expired_optimizations.append(opt_id)
        
        for opt_id in expired_optimizations:
            del self.active_optimizations[opt_id]
    
    async def optimize_quantum_task_batch(self, 
                                        tasks: List[Tuple[str, Callable, Tuple]],
                                        execution_mode: str = "concurrent") -> List[Any]:
        """Optimize execution of quantum task batch."""
        
        if not tasks:
            return []
        
        batch_size = len(tasks)
        start_time = time.time()
        
        try:
            if execution_mode == "concurrent":
                # Execute tasks concurrently with optimal concurrency
                semaphore = asyncio.Semaphore(min(batch_size, 10))  # Limit concurrent tasks
                
                async def execute_task(task_id, task_func, args):
                    async with semaphore:
                        return await self.resource_pool.submit_quantum_task(
                            task_id, task_func, *args, use_cache=self.enable_caching
                        )
                
                tasks_coroutines = [
                    execute_task(task_id, task_func, args)
                    for task_id, task_func, args in tasks
                ]
                
                results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
                
            elif execution_mode == "batch":
                # Execute as single batch operation
                results = []
                for task_id, task_func, args in tasks:
                    result = await self.resource_pool.submit_quantum_task(
                        task_id, task_func, *args, use_cache=self.enable_caching
                    )
                    results.append(result)
            
            else:  # sequential
                results = []
                for task_id, task_func, args in tasks:
                    result = await self.resource_pool.submit_quantum_task(
                        task_id, task_func, *args, use_cache=self.enable_caching
                    )
                    results.append(result)
            
            # Record batch execution metrics
            execution_time = time.time() - start_time
            throughput = batch_size / execution_time if execution_time > 0 else 0
            
            self.active_optimizations[f"batch_{int(start_time)}"] = {
                "type": "batch_execution",
                "batch_size": batch_size,
                "execution_time": execution_time,
                "throughput": throughput,
                "execution_mode": execution_mode,
                "timestamp": start_time
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Batch optimization failed: {str(e)}")
            raise e
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        
        current_metrics = self.performance_history[-1] if self.performance_history else None
        
        # Calculate optimization effectiveness
        effectiveness_scores = {}
        if len(self.performance_history) >= 2:
            recent = self.performance_history[-1]
            baseline = self.performance_history[0]
            
            effectiveness_scores = {
                "latency_improvement": (baseline.avg_planning_time - recent.avg_planning_time) / baseline.avg_planning_time,
                "throughput_improvement": (recent.tasks_per_second - baseline.tasks_per_second) / baseline.tasks_per_second,
                "resource_efficiency": 1.0 - recent.memory_utilization / baseline.memory_utilization if baseline.memory_utilization > 0 else 0.0
            }
        
        return {
            "optimization_enabled": self.optimization_enabled,
            "optimization_strategy": self.optimization_strategy.value,
            "auto_scaling_enabled": self.enable_auto_scaling,
            "caching_enabled": self.enable_caching,
            "current_metrics": current_metrics.__dict__ if current_metrics else {},
            "resource_pool_stats": self.resource_pool.get_pool_stats(),
            "active_optimizations": len(self.active_optimizations),
            "optimization_effectiveness": effectiveness_scores,
            "scaling_policies": len(self.scaling_policies),
            "recent_scaling_events": self.scaling_history[-5:],
            "current_instance_counts": dict(self.current_instances),
            "performance_targets": self.optimization_targets,
            "optimization_features": {
                "adaptive_batching": self.adaptive_batching_enabled,
                "dynamic_load_balancing": self.dynamic_load_balancing_enabled,
                "predictive_scaling": self.predictive_scaling_enabled
            }
        }