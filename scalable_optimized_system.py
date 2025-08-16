#!/usr/bin/env python3
"""
Scalable Optimized System - Generation 3: MAKE IT SCALE
Advanced performance optimization, caching, concurrency, and auto-scaling.
"""

import asyncio
import time
import uuid
import json
import logging
import statistics
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from federated_dp_llm import (
    PrivacyAccountant, DPConfig, FederatedRouter, 
    HospitalNode, PrivateInferenceClient, BudgetManager
)
from federated_dp_llm.routing.load_balancer import InferenceRequest, InferenceResponse
from federated_dp_llm.optimization.advanced_performance_optimizer import AdvancedPerformanceOptimizer
from federated_dp_llm.optimization.caching import IntelligentCache, CacheStrategy
from federated_dp_llm.optimization.connection_pool import ConnectionPoolManager


class ScalingStrategy(Enum):
    """Auto-scaling strategies for system optimization."""
    REACTIVE = "reactive"  # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    PROACTIVE = "proactive"  # Scale preemptively
    ADAPTIVE = "adaptive"  # Learn and adapt scaling patterns


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms for optimal distribution."""
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME_BASED = "response_time_based"
    RESOURCE_BASED = "resource_based"
    QUANTUM_COHERENCE = "quantum_coherence"


class CacheLevel(Enum):
    """Multi-level caching hierarchy."""
    L1_MEMORY = "l1_memory"  # In-memory cache
    L2_DISTRIBUTED = "l2_distributed"  # Distributed cache
    L3_PERSISTENT = "l3_persistent"  # Persistent storage
    L4_CDN = "l4_cdn"  # Content delivery network


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics tracking."""
    timestamp: float
    requests_per_second: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    memory_usage_mb: float
    cpu_utilization: float
    cache_hit_ratio: float
    error_rate: float
    throughput_mbps: float
    concurrent_requests: int
    queue_depth: int


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    min_instances: int = 2
    max_instances: int = 20
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.8
    target_response_time_ms: float = 200.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    predictive_window: int = 3600  # 1 hour
    scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE


@dataclass
class OptimizationConfig:
    """Advanced optimization configuration."""
    enable_intelligent_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 1024
    enable_connection_pooling: bool = True
    max_connections_per_node: int = 100
    connection_timeout_seconds: int = 30
    enable_request_batching: bool = True
    batch_size: int = 10
    batch_timeout_ms: int = 50
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    enable_async_processing: bool = True
    worker_thread_count: int = None  # Auto-detect
    enable_load_prediction: bool = True
    prediction_model: str = "exponential_smoothing"


class IntelligentLoadBalancer:
    """Advanced load balancer with multiple algorithms and optimization."""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.QUANTUM_COHERENCE):
        self.algorithm = algorithm
        self.node_weights: Dict[str, float] = {}
        self.node_connections: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.node_resources: Dict[str, Dict[str, float]] = {}
        self.quantum_coherence_scores: Dict[str, float] = {}
        
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update metrics for load balancing decisions."""
        self.node_resources[node_id] = {
            "cpu_usage": metrics.get("cpu_usage", 0.0),
            "memory_usage": metrics.get("memory_usage", 0.0),
            "network_usage": metrics.get("network_usage", 0.0),
            "queue_depth": metrics.get("queue_depth", 0),
            "active_requests": metrics.get("active_requests", 0)
        }
        
        # Update response time history
        if "response_time" in metrics:
            self.response_times[node_id].append(metrics["response_time"])
    
    def select_optimal_node(self, available_nodes: List[str], request_context: Dict[str, Any]) -> str:
        """Select optimal node based on configured algorithm."""
        if not available_nodes:
            raise ValueError("No available nodes for load balancing")
        
        if self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(available_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(available_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.RESPONSE_TIME_BASED:
            return self._response_time_based(available_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            return self._resource_based(available_nodes)
        elif self.algorithm == LoadBalancingAlgorithm.QUANTUM_COHERENCE:
            return self._quantum_coherence_based(available_nodes, request_context)
        else:
            return available_nodes[0]  # Fallback
    
    def _weighted_round_robin(self, nodes: List[str]) -> str:
        """Weighted round-robin based on node capacity."""
        best_node = nodes[0]
        best_score = float('-inf')
        
        for node in nodes:
            weight = self.node_weights.get(node, 1.0)
            connections = self.node_connections.get(node, 0)
            score = weight / (connections + 1)
            
            if score > best_score:
                best_score = score
                best_node = node
        
        self.node_connections[best_node] += 1
        return best_node
    
    def _least_connections(self, nodes: List[str]) -> str:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: self.node_connections.get(n, 0))
    
    def _response_time_based(self, nodes: List[str]) -> str:
        """Select node with best average response time."""
        best_node = nodes[0]
        best_avg_time = float('inf')
        
        for node in nodes:
            times = list(self.response_times.get(node, []))
            if times:
                avg_time = statistics.mean(times)
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_node = node
        
        return best_node
    
    def _resource_based(self, nodes: List[str]) -> str:
        """Select node with best resource availability."""
        best_node = nodes[0]
        best_score = float('-inf')
        
        for node in nodes:
            resources = self.node_resources.get(node, {})
            cpu_available = 1.0 - resources.get("cpu_usage", 0.0)
            memory_available = 1.0 - resources.get("memory_usage", 0.0)
            
            # Composite resource score
            score = (cpu_available * 0.4 + memory_available * 0.4 + 
                    (1.0 / (resources.get("queue_depth", 1) + 1)) * 0.2)
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _quantum_coherence_based(self, nodes: List[str], request_context: Dict[str, Any]) -> str:
        """Quantum-inspired coherence-based selection."""
        # Calculate coherence scores based on request affinity and node state
        request_department = request_context.get("department", "general")
        request_priority = request_context.get("priority", 5)
        
        best_node = nodes[0]
        best_coherence = float('-inf')
        
        for node in nodes:
            resources = self.node_resources.get(node, {})
            
            # Base coherence from resource availability
            resource_coherence = (
                (1.0 - resources.get("cpu_usage", 0.0)) * 0.3 +
                (1.0 - resources.get("memory_usage", 0.0)) * 0.3 +
                (1.0 / (resources.get("active_requests", 1) + 1)) * 0.4
            )
            
            # Department affinity (simulate quantum entanglement)
            dept_affinity = 1.0
            if hasattr(self, f"{node}_department_affinity"):
                dept_affinity = getattr(self, f"{node}_department_affinity", {}).get(request_department, 1.0)
            
            # Priority resonance (simulate quantum interference) 
            priority_resonance = 1.0 - abs(request_priority - 5) / 10.0
            
            # Temporal coherence (recent performance)
            temporal_coherence = 1.0
            if node in self.response_times and self.response_times[node]:
                recent_times = list(self.response_times[node])[-10:]  # Last 10 requests
                if recent_times:
                    temporal_coherence = 1.0 / (1.0 + statistics.stdev(recent_times)) if len(recent_times) > 1 else 1.0
            
            # Composite quantum coherence
            coherence = (resource_coherence * 0.4 + dept_affinity * 0.3 + 
                        priority_resonance * 0.2 + temporal_coherence * 0.1)
            
            if coherence > best_coherence:
                best_coherence = coherence
                best_node = node
        
        self.quantum_coherence_scores[best_node] = best_coherence
        return best_node


class AdvancedCacheManager:
    """Multi-level intelligent caching system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.l1_cache: Dict[str, Any] = {}  # In-memory
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self.cache_timestamps: Dict[str, float] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.cache_levels = {
            CacheLevel.L1_MEMORY: self.l1_cache
        }
        
    def _generate_cache_key(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate intelligent cache key based on semantic content."""
        # Create normalized prompt for caching
        normalized_prompt = prompt.lower().strip()
        
        # Include relevant context
        context_str = json.dumps({
            k: v for k, v in context.items() 
            if k in ["department", "model_name", "priority"]
        }, sort_keys=True)
        
        # Generate hash
        cache_content = f"{normalized_prompt}|{context_str}"
        return hashlib.sha256(cache_content.encode()).hexdigest()[:16]
    
    async def get(self, prompt: str, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached result with intelligent cache strategies."""
        cache_key = self._generate_cache_key(prompt, context)
        
        # Check L1 cache
        if cache_key in self.l1_cache:
            # Verify TTL
            if time.time() - self.cache_timestamps.get(cache_key, 0) < self.config.cache_ttl_seconds:
                self.cache_stats[cache_key]["hits"] += 1
                self._update_access_pattern(cache_key)
                return self.l1_cache[cache_key]
            else:
                # Expired
                del self.l1_cache[cache_key]
                del self.cache_timestamps[cache_key]
        
        self.cache_stats[cache_key]["misses"] += 1
        return None
    
    async def set(self, prompt: str, context: Dict[str, Any], result: Any):
        """Cache result with intelligent eviction policies."""
        cache_key = self._generate_cache_key(prompt, context)
        
        # Check cache size limits
        if len(self.l1_cache) >= 1000:  # Basic size limit
            await self._evict_cache_entries()
        
        self.l1_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        self._update_access_pattern(cache_key)
    
    def _update_access_pattern(self, cache_key: str):
        """Track access patterns for intelligent caching."""
        self.access_patterns[cache_key].append(time.time())
        
        # Keep only recent accesses
        cutoff_time = time.time() - 3600  # Last hour
        self.access_patterns[cache_key] = [
            t for t in self.access_patterns[cache_key] if t > cutoff_time
        ]
    
    async def _evict_cache_entries(self):
        """Intelligent cache eviction based on access patterns."""
        # Calculate access scores
        access_scores = {}
        current_time = time.time()
        
        for cache_key in list(self.l1_cache.keys()):
            accesses = self.access_patterns.get(cache_key, [])
            age = current_time - self.cache_timestamps.get(cache_key, current_time)
            
            # Score based on frequency, recency, and age
            frequency_score = len(accesses)
            recency_score = 1.0 / (age + 1)
            access_scores[cache_key] = frequency_score * recency_score
        
        # Evict lowest scoring entries
        sorted_entries = sorted(access_scores.items(), key=lambda x: x[1])
        evict_count = len(self.l1_cache) // 4  # Evict 25%
        
        for cache_key, _ in sorted_entries[:evict_count]:
            self.l1_cache.pop(cache_key, None)
            self.cache_timestamps.pop(cache_key, None)
            self.access_patterns.pop(cache_key, None)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_hits = sum(stats["hits"] for stats in self.cache_stats.values())
        total_misses = sum(stats["misses"] for stats in self.cache_stats.values())
        total_requests = total_hits + total_misses
        
        hit_ratio = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_requests": total_requests,
            "cache_hits": total_hits,
            "cache_misses": total_misses,
            "hit_ratio": hit_ratio,
            "cache_size": len(self.l1_cache),
            "unique_keys": len(self.cache_stats),
            "memory_usage_mb": len(str(self.l1_cache).encode()) / (1024 * 1024)
        }


class PredictiveScaler:
    """Predictive auto-scaling based on historical patterns and ML."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.historical_metrics: List[PerformanceMetrics] = []
        self.current_instances = config.min_instances
        self.last_scale_action = 0
        self.predicted_load: Optional[float] = None
        
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add performance metrics for prediction."""
        self.historical_metrics.append(metrics)
        
        # Keep only recent history
        cutoff_time = time.time() - self.config.predictive_window
        self.historical_metrics = [
            m for m in self.historical_metrics if m.timestamp > cutoff_time
        ]
    
    def predict_future_load(self, horizon_minutes: int = 30) -> float:
        """Predict future load using exponential smoothing."""
        if len(self.historical_metrics) < 3:
            return 0.5  # Default prediction
        
        # Extract load indicators
        loads = [
            (m.cpu_utilization * 0.4 + 
             m.memory_usage_mb / 1024 * 0.3 +  # Normalize to 0-1
             m.requests_per_second / 100 * 0.3)  # Normalize to 0-1
            for m in self.historical_metrics[-50:]  # Last 50 measurements
        ]
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        prediction = loads[0]
        
        for load in loads[1:]:
            prediction = alpha * load + (1 - alpha) * prediction
        
        # Add trend component
        if len(loads) >= 10:
            recent_trend = statistics.mean(loads[-5:]) - statistics.mean(loads[-10:-5])
            prediction += recent_trend * horizon_minutes / 60
        
        self.predicted_load = max(0.0, min(1.0, prediction))
        return self.predicted_load
    
    def should_scale_up(self, current_load: float) -> bool:
        """Determine if scaling up is needed."""
        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_action
        
        # Check cooldown
        if time_since_last_scale < self.config.scale_up_cooldown:
            return False
        
        # Check if at max instances
        if self.current_instances >= self.config.max_instances:
            return False
        
        # Reactive scaling
        if current_load > self.config.scale_up_threshold:
            return True
        
        # Predictive scaling
        if self.config.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.ADAPTIVE]:
            predicted_load = self.predict_future_load()
            if predicted_load > self.config.scale_up_threshold:
                return True
        
        return False
    
    def should_scale_down(self, current_load: float) -> bool:
        """Determine if scaling down is needed."""
        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_action
        
        # Check cooldown
        if time_since_last_scale < self.config.scale_down_cooldown:
            return False
        
        # Check if at min instances
        if self.current_instances <= self.config.min_instances:
            return False
        
        # Conservative scaling down
        if current_load < self.config.scale_down_threshold:
            # Ensure predicted load also suggests scaling down
            if self.config.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.ADAPTIVE]:
                predicted_load = self.predict_future_load()
                return predicted_load < self.config.scale_down_threshold
            return True
        
        return False
    
    def execute_scale_action(self, scale_up: bool) -> int:
        """Execute scaling action and return new instance count."""
        if scale_up:
            self.current_instances = min(
                self.config.max_instances,
                int(self.current_instances * 1.5)  # Scale up by 50%
            )
        else:
            self.current_instances = max(
                self.config.min_instances,
                int(self.current_instances * 0.8)  # Scale down by 20%
            )
        
        self.last_scale_action = time.time()
        return self.current_instances


class ScalableOptimizedSystem:
    """Highly scalable and optimized federated system."""
    
    def __init__(
        self,
        optimization_config: OptimizationConfig,
        scaling_config: ScalingConfig
    ):
        self.optimization_config = optimization_config
        self.scaling_config = scaling_config
        self.system_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Core components
        self.privacy_accountant = PrivacyAccountant(DPConfig())
        self.budget_manager = BudgetManager({
            "emergency": 20.0,
            "critical_care": 15.0,
            "radiology": 10.0,
            "general": 5.0,
            "research": 2.0
        })
        
        # Optimization components
        self.load_balancer = IntelligentLoadBalancer(
            LoadBalancingAlgorithm.QUANTUM_COHERENCE
        )
        self.cache_manager = AdvancedCacheManager(optimization_config)
        self.predictive_scaler = PredictiveScaler(scaling_config)
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=1000)
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.active_requests: Set[str] = set()
        
        # Connection and resource management
        self.connection_pools: Dict[str, Any] = {}
        if optimization_config.worker_thread_count is None:
            self.worker_thread_count = mp.cpu_count()
        else:
            self.worker_thread_count = optimization_config.worker_thread_count
        
        self.thread_executor = ThreadPoolExecutor(max_workers=self.worker_thread_count)
        
        # Request batching
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.ScalableSystem")
        self.logger.info(f"ScalableOptimizedSystem initialized with ID: {self.system_id}")
    
    async def initialize_optimized_network(self, hospitals: List[Dict[str, Any]]) -> bool:
        """Initialize hospital network with optimization features."""
        try:
            # Create router with optimization
            self.router = FederatedRouter(
                model_name="medllama-7b",
                num_shards=min(8, len(hospitals))  # Increased sharding
            )
            
            # Create optimized hospital nodes
            hospital_nodes = []
            for hospital_config in hospitals:
                node = HospitalNode(
                    id=hospital_config["id"],
                    endpoint=hospital_config["endpoint"],
                    data_size=hospital_config.get("data_size", 50000),
                    compute_capacity=hospital_config.get("compute_capacity", "4xA100"),
                    department=hospital_config.get("department"),
                    region=hospital_config.get("region")
                )
                hospital_nodes.append(node)
                
                # Initialize connection pools
                if self.optimization_config.enable_connection_pooling:
                    self.connection_pools[node.id] = {
                        "max_connections": self.optimization_config.max_connections_per_node,
                        "active_connections": 0,
                        "connection_queue": asyncio.Queue(maxsize=50)
                    }
            
            await self.router.register_nodes(hospital_nodes)
            
            # Start optimization tasks
            await self._start_optimization_tasks()
            
            self.logger.info(f"Optimized network initialized with {len(hospital_nodes)} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optimized network: {str(e)}")
            return False
    
    async def process_request_optimized(
        self,
        user_id: str,
        clinical_prompt: str,
        department: str = "general",
        priority: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Process request with full optimization stack."""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.active_requests.add(request_id)
            
            # Intelligent caching check
            cache_context = {
                "department": department,
                "model_name": "medllama-7b",
                "priority": priority
            }
            
            cached_result = await self.cache_manager.get(clinical_prompt, cache_context)
            if cached_result:
                self.logger.info(f"Cache hit for request {request_id[:8]}")
                
                # Update performance metrics
                latency = time.time() - start_time
                await self._record_performance_metrics(latency, True, cache_hit=True)
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "response": cached_result["response"],
                    "privacy_cost": cached_result["privacy_cost"],
                    "latency": latency,
                    "cache_hit": True,
                    "processing_nodes": cached_result.get("processing_nodes", []),
                    "optimization_applied": True
                }
            
            # Request batching optimization
            if self.optimization_config.enable_request_batching and priority <= 5:
                return await self._process_with_batching(
                    request_id, user_id, clinical_prompt, department, priority, **kwargs
                )
            
            # Direct processing for high-priority requests
            return await self._process_direct_optimized(
                request_id, user_id, clinical_prompt, department, priority, **kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Optimized request processing failed: {str(e)}")
            latency = time.time() - start_time
            await self._record_performance_metrics(latency, False)
            
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "latency": latency
            }
        finally:
            self.active_requests.discard(request_id)
    
    async def _process_with_batching(
        self, request_id: str, user_id: str, prompt: str, 
        department: str, priority: int, **kwargs
    ) -> Dict[str, Any]:
        """Process request with intelligent batching."""
        
        batch_request = {
            "request_id": request_id,
            "user_id": user_id,
            "prompt": prompt,
            "department": department,
            "priority": priority,
            "timestamp": time.time(),
            "future": asyncio.Future(),
            **kwargs
        }
        
        async with self.batch_lock:
            self.batch_queue.append(batch_request)
            
            # Trigger batch processing if conditions met
            if (len(self.batch_queue) >= self.optimization_config.batch_size or
                time.time() - self.batch_queue[0]["timestamp"] > 
                self.optimization_config.batch_timeout_ms / 1000):
                
                await self._process_batch()
        
        # Wait for batch processing result
        return await batch_request["future"]
    
    async def _process_batch(self):
        """Process accumulated batch of requests."""
        if not self.batch_queue:
            return
        
        current_batch = self.batch_queue.copy()
        self.batch_queue.clear()
        
        # Group by department for efficient processing
        dept_groups = defaultdict(list)
        for req in current_batch:
            dept_groups[req["department"]].append(req)
        
        # Process each department group
        tasks = []
        for department, requests in dept_groups.items():
            task = self._process_department_batch(department, requests)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _process_department_batch(self, department: str, requests: List[Dict]):
        """Process batch of requests for a specific department."""
        try:
            # Select optimal node for this department
            available_nodes = list(self.connection_pools.keys())
            if available_nodes:
                optimal_node = self.load_balancer.select_optimal_node(
                    available_nodes, {"department": department}
                )
                
                # Process requests in parallel with node affinity
                tasks = []
                for request in requests:
                    task = self._process_single_request_on_node(request, optimal_node)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Set futures with results
                for request, result in zip(requests, results):
                    if isinstance(result, Exception):
                        request["future"].set_exception(result)
                    else:
                        request["future"].set_result(result)
        
        except Exception as e:
            # Set all futures to failed
            for request in requests:
                if not request["future"].done():
                    request["future"].set_exception(e)
    
    async def _process_single_request_on_node(
        self, request: Dict[str, Any], node_id: str
    ) -> Dict[str, Any]:
        """Process single request on specified node with optimization."""
        
        start_time = time.time()
        
        # Create inference request
        inference_request = InferenceRequest(
            request_id=request["request_id"],
            user_id=request["user_id"],
            prompt=request["prompt"],
            model_name="medllama-7b",
            max_privacy_budget=request.get("max_privacy_budget", 0.1),
            department=request["department"],
            priority=request["priority"]
        )
        
        # Route through optimized router
        response = await self.router.route_request(inference_request)
        
        # Cache successful responses
        if response and hasattr(response, 'text'):
            cache_context = {
                "department": request["department"],
                "model_name": "medllama-7b",
                "priority": request["priority"]
            }
            
            cache_data = {
                "response": response.text,
                "privacy_cost": response.privacy_cost,
                "processing_nodes": response.processing_nodes
            }
            
            await self.cache_manager.set(request["prompt"], cache_context, cache_data)
        
        # Record performance
        latency = time.time() - start_time
        await self._record_performance_metrics(latency, True)
        
        return {
            "request_id": request["request_id"],
            "success": True,
            "response": response.text if response else "No response",
            "privacy_cost": response.privacy_cost if response else 0.0,
            "latency": latency,
            "processing_nodes": response.processing_nodes if response else [],
            "batch_processed": True
        }
    
    async def _process_direct_optimized(
        self, request_id: str, user_id: str, prompt: str,
        department: str, priority: int, **kwargs
    ) -> Dict[str, Any]:
        """Process high-priority request directly with optimizations."""
        
        start_time = time.time()
        
        # Select optimal node
        available_nodes = list(self.connection_pools.keys())
        if not available_nodes:
            raise RuntimeError("No available nodes")
        
        optimal_node = self.load_balancer.select_optimal_node(
            available_nodes, {
                "department": department,
                "priority": priority,
                "user_id": user_id
            }
        )
        
        # Create and process request
        inference_request = InferenceRequest(
            request_id=request_id,
            user_id=user_id,
            prompt=prompt,
            model_name="medllama-7b",
            max_privacy_budget=kwargs.get("max_privacy_budget", 0.1),
            department=department,
            priority=priority
        )
        
        # Execute with connection pooling
        async with self._get_connection(optimal_node):
            response = await self.router.route_request(inference_request)
        
        # Cache result
        cache_context = {
            "department": department,
            "model_name": "medllama-7b",
            "priority": priority
        }
        
        if response:
            cache_data = {
                "response": response.text,
                "privacy_cost": response.privacy_cost,
                "processing_nodes": response.processing_nodes
            }
            await self.cache_manager.set(prompt, cache_context, cache_data)
        
        # Record performance
        latency = time.time() - start_time
        await self._record_performance_metrics(latency, True)
        
        return {
            "request_id": request_id,
            "success": True,
            "response": response.text if response else "No response",
            "privacy_cost": response.privacy_cost if response else 0.0,
            "latency": latency,
            "processing_nodes": response.processing_nodes if response else [],
            "optimized_routing": True,
            "selected_node": optimal_node
        }
    
    async def _get_connection(self, node_id: str):
        """Get connection from pool with proper resource management."""
        
        class ConnectionContext:
            def __init__(self, pool_info, node_id):
                self.pool_info = pool_info
                self.node_id = node_id
            
            async def __aenter__(self):
                self.pool_info["active_connections"] += 1
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.pool_info["active_connections"] -= 1
        
        pool_info = self.connection_pools.get(node_id, {"active_connections": 0})
        return ConnectionContext(pool_info, node_id)
    
    async def _record_performance_metrics(
        self, latency: float, success: bool, cache_hit: bool = False
    ):
        """Record comprehensive performance metrics."""
        
        current_time = time.time()
        
        # Calculate current system metrics
        cpu_utilization = len(self.active_requests) / 100.0  # Simulated
        memory_usage_mb = len(str(self.cache_manager.l1_cache).encode()) / (1024 * 1024)
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_statistics()
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            requests_per_second=len(self.active_requests),  # Approximate
            avg_response_time=latency,
            p95_response_time=latency * 1.2,  # Estimate
            p99_response_time=latency * 1.5,  # Estimate
            memory_usage_mb=memory_usage_mb,
            cpu_utilization=cpu_utilization,
            cache_hit_ratio=cache_stats["hit_ratio"],
            error_rate=0.0 if success else 1.0,
            throughput_mbps=1.0,  # Estimate
            concurrent_requests=len(self.active_requests),
            queue_depth=self.request_queue.qsize()
        )
        
        self.performance_history.append(metrics)
        
        # Update predictive scaler
        self.predictive_scaler.add_metrics(metrics)
        
        # Update load balancer metrics
        for node_id in self.connection_pools.keys():
            pool_info = self.connection_pools[node_id]
            self.load_balancer.update_node_metrics(node_id, {
                "response_time": latency,
                "active_requests": pool_info["active_connections"],
                "cpu_usage": cpu_utilization,
                "memory_usage": memory_usage_mb / 1024,  # Normalize
                "queue_depth": 0
            })
    
    async def _start_optimization_tasks(self):
        """Start background optimization tasks."""
        
        # Batch processing task
        if self.optimization_config.enable_request_batching:
            asyncio.create_task(self._batch_processing_loop())
        
        # Auto-scaling task
        asyncio.create_task(self._auto_scaling_loop())
        
        # Performance monitoring task
        asyncio.create_task(self._performance_monitoring_loop())
    
    async def _batch_processing_loop(self):
        """Background task for periodic batch processing."""
        while True:
            try:
                await asyncio.sleep(self.optimization_config.batch_timeout_ms / 1000)
                
                async with self.batch_lock:
                    if self.batch_queue:
                        await self._process_batch()
                        
            except Exception as e:
                self.logger.error(f"Batch processing loop error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _auto_scaling_loop(self):
        """Background task for predictive auto-scaling."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.performance_history:
                    continue
                
                # Calculate current load
                recent_metrics = list(self.performance_history)[-10:]  # Last 10 measurements
                avg_cpu = statistics.mean(m.cpu_utilization for m in recent_metrics)
                avg_memory = statistics.mean(m.memory_usage_mb for m in recent_metrics)
                
                current_load = max(avg_cpu, avg_memory / 1024)  # Normalize memory to 0-1
                
                # Check scaling decisions
                if self.predictive_scaler.should_scale_up(current_load):
                    new_instances = self.predictive_scaler.execute_scale_action(scale_up=True)
                    self.logger.info(f"Scaled up to {new_instances} instances (load: {current_load:.2f})")
                    
                elif self.predictive_scaler.should_scale_down(current_load):
                    new_instances = self.predictive_scaler.execute_scale_action(scale_up=False)
                    self.logger.info(f"Scaled down to {new_instances} instances (load: {current_load:.2f})")
                
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring and optimization."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Log performance summary
                if self.performance_history:
                    recent_metrics = list(self.performance_history)[-10:]
                    avg_latency = statistics.mean(m.avg_response_time for m in recent_metrics)
                    avg_cpu = statistics.mean(m.cpu_utilization for m in recent_metrics)
                    
                    cache_stats = self.cache_manager.get_cache_statistics()
                    
                    self.logger.info(
                        f"Performance: {avg_latency:.3f}s latency, "
                        f"{avg_cpu:.2%} CPU, "
                        f"{cache_stats['hit_ratio']:.2%} cache hit ratio, "
                        f"{len(self.active_requests)} active requests"
                    )
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(30)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization and scaling statistics."""
        
        # Performance statistics
        perf_stats = {}
        if self.performance_history:
            recent_metrics = list(self.performance_history)[-50:]  # Last 50 measurements
            perf_stats = {
                "avg_latency": statistics.mean(m.avg_response_time for m in recent_metrics),
                "p95_latency": sorted(m.p95_response_time for m in recent_metrics)[int(len(recent_metrics) * 0.95)],
                "avg_cpu_utilization": statistics.mean(m.cpu_utilization for m in recent_metrics),
                "avg_memory_usage_mb": statistics.mean(m.memory_usage_mb for m in recent_metrics),
                "avg_concurrent_requests": statistics.mean(m.concurrent_requests for m in recent_metrics)
            }
        
        # Cache statistics
        cache_stats = self.cache_manager.get_cache_statistics()
        
        # Scaling statistics
        scaling_stats = {
            "current_instances": self.predictive_scaler.current_instances,
            "predicted_load": self.predictive_scaler.predicted_load,
            "scaling_strategy": self.scaling_config.scaling_strategy.value,
            "last_scale_action": self.predictive_scaler.last_scale_action
        }
        
        # Load balancer statistics
        lb_stats = {
            "algorithm": self.load_balancer.algorithm.value,
            "node_connections": dict(self.load_balancer.node_connections),
            "quantum_coherence_scores": dict(self.load_balancer.quantum_coherence_scores)
        }
        
        return {
            "system_id": self.system_id,
            "uptime": time.time() - self.start_time,
            "optimization_config": {
                "caching_enabled": self.optimization_config.enable_intelligent_caching,
                "connection_pooling": self.optimization_config.enable_connection_pooling,
                "request_batching": self.optimization_config.enable_request_batching,
                "async_processing": self.optimization_config.enable_async_processing,
                "worker_threads": self.worker_thread_count
            },
            "performance_statistics": perf_stats,
            "cache_statistics": cache_stats,
            "scaling_statistics": scaling_stats,
            "load_balancer_statistics": lb_stats,
            "active_requests": len(self.active_requests),
            "batch_queue_size": len(self.batch_queue),
            "timestamp": time.time()
        }
    
    async def shutdown_optimized(self):
        """Graceful shutdown with optimization cleanup."""
        self.logger.info("Starting optimized shutdown...")
        
        # Process remaining batches
        async with self.batch_lock:
            if self.batch_queue:
                await self._process_batch()
        
        # Wait for active requests
        timeout = 30
        start_shutdown = time.time()
        
        while self.active_requests and (time.time() - start_shutdown) < timeout:
            await asyncio.sleep(1)
        
        # Cleanup thread executor
        self.thread_executor.shutdown(wait=True)
        
        self.logger.info("Optimized shutdown completed")


async def demo_scalable_system():
    """Demonstrate scalable optimized system capabilities."""
    print("\n⚡ Scalable Optimized System - Generation 3: MAKE IT SCALE")
    print("=" * 80)
    
    # Configure optimization
    opt_config = OptimizationConfig(
        enable_intelligent_caching=True,
        cache_ttl_seconds=1800,
        enable_connection_pooling=True,
        enable_request_batching=True,
        batch_size=5,
        batch_timeout_ms=100,
        enable_async_processing=True,
        enable_load_prediction=True
    )
    
    # Configure scaling
    scale_config = ScalingConfig(
        min_instances=2,
        max_instances=10,
        target_cpu_utilization=0.7,
        scaling_strategy=ScalingStrategy.ADAPTIVE,
        scale_up_cooldown=60,  # Faster for demo
        scale_down_cooldown=120
    )
    
    # Create scalable system
    system = ScalableOptimizedSystem(opt_config, scale_config)
    
    # Define optimized hospital network
    hospitals = [
        {
            "id": "hospital_scalable_1",
            "endpoint": "https://scalable1.federated.health:8443",
            "data_size": 120000,
            "compute_capacity": "12xA100",
            "department": "multi_specialty",
            "region": "northeast_us"
        },
        {
            "id": "hospital_scalable_2", 
            "endpoint": "https://scalable2.federated.health:8443",
            "data_size": 100000,
            "compute_capacity": "10xA100",
            "department": "research",
            "region": "west_us"
        },
        {
            "id": "hospital_scalable_3",
            "endpoint": "https://scalable3.federated.health:8443",
            "data_size": 80000,
            "compute_capacity": "8xA100",
            "department": "emergency",
            "region": "south_us"
        }
    ]
    
    # Initialize optimized network
    print("🏥 Initializing scalable optimized network...")
    success = await system.initialize_optimized_network(hospitals)
    
    if not success:
        print("❌ Failed to initialize optimized network")
        return
    
    print("✅ Scalable network initialized with optimization features")
    
    # Demonstrate intelligent caching
    print("\n🧠 Testing intelligent caching...")
    
    # First request (cache miss)
    cache_test_prompt = "Patient with chest pain and elevated troponins, assess for MI"
    result1 = await system.process_request_optimized(
        user_id="dr_cache_test",
        clinical_prompt=cache_test_prompt,
        department="emergency",
        priority=8
    )
    
    # Second identical request (cache hit)
    result2 = await system.process_request_optimized(
        user_id="dr_cache_test",
        clinical_prompt=cache_test_prompt,
        department="emergency", 
        priority=8
    )
    
    print(f"✅ Cache test: First request {result1['latency']:.3f}s (miss), "
          f"Second request {result2['latency']:.3f}s ({'hit' if result2.get('cache_hit') else 'miss'})")
    
    # Demonstrate batch processing
    print("\n📦 Testing request batching...")
    
    batch_requests = []
    for i in range(8):  # Batch size trigger
        task = system.process_request_optimized(
            user_id=f"dr_batch_{i:02d}",
            clinical_prompt=f"Patient {i+1} requires differential diagnosis evaluation",
            department="general",
            priority=3  # Low priority for batching
        )
        batch_requests.append(task)
    
    batch_results = await asyncio.gather(*batch_requests)
    batch_processed = sum(1 for r in batch_results if r.get("batch_processed"))
    
    print(f"✅ Batch processing: {batch_processed}/{len(batch_results)} requests batched")
    
    # Demonstrate load balancing
    print("\n⚖️  Testing quantum coherence load balancing...")
    
    load_balance_tasks = []
    departments = ["emergency", "critical_care", "radiology", "general"]
    
    for i in range(12):
        dept = departments[i % len(departments)]
        task = system.process_request_optimized(
            user_id=f"dr_load_{i:02d}",
            clinical_prompt=f"Load balancing test request {i+1}",
            department=dept,
            priority=7  # High priority for direct processing
        )
        load_balance_tasks.append(task)
    
    load_results = await asyncio.gather(*load_balance_tasks)
    
    # Analyze node distribution
    node_distribution = {}
    for result in load_results:
        if result.get("selected_node"):
            node = result["selected_node"]
            node_distribution[node] = node_distribution.get(node, 0) + 1
    
    print("✅ Load balancing distribution:")
    for node, count in node_distribution.items():
        print(f"   - {node}: {count} requests")
    
    # Demonstrate auto-scaling simulation
    print("\n📈 Testing predictive auto-scaling...")
    
    # Simulate load spike
    high_load_tasks = []
    for i in range(20):  # High load
        task = system.process_request_optimized(
            user_id=f"dr_scale_{i:03d}",
            clinical_prompt=f"High load scenario request {i+1}",
            department="emergency",
            priority=9
        )
        high_load_tasks.append(task)
    
    # Process in waves to trigger scaling
    wave1 = high_load_tasks[:10]
    wave2 = high_load_tasks[10:]
    
    results1 = await asyncio.gather(*wave1)
    await asyncio.sleep(2)  # Allow scaling decision
    results2 = await asyncio.gather(*wave2)
    
    print(f"✅ Auto-scaling test: Wave 1: {len(results1)} requests, Wave 2: {len(results2)} requests")
    
    # Show comprehensive statistics
    print(f"\n📊 Comprehensive Optimization Statistics:")
    stats = system.get_optimization_statistics()
    
    print(f"   System uptime: {stats['uptime']:.1f}s")
    print(f"   Active requests: {stats['active_requests']}")
    print(f"   Current instances: {stats['scaling_statistics']['current_instances']}")
    print(f"   Predicted load: {stats['scaling_statistics'].get('predicted_load', 'N/A')}")
    
    if 'performance_statistics' in stats and stats['performance_statistics']:
        perf = stats['performance_statistics']
        print(f"   Average latency: {perf.get('avg_latency', 0):.3f}s")
        print(f"   P95 latency: {perf.get('p95_latency', 0):.3f}s") 
        print(f"   CPU utilization: {perf.get('avg_cpu_utilization', 0):.2%}")
        print(f"   Memory usage: {perf.get('avg_memory_usage_mb', 0):.1f}MB")
    
    cache_stats = stats['cache_statistics']
    print(f"   Cache hit ratio: {cache_stats['hit_ratio']:.2%}")
    print(f"   Cache size: {cache_stats['cache_size']} entries")
    print(f"   Cache memory: {cache_stats['memory_usage_mb']:.1f}MB")
    
    lb_stats = stats['load_balancer_statistics']
    print(f"   Load balancing: {lb_stats['algorithm']}")
    
    opt_config = stats['optimization_config']
    print(f"   Optimizations enabled:")
    print(f"     - Caching: {opt_config['caching_enabled']}")
    print(f"     - Connection pooling: {opt_config['connection_pooling']}")
    print(f"     - Request batching: {opt_config['request_batching']}")
    print(f"     - Async processing: {opt_config['async_processing']}")
    print(f"     - Worker threads: {opt_config['worker_threads']}")
    
    # Graceful shutdown
    print(f"\n🛑 Initiating optimized shutdown...")
    await system.shutdown_optimized()
    print("✅ Optimized shutdown completed successfully")
    
    print(f"\n🎉 Scalable Optimized System Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_scalable_system())