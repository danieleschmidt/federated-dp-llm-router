"""
Advanced Performance Optimizer for Federated DP-LLM System

Implements intelligent caching, auto-scaling, load balancing optimization,
and quantum-enhanced performance tuning for production scalability.
"""

import asyncio
import time
import statistics
import psutil
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"      # Maximum performance, higher resource usage
    BALANCED = "balanced"          # Balance between performance and resources
    CONSERVATIVE = "conservative"  # Resource-conscious optimization
    ADAPTIVE = "adaptive"          # AI-driven adaptive optimization


class ScalingDirection(Enum):
    """Auto-scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"      # Horizontal scaling
    SCALE_IN = "scale_in"        # Horizontal scaling down
    MAINTAIN = "maintain"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    timestamp: float
    request_latency: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    network_latency: float = 0.0
    privacy_computation_time: float = 0.0
    quantum_coherence_score: float = 1.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "timestamp": self.timestamp,
            "request_latency": self.request_latency,
            "throughput": self.throughput,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "network_latency": self.network_latency,
            "privacy_computation_time": self.privacy_computation_time,
            "quantum_coherence_score": self.quantum_coherence_score,
            "error_rate": self.error_rate
        }


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning."""
    direction: ScalingDirection
    confidence: float
    target_replicas: int
    estimated_impact: Dict[str, float]
    reasoning: List[str]
    timestamp: float = field(default_factory=time.time)


class IntelligentCache:
    """Intelligent caching system with ML-based eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _generate_cache_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """Generate cache key from request parameters."""
        # Normalize prompt (remove extra whitespace, lowercase)
        normalized_prompt = " ".join(prompt.lower().split())
        
        # Create deterministic key
        param_str = json.dumps(params, sort_keys=True)
        key_data = f"{normalized_prompt}:{model}:{param_str}"
        
        # Use hash for fixed-length key
        import hashlib
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    async def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[str]:
        """Get cached response if available."""
        key = self._generate_cache_key(prompt, model, params)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry["timestamp"] <= self.ttl:
                # Update access pattern
                current_time = time.time()
                if key not in self.access_patterns:
                    self.access_patterns[key] = []
                self.access_patterns[key].append(current_time)
                
                # Keep only recent accesses (last hour)
                self.access_patterns[key] = [
                    t for t in self.access_patterns[key] 
                    if current_time - t <= 3600
                ]
                
                self.cache_hits += 1
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return entry["response"]
            else:
                # Expired entry
                del self.cache[key]
                if key in self.access_patterns:
                    del self.access_patterns[key]
        
        self.cache_misses += 1
        return None
    
    async def put(self, prompt: str, model: str, params: Dict[str, Any], response: str):
        """Store response in cache with intelligent eviction."""
        key = self._generate_cache_key(prompt, model, params)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size:
            await self._evict_least_valuable()
        
        # Store entry
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1
        }
        
        logger.debug(f"Cached response for key: {key[:8]}...")
    
    async def _evict_least_valuable(self):
        """Evict least valuable cache entries using ML-based scoring."""
        if not self.cache:
            return
        
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            # Calculate value score based on:
            # 1. Recency of access
            # 2. Frequency of access
            # 3. Age of entry
            
            age = current_time - entry["timestamp"]
            access_pattern = self.access_patterns.get(key, [])
            
            # Recency score (higher = more recent)
            recency_score = 1.0 / (1.0 + age / 3600)  # Decay over hours
            
            # Frequency score
            frequency_score = len(access_pattern) / max(1, age / 3600)  # Accesses per hour
            
            # Combined score (higher = more valuable)
            scores[key] = recency_score * 0.3 + frequency_score * 0.7
        
        # Evict 20% of least valuable entries
        evict_count = max(1, len(self.cache) // 5)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        
        for key in sorted_keys[:evict_count]:
            del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
        
        logger.info(f"Evicted {evict_count} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "utilization": len(self.cache) / self.max_size
        }


class AdaptiveLoadBalancer:
    """Adaptive load balancer with ML-based node selection."""
    
    def __init__(self):
        self.node_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.node_weights: Dict[str, float] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        
    def update_node_metrics(self, node_id: str, metrics: PerformanceMetrics):
        """Update performance metrics for a node."""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = deque(maxlen=100)  # Keep last 100 metrics
        
        self.node_metrics[node_id].append(metrics)
        self._update_node_weight(node_id)
    
    def _update_node_weight(self, node_id: str):
        """Update node weight based on recent performance."""
        metrics = list(self.node_metrics[node_id])
        if not metrics:
            self.node_weights[node_id] = 0.5  # Default weight
            return
        
        # Get recent metrics (last 10)
        recent_metrics = metrics[-10:]
        
        # Calculate performance score
        avg_latency = statistics.mean(m.request_latency for m in recent_metrics)
        avg_cpu = statistics.mean(m.cpu_usage for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage for m in recent_metrics)
        avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
        
        # Compute weight (higher = better performance)
        # Lower latency, CPU, memory usage and error rate = higher weight
        latency_score = 1.0 / (1.0 + avg_latency)  # Inverse latency
        resource_score = 1.0 - min(0.9, (avg_cpu + avg_memory) / 2)  # Lower resource usage
        reliability_score = 1.0 - min(0.9, avg_error_rate)  # Lower error rate
        
        weight = (latency_score * 0.4 + resource_score * 0.3 + reliability_score * 0.3)
        self.node_weights[node_id] = max(0.1, weight)  # Minimum weight of 0.1
        
        logger.debug(f"Updated weight for {node_id}: {weight:.3f}")
    
    async def select_optimal_nodes(self, request_requirements: Dict[str, Any], 
                                 num_nodes: int = 1) -> List[str]:
        """Select optimal nodes for request processing."""
        available_nodes = list(self.node_weights.keys())
        
        if len(available_nodes) <= num_nodes:
            return available_nodes
        
        # Weighted selection based on performance
        weights = [self.node_weights[node] for node in available_nodes]
        
        # Add some randomness to prevent all requests going to single best node
        import random
        adjusted_weights = []
        for w in weights:
            # Add 10% randomness
            randomness = random.uniform(0.9, 1.1)
            adjusted_weights.append(w * randomness)
        
        # Select top nodes by adjusted weight
        node_weight_pairs = list(zip(available_nodes, adjusted_weights))
        node_weight_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_nodes = [node for node, _ in node_weight_pairs[:num_nodes]]
        
        logger.debug(f"Selected nodes: {selected_nodes}")
        return selected_nodes
    
    def predict_load_distribution(self) -> Dict[str, float]:
        """Predict optimal load distribution across nodes."""
        if not self.node_weights:
            return {}
        
        total_weight = sum(self.node_weights.values())
        
        distribution = {
            node_id: weight / total_weight
            for node_id, weight in self.node_weights.items()
        }
        
        return distribution


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, min_replicas: int = 1, max_replicas: int = 10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        self.metrics_history: List[PerformanceMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        self.cooldown_period = 300  # 5 minutes between scaling decisions
        
    def analyze_scaling_need(self, current_metrics: PerformanceMetrics) -> ScalingDecision:
        """Analyze if scaling is needed based on current metrics."""
        self.metrics_history.append(current_metrics)
        
        # Keep only recent metrics (last hour)
        cutoff_time = time.time() - 3600
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Check cooldown period
        if self.scaling_history:
            last_decision = self.scaling_history[-1]
            if time.time() - last_decision.timestamp < self.cooldown_period:
                return ScalingDecision(
                    direction=ScalingDirection.MAINTAIN,
                    confidence=0.0,
                    target_replicas=self.current_replicas,
                    estimated_impact={},
                    reasoning=["Cooling down from previous scaling decision"]
                )
        
        # Analyze recent performance
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        if not recent_metrics:
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                confidence=0.0,
                target_replicas=self.current_replicas,
                estimated_impact={},
                reasoning=["Insufficient metrics for scaling decision"]
            )
        
        # Calculate averages
        avg_latency = statistics.mean(m.request_latency for m in recent_metrics)
        avg_cpu = statistics.mean(m.cpu_usage for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage for m in recent_metrics)
        avg_throughput = statistics.mean(m.throughput for m in recent_metrics)
        avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
        
        # Scaling decision logic
        reasoning = []
        confidence = 0.0
        direction = ScalingDirection.MAINTAIN
        
        # Scale up conditions
        scale_up_signals = 0
        if avg_latency > 1000:  # >1 second latency
            reasoning.append(f"High latency: {avg_latency:.1f}ms")
            scale_up_signals += 2
        elif avg_latency > 500:  # >500ms latency
            reasoning.append(f"Elevated latency: {avg_latency:.1f}ms")
            scale_up_signals += 1
        
        if avg_cpu > 80:  # >80% CPU
            reasoning.append(f"High CPU usage: {avg_cpu:.1f}%")
            scale_up_signals += 2
        elif avg_cpu > 60:  # >60% CPU
            reasoning.append(f"Elevated CPU usage: {avg_cpu:.1f}%")
            scale_up_signals += 1
        
        if avg_memory > 85:  # >85% memory
            reasoning.append(f"High memory usage: {avg_memory:.1f}%")
            scale_up_signals += 2
        
        if avg_error_rate > 0.05:  # >5% error rate
            reasoning.append(f"High error rate: {avg_error_rate:.2%}")
            scale_up_signals += 3
        
        # Scale down conditions
        scale_down_signals = 0
        if avg_latency < 100 and avg_cpu < 30 and avg_memory < 50:
            reasoning.append("Low resource utilization across all metrics")
            scale_down_signals += 2
        elif avg_cpu < 20 and avg_memory < 30:
            reasoning.append("Very low resource utilization")
            scale_down_signals += 1
        
        # Make scaling decision
        if scale_up_signals >= 3 and self.current_replicas < self.max_replicas:
            direction = ScalingDirection.SCALE_OUT
            confidence = min(0.9, scale_up_signals / 5.0)
            target_replicas = min(self.max_replicas, self.current_replicas + 1)
        elif scale_up_signals >= 2 and self.current_replicas < self.max_replicas:
            direction = ScalingDirection.SCALE_OUT
            confidence = min(0.7, scale_up_signals / 4.0)
            target_replicas = min(self.max_replicas, self.current_replicas + 1)
        elif scale_down_signals >= 2 and self.current_replicas > self.min_replicas:
            direction = ScalingDirection.SCALE_IN
            confidence = 0.6
            target_replicas = max(self.min_replicas, self.current_replicas - 1)
        else:
            target_replicas = self.current_replicas
        
        # Estimate impact
        estimated_impact = {}
        if direction == ScalingDirection.SCALE_OUT:
            estimated_impact = {
                "latency_reduction": 0.3,  # 30% latency reduction
                "cpu_reduction": 0.4,      # 40% CPU reduction per node
                "cost_increase": 1.0 / target_replicas  # Cost per additional replica
            }
        elif direction == ScalingDirection.SCALE_IN:
            estimated_impact = {
                "latency_increase": 0.2,   # 20% latency increase
                "cpu_increase": 0.3,       # 30% CPU increase per remaining node
                "cost_reduction": 1.0 / self.current_replicas  # Cost savings
            }
        
        decision = ScalingDecision(
            direction=direction,
            confidence=confidence,
            target_replicas=target_replicas,
            estimated_impact=estimated_impact,
            reasoning=reasoning
        )
        
        self.scaling_history.append(decision)
        return decision
    
    def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply scaling decision if confidence is sufficient."""
        if decision.confidence < 0.5:
            logger.info(f"Scaling decision confidence too low: {decision.confidence:.2f}")
            return False
        
        if decision.direction in [ScalingDirection.SCALE_OUT, ScalingDirection.SCALE_IN]:
            old_replicas = self.current_replicas
            self.current_replicas = decision.target_replicas
            
            logger.info(f"Scaling {old_replicas} -> {self.current_replicas} replicas "
                       f"(confidence: {decision.confidence:.2f})")
            logger.info(f"Reasoning: {'; '.join(decision.reasoning)}")
            
            return True
        
        return False


class QuantumPerformanceOptimizer:
    """Quantum-enhanced performance optimization."""
    
    def __init__(self):
        self.quantum_coherence_threshold = 0.8
        self.optimization_history: List[Dict[str, Any]] = []
        
    def optimize_quantum_parameters(self, current_performance: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize quantum planning parameters based on performance."""
        optimizations = {}
        
        # Adjust quantum coherence based on performance
        if current_performance.request_latency > 500:
            # High latency - reduce quantum complexity
            optimizations["superposition_depth"] = 3  # Reduce from default 5
            optimizations["entanglement_pairs"] = 2   # Reduce from default 4
            optimizations["coherence_time"] = 10      # Reduce from default 20
        elif current_performance.request_latency < 100:
            # Low latency - can afford more quantum complexity
            optimizations["superposition_depth"] = 7  # Increase from default 5
            optimizations["entanglement_pairs"] = 6   # Increase from default 4
            optimizations["coherence_time"] = 30      # Increase from default 20
        
        # Adjust based on quantum coherence score
        if current_performance.quantum_coherence_score < self.quantum_coherence_threshold:
            optimizations["decoherence_mitigation"] = True
            optimizations["error_correction"] = "active"
        
        return optimizations
    
    def predict_quantum_performance(self, proposed_optimizations: Dict[str, Any]) -> float:
        """Predict performance improvement from quantum optimizations."""
        # Simple heuristic model for quantum performance prediction
        base_improvement = 0.1  # 10% base improvement
        
        # Factor in optimization parameters
        depth_factor = proposed_optimizations.get("superposition_depth", 5) / 5.0
        entanglement_factor = proposed_optimizations.get("entanglement_pairs", 4) / 4.0
        coherence_factor = proposed_optimizations.get("coherence_time", 20) / 20.0
        
        # Quantum advantage scales with complexity but has diminishing returns
        quantum_advantage = np.tanh(depth_factor * entanglement_factor * coherence_factor)
        
        predicted_improvement = base_improvement * quantum_advantage
        return min(0.5, predicted_improvement)  # Cap at 50% improvement


class AdvancedPerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.cache = IntelligentCache()
        self.load_balancer = AdaptiveLoadBalancer()
        self.auto_scaler = AutoScaler()
        self.quantum_optimizer = QuantumPerformanceOptimizer()
        
        self.optimization_metrics: List[PerformanceMetrics] = []
        self.last_optimization_time = 0
        self.optimization_interval = 60  # Optimize every minute
        
    async def optimize_request_processing(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize request processing with intelligent routing and caching."""
        # Check cache first
        cached_response = await self.cache.get(
            request_data.get("prompt", ""),
            request_data.get("model", ""),
            request_data.get("parameters", {})
        )
        
        if cached_response:
            return {
                "response": cached_response,
                "cached": True,
                "optimization_applied": "cache_hit"
            }
        
        # Select optimal nodes for processing
        optimal_nodes = await self.load_balancer.select_optimal_nodes(
            request_data,
            num_nodes=request_data.get("consensus_required", False) and 2 or 1
        )
        
        return {
            "selected_nodes": optimal_nodes,
            "cached": False,
            "optimization_applied": "intelligent_routing"
        }
    
    async def store_response_in_cache(self, request_data: Dict[str, Any], response: str):
        """Store response in intelligent cache."""
        await self.cache.put(
            request_data.get("prompt", ""),
            request_data.get("model", ""),
            request_data.get("parameters", {}),
            response
        )
    
    def record_performance_metrics(self, metrics: PerformanceMetrics, node_id: str = "default"):
        """Record performance metrics for optimization."""
        self.optimization_metrics.append(metrics)
        self.load_balancer.update_node_metrics(node_id, metrics)
        
        # Keep only recent metrics
        cutoff_time = time.time() - 3600  # Last hour
        self.optimization_metrics = [
            m for m in self.optimization_metrics 
            if m.timestamp > cutoff_time
        ]
    
    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run comprehensive optimization cycle."""
        current_time = time.time()
        
        if current_time - self.last_optimization_time < self.optimization_interval:
            return {"status": "skipped", "reason": "optimization_interval_not_reached"}
        
        if not self.optimization_metrics:
            return {"status": "skipped", "reason": "insufficient_metrics"}
        
        self.last_optimization_time = current_time
        
        # Get latest metrics
        latest_metrics = self.optimization_metrics[-1]
        
        # Auto-scaling analysis
        scaling_decision = self.auto_scaler.analyze_scaling_need(latest_metrics)
        scaling_applied = self.auto_scaler.apply_scaling_decision(scaling_decision)
        
        # Quantum optimization
        quantum_optimizations = self.quantum_optimizer.optimize_quantum_parameters(latest_metrics)
        quantum_improvement = self.quantum_optimizer.predict_quantum_performance(quantum_optimizations)
        
        # Load balancing optimization
        load_distribution = self.load_balancer.predict_load_distribution()
        
        # Cache optimization
        cache_stats = self.cache.get_cache_stats()
        
        optimization_results = {
            "timestamp": current_time,
            "scaling": {
                "decision": scaling_decision.direction.value,
                "confidence": scaling_decision.confidence,
                "applied": scaling_applied,
                "target_replicas": scaling_decision.target_replicas,
                "reasoning": scaling_decision.reasoning
            },
            "quantum": {
                "optimizations": quantum_optimizations,
                "predicted_improvement": quantum_improvement
            },
            "load_balancing": {
                "distribution": load_distribution
            },
            "caching": cache_stats,
            "current_performance": latest_metrics.to_dict()
        }
        
        logger.info(f"Optimization cycle completed. Scaling: {scaling_decision.direction.value}, "
                   f"Quantum improvement: {quantum_improvement:.2%}")
        
        return optimization_results
    
    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive optimization dashboard."""
        if not self.optimization_metrics:
            return {"status": "no_metrics"}
        
        # Recent performance trends
        recent_metrics = self.optimization_metrics[-10:] if len(self.optimization_metrics) >= 10 else self.optimization_metrics
        
        avg_latency = statistics.mean(m.request_latency for m in recent_metrics)
        avg_throughput = statistics.mean(m.throughput for m in recent_metrics)
        avg_cpu = statistics.mean(m.cpu_usage for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage for m in recent_metrics)
        
        # Performance trends
        latency_trend = "stable"
        if len(recent_metrics) >= 5:
            early_latency = statistics.mean(m.request_latency for m in recent_metrics[:len(recent_metrics)//2])
            late_latency = statistics.mean(m.request_latency for m in recent_metrics[len(recent_metrics)//2:])
            
            if late_latency > early_latency * 1.1:
                latency_trend = "increasing"
            elif late_latency < early_latency * 0.9:
                latency_trend = "decreasing"
        
        return {
            "strategy": self.strategy.value,
            "current_performance": {
                "avg_latency": avg_latency,
                "avg_throughput": avg_throughput,
                "avg_cpu": avg_cpu,
                "avg_memory": avg_memory,
                "latency_trend": latency_trend
            },
            "scaling_status": {
                "current_replicas": self.auto_scaler.current_replicas,
                "min_replicas": self.auto_scaler.min_replicas,
                "max_replicas": self.auto_scaler.max_replicas
            },
            "cache_performance": self.cache.get_cache_stats(),
            "load_balancing": {
                "active_nodes": len(self.load_balancer.node_weights),
                "node_weights": dict(self.load_balancer.node_weights)
            },
            "optimization_history": len(self.optimization_metrics),
            "dashboard_generated": time.time()
        }


# Global optimizer instance
global_optimizer = AdvancedPerformanceOptimizer()


def get_performance_optimizer() -> AdvancedPerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return global_optimizer