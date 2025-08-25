"""
Global Load Balancer for Generation 3 Scalability

Implements intelligent global load balancing with quantum-aware routing,
geo-distributed optimization, and adaptive traffic management for
massive-scale federated healthcare LLM deployments.
"""

import asyncio
import time
import logging
import threading
import statistics
try:
    from ..quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
except ImportError:
    # For files outside quantum_planning module
    from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import concurrent.futures
from abc import ABC, abstractmethod
import hashlib
import heapq

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASHING = "consistent_hashing"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    QUANTUM_OPTIMIZED = "quantum_optimized"
    PRIVACY_AWARE = "privacy_aware"
    GEO_DISTRIBUTED = "geo_distributed"


class HealthStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class NodeMetrics:
    """Real-time node performance metrics."""
    node_id: str
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    network_latency: float
    active_connections: int
    requests_per_second: float
    error_rate: float
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    privacy_budget_remaining: float
    quantum_coherence: float
    health_status: HealthStatus
    geographic_region: str
    capacity_weight: float = 1.0


@dataclass
class LoadBalancingRequest:
    """Request for load balancing."""
    request_id: str
    user_id: str
    session_id: Optional[str]
    privacy_requirements: Dict[str, Any]
    geographic_preference: Optional[str]
    quantum_requirements: Dict[str, Any]
    priority_level: int
    estimated_complexity: float
    routing_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Load balancing routing decision."""
    selected_node: str
    reasoning: str
    confidence: float
    backup_nodes: List[str]
    estimated_latency: float
    expected_load: float
    routing_metadata: Dict[str, Any] = field(default_factory=dict)


class ConsistentHashRing:
    """Consistent hashing implementation for load balancing."""
    
    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()
        self._lock = threading.RLock()
    
    def add_node(self, node_id: str):
        """Add a node to the hash ring."""
        with self._lock:
            if node_id in self.nodes:
                return
            
            self.nodes.add(node_id)
            
            for i in range(self.num_virtual_nodes):
                virtual_key = self._hash(f"{node_id}:{i}")
                self.ring[virtual_key] = node_id
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str):
        """Remove a node from the hash ring."""
        with self._lock:
            if node_id not in self.nodes:
                return
            
            self.nodes.remove(node_id)
            
            # Remove all virtual nodes
            keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
            for key in keys_to_remove:
                del self.ring[key]
            
            self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a given key."""
        with self._lock:
            if not self.sorted_keys:
                return None
            
            key_hash = self._hash(key)
            
            # Binary search for the first node clockwise from the key
            idx = self._binary_search_right(self.sorted_keys, key_hash)
            
            if idx == len(self.sorted_keys):
                idx = 0
            
            return self.ring[self.sorted_keys[idx]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for replication."""
        with self._lock:
            if not self.sorted_keys or count <= 0:
                return []
            
            nodes = []
            key_hash = self._hash(key)
            
            idx = self._binary_search_right(self.sorted_keys, key_hash)
            
            seen_nodes = set()
            
            for i in range(len(self.sorted_keys)):
                actual_idx = (idx + i) % len(self.sorted_keys)
                node = self.ring[self.sorted_keys[actual_idx]]
                
                if node not in seen_nodes:
                    nodes.append(node)
                    seen_nodes.add(node)
                    
                    if len(nodes) >= count:
                        break
            
            return nodes
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def _binary_search_right(self, arr: List[int], target: int) -> int:
        """Binary search for insertion point."""
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left


class AdaptiveWeightCalculator:
    """Calculates adaptive weights for nodes based on performance."""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.weight_cache: Dict[str, float] = {}
        self.cache_timeout = 60.0  # Cache weights for 1 minute
        self.last_calculation: Dict[str, float] = {}
    
    def update_performance(self, node_id: str, metrics: NodeMetrics):
        """Update performance data for a node."""
        performance_score = self._calculate_performance_score(metrics)
        self.performance_history[node_id].append({
            'timestamp': metrics.timestamp,
            'score': performance_score,
            'response_time': metrics.response_time_p50,
            'error_rate': metrics.error_rate,
            'utilization': (metrics.cpu_utilization + metrics.memory_utilization) / 2
        })
        
        # Invalidate cache for this node
        self.weight_cache.pop(node_id, None)
        self.last_calculation.pop(node_id, None)
    
    def get_weight(self, node_id: str) -> float:
        """Get adaptive weight for a node."""
        current_time = time.time()
        
        # Check cache
        if (node_id in self.weight_cache and 
            node_id in self.last_calculation and
            current_time - self.last_calculation[node_id] < self.cache_timeout):
            return self.weight_cache[node_id]
        
        # Calculate new weight
        weight = self._calculate_adaptive_weight(node_id)
        
        self.weight_cache[node_id] = weight
        self.last_calculation[node_id] = current_time
        
        return weight
    
    def _calculate_performance_score(self, metrics: NodeMetrics) -> float:
        """Calculate performance score from metrics."""
        # Base score from resource utilization (inverse - lower utilization is better)
        utilization_score = 1.0 - ((metrics.cpu_utilization + metrics.memory_utilization) / 200.0)
        
        # Response time score (lower is better)
        response_time_score = max(0.0, 1.0 - (metrics.response_time_p50 / 1000.0))
        
        # Error rate score (lower is better)
        error_rate_score = max(0.0, 1.0 - (metrics.error_rate * 10))
        
        # Health status score
        health_scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.MAINTENANCE: 0.1,
            HealthStatus.OFFLINE: 0.0
        }
        health_score = health_scores.get(metrics.health_status, 0.5)
        
        # Privacy budget score
        privacy_score = min(1.0, metrics.privacy_budget_remaining / 100.0)
        
        # Quantum coherence score
        quantum_score = metrics.quantum_coherence
        
        # Weighted combination
        composite_score = (
            utilization_score * 0.25 +
            response_time_score * 0.25 +
            error_rate_score * 0.15 +
            health_score * 0.15 +
            privacy_score * 0.1 +
            quantum_score * 0.1
        )
        
        return max(0.0, min(1.0, composite_score))
    
    def _calculate_adaptive_weight(self, node_id: str) -> float:
        """Calculate adaptive weight based on historical performance."""
        history = self.performance_history[node_id]
        
        if not history:
            return 1.0  # Default weight
        
        # Recent performance (last 10 entries)
        recent_scores = [entry['score'] for entry in list(history)[-10:]]
        recent_avg = statistics.mean(recent_scores) if recent_scores else 0.5
        
        # Overall performance trend
        if len(history) >= 20:
            early_scores = [entry['score'] for entry in list(history)[:10]]
            late_scores = [entry['score'] for entry in list(history)[-10:]]
            
            early_avg = statistics.mean(early_scores)
            late_avg = statistics.mean(late_scores)
            
            trend_factor = (late_avg - early_avg) + 1.0  # +1 to keep positive
        else:
            trend_factor = 1.0
        
        # Stability factor (lower variance is better)
        if len(recent_scores) > 1:
            variance = statistics.variance(recent_scores)
            stability_factor = max(0.5, 1.0 - variance)
        else:
            stability_factor = 1.0
        
        # Combined adaptive weight
        adaptive_weight = recent_avg * trend_factor * stability_factor
        
        # Ensure weight is within reasonable bounds
        return max(0.1, min(10.0, adaptive_weight))


class QuantumAwareRouter:
    """Quantum-aware routing component."""
    
    def __init__(self):
        self.quantum_coherence_threshold = 0.7
        self.entanglement_preferences: Dict[str, List[str]] = {}
        self.superposition_routing: Dict[str, List[str]] = {}
    
    def should_use_quantum_routing(self, request: LoadBalancingRequest) -> bool:
        """Determine if quantum routing should be used."""
        quantum_reqs = request.quantum_requirements
        
        return (
            quantum_reqs.get('requires_quantum_optimization', False) or
            quantum_reqs.get('entangled_processing', False) or
            quantum_reqs.get('superposition_scheduling', False)
        )
    
    def get_quantum_optimized_nodes(
        self, 
        request: LoadBalancingRequest,
        available_nodes: List[str],
        node_metrics: Dict[str, NodeMetrics]
    ) -> List[str]:
        """Get quantum-optimized node selection."""
        if not self.should_use_quantum_routing(request):
            return available_nodes
        
        # Filter nodes by quantum coherence
        quantum_capable_nodes = [
            node_id for node_id in available_nodes
            if (node_id in node_metrics and 
                node_metrics[node_id].quantum_coherence >= self.quantum_coherence_threshold)
        ]
        
        if not quantum_capable_nodes:
            return available_nodes  # Fallback to all nodes
        
        # Check for entanglement requirements
        if request.quantum_requirements.get('entangled_processing', False):
            return self._get_entangled_nodes(request.user_id, quantum_capable_nodes)
        
        # Check for superposition requirements
        if request.quantum_requirements.get('superposition_scheduling', False):
            return self._get_superposition_nodes(request.request_id, quantum_capable_nodes)
        
        return quantum_capable_nodes
    
    def _get_entangled_nodes(self, user_id: str, available_nodes: List[str]) -> List[str]:
        """Get nodes that are entangled for this user."""
        if user_id in self.entanglement_preferences:
            entangled_nodes = [
                node for node in self.entanglement_preferences[user_id]
                if node in available_nodes
            ]
            if entangled_nodes:
                return entangled_nodes
        
        # Create new entanglement if none exists
        if len(available_nodes) >= 2:
            entangled_pair = available_nodes[:2]
            self.entanglement_preferences[user_id] = entangled_pair
            return entangled_pair
        
        return available_nodes
    
    def _get_superposition_nodes(self, request_id: str, available_nodes: List[str]) -> List[str]:
        """Get nodes for superposition-based routing."""
        # Use request ID to deterministically select nodes for superposition
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        
        # Select up to 3 nodes for superposition
        num_nodes = min(3, len(available_nodes))
        selected_indices = []
        
        for i in range(num_nodes):
            idx = (hash_val + i) % len(available_nodes)
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        return [available_nodes[i] for i in selected_indices]


class GeoDistributedRouter:
    """Geographic distribution aware routing."""
    
    def __init__(self):
        self.region_latencies: Dict[Tuple[str, str], float] = {}
        self.region_preferences: Dict[str, str] = {}
        self.compliance_zones: Dict[str, List[str]] = {}
    
    def add_region_latency(self, from_region: str, to_region: str, latency_ms: float):
        """Add latency measurement between regions."""
        self.region_latencies[(from_region, to_region)] = latency_ms
        self.region_latencies[(to_region, from_region)] = latency_ms
    
    def set_compliance_zone(self, data_type: str, allowed_regions: List[str]):
        """Set compliance restrictions for data types."""
        self.compliance_zones[data_type] = allowed_regions
    
    def filter_compliant_nodes(
        self, 
        request: LoadBalancingRequest,
        available_nodes: List[str],
        node_metrics: Dict[str, NodeMetrics]
    ) -> List[str]:
        """Filter nodes based on geographic compliance."""
        # Check for data sovereignty requirements
        privacy_reqs = request.privacy_requirements
        data_types = privacy_reqs.get('data_types', [])
        
        if not data_types:
            return available_nodes
        
        compliant_nodes = []
        
        for node_id in available_nodes:
            if node_id not in node_metrics:
                continue
            
            node_region = node_metrics[node_id].geographic_region
            
            # Check if node region is compliant for all data types
            is_compliant = True
            for data_type in data_types:
                if data_type in self.compliance_zones:
                    allowed_regions = self.compliance_zones[data_type]
                    if node_region not in allowed_regions:
                        is_compliant = False
                        break
            
            if is_compliant:
                compliant_nodes.append(node_id)
        
        return compliant_nodes if compliant_nodes else available_nodes
    
    def sort_by_geographic_preference(
        self,
        request: LoadBalancingRequest,
        candidate_nodes: List[str],
        node_metrics: Dict[str, NodeMetrics]
    ) -> List[str]:
        """Sort nodes by geographic preference."""
        user_region = request.geographic_preference
        
        if not user_region:
            return candidate_nodes
        
        # Calculate scores based on latency
        node_scores = []
        
        for node_id in candidate_nodes:
            if node_id not in node_metrics:
                continue
            
            node_region = node_metrics[node_id].geographic_region
            
            # Get latency to this region
            latency = self.region_latencies.get((user_region, node_region), 1000.0)
            
            # Lower latency = higher score
            score = max(0.0, 1000.0 - latency) / 1000.0
            
            node_scores.append((node_id, score))
        
        # Sort by score (descending)
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [node_id for node_id, score in node_scores]


class GlobalLoadBalancer:
    """Advanced global load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_WEIGHTED):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.hash_ring = ConsistentHashRing()
        self.weight_calculator = AdaptiveWeightCalculator()
        self.quantum_router = QuantumAwareRouter()
        self.geo_router = GeoDistributedRouter()
        
        # Node state tracking
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.node_connections: Dict[str, int] = defaultdict(int)
        self.node_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Round-robin state
        self.round_robin_index = 0
        
        # Request tracking
        self.request_history: deque = deque(maxlen=10000)
        self.routing_stats: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance monitoring
        self.total_requests = 0
        self.successful_routes = 0
        self.average_routing_time = 0.0
    
    def register_node(self, node_id: str, initial_metrics: NodeMetrics):
        """Register a new node with the load balancer."""
        with self._lock:
            self.node_metrics[node_id] = initial_metrics
            self.hash_ring.add_node(node_id)
            self.weight_calculator.update_performance(node_id, initial_metrics)
            
            self.logger.info(f"Registered node {node_id} in region {initial_metrics.geographic_region}")
    
    def unregister_node(self, node_id: str):
        """Unregister a node from the load balancer."""
        with self._lock:
            self.node_metrics.pop(node_id, None)
            self.hash_ring.remove_node(node_id)
            self.node_connections.pop(node_id, None)
            self.node_weights.pop(node_id, None)
            
            self.logger.info(f"Unregistered node {node_id}")
    
    def update_node_metrics(self, node_id: str, metrics: NodeMetrics):
        """Update metrics for a node."""
        with self._lock:
            if node_id in self.node_metrics:
                self.node_metrics[node_id] = metrics
                self.weight_calculator.update_performance(node_id, metrics)
                
                # Update cached weight
                self.node_weights[node_id] = self.weight_calculator.get_weight(node_id)
    
    def route_request(self, request: LoadBalancingRequest) -> RoutingDecision:
        """Route a request to the best available node."""
        routing_start = time.time()
        
        with self._lock:
            self.total_requests += 1
        
        try:
            # Get available nodes
            available_nodes = self._get_available_nodes()
            
            if not available_nodes:
                raise RuntimeError("No available nodes for routing")
            
            # Apply filtering based on requirements
            candidate_nodes = self._filter_candidate_nodes(request, available_nodes)
            
            if not candidate_nodes:
                candidate_nodes = available_nodes  # Fallback
            
            # Select best node based on strategy
            selected_node = self._select_node(request, candidate_nodes)
            
            # Generate routing decision
            decision = self._create_routing_decision(request, selected_node, candidate_nodes)
            
            # Update tracking
            self._record_routing_decision(decision, time.time() - routing_start)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Routing failed for request {request.request_id}: {e}")
            raise
    
    def _get_available_nodes(self) -> List[str]:
        """Get list of available healthy nodes."""
        return [
            node_id for node_id, metrics in self.node_metrics.items()
            if metrics.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        ]
    
    def _filter_candidate_nodes(
        self, 
        request: LoadBalancingRequest, 
        available_nodes: List[str]
    ) -> List[str]:
        """Filter nodes based on request requirements."""
        candidate_nodes = available_nodes
        
        # Geographic compliance filtering
        candidate_nodes = self.geo_router.filter_compliant_nodes(
            request, candidate_nodes, self.node_metrics
        )
        
        # Quantum requirements filtering
        if self.strategy == LoadBalancingStrategy.QUANTUM_OPTIMIZED:
            candidate_nodes = self.quantum_router.get_quantum_optimized_nodes(
                request, candidate_nodes, self.node_metrics
            )
        
        # Privacy budget filtering
        if request.privacy_requirements.get('min_budget_required', 0) > 0:
            min_budget = request.privacy_requirements['min_budget_required']
            candidate_nodes = [
                node_id for node_id in candidate_nodes
                if self.node_metrics[node_id].privacy_budget_remaining >= min_budget
            ]
        
        return candidate_nodes
    
    def _select_node(self, request: LoadBalancingRequest, candidate_nodes: List[str]) -> str:
        """Select the best node using the configured strategy."""
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASHING:
            return self._consistent_hash_selection(request, candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE_WEIGHTED:
            return self._adaptive_weighted_selection(candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.QUANTUM_OPTIMIZED:
            return self._quantum_optimized_selection(request, candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.PRIVACY_AWARE:
            return self._privacy_aware_selection(request, candidate_nodes)
        
        elif self.strategy == LoadBalancingStrategy.GEO_DISTRIBUTED:
            return self._geo_distributed_selection(request, candidate_nodes)
        
        else:
            # Default to round robin
            return self._round_robin_selection(candidate_nodes)
    
    def _round_robin_selection(self, candidate_nodes: List[str]) -> str:
        """Simple round-robin selection."""
        if not candidate_nodes:
            raise RuntimeError("No candidate nodes available")
        
        selected_idx = self.round_robin_index % len(candidate_nodes)
        self.round_robin_index += 1
        
        return candidate_nodes[selected_idx]
    
    def _weighted_round_robin_selection(self, candidate_nodes: List[str]) -> str:
        """Weighted round-robin selection based on node weights."""
        if not candidate_nodes:
            raise RuntimeError("No candidate nodes available")
        
        # Create weighted list
        weighted_nodes = []
        for node_id in candidate_nodes:
            weight = int(self.node_weights[node_id] * 10)  # Scale up for integer weights
            weighted_nodes.extend([node_id] * max(1, weight))
        
        if not weighted_nodes:
            return candidate_nodes[0]
        
        selected_idx = self.round_robin_index % len(weighted_nodes)
        self.round_robin_index += 1
        
        return weighted_nodes[selected_idx]
    
    def _least_connections_selection(self, candidate_nodes: List[str]) -> str:
        """Select node with least active connections."""
        if not candidate_nodes:
            raise RuntimeError("No candidate nodes available")
        
        return min(candidate_nodes, key=lambda node: self.node_connections[node])
    
    def _least_response_time_selection(self, candidate_nodes: List[str]) -> str:
        """Select node with lowest response time."""
        if not candidate_nodes:
            raise RuntimeError("No candidate nodes available")
        
        def get_response_time(node_id: str) -> float:
            if node_id in self.node_metrics:
                return self.node_metrics[node_id].response_time_p50
            return float('inf')
        
        return min(candidate_nodes, key=get_response_time)
    
    def _consistent_hash_selection(self, request: LoadBalancingRequest, candidate_nodes: List[str]) -> str:
        """Select node using consistent hashing."""
        # Use session ID if available, otherwise user ID
        hash_key = request.session_id or request.user_id
        
        # Filter hash ring to only include candidate nodes
        for node in candidate_nodes:
            if node not in self.hash_ring.nodes:
                self.hash_ring.add_node(node)
        
        selected_node = self.hash_ring.get_node(hash_key)
        
        if selected_node and selected_node in candidate_nodes:
            return selected_node
        
        # Fallback to first candidate
        return candidate_nodes[0]
    
    def _adaptive_weighted_selection(self, candidate_nodes: List[str]) -> str:
        """Select node using adaptive weights based on performance."""
        if not candidate_nodes:
            raise RuntimeError("No candidate nodes available")
        
        # Calculate selection probabilities based on weights
        weights = [self.weight_calculator.get_weight(node) for node in candidate_nodes]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            return candidate_nodes[0]
        
        # Weighted random selection
        import random
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return candidate_nodes[i]
        
        return candidate_nodes[-1]  # Fallback
    
    def _quantum_optimized_selection(self, request: LoadBalancingRequest, candidate_nodes: List[str]) -> str:
        """Select node using quantum optimization."""
        quantum_nodes = self.quantum_router.get_quantum_optimized_nodes(
            request, candidate_nodes, self.node_metrics
        )
        
        if not quantum_nodes:
            return self._adaptive_weighted_selection(candidate_nodes)
        
        # Among quantum-capable nodes, select based on quantum coherence
        def quantum_score(node_id: str) -> float:
            if node_id in self.node_metrics:
                metrics = self.node_metrics[node_id]
                return metrics.quantum_coherence * self.weight_calculator.get_weight(node_id)
            return 0.0
        
        return max(quantum_nodes, key=quantum_score)
    
    def _privacy_aware_selection(self, request: LoadBalancingRequest, candidate_nodes: List[str]) -> str:
        """Select node with privacy optimization."""
        # Weight by privacy budget remaining
        def privacy_score(node_id: str) -> float:
            if node_id in self.node_metrics:
                metrics = self.node_metrics[node_id]
                base_weight = self.weight_calculator.get_weight(node_id)
                privacy_factor = metrics.privacy_budget_remaining / 100.0
                return base_weight * privacy_factor
            return 0.0
        
        return max(candidate_nodes, key=privacy_score)
    
    def _geo_distributed_selection(self, request: LoadBalancingRequest, candidate_nodes: List[str]) -> str:
        """Select node using geographic optimization."""
        geo_sorted = self.geo_router.sort_by_geographic_preference(
            request, candidate_nodes, self.node_metrics
        )
        
        if geo_sorted:
            return geo_sorted[0]
        
        return candidate_nodes[0]
    
    def _create_routing_decision(
        self, 
        request: LoadBalancingRequest,
        selected_node: str,
        candidate_nodes: List[str]
    ) -> RoutingDecision:
        """Create routing decision with metadata."""
        
        # Get backup nodes
        backup_nodes = [node for node in candidate_nodes if node != selected_node][:3]
        
        # Estimate latency and load
        if selected_node in self.node_metrics:
            metrics = self.node_metrics[selected_node]
            estimated_latency = metrics.response_time_p50 + metrics.network_latency
            expected_load = metrics.active_connections / max(metrics.capacity_weight, 1.0)
        else:
            estimated_latency = 100.0
            expected_load = 0.5
        
        # Calculate confidence based on node health and weight
        confidence = 0.5
        if selected_node in self.node_metrics:
            metrics = self.node_metrics[selected_node]
            health_confidence = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.DEGRADED: 0.7,
                HealthStatus.UNHEALTHY: 0.3,
                HealthStatus.MAINTENANCE: 0.2,
                HealthStatus.OFFLINE: 0.0
            }.get(metrics.health_status, 0.5)
            
            weight_confidence = min(1.0, self.weight_calculator.get_weight(selected_node))
            confidence = (health_confidence + weight_confidence) / 2
        
        return RoutingDecision(
            selected_node=selected_node,
            reasoning=f"Selected using {self.strategy.value} strategy",
            confidence=confidence,
            backup_nodes=backup_nodes,
            estimated_latency=estimated_latency,
            expected_load=expected_load,
            routing_metadata={
                'strategy': self.strategy.value,
                'candidate_count': len(candidate_nodes),
                'total_nodes': len(self.node_metrics),
                'timestamp': time.time()
            }
        )
    
    def _record_routing_decision(self, decision: RoutingDecision, routing_time: float):
        """Record routing decision for analytics."""
        with self._lock:
            self.successful_routes += 1
            
            # Update average routing time
            total_routes = self.successful_routes
            current_avg = self.average_routing_time
            self.average_routing_time = (
                (current_avg * (total_routes - 1) + routing_time) / total_routes
            )
            
            # Update routing stats
            self.routing_stats[decision.selected_node] += 1
            
            # Store in history
            self.request_history.append({
                'timestamp': time.time(),
                'selected_node': decision.selected_node,
                'routing_time': routing_time,
                'confidence': decision.confidence,
                'strategy': self.strategy.value
            })
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        with self._lock:
            node_utilization = {}
            for node_id, count in self.routing_stats.items():
                utilization = count / max(self.total_requests, 1)
                node_utilization[node_id] = {
                    'requests': count,
                    'utilization': utilization,
                    'connections': self.node_connections[node_id],
                    'weight': self.node_weights[node_id]
                }
            
            recent_requests = list(self.request_history)[-100:]  # Last 100
            avg_confidence = (
                sum(req['confidence'] for req in recent_requests) / 
                max(len(recent_requests), 1)
            )
            
            return {
                'strategy': self.strategy.value,
                'total_requests': self.total_requests,
                'successful_routes': self.successful_routes,
                'success_rate': self.successful_routes / max(self.total_requests, 1),
                'average_routing_time': self.average_routing_time,
                'average_confidence': avg_confidence,
                'registered_nodes': len(self.node_metrics),
                'healthy_nodes': len(self._get_available_nodes()),
                'node_utilization': node_utilization,
                'recent_requests': len(recent_requests)
            }


# Global load balancer instance
global_load_balancer = GlobalLoadBalancer()


def get_global_load_balancer() -> GlobalLoadBalancer:
    """Get the global load balancer instance."""
    return global_load_balancer