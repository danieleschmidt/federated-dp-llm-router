"""
Privacy-Aware Federated Router

Implements intelligent routing of inference requests across federated nodes
while considering privacy budgets, node capabilities, and load balancing.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from ..core.privacy_accountant import PrivacyAccountant, DPConfig
from ..core.model_sharding import ModelSharder, ShardingStrategy


class RoutingStrategy(Enum):
    """Routing strategies for federated inference."""
    ROUND_ROBIN = "round_robin"
    PRIVACY_AWARE = "privacy_aware"
    LOAD_BALANCED = "load_balanced"
    CONSENSUS_REQUIRED = "consensus_required"


@dataclass
class NodeCapability:
    """Describes the computational capabilities of a federated node."""
    node_id: str
    gpu_memory: int  # in MB
    cpu_cores: int
    network_bandwidth: int  # in Mbps
    max_concurrent_requests: int
    supported_models: List[str]
    privacy_budget_remaining: float
    last_health_check: float = field(default_factory=time.time)
    is_healthy: bool = True


@dataclass
class InferenceRequest:
    """Represents an inference request with privacy requirements."""
    request_id: str
    user_id: str
    prompt: str
    model_name: str
    max_privacy_budget: float
    require_consensus: bool = False
    priority: int = 1  # 1-10, higher is more urgent
    timeout: float = 30.0
    department: Optional[str] = None


@dataclass
class InferenceResponse:
    """Response from federated inference."""
    request_id: str
    text: str
    privacy_cost: float
    remaining_budget: float
    processing_nodes: List[str]
    latency: float
    confidence_score: float = 0.0
    consensus_achieved: bool = False


class NodeLoadTracker:
    """Tracks load and performance metrics for nodes."""
    
    def __init__(self):
        self.node_loads: Dict[str, float] = {}
        self.response_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_requests: Dict[str, float] = {}
    
    def update_load(self, node_id: str, current_load: float):
        """Update current load for a node (0.0 to 1.0)."""
        self.node_loads[node_id] = current_load
    
    def record_response_time(self, node_id: str, response_time: float):
        """Record response time for performance tracking."""
        if node_id not in self.response_times:
            self.response_times[node_id] = []
        
        self.response_times[node_id].append(response_time)
        
        # Keep only recent responses (last 100)
        if len(self.response_times[node_id]) > 100:
            self.response_times[node_id] = self.response_times[node_id][-100:]
    
    def record_error(self, node_id: str):
        """Record error for reliability tracking."""
        self.error_counts[node_id] = self.error_counts.get(node_id, 0) + 1
    
    def get_average_response_time(self, node_id: str) -> float:
        """Get average response time for a node."""
        times = self.response_times.get(node_id, [])
        return np.mean(times) if times else float('inf')
    
    def get_error_rate(self, node_id: str) -> float:
        """Get error rate for a node."""
        errors = self.error_counts.get(node_id, 0)
        requests = len(self.response_times.get(node_id, []))
        return errors / max(requests, 1)
    
    def get_node_score(self, node_id: str) -> float:
        """Calculate overall node performance score (0.0 to 1.0)."""
        load = self.node_loads.get(node_id, 1.0)
        avg_response = self.get_average_response_time(node_id)
        error_rate = self.get_error_rate(node_id)
        
        # Normalize response time (assume 1000ms is very poor)
        response_score = max(0.0, 1.0 - (avg_response / 1000.0))
        
        # Calculate composite score
        load_score = 1.0 - load
        reliability_score = 1.0 - error_rate
        
        return (load_score * 0.4 + response_score * 0.4 + reliability_score * 0.2)


class FederatedRouter:
    """Main federated router for privacy-aware LLM inference."""
    
    def __init__(
        self,
        model_name: str,
        num_shards: int = 4,
        aggregation_protocol: str = "secure_aggregation",
        encryption: str = "homomorphic",
        routing_strategy: RoutingStrategy = RoutingStrategy.PRIVACY_AWARE
    ):
        self.model_name = model_name
        self.num_shards = num_shards
        self.aggregation_protocol = aggregation_protocol
        self.encryption = encryption
        self.routing_strategy = routing_strategy
        
        # Initialize components
        self.nodes: Dict[str, NodeCapability] = {}
        self.model_sharder = ModelSharder(model_name)
        self.load_tracker = NodeLoadTracker()
        self.privacy_accountant = PrivacyAccountant(DPConfig())
        
        # Request tracking
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.request_history: List[Tuple[str, float, bool]] = []  # (node_id, timestamp, success)
        
        # Round-robin counter
        self._round_robin_index = 0
    
    def register_nodes(self, hospital_nodes: List['HospitalNode']):
        """Register hospital nodes with the router."""
        for node in hospital_nodes:
            capability = NodeCapability(
                node_id=node.id,
                gpu_memory=self._parse_compute_capacity(node.compute_capacity),
                cpu_cores=16,  # Default
                network_bandwidth=1000,  # Default 1Gbps
                max_concurrent_requests=10,
                supported_models=[self.model_name],
                privacy_budget_remaining=100.0  # Default budget
            )
            self.nodes[node.id] = capability
    
    def _parse_compute_capacity(self, capacity_str: str) -> int:
        """Parse compute capacity string to estimate GPU memory."""
        if "A100" in capacity_str:
            gpu_count = int(capacity_str.split('x')[0]) if 'x' in capacity_str else 1
            return gpu_count * 80 * 1024  # A100 has 80GB
        elif "V100" in capacity_str:
            gpu_count = int(capacity_str.split('x')[0]) if 'x' in capacity_str else 1
            return gpu_count * 32 * 1024  # V100 has 32GB
        else:
            return 16 * 1024  # Default 16GB
    
    async def route_request(self, request: InferenceRequest) -> InferenceResponse:
        """Route inference request to appropriate node(s)."""
        start_time = time.time()
        
        # Check privacy budget
        if not self.privacy_accountant.check_budget(request.user_id, request.max_privacy_budget):
            raise ValueError(f"Insufficient privacy budget for user {request.user_id}")
        
        # Store active request
        self.active_requests[request.request_id] = request
        
        try:
            if request.require_consensus:
                response = await self._handle_consensus_request(request)
            else:
                response = await self._handle_single_node_request(request)
            
            # Record successful processing
            response.latency = time.time() - start_time
            self._record_request_success(response.processing_nodes, response.latency)
            
            return response
            
        except Exception as e:
            # Record failure
            self._record_request_failure(request)
            raise e
        finally:
            # Clean up
            self.active_requests.pop(request.request_id, None)
    
    async def _handle_single_node_request(self, request: InferenceRequest) -> InferenceResponse:
        """Handle request using single best node."""
        selected_node = self._select_best_node(request)
        
        if not selected_node:
            raise RuntimeError("No suitable node available")
        
        # Simulate inference (in practice, would make HTTP request to node)
        response_text = await self._simulate_inference(selected_node.node_id, request)
        
        # Spend privacy budget
        privacy_spent = min(request.max_privacy_budget, self.privacy_accountant.config.epsilon_per_query)
        self.privacy_accountant.spend_budget(request.user_id, privacy_spent, "inference")
        
        return InferenceResponse(
            request_id=request.request_id,
            text=response_text,
            privacy_cost=privacy_spent,
            remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id),
            processing_nodes=[selected_node.node_id],
            latency=0.0,  # Will be set by caller
            confidence_score=0.85  # Simulated
        )
    
    async def _handle_consensus_request(self, request: InferenceRequest) -> InferenceResponse:
        """Handle request requiring consensus from multiple nodes."""
        # Select multiple nodes for consensus
        min_nodes = min(3, len(self.nodes))  # At least 3 nodes for consensus
        selected_nodes = self._select_consensus_nodes(request, min_nodes)
        
        if len(selected_nodes) < min_nodes:
            raise RuntimeError("Insufficient nodes for consensus")
        
        # Send request to all selected nodes
        tasks = []
        for node in selected_nodes:
            task = self._simulate_inference(node.node_id, request)
            tasks.append(task)
        
        # Wait for responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate responses and check consensus
        valid_responses = [r for r in responses if isinstance(r, str)]
        
        if len(valid_responses) < min_nodes // 2 + 1:
            raise RuntimeError("Failed to achieve consensus - insufficient valid responses")
        
        # Simple consensus: majority vote on response similarity
        consensus_response = self._achieve_consensus(valid_responses)
        
        # Spend privacy budget (higher cost for consensus)
        privacy_spent = min(request.max_privacy_budget, self.privacy_accountant.config.epsilon_per_query * 1.5)
        self.privacy_accountant.spend_budget(request.user_id, privacy_spent, "consensus_inference")
        
        return InferenceResponse(
            request_id=request.request_id,
            text=consensus_response,
            privacy_cost=privacy_spent,
            remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id),
            processing_nodes=[node.node_id for node in selected_nodes],
            latency=0.0,
            confidence_score=0.95,  # Higher confidence for consensus
            consensus_achieved=True
        )
    
    def _select_best_node(self, request: InferenceRequest) -> Optional[NodeCapability]:
        """Select best node based on routing strategy."""
        available_nodes = [
            node for node in self.nodes.values()
            if node.is_healthy and self.model_name in node.supported_models
        ]
        
        if not available_nodes:
            return None
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            selected = available_nodes[self._round_robin_index % len(available_nodes)]
            self._round_robin_index += 1
            return selected
        
        elif self.routing_strategy == RoutingStrategy.PRIVACY_AWARE:
            # Prefer nodes with more privacy budget remaining
            return max(available_nodes, key=lambda n: n.privacy_budget_remaining)
        
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            # Select node with best performance score
            return max(available_nodes, key=lambda n: self.load_tracker.get_node_score(n.node_id))
        
        else:
            # Default to first available
            return available_nodes[0]
    
    def _select_consensus_nodes(self, request: InferenceRequest, num_nodes: int) -> List[NodeCapability]:
        """Select multiple nodes for consensus-based inference."""
        available_nodes = [
            node for node in self.nodes.values()
            if node.is_healthy and self.model_name in node.supported_models
        ]
        
        if len(available_nodes) < num_nodes:
            return available_nodes
        
        # Sort by combined score of privacy budget and performance
        scored_nodes = [
            (node, self.load_tracker.get_node_score(node.node_id) + node.privacy_budget_remaining / 100.0)
            for node in available_nodes
        ]
        
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, score in scored_nodes[:num_nodes]]
    
    async def _simulate_inference(self, node_id: str, request: InferenceRequest) -> str:
        """Simulate inference on a node (replace with actual HTTP call)."""
        # Simulate network delay
        await asyncio.sleep(0.1 + np.random.exponential(0.05))
        
        # Simulate response based on request
        response_templates = [
            "Based on the clinical presentation, consider differential diagnosis including...",
            "The symptoms suggest a pattern consistent with...",
            "Recommended next steps include further evaluation with...",
            "Patient history indicates possible...",
        ]
        
        # Simple response generation (in practice, would be actual LLM inference)
        response = np.random.choice(response_templates)
        return f"{response} [Generated by node {node_id}]"
    
    def _achieve_consensus(self, responses: List[str]) -> str:
        """Achieve consensus from multiple node responses."""
        # Simplified consensus: return most common response or first if all different
        from collections import Counter
        
        # In practice, would use semantic similarity
        response_counts = Counter(responses)
        most_common = response_counts.most_common(1)[0]
        
        if most_common[1] > 1:  # More than one node gave same response
            return most_common[0]
        else:
            # No exact matches, return longest response (proxy for most detailed)
            return max(responses, key=len)
    
    def _record_request_success(self, node_ids: List[str], latency: float):
        """Record successful request for performance tracking."""
        timestamp = time.time()
        for node_id in node_ids:
            self.load_tracker.record_response_time(node_id, latency)
            self.request_history.append((node_id, timestamp, True))
    
    def _record_request_failure(self, request: InferenceRequest):
        """Record failed request."""
        timestamp = time.time()
        # Record failure for all nodes that might have been involved
        for node_id in self.nodes.keys():
            self.load_tracker.record_error(node_id)
            self.request_history.append((node_id, timestamp, False))
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all registered nodes."""
        health_status = {}
        
        for node_id, node in self.nodes.items():
            try:
                # Simulate health check (in practice, would ping node)
                await asyncio.sleep(0.01)
                
                # Update health status
                node.is_healthy = True
                node.last_health_check = time.time()
                
                health_status[node_id] = {
                    "healthy": True,
                    "load": self.load_tracker.node_loads.get(node_id, 0.0),
                    "avg_response_time": self.load_tracker.get_average_response_time(node_id),
                    "error_rate": self.load_tracker.get_error_rate(node_id),
                    "privacy_budget": node.privacy_budget_remaining
                }
                
            except Exception as e:
                node.is_healthy = False
                health_status[node_id] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        return health_status
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_requests = len(self.request_history)
        successful_requests = sum(1 for _, _, success in self.request_history if success)
        
        node_stats = {}
        for node_id in self.nodes.keys():
            node_requests = [r for r in self.request_history if r[0] == node_id]
            node_successes = sum(1 for _, _, success in node_requests if success)
            
            node_stats[node_id] = {
                "requests": len(node_requests),
                "success_rate": node_successes / max(len(node_requests), 1),
                "avg_response_time": self.load_tracker.get_average_response_time(node_id),
                "current_load": self.load_tracker.node_loads.get(node_id, 0.0),
                "performance_score": self.load_tracker.get_node_score(node_id)
            }
        
        return {
            "total_requests": total_requests,
            "success_rate": successful_requests / max(total_requests, 1),
            "active_requests": len(self.active_requests),
            "registered_nodes": len(self.nodes),
            "healthy_nodes": sum(1 for node in self.nodes.values() if node.is_healthy),
            "routing_strategy": self.routing_strategy.value,
            "node_statistics": node_stats
        }