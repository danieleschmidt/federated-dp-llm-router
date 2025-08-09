"""
Simple Load Balancer

A basic, practical load balancer for federated inference without quantum complexity.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random

from ..federation.http_client import FederatedHTTPClient, NodeInferenceRequest, NodeInferenceResponse


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    FASTEST_RESPONSE = "fastest_response"


@dataclass
class NodeStats:
    """Statistics for a node."""
    node_id: str
    total_requests: int = 0
    active_requests: int = 0
    total_response_time: float = 0.0
    last_response_time: float = 0.0
    error_count: int = 0
    last_error_time: Optional[float] = None
    health_score: float = 1.0
    
    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Get error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests


class SimpleLoadBalancer:
    """Simple, practical load balancer for federated nodes."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.http_client = FederatedHTTPClient()
        self.node_stats: Dict[str, NodeStats] = {}
        self.round_robin_index = 0
        self.logger = logging.getLogger(__name__)
        
        # Health check configuration
        self.health_check_interval = 30.0  # seconds
        self.unhealthy_threshold = 3  # consecutive failures
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Start health checking
        self._start_health_monitoring()
    
    def register_node(self, node_id: str, host: str, port: int, weight: float = 1.0):
        """Register a node with the load balancer."""
        self.http_client.register_node(node_id, host, port)
        self.node_stats[node_id] = NodeStats(node_id=node_id)
        self.logger.info(f"Registered node {node_id} at {host}:{port}")
    
    def remove_node(self, node_id: str):
        """Remove a node from the load balancer."""
        self.http_client.remove_node(node_id)
        if node_id in self.node_stats:
            del self.node_stats[node_id]
        self.logger.info(f"Removed node {node_id}")
    
    def _start_health_monitoring(self):
        """Start periodic health monitoring."""
        async def health_monitor():
            while True:
                try:
                    await self.http_client.health_check_all_nodes()
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(5)  # Short retry on error
        
        self.health_check_task = asyncio.create_task(health_monitor())
    
    async def route_inference(self, request: NodeInferenceRequest) -> Optional[NodeInferenceResponse]:
        """Route an inference request to the best available node."""
        healthy_nodes = self.http_client.get_healthy_nodes()
        
        if not healthy_nodes:
            self.logger.warning("No healthy nodes available")
            return None
        
        # Select node based on strategy
        selected_node = self._select_node(healthy_nodes)
        
        if not selected_node:
            self.logger.warning("No node selected by load balancing strategy")
            return None
        
        # Update stats - request starting
        stats = self.node_stats.get(selected_node)
        if stats:
            stats.active_requests += 1
            stats.total_requests += 1
        
        start_time = time.time()
        
        try:
            # Send request
            response = await self.http_client.inference_request(selected_node, request)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update stats - request completed
            if stats:
                stats.active_requests = max(0, stats.active_requests - 1)
                stats.total_response_time += response_time
                stats.last_response_time = response_time
                
                if response and response.success:
                    # Success - improve health score
                    stats.health_score = min(1.0, stats.health_score + 0.01)
                else:
                    # Error - degrade health score
                    stats.error_count += 1
                    stats.last_error_time = time.time()
                    stats.health_score = max(0.0, stats.health_score - 0.1)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error routing to node {selected_node}: {e}")
            
            # Update error stats
            if stats:
                stats.active_requests = max(0, stats.active_requests - 1)
                stats.error_count += 1
                stats.last_error_time = time.time()
                stats.health_score = max(0.0, stats.health_score - 0.1)
            
            return None
    
    def _select_node(self, healthy_nodes: List[str]) -> Optional[str]:
        """Select a node based on the configured strategy."""
        if not healthy_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(healthy_nodes)
        else:
            # Fallback to round robin
            return self._round_robin_select(healthy_nodes)
    
    def _round_robin_select(self, nodes: List[str]) -> str:
        """Simple round-robin selection."""
        if not nodes:
            return None
        
        selected = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index = (self.round_robin_index + 1) % len(nodes)
        return selected
    
    def _weighted_round_robin_select(self, nodes: List[str]) -> str:
        """Weighted round-robin based on health scores."""
        if not nodes:
            return None
        
        # Weight by health score
        weights = []
        for node in nodes:
            stats = self.node_stats.get(node)
            health_score = stats.health_score if stats else 1.0
            weights.append(max(0.1, health_score))  # Minimum weight of 0.1
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(nodes)
        
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return nodes[i]
        
        return nodes[-1]  # Fallback
    
    def _least_connections_select(self, nodes: List[str]) -> str:
        """Select node with fewest active connections."""
        if not nodes:
            return None
        
        best_node = None
        min_connections = float('inf')
        
        for node in nodes:
            stats = self.node_stats.get(node)
            connections = stats.active_requests if stats else 0
            
            if connections < min_connections:
                min_connections = connections
                best_node = node
        
        return best_node or nodes[0]
    
    def _random_select(self, nodes: List[str]) -> str:
        """Random selection."""
        return random.choice(nodes) if nodes else None
    
    def _fastest_response_select(self, nodes: List[str]) -> str:
        """Select node with fastest average response time."""
        if not nodes:
            return None
        
        best_node = None
        best_response_time = float('inf')
        
        for node in nodes:
            stats = self.node_stats.get(node)
            if not stats or stats.total_requests == 0:
                # New node gets priority
                return node
            
            avg_response_time = stats.average_response_time
            if avg_response_time < best_response_time:
                best_response_time = avg_response_time
                best_node = node
        
        return best_node or nodes[0]
    
    async def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        stats = {
            "strategy": self.strategy.value,
            "total_nodes": len(self.node_stats),
            "healthy_nodes": len(self.http_client.get_healthy_nodes()),
            "node_stats": {}
        }
        
        for node_id, node_stats in self.node_stats.items():
            stats["node_stats"][node_id] = {
                "total_requests": node_stats.total_requests,
                "active_requests": node_stats.active_requests,
                "average_response_time": node_stats.average_response_time,
                "error_rate": node_stats.error_rate,
                "health_score": node_stats.health_score,
                "last_response_time": node_stats.last_response_time
            }
        
        return stats
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all nodes."""
        return await self.http_client.health_check_all_nodes()
    
    def update_strategy(self, new_strategy: LoadBalancingStrategy):
        """Update the load balancing strategy."""
        self.strategy = new_strategy
        self.logger.info(f"Updated load balancing strategy to {new_strategy.value}")
    
    async def shutdown(self):
        """Shutdown the load balancer."""
        if self.health_check_task:
            self.health_check_task.cancel()
        
        await self.http_client.close()
        self.logger.info("Load balancer shutdown completed")


class FederatedInferenceCoordinator:
    """High-level coordinator for federated inference."""
    
    def __init__(self, load_balancer: SimpleLoadBalancer):
        self.load_balancer = load_balancer
        self.logger = logging.getLogger(__name__)
    
    async def single_inference(self, text: str, user_id: str, request_id: str) -> Optional[str]:
        """Perform inference on a single node."""
        request = NodeInferenceRequest(
            text=text,
            user_id=user_id,
            request_id=request_id
        )
        
        response = await self.load_balancer.route_inference(request)
        
        if response and response.success:
            return response.generated_text
        
        return None
    
    async def redundant_inference(self, text: str, user_id: str, request_id: str, redundancy: int = 2) -> Optional[str]:
        """Perform inference with redundancy across multiple nodes."""
        if redundancy < 1:
            redundancy = 1
        
        # Create multiple requests
        tasks = []
        for i in range(redundancy):
            request = NodeInferenceRequest(
                text=text,
                user_id=user_id,
                request_id=f"{request_id}_{i}"
            )
            tasks.append(self.load_balancer.route_inference(request))
        
        try:
            # Wait for all responses
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Find successful responses
            successful_responses = []
            for response in responses:
                if (isinstance(response, NodeInferenceResponse) and 
                    response.success and response.generated_text):
                    successful_responses.append(response.generated_text)
            
            if successful_responses:
                # For now, return the first successful response
                # Could implement voting or consensus here
                return successful_responses[0]
            
        except Exception as e:
            self.logger.error(f"Redundant inference failed: {e}")
        
        return None
    
    async def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        lb_stats = await self.load_balancer.get_load_balancing_stats()
        health_stats = await self.load_balancer.health_check_all()
        
        return {
            "load_balancer": lb_stats,
            "node_health": health_stats,
            "timestamp": time.time()
        }