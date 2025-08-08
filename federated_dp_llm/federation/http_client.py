"""
HTTP Client for Federated Node Communication

Simple HTTP client for communicating with federated nodes over the network.
"""

import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .node_server import NodeInferenceRequest, NodeInferenceResponse, NodeHealthResponse


class NodeStatus(Enum):
    """Node status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"


@dataclass
class NodeEndpoint:
    """Represents a federated node endpoint."""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.UNREACHABLE
    last_health_check: float = 0.0
    
    @property
    def base_url(self) -> str:
        """Get the base URL for this node."""
        return f"http://{self.host}:{self.port}"


class FederatedHTTPClient:
    """HTTP client for communicating with federated nodes."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, NodeEndpoint] = {}
        self.client = httpx.AsyncClient(timeout=timeout)
    
    def register_node(self, node_id: str, host: str, port: int):
        """Register a federated node."""
        endpoint = NodeEndpoint(node_id=node_id, host=host, port=port)
        self.nodes[node_id] = endpoint
        self.logger.info(f"Registered node {node_id} at {endpoint.base_url}")
    
    def remove_node(self, node_id: str):
        """Remove a federated node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Removed node {node_id}")
    
    async def health_check_node(self, node_id: str) -> Optional[NodeHealthResponse]:
        """Check health of a specific node."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        try:
            response = await self.client.get(f"{node.base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            health_response = NodeHealthResponse(**health_data)
            
            # Update node status
            node.status = NodeStatus.HEALTHY if health_response.status == "healthy" else NodeStatus.UNHEALTHY
            node.last_health_check = asyncio.get_event_loop().time()
            
            return health_response
            
        except Exception as e:
            self.logger.warning(f"Health check failed for node {node_id}: {e}")
            node.status = NodeStatus.UNREACHABLE
            node.last_health_check = asyncio.get_event_loop().time()
            return None
    
    async def health_check_all_nodes(self) -> Dict[str, Optional[NodeHealthResponse]]:
        """Check health of all registered nodes."""
        tasks = [
            self.health_check_node(node_id) 
            for node_id in self.nodes.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            node_id: result if not isinstance(result, Exception) else None
            for node_id, result in zip(self.nodes.keys(), results)
        }
    
    async def inference_request(self, node_id: str, request: NodeInferenceRequest) -> Optional[NodeInferenceResponse]:
        """Send inference request to a specific node."""
        if node_id not in self.nodes:
            self.logger.error(f"Node {node_id} not registered")
            return None
        
        node = self.nodes[node_id]
        
        try:
            response = await self.client.post(
                f"{node.base_url}/inference",
                json=request.dict()
            )
            response.raise_for_status()
            
            response_data = response.json()
            return NodeInferenceResponse(**response_data)
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error for node {node_id}: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Inference request failed for node {node_id}: {e}")
            return None
    
    async def distributed_inference(self, request: NodeInferenceRequest, node_ids: Optional[List[str]] = None) -> Dict[str, Optional[NodeInferenceResponse]]:
        """Send inference request to multiple nodes."""
        target_nodes = node_ids or list(self.nodes.keys())
        
        # Filter to only healthy nodes
        healthy_nodes = []
        for node_id in target_nodes:
            if node_id in self.nodes and self.nodes[node_id].status == NodeStatus.HEALTHY:
                healthy_nodes.append(node_id)
        
        if not healthy_nodes:
            self.logger.warning("No healthy nodes available for distributed inference")
            return {}
        
        # Send requests in parallel
        tasks = [
            self.inference_request(node_id, request)
            for node_id in healthy_nodes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            node_id: result if not isinstance(result, Exception) else None
            for node_id, result in zip(healthy_nodes, results)
        }
    
    async def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a node."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        try:
            response = await self.client.get(f"{node.base_url}/info")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get info for node {node_id}: {e}")
            return None
    
    def get_healthy_nodes(self) -> List[str]:
        """Get list of healthy node IDs."""
        return [
            node_id for node_id, node in self.nodes.items()
            if node.status == NodeStatus.HEALTHY
        ]
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get statistics about registered nodes."""
        total_nodes = len(self.nodes)
        healthy_nodes = len(self.get_healthy_nodes())
        
        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "unhealthy_nodes": total_nodes - healthy_nodes,
            "nodes": {
                node_id: {
                    "status": node.status.value,
                    "endpoint": node.base_url,
                    "last_health_check": node.last_health_check
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class SimpleDistributedInference:
    """Simple distributed inference coordinator."""
    
    def __init__(self, client: FederatedHTTPClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    async def consensus_inference(self, text: str, user_id: str, request_id: str, min_consensus: int = 2) -> Optional[str]:
        """Perform inference with consensus from multiple nodes."""
        # Create request
        request = NodeInferenceRequest(
            text=text,
            user_id=user_id,
            request_id=request_id
        )
        
        # Get responses from all healthy nodes
        responses = await self.client.distributed_inference(request)
        
        # Filter successful responses
        successful_responses = {
            node_id: response for node_id, response in responses.items()
            if response and response.success
        }
        
        if len(successful_responses) < min_consensus:
            self.logger.warning(f"Only {len(successful_responses)} successful responses, need {min_consensus} for consensus")
            return None
        
        # Simple consensus: return the most common response
        # In a real implementation, this would be more sophisticated
        response_texts = [resp.generated_text for resp in successful_responses.values()]
        
        if response_texts:
            # For now, just return the first response
            # Could implement voting, similarity scoring, etc.
            return response_texts[0]
        
        return None
    
    async def fastest_inference(self, text: str, user_id: str, request_id: str) -> Optional[str]:
        """Get inference from the fastest responding node."""
        # Create request
        request = NodeInferenceRequest(
            text=text,
            user_id=user_id,
            request_id=request_id
        )
        
        # Get healthy nodes
        healthy_nodes = self.client.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Race: return first successful response
        tasks = [
            self.client.inference_request(node_id, request)
            for node_id in healthy_nodes
        ]
        
        try:
            # Wait for first successful result
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
            # Get result from first completed task
            for task in done:
                result = await task
                if result and result.success:
                    return result.generated_text
            
        except Exception as e:
            self.logger.error(f"Fastest inference failed: {e}")
        
        return None