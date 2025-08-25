"""
Federated Client Components

Hospital nodes and inference clients for participating in federated learning
and privacy-preserving inference.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import httpx
try:
    from ..quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
except ImportError:
    # For files outside quantum_planning module
    from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
from ..core.privacy_accountant import PrivacyAccountant, DPConfig
from ..routing.load_balancer import InferenceRequest, InferenceResponse


@dataclass
class HospitalNode:
    """Represents a hospital node in the federated network."""
    id: str
    endpoint: str
    data_size: int
    compute_capacity: str
    department: Optional[str] = None
    region: Optional[str] = None
    last_seen: Optional[float] = None
    is_active: bool = True
    model_versions: Dict[str, str] = None
    
    def __post_init__(self):
        if self.model_versions is None:
            self.model_versions = {}
        if self.last_seen is None:
            self.last_seen = time.time()


@dataclass
class ModelUpdate:
    """Represents a model update for federated learning."""
    node_id: str
    model_name: str
    parameters: Dict[str, List]
    metadata: Dict[str, Any]
    privacy_spent: float
    training_samples: int
    update_round: int
    timestamp: float


class PrivateInferenceClient:
    """Client for making privacy-preserving inference requests."""
    
    def __init__(
        self,
        router_endpoint: str,
        user_id: str,
        department: str = "general",
        auth_token: Optional[str] = None
    ):
        self.router_endpoint = router_endpoint.rstrip('/')
        self.user_id = user_id
        self.department = department
        self.auth_token = auth_token
        self.request_counter = 0
        
        # HTTP client with timeout and security settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            verify=True,  # Verify SSL certificates
            headers={
                "User-Agent": "FederatedDP-LLM-Client/0.1.0",
                "Content-Type": "application/json"
            }
        )
        
        if auth_token:
            self.client.headers["Authorization"] = f"Bearer {auth_token}"
    
    async def query(
        self,
        prompt: str,
        model_name: str = "medllama-7b",
        max_privacy_budget: float = 0.1,
        require_consensus: bool = False,
        priority: int = 1,
        timeout: float = 30.0,
        audit_trail: bool = True
    ) -> InferenceResponse:
        """Make a privacy-preserving inference query."""
        
        self.request_counter += 1
        
        # Prepare request payload
        request_data = {
            "prompt": prompt,
            "model_name": model_name,
            "max_privacy_budget": max_privacy_budget,
            "require_consensus": require_consensus,
            "priority": priority,
            "timeout": timeout,
            "department": self.department
        }
        
        # Add audit metadata if requested
        if audit_trail:
            request_data["audit_metadata"] = {
                "client_id": f"{self.user_id}_{self.request_counter}",
                "timestamp": time.time(),
                "department": self.department
            }
        
        try:
            response = await self.client.post(
                f"{self.router_endpoint}/inference",
                json=request_data,
                timeout=timeout + 5.0  # Add buffer for network overhead
            )
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            return InferenceResponse(
                request_id=response_data["request_id"],
                text=response_data["text"],
                privacy_cost=response_data["privacy_cost"],
                remaining_budget=response_data["remaining_budget"],
                processing_nodes=response_data["processing_nodes"],
                latency=response_data["latency"],
                confidence_score=response_data.get("confidence_score", 0.0),
                consensus_achieved=response_data.get("consensus_achieved", False)
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                error_detail = e.response.json().get("detail", "Bad request")
                raise ValueError(f"Invalid request: {error_detail}")
            elif e.response.status_code == 401:
                raise ValueError("Authentication failed - invalid token")
            elif e.response.status_code == 403:
                raise ValueError("Access forbidden - insufficient permissions")
            elif e.response.status_code == 408:
                raise TimeoutError("Request timed out")
            else:
                raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")
        
        except httpx.TimeoutException:
            raise TimeoutError("Request timed out")
        
        except Exception as e:
            raise RuntimeError(f"Inference request failed: {str(e)}")
    
    async def get_privacy_budget(self) -> Dict[str, Any]:
        """Get current privacy budget information."""
        try:
            response = await self.client.get(
                f"{self.router_endpoint}/privacy/budget/{self.user_id}"
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Failed to get privacy budget: {e.response.text}")
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models for inference."""
        try:
            response = await self.client.get(f"{self.router_endpoint}/models")
            response.raise_for_status()
            return response.json()["models"]
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Failed to list models: {e.response.text}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the federated router."""
        try:
            response = await self.client.get(f"{self.router_endpoint}/health")
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Health check failed: {e.response.text}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class FederatedLearningClient:
    """Client for participating in federated learning rounds."""
    
    def __init__(
        self,
        node: HospitalNode,
        privacy_config: DPConfig,
        coordinator_endpoint: str
    ):
        self.node = node
        self.privacy_accountant = PrivacyAccountant(privacy_config)
        self.coordinator_endpoint = coordinator_endpoint.rstrip('/')
        
        # Local model state
        self.local_model_parameters: Dict[str, List] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        # HTTP client for coordinator communication
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),  # Longer timeout for training
            verify=True
        )
    
    async def register_with_coordinator(self) -> bool:
        """Register this node with the federated learning coordinator."""
        registration_data = {
            "node_info": asdict(self.node),
            "privacy_config": {
                "max_epsilon": self.privacy_accountant.config.max_budget_per_user,
                "delta": self.privacy_accountant.config.delta,
                "mechanism": self.privacy_accountant.config.mechanism.value
            },
            "timestamp": time.time()
        }
        
        try:
            response = await self.client.post(
                f"{self.coordinator_endpoint}/register",
                json=registration_data
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Registration failed: {str(e)}")
            return False
    
    async def participate_in_round(self, round_id: str, global_model: Dict[str, List]) -> ModelUpdate:
        """Participate in a federated learning round."""
        start_time = time.time()
        
        # Simulate local training (in practice, would train actual model)
        local_updates = await self._simulate_local_training(global_model)
        
        # Calculate privacy cost
        privacy_cost = self.privacy_accountant.config.epsilon_per_query
        
        # Create model update
        model_update = ModelUpdate(
            node_id=self.node.id,
            model_name="medllama-7b",  # Default model
            parameters=local_updates,
            metadata={
                "training_time": time.time() - start_time,
                "data_samples": self.node.data_size,
                "node_capacity": self.node.compute_capacity,
                "department": self.node.department
            },
            privacy_spent=privacy_cost,
            training_samples=min(1000, self.node.data_size),  # Cap for privacy
            update_round=len(self.training_history) + 1,
            timestamp=time.time()
        )
        
        # Record training history
        self.training_history.append({
            "round_id": round_id,
            "privacy_spent": privacy_cost,
            "samples_used": model_update.training_samples,
            "timestamp": model_update.timestamp
        })
        
        return model_update
    
    async def _simulate_local_training(self, global_model: Dict[str, List]) -> Dict[str, List]:
        """Simulate local model training with differential privacy."""
        # Simulate training delay
        training_time = np.random.uniform(1.0, 5.0)  # 1-5 seconds
        await asyncio.sleep(training_time)
        
        # Create simulated parameter updates
        updates = {}
        
        for param_name, param_tensor in global_model.items():
            # Simulate gradient-based updates
            gradient = np.random.normal(0, 0.01, param_tensor.shape)
            
            # Add differential privacy noise
            dp_noise = self.privacy_accountant.add_noise_to_query(
                gradient,
                sensitivity=1.0,  # L2 sensitivity
                epsilon=self.privacy_accountant.config.epsilon_per_query
            )
            
            # Apply update
            updates[param_name] = param_tensor + 0.01 * dp_noise  # Learning rate 0.01
        
        return updates
    
    async def submit_model_update(self, round_id: str, model_update: ModelUpdate) -> bool:
        """Submit model update to coordinator."""
        # Serialize numpy arrays to lists for JSON
        serialized_parameters = {
            name: param.tolist() if isinstance(param, List) else param
            for name, param in model_update.parameters.items()
        }
        
        update_data = {
            "round_id": round_id,
            "node_id": model_update.node_id,
            "model_name": model_update.model_name,
            "parameters": serialized_parameters,
            "metadata": model_update.metadata,
            "privacy_spent": model_update.privacy_spent,
            "training_samples": model_update.training_samples,
            "timestamp": model_update.timestamp
        }
        
        try:
            response = await self.client.post(
                f"{self.coordinator_endpoint}/submit_update",
                json=update_data
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Failed to submit model update: {str(e)}")
            return False
    
    async def get_aggregated_model(self, round_id: str) -> Optional[Dict[str, List]]:
        """Get aggregated model after federated learning round."""
        try:
            response = await self.client.get(
                f"{self.coordinator_endpoint}/aggregated_model/{round_id}"
            )
            response.raise_for_status()
            
            model_data = response.json()
            
            # Deserialize parameters back to numpy arrays
            parameters = {
                name: np.array(param) if isinstance(param, list) else param
                for name, param in model_data["parameters"].items()
            }
            
            return parameters
            
        except Exception as e:
            print(f"Failed to get aggregated model: {str(e)}")
            return None
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for this node."""
        if not self.training_history:
            return {"rounds_participated": 0}
        
        total_privacy_spent = sum(h["privacy_spent"] for h in self.training_history)
        total_samples = sum(h["samples_used"] for h in self.training_history)
        
        return {
            "rounds_participated": len(self.training_history),
            "total_privacy_spent": total_privacy_spent,
            "remaining_privacy_budget": self.privacy_accountant.get_remaining_budget(self.node.id),
            "total_samples_used": total_samples,
            "average_samples_per_round": total_samples / len(self.training_history),
            "last_participation": self.training_history[-1]["timestamp"],
            "node_info": asdict(self.node)
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()