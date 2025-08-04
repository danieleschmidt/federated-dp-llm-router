"""
Federated Learning Protocols

Implements various federated learning protocols and communication patterns
for distributed model training with privacy preservation.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Protocol, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class CommunicationPattern(Enum):
    """Communication patterns for federated learning."""
    CENTRALIZED = "centralized"  # Central server coordinates
    DECENTRALIZED = "decentralized"  # Peer-to-peer communication
    HIERARCHICAL = "hierarchical"  # Multi-level coordination
    GOSSIP = "gossip"  # Gossip-based propagation


class AggregationStrategy(Enum):
    """Model aggregation strategies."""
    FEDERATED_AVERAGING = "fedavg"
    FEDERATED_PROXIMAL = "fedprox"
    SCAFFOLD = "scaffold"
    FEDERATED_NOVA = "fednova"
    CUSTOM = "custom"


T = TypeVar('T')


class FederatedProtocol(Protocol, Generic[T]):
    """Protocol interface for federated learning algorithms."""
    
    def initialize_round(self, participants: List[str]) -> str:
        """Initialize a new federated learning round."""
        ...
    
    def aggregate_updates(self, updates: List[T]) -> T:
        """Aggregate model updates from participants."""
        ...
    
    def validate_update(self, update: T, participant_id: str) -> bool:
        """Validate an update from a participant."""
        ...


@dataclass
class FedAvgUpdate:
    """FedAvg algorithm update structure."""
    participant_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    training_loss: float
    privacy_cost: float
    timestamp: float


@dataclass 
class FedProxUpdate:
    """FedProx algorithm update structure."""
    participant_id: str
    model_weights: Dict[str, np.ndarray]
    proximal_term: Dict[str, np.ndarray]
    num_samples: int
    mu: float  # Proximal regularization parameter
    training_loss: float
    privacy_cost: float
    timestamp: float


class FederatedAveragingProtocol:
    """Implementation of the FedAvg protocol."""
    
    def __init__(self, min_participants: int = 3):
        self.min_participants = min_participants
        self.active_rounds: Dict[str, Dict[str, Any]] = {}
        self.global_model: Optional[Dict[str, np.ndarray]] = None
        self.round_counter = 0
    
    def initialize_round(self, participants: List[str]) -> str:
        """Initialize a new FedAvg round."""
        if len(participants) < self.min_participants:
            raise ValueError(f"Need at least {self.min_participants} participants")
        
        self.round_counter += 1
        round_id = f"fedavg_round_{self.round_counter}_{int(time.time())}"
        
        self.active_rounds[round_id] = {
            "participants": participants,
            "updates": {},
            "status": "collecting",
            "start_time": time.time(),
            "global_model_snapshot": self.global_model.copy() if self.global_model else None
        }
        
        return round_id
    
    def submit_update(self, round_id: str, update: FedAvgUpdate) -> bool:
        """Submit an update for a round."""
        if round_id not in self.active_rounds:
            return False
        
        round_info = self.active_rounds[round_id]
        
        if update.participant_id not in round_info["participants"]:
            return False
        
        if round_info["status"] != "collecting":
            return False
        
        # Validate update
        if not self.validate_update(update, update.participant_id):
            return False
        
        round_info["updates"][update.participant_id] = update
        
        # Check if we have enough updates to aggregate
        if len(round_info["updates"]) >= self.min_participants:
            round_info["status"] = "ready_for_aggregation"
        
        return True
    
    def aggregate_updates(self, round_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate updates using FedAvg algorithm."""
        if round_id not in self.active_rounds:
            return None
        
        round_info = self.active_rounds[round_id]
        updates = list(round_info["updates"].values())
        
        if len(updates) < self.min_participants:
            return None
        
        # Calculate total samples
        total_samples = sum(update.num_samples for update in updates)
        
        if total_samples == 0:
            return None
        
        # Weighted average based on number of samples
        aggregated_weights = {}
        
        # Get parameter names from first update
        param_names = list(updates[0].model_weights.keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_param = update.model_weights[param_name] * weight
                
                if weighted_sum is None:
                    weighted_sum = weighted_param.copy()
                else:
                    weighted_sum += weighted_param
            
            aggregated_weights[param_name] = weighted_sum
        
        # Update global model
        self.global_model = aggregated_weights
        round_info["status"] = "completed"
        round_info["aggregated_model"] = aggregated_weights
        
        return aggregated_weights
    
    def validate_update(self, update: FedAvgUpdate, participant_id: str) -> bool:
        """Validate a FedAvg update."""
        # Basic validation checks
        if update.participant_id != participant_id:
            return False
        
        if update.num_samples <= 0:
            return False
        
        if not update.model_weights:
            return False
        
        # Check for reasonable parameter values
        for param_name, param_value in update.model_weights.items():
            if not isinstance(param_value, np.ndarray):
                return False
            
            if np.any(np.isnan(param_value)) or np.any(np.isinf(param_value)):
                return False
            
            # Check for suspiciously large values (potential attack)
            if np.max(np.abs(param_value)) > 100.0:
                return False
        
        return True


class FederatedProximalProtocol:
    """Implementation of the FedProx protocol."""
    
    def __init__(self, mu: float = 0.01, min_participants: int = 3):
        self.mu = mu  # Proximal regularization parameter
        self.min_participants = min_participants
        self.active_rounds: Dict[str, Dict[str, Any]] = {}
        self.global_model: Optional[Dict[str, np.ndarray]] = None
        self.round_counter = 0
    
    def initialize_round(self, participants: List[str]) -> str:
        """Initialize a new FedProx round."""
        if len(participants) < self.min_participants:
            raise ValueError(f"Need at least {self.min_participants} participants")
        
        self.round_counter += 1
        round_id = f"fedprox_round_{self.round_counter}_{int(time.time())}"
        
        self.active_rounds[round_id] = {
            "participants": participants,
            "updates": {},
            "status": "collecting",
            "start_time": time.time(),
            "global_model_snapshot": self.global_model.copy() if self.global_model else None,
            "mu": self.mu
        }
        
        return round_id
    
    def submit_update(self, round_id: str, update: FedProxUpdate) -> bool:
        """Submit a FedProx update."""
        if round_id not in self.active_rounds:
            return False
        
        round_info = self.active_rounds[round_id]
        
        if update.participant_id not in round_info["participants"]:
            return False
        
        if round_info["status"] != "collecting":
            return False
        
        if not self.validate_update(update, update.participant_id):
            return False
        
        round_info["updates"][update.participant_id] = update
        
        if len(round_info["updates"]) >= self.min_participants:
            round_info["status"] = "ready_for_aggregation"
        
        return True
    
    def aggregate_updates(self, round_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate updates using FedProx algorithm."""
        if round_id not in self.active_rounds:
            return None
        
        round_info = self.active_rounds[round_id]
        updates = list(round_info["updates"].values())
        
        if len(updates) < self.min_participants:
            return None
        
        # FedProx aggregation with proximal terms
        total_samples = sum(update.num_samples for update in updates)
        
        if total_samples == 0:
            return None
        
        aggregated_weights = {}
        param_names = list(updates[0].model_weights.keys())
        
        for param_name in param_names:
            weighted_sum = None
            proximal_sum = None
            
            for update in updates:
                weight = update.num_samples / total_samples
                
                # Regular weighted average
                weighted_param = update.model_weights[param_name] * weight
                
                # Proximal term
                proximal_param = update.proximal_term[param_name] * weight * update.mu
                
                if weighted_sum is None:
                    weighted_sum = weighted_param.copy()
                    proximal_sum = proximal_param.copy()
                else:
                    weighted_sum += weighted_param
                    proximal_sum += proximal_param
            
            # Apply proximal regularization
            aggregated_weights[param_name] = weighted_sum - proximal_sum
        
        self.global_model = aggregated_weights
        round_info["status"] = "completed"
        round_info["aggregated_model"] = aggregated_weights
        
        return aggregated_weights
    
    def validate_update(self, update: FedProxUpdate, participant_id: str) -> bool:
        """Validate a FedProx update."""
        # Basic validation
        if update.participant_id != participant_id:
            return False
        
        if update.num_samples <= 0:
            return False
        
        if not update.model_weights or not update.proximal_term:
            return False
        
        # Check parameter consistency
        if set(update.model_weights.keys()) != set(update.proximal_term.keys()):
            return False
        
        # Validate parameter values
        for param_name in update.model_weights.keys():
            model_param = update.model_weights[param_name]
            proximal_param = update.proximal_term[param_name]
            
            if not isinstance(model_param, np.ndarray) or not isinstance(proximal_param, np.ndarray):
                return False
            
            if model_param.shape != proximal_param.shape:
                return False
            
            if (np.any(np.isnan(model_param)) or np.any(np.isinf(model_param)) or
                np.any(np.isnan(proximal_param)) or np.any(np.isinf(proximal_param))):
                return False
        
        return True


class DecentralizedProtocol:
    """Decentralized federated learning protocol using gossip communication."""
    
    def __init__(self, gossip_probability: float = 0.5):
        self.gossip_probability = gossip_probability
        self.peer_connections: Dict[str, List[str]] = {}
        self.node_models: Dict[str, Dict[str, np.ndarray]] = {}
        self.communication_rounds: int = 0
    
    def register_node(self, node_id: str, neighbors: List[str]):
        """Register a node with its neighbors."""
        self.peer_connections[node_id] = neighbors
        
        # Initialize with random model if not exists
        if node_id not in self.node_models:
            # Placeholder model initialization
            self.node_models[node_id] = {
                "layer1": np.random.normal(0, 0.1, (100, 50)),
                "layer2": np.random.normal(0, 0.1, (50, 10))
            }
    
    async def gossip_round(self, max_iterations: int = 10) -> Dict[str, Any]:
        """Execute a gossip-based communication round."""
        self.communication_rounds += 1
        
        messages_sent = 0
        nodes_participating = set()
        
        for iteration in range(max_iterations):
            # Each node decides whether to gossip
            for node_id in self.peer_connections.keys():
                if np.random.random() < self.gossip_probability:
                    await self._node_gossip(node_id)
                    messages_sent += 1
                    nodes_participating.add(node_id)
            
            # Small delay between iterations
            await asyncio.sleep(0.01)
        
        return {
            "round": self.communication_rounds,
            "messages_sent": messages_sent,
            "nodes_participating": len(nodes_participating),
            "total_nodes": len(self.peer_connections)
        }
    
    async def _node_gossip(self, node_id: str):
        """Have a node gossip with a random neighbor."""
        neighbors = self.peer_connections.get(node_id, [])
        
        if not neighbors:
            return
        
        # Select random neighbor
        neighbor_id = np.random.choice(neighbors)
        
        if neighbor_id not in self.node_models:
            return
        
        # Exchange and average models
        node_model = self.node_models[node_id]
        neighbor_model = self.node_models[neighbor_id]
        
        # Simple averaging (in practice, would be more sophisticated)
        for param_name in node_model.keys():
            if param_name in neighbor_model:
                averaged_param = (node_model[param_name] + neighbor_model[param_name]) / 2.0
                
                # Update both nodes with averaged parameters
                self.node_models[node_id][param_name] = averaged_param.copy()
                self.node_models[neighbor_id][param_name] = averaged_param.copy()
    
    def get_consensus_level(self) -> float:
        """Calculate how close all node models are to consensus."""
        if len(self.node_models) < 2:
            return 1.0
        
        # Calculate pairwise distances between all node models
        node_ids = list(self.node_models.keys())
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                node1_model = self.node_models[node_ids[i]]
                node2_model = self.node_models[node_ids[j]]
                
                # Calculate L2 distance between models
                distance = 0.0
                for param_name in node1_model.keys():
                    if param_name in node2_model:
                        param_dist = np.linalg.norm(
                            node1_model[param_name] - node2_model[param_name]
                        )
                        distance += param_dist
                
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        avg_distance = total_distance / comparisons
        
        # Convert to consensus level (closer to 0 distance = higher consensus)
        consensus_level = 1.0 / (1.0 + avg_distance)
        
        return consensus_level


class ProtocolManager:
    """Manages multiple federated learning protocols."""
    
    def __init__(self):
        self.protocols: Dict[str, Any] = {}
        self.active_protocol: Optional[str] = None
    
    def register_protocol(self, name: str, protocol: Any):
        """Register a federated learning protocol."""
        self.protocols[name] = protocol
    
    def set_active_protocol(self, name: str) -> bool:
        """Set the active protocol."""
        if name not in self.protocols:
            return False
        
        self.active_protocol = name
        return True
    
    def get_protocol(self, name: str) -> Optional[Any]:
        """Get a registered protocol."""
        return self.protocols.get(name)
    
    def list_protocols(self) -> List[str]:
        """List all registered protocols."""
        return list(self.protocols.keys())
    
    async def execute_round(self, protocol_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a round using specified protocol."""
        if protocol_name not in self.protocols:
            raise ValueError(f"Protocol {protocol_name} not found")
        
        protocol = self.protocols[protocol_name]
        
        # Execute based on protocol type
        if hasattr(protocol, 'gossip_round'):
            # Decentralized protocol
            return await protocol.gossip_round(**kwargs)
        else:
            # Centralized protocol
            round_id = protocol.initialize_round(kwargs.get('participants', []))
            return {"round_id": round_id, "status": "initialized"}
    
    def get_protocol_stats(self) -> Dict[str, Any]:
        """Get statistics for all protocols."""
        stats = {}
        
        for name, protocol in self.protocols.items():
            if hasattr(protocol, 'round_counter'):
                stats[name] = {
                    "rounds_completed": protocol.round_counter,
                    "active_rounds": len(getattr(protocol, 'active_rounds', {})),
                    "type": "centralized"
                }
            elif hasattr(protocol, 'communication_rounds'):
                stats[name] = {
                    "communication_rounds": protocol.communication_rounds,
                    "registered_nodes": len(getattr(protocol, 'peer_connections', {})),
                    "consensus_level": protocol.get_consensus_level() if hasattr(protocol, 'get_consensus_level') else 0.0,
                    "type": "decentralized"
                }
            else:
                stats[name] = {"type": "unknown"}
        
        return {
            "total_protocols": len(self.protocols),
            "active_protocol": self.active_protocol,
            "protocol_details": stats
        }