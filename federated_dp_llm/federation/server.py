"""
Federated Learning Server/Coordinator

Coordinates federated learning rounds, aggregates model updates, and manages
the global model state across distributed hospital nodes.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from ..core.privacy_accountant import PrivacyAccountant, DPConfig
from ..core.secure_aggregation import SecureAggregator
from ..federation.client import HospitalNode, ModelUpdate


@dataclass
class TrainingRound:
    """Represents a federated learning training round."""
    round_id: str
    participants: List[str]
    global_model: Dict[str, np.ndarray]
    submitted_updates: Dict[str, ModelUpdate]
    aggregated_model: Optional[Dict[str, np.ndarray]]
    status: str = "active"  # active, aggregating, completed, failed
    start_time: float = time.time()
    deadline: float = time.time() + 300  # 5 minutes default
    min_participants: int = 3
    privacy_budget_per_round: float = 1.0


@dataclass
class FederatedModel:
    """Represents the global federated model."""
    model_name: str
    version: int
    parameters: Dict[str, np.ndarray]
    training_rounds: int
    total_participants: int
    last_updated: float
    performance_metrics: Dict[str, float]


class FederatedAggregator:
    """Handles aggregation of model updates from federated nodes."""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.secure_aggregator = SecureAggregator()
    
    def aggregate_updates(
        self,
        updates: List[ModelUpdate],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate model updates using specified method."""
        
        if not updates:
            raise ValueError("No updates to aggregate")
        
        if self.aggregation_method == "fedavg":
            return self._federated_averaging(updates, weights)
        elif self.aggregation_method == "weighted_average":
            return self._weighted_averaging(updates, weights)
        elif self.aggregation_method == "secure_aggregation":
            return self._secure_aggregation(updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _federated_averaging(
        self,
        updates: List[ModelUpdate],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """FedAvg algorithm - weighted average by number of samples."""
        
        if weights is None:
            # Use training samples as weights
            weights = [update.training_samples for update in updates]
        
        total_samples = sum(weights)
        aggregated_params = {}
        
        # Get parameter names from first update
        param_names = list(updates[0].parameters.keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update, weight in zip(updates, weights):
                param_value = update.parameters[param_name]
                weighted_param = param_value * (weight / total_samples)
                
                if weighted_sum is None:
                    weighted_sum = weighted_param.copy()
                else:
                    weighted_sum += weighted_param
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def _weighted_averaging(
        self,
        updates: List[ModelUpdate],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Weighted averaging with custom weights."""
        
        if weights is None:
            weights = [1.0] * len(updates)  # Equal weights
        
        total_weight = sum(weights)
        aggregated_params = {}
        
        param_names = list(updates[0].parameters.keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update, weight in zip(updates, weights):
                param_value = update.parameters[param_name]
                weighted_param = param_value * (weight / total_weight)
                
                if weighted_sum is None:
                    weighted_sum = weighted_param.copy()
                else:
                    weighted_sum += weighted_param
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def _secure_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Secure aggregation using cryptographic protocols."""
        
        # Start secure aggregation round
        round_id = f"secure_agg_{int(time.time())}"
        participants = [update.node_id for update in updates]
        
        agg_round = self.secure_aggregator.start_aggregation_round(
            round_id, participants
        )
        
        # Submit encrypted shares
        for update in updates:
            # Convert parameters to numpy array for aggregation
            param_array = np.concatenate([
                param.flatten() for param in update.parameters.values()
            ])
            
            self.secure_aggregator.submit_share(
                round_id, update.node_id, param_array
            )
        
        # Perform secure aggregation
        aggregated_array = self.secure_aggregator.aggregate_shares(round_id)
        
        if aggregated_array is None:
            raise RuntimeError("Secure aggregation failed")
        
        # Reconstruct parameter structure
        aggregated_params = self._reconstruct_parameters(
            aggregated_array, updates[0].parameters
        )
        
        return aggregated_params
    
    def _reconstruct_parameters(
        self,
        flat_array: np.ndarray,
        template_params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Reconstruct parameter dictionary from flattened array."""
        
        reconstructed = {}
        offset = 0
        
        for param_name, param_template in template_params.items():
            param_size = param_template.size
            param_shape = param_template.shape
            
            param_data = flat_array[offset:offset + param_size]
            reconstructed[param_name] = param_data.reshape(param_shape)
            
            offset += param_size
        
        return reconstructed


class FederatedTrainer:
    """Main federated learning coordinator/trainer."""
    
    def __init__(
        self,
        base_model: str,
        dp_config: DPConfig,
        rounds: int = 100,
        clients_per_round: int = 5,
        local_epochs: int = 1,
        aggregation_method: str = "fedavg"
    ):
        self.base_model = base_model
        self.dp_config = dp_config
        self.max_rounds = rounds
        self.clients_per_round = clients_per_round
        self.local_epochs = local_epochs
        
        # Components
        self.privacy_accountant = PrivacyAccountant(dp_config)
        self.aggregator = FederatedAggregator(aggregation_method)
        
        # State management
        self.registered_nodes: Dict[str, HospitalNode] = {}
        self.active_rounds: Dict[str, TrainingRound] = {}
        self.global_models: Dict[str, FederatedModel] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.round_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize global model
        self._initialize_global_model()
    
    def _initialize_global_model(self):
        """Initialize the global model with random parameters."""
        
        # Simulate model parameters (in practice, load from actual model)
        model_params = {
            "embedding.weight": np.random.normal(0, 0.02, (50000, 768)),  # Vocab embedding
            "transformer.layer.0.attention.weight": np.random.normal(0, 0.02, (768, 768)),
            "transformer.layer.0.feed_forward.weight": np.random.normal(0, 0.02, (768, 3072)),
            "output_projection.weight": np.random.normal(0, 0.02, (768, 50000))
        }
        
        global_model = FederatedModel(
            model_name=self.base_model,
            version=1,
            parameters=model_params,
            training_rounds=0,
            total_participants=0,
            last_updated=time.time(),
            performance_metrics={"accuracy": 0.0, "loss": float('inf')}
        )
        
        self.global_models[self.base_model] = global_model
    
    def register_node(self, node: HospitalNode) -> bool:
        """Register a hospital node for federated learning."""
        
        if node.id in self.registered_nodes:
            print(f"Node {node.id} already registered, updating info")
        
        self.registered_nodes[node.id] = node
        print(f"Registered node {node.id} with {node.data_size} samples")
        
        return True
    
    async def train_federated(
        self,
        hospital_nodes: List[HospitalNode],
        validation_strategy: str = "cross_hospital",
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """Execute federated training across hospital nodes."""
        
        # Register all nodes
        for node in hospital_nodes:
            self.register_node(node)
        
        training_start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting federated training with {len(hospital_nodes)} nodes")
        print(f"Target rounds: {self.max_rounds}, Clients per round: {self.clients_per_round}")
        
        for round_num in range(self.max_rounds):
            print(f"\n=== Federated Learning Round {round_num + 1}/{self.max_rounds} ===")
            
            # Start training round
            round_result = await self._execute_training_round(
                round_num + 1,
                validation_strategy
            )
            
            if not round_result["success"]:
                print(f"Round {round_num + 1} failed: {round_result['error']}")
                continue
            
            # Update global model
            current_model = self.global_models[self.base_model]
            current_model.parameters = round_result["aggregated_model"]
            current_model.version += 1
            current_model.training_rounds += 1
            current_model.last_updated = time.time()
            
            # Update performance metrics
            round_loss = round_result["metrics"]["loss"]
            current_model.performance_metrics["loss"] = round_loss
            current_model.performance_metrics["accuracy"] = round_result["metrics"]["accuracy"]
            
            # Record round metrics
            self.round_metrics["loss"].append(round_loss)
            self.round_metrics["accuracy"].append(round_result["metrics"]["accuracy"])
            self.round_metrics["participants"].append(round_result["participants"])
            self.round_metrics["privacy_spent"].append(round_result["total_privacy_spent"])
            
            # Early stopping check
            if round_loss < best_loss:
                best_loss = round_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Round {round_num + 1} completed - Loss: {round_loss:.4f}, "
                  f"Accuracy: {round_result['metrics']['accuracy']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {round_num + 1} rounds")
                break
        
        training_end_time = time.time()
        
        # Compile training history
        history = {
            "total_rounds": len(self.round_metrics["loss"]),
            "total_training_time": training_end_time - training_start_time,
            "final_loss": self.round_metrics["loss"][-1] if self.round_metrics["loss"] else float('inf'),
            "final_accuracy": self.round_metrics["accuracy"][-1] if self.round_metrics["accuracy"] else 0.0,
            "best_loss": best_loss,
            "total_participants": len(self.registered_nodes),
            "average_participants_per_round": np.mean(self.round_metrics["participants"]) if self.round_metrics["participants"] else 0,
            "total_privacy_spent": sum(self.round_metrics["privacy_spent"]),
            "metrics_history": dict(self.round_metrics)
        }
        
        print(f"\nFederated training completed!")
        print(f"Final model performance - Loss: {history['final_loss']:.4f}, Accuracy: {history['final_accuracy']:.4f}")
        
        return history
    
    async def _execute_training_round(
        self,
        round_num: int,
        validation_strategy: str
    ) -> Dict[str, Any]:
        """Execute a single federated training round."""
        
        round_id = f"round_{round_num}_{int(time.time())}"
        
        # Select participants for this round
        available_nodes = [
            node for node in self.registered_nodes.values()
            if node.is_active
        ]
        
        if len(available_nodes) < self.clients_per_round:
            participants = available_nodes
        else:
            # Randomly select participants (in practice, might use more sophisticated selection)
            import random
            participants = random.sample(available_nodes, self.clients_per_round)
        
        participant_ids = [node.id for node in participants]
        
        print(f"Selected {len(participants)} participants: {participant_ids}")
        
        # Create training round
        current_model = self.global_models[self.base_model]
        training_round = TrainingRound(
            round_id=round_id,
            participants=participant_ids,
            global_model=current_model.parameters.copy(),
            submitted_updates={},
            aggregated_model=None,
            min_participants=min(3, len(participants))
        )
        
        self.active_rounds[round_id] = training_round
        
        try:
            # Simulate local training and collect updates
            model_updates = await self._collect_model_updates(
                training_round,
                participants
            )
            
            if len(model_updates) < training_round.min_participants:
                return {
                    "success": False,
                    "error": f"Insufficient participants: {len(model_updates)}/{training_round.min_participants}"
                }
            
            # Aggregate model updates
            aggregated_model = self.aggregator.aggregate_updates(model_updates)
            training_round.aggregated_model = aggregated_model
            training_round.status = "completed"
            
            # Calculate round metrics
            metrics = self._calculate_round_metrics(
                model_updates,
                training_round.global_model,
                aggregated_model
            )
            
            # Calculate total privacy spent
            total_privacy_spent = sum(update.privacy_spent for update in model_updates)
            
            return {
                "success": True,
                "round_id": round_id,
                "aggregated_model": aggregated_model,
                "participants": len(model_updates),
                "metrics": metrics,
                "total_privacy_spent": total_privacy_spent
            }
            
        except Exception as e:
            training_round.status = "failed"
            return {
                "success": False,
                "error": str(e),
                "round_id": round_id
            }
        
        finally:
            # Clean up completed round after some time
            asyncio.create_task(self._cleanup_round(round_id, delay=3600))
    
    async def _collect_model_updates(
        self,
        training_round: TrainingRound,
        participants: List[HospitalNode]
    ) -> List[ModelUpdate]:
        """Collect model updates from participating nodes."""
        
        model_updates = []
        
        # Simulate parallel local training
        tasks = []
        for node in participants:
            task = self._simulate_node_training(node, training_round.global_model)
            tasks.append(task)
        
        # Wait for all training to complete
        updates = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful updates
        for update in updates:
            if isinstance(update, ModelUpdate):
                model_updates.append(update)
                training_round.submitted_updates[update.node_id] = update
            else:
                print(f"Node training failed: {update}")
        
        return model_updates
    
    async def _simulate_node_training(
        self,
        node: HospitalNode,
        global_model: Dict[str, np.ndarray]
    ) -> ModelUpdate:
        """Simulate local training on a node."""
        
        # Simulate training time based on node capacity and data size
        base_time = 2.0  # Base training time
        capacity_factor = 1.0 if "A100" in node.compute_capacity else 1.5
        data_factor = min(2.0, node.data_size / 10000)  # Scale with data size
        
        training_time = base_time * capacity_factor * data_factor
        await asyncio.sleep(min(training_time, 10.0))  # Cap at 10 seconds for simulation
        
        # Create simulated model updates
        updated_params = {}
        for param_name, param_value in global_model.items():
            # Simulate gradient descent updates
            gradient = np.random.normal(0, 0.01, param_value.shape)
            
            # Add differential privacy noise
            dp_gradient = self.privacy_accountant.add_noise_to_query(
                gradient,
                sensitivity=1.0,
                epsilon=self.dp_config.epsilon_per_query
            )
            
            # Apply update with learning rate
            learning_rate = 0.01
            updated_params[param_name] = param_value - learning_rate * dp_gradient
        
        # Create model update
        model_update = ModelUpdate(
            node_id=node.id,
            model_name=self.base_model,
            parameters=updated_params,
            metadata={
                "training_time": training_time,
                "local_epochs": self.local_epochs,
                "data_samples": node.data_size,
                "compute_capacity": node.compute_capacity
            },
            privacy_spent=self.dp_config.epsilon_per_query,
            training_samples=min(1000, node.data_size),  # Cap for privacy
            update_round=len(self.training_history) + 1,
            timestamp=time.time()
        )
        
        return model_update
    
    def _calculate_round_metrics(
        self,
        model_updates: List[ModelUpdate],
        global_model: Dict[str, np.ndarray],
        aggregated_model: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate performance metrics for the training round."""
        
        # Simulate metrics calculation (in practice, would evaluate on validation set)
        
        # Calculate parameter change magnitude
        total_change = 0.0
        total_params = 0
        
        for param_name in global_model.keys():
            old_param = global_model[param_name]
            new_param = aggregated_model[param_name]
            
            change = np.linalg.norm(new_param - old_param)
            total_change += change
            total_params += old_param.size
        
        avg_change = total_change / len(global_model)
        
        # Simulate loss and accuracy (in practice, would evaluate model)
        base_loss = 2.5
        loss_improvement = min(0.1, avg_change * 0.01)
        simulated_loss = base_loss - (len(self.training_history) * 0.01) - loss_improvement
        simulated_loss = max(0.1, simulated_loss + np.random.normal(0, 0.05))
        
        simulated_accuracy = min(0.95, 0.3 + (len(self.training_history) * 0.005) + np.random.normal(0, 0.02))
        simulated_accuracy = max(0.0, simulated_accuracy)
        
        return {
            "loss": simulated_loss,
            "accuracy": simulated_accuracy,
            "parameter_change": avg_change,
            "participating_nodes": len(model_updates),
            "total_samples": sum(update.training_samples for update in model_updates)
        }
    
    async def _cleanup_round(self, round_id: str, delay: float = 3600):
        """Clean up completed training round after delay."""
        await asyncio.sleep(delay)
        if round_id in self.active_rounds:
            del self.active_rounds[round_id]
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        include_privacy_metadata: bool = True
    ):
        """Save federated model checkpoint."""
        
        current_model = self.global_models[self.base_model]
        
        checkpoint_data = {
            "model_name": current_model.model_name,
            "version": current_model.version,
            "training_rounds": current_model.training_rounds,
            "performance_metrics": current_model.performance_metrics,
            "last_updated": current_model.last_updated,
            "parameters": {},  # Will be saved separately for large models
            "training_history": self.training_history[-100:],  # Last 100 rounds
            "round_metrics": {k: v[-100:] for k, v in self.round_metrics.items()}
        }
        
        if include_privacy_metadata:
            total_privacy_spent, total_delta = self.privacy_accountant.get_privacy_spent_total()
            checkpoint_data["privacy_metadata"] = {
                "total_epsilon_spent": total_privacy_spent,
                "total_delta": total_delta,
                "dp_config": {
                    "epsilon_per_query": self.dp_config.epsilon_per_query,
                    "delta": self.dp_config.delta,
                    "mechanism": self.dp_config.mechanism.value,
                    "noise_multiplier": self.dp_config.noise_multiplier
                }
            }
        
        # Save checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        # Save model parameters separately (due to size)
        import pickle
        params_path = checkpoint_path.replace('.json', '_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(current_model.parameters, f)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current federated training status."""
        
        current_model = self.global_models.get(self.base_model)
        
        return {
            "model_info": {
                "name": current_model.model_name if current_model else "none",
                "version": current_model.version if current_model else 0,
                "training_rounds": current_model.training_rounds if current_model else 0,
                "last_updated": current_model.last_updated if current_model else 0
            },
            "registered_nodes": len(self.registered_nodes),
            "active_rounds": len(self.active_rounds),
            "total_training_history": len(self.training_history),
            "current_performance": current_model.performance_metrics if current_model else {},
            "privacy_budget_status": {
                "total_spent": sum(self.round_metrics["privacy_spent"]) if self.round_metrics["privacy_spent"] else 0,
                "epsilon_per_round": self.dp_config.epsilon_per_query,
                "delta": self.dp_config.delta
            }
        }