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

# Conditional imports for model sharding (requires torch)
try:
    from ..core.model_sharding import ModelSharder, ShardingStrategy
    MODEL_SHARDING_AVAILABLE = True
except ImportError:
    ModelSharder = None
    ShardingStrategy = None
    MODEL_SHARDING_AVAILABLE = False

from ..quantum_planning.quantum_planner import QuantumTaskPlanner, QuantumTask, TaskPriority
from ..quantum_planning.superposition_scheduler import SuperpositionScheduler, TaskSuperposition
from ..quantum_planning.entanglement_optimizer import EntanglementOptimizer, EntanglementType
from ..quantum_planning.interference_balancer import InterferenceBalancer, InterferenceType
from ..quantum_planning.quantum_validators import QuantumComponentValidator, QuantumErrorHandler, ValidationLevel
from ..quantum_planning.quantum_monitor import QuantumMonitor, AlertSeverity, HealthStatus
from ..quantum_planning.quantum_security import QuantumSecurityController, SecurityLevel, QuantumSecurityContext


class RoutingStrategy(Enum):
    """Routing strategies for federated inference."""
    ROUND_ROBIN = "round_robin"
    PRIVACY_AWARE = "privacy_aware"
    LOAD_BALANCED = "load_balanced"
    CONSENSUS_REQUIRED = "consensus_required"
    QUANTUM_OPTIMIZED = "quantum_optimized"
    SUPERPOSITION_BASED = "superposition_based"
    ENTANGLEMENT_AWARE = "entanglement_aware"
    INTERFERENCE_BALANCED = "interference_balanced"


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
        routing_strategy: RoutingStrategy = RoutingStrategy.QUANTUM_OPTIMIZED
    ):
        self.model_name = model_name
        self.num_shards = num_shards
        self.aggregation_protocol = aggregation_protocol
        self.encryption = encryption
        self.routing_strategy = routing_strategy
        
        # Initialize components
        self.nodes: Dict[str, NodeCapability] = {}
        if MODEL_SHARDING_AVAILABLE:
            from ..core.model_sharding import ShardingConfig, ShardingStrategy
            shard_config = ShardingConfig(
                strategy=ShardingStrategy.LAYER_WISE,
                num_shards=num_shards,
                overlap_layers=0,
                load_balancing=True,
                privacy_constraints=True
            )
            self.model_sharder = ModelSharder(shard_config)
        else:
            self.model_sharder = None
        self.load_tracker = NodeLoadTracker()
        self.privacy_accountant = PrivacyAccountant(DPConfig())
        
        # Initialize quantum planning components
        self.quantum_planner = QuantumTaskPlanner(
            privacy_accountant=self.privacy_accountant,
            coherence_threshold=0.8,
            max_entangled_tasks=5
        )
        self.superposition_scheduler = SuperpositionScheduler(
            max_superposition_time=300.0,
            interference_strength=0.5,
            decoherence_rate=0.01
        )
        self.entanglement_optimizer = EntanglementOptimizer(
            max_entanglement_distance=1000.0,
            bell_inequality_threshold=2.0,
            decoherence_mitigation_enabled=True
        )
        self.interference_balancer = InterferenceBalancer(
            interference_resolution=0.1,
            coherence_threshold=0.7,
            phase_locked_loop_enabled=True
        )
        
        # Request tracking
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.request_history: List[Tuple[str, float, bool]] = []  # (node_id, timestamp, success)
        
        # Round-robin counter
        self._round_robin_index = 0
        
        # Quantum planning state
        self.quantum_task_counter = 0
    
    async def register_nodes(self, hospital_nodes: List['HospitalNode']):
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
            
            # Register with quantum planning components
            node_characteristics = {
                'current_load': 0.0,
                'privacy_budget': capability.privacy_budget_remaining,
                'compute_capacity': {
                    'gpu_memory': capability.gpu_memory,
                    'cpu_cores': capability.cpu_cores,
                    'network_bandwidth': capability.network_bandwidth
                },
                'critical_affinity': 1.0,
                'high_affinity': 0.9,
                'medium_affinity': 0.8,
                'low_affinity': 0.7,
                'background_affinity': 0.5,
                'processing_frequency': 1.0,
                'load_capacity': 1.0,
                'network_latency': 0.1
            }
            
            # Register with quantum planner
            self.quantum_planner.register_node(node.id, node_characteristics)
            
            # Initialize node wave state for interference balancer
            await self.interference_balancer.initialize_node_wave_state(node.id, node_characteristics)
            
            # Register with entanglement optimizer
            await self.entanglement_optimizer.create_resource_entanglement(
                resource_pairs=[(node.id, 'compute'), (node.id, 'memory')],
                entanglement_type=EntanglementType.RESOURCE,
                target_correlation=0.8
            )
    
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
        """Route inference request to appropriate node(s) using quantum-inspired optimization."""
        start_time = time.time()
        
        # Check privacy budget
        if not self.privacy_accountant.check_budget(request.user_id, request.max_privacy_budget):
            raise ValueError(f"Insufficient privacy budget for user {request.user_id}")
        
        # Store active request
        self.active_requests[request.request_id] = request
        
        try:
            # Use quantum-enhanced routing based on strategy
            if self.routing_strategy in [
                RoutingStrategy.QUANTUM_OPTIMIZED,
                RoutingStrategy.SUPERPOSITION_BASED,
                RoutingStrategy.ENTANGLEMENT_AWARE,
                RoutingStrategy.INTERFERENCE_BALANCED
            ]:
                response = await self._handle_quantum_enhanced_request(request)
            elif request.require_consensus:
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
    
    async def _handle_quantum_enhanced_request(self, request: InferenceRequest) -> InferenceResponse:
        """Handle request using quantum-inspired optimization."""
        
        # Create quantum task
        self.quantum_task_counter += 1
        quantum_task_id = f"qt_{self.quantum_task_counter}_{request.request_id}"
        
        # Map request priority to quantum priority
        priority_mapping = {
            10: TaskPriority.CRITICAL,   # Emergency
            7: TaskPriority.HIGH,        # Urgent
            5: TaskPriority.MEDIUM,      # Normal
            3: TaskPriority.LOW,         # Routine
            1: TaskPriority.BACKGROUND   # Background
        }
        quantum_priority = priority_mapping.get(request.priority, TaskPriority.MEDIUM)
        
        # Prepare quantum task data
        task_data = {
            'task_id': quantum_task_id,
            'user_id': request.user_id,
            'prompt': request.prompt,
            'priority': quantum_priority.value,
            'privacy_budget': request.max_privacy_budget,
            'estimated_duration': request.timeout,
            'resource_requirements': {
                'compute': 0.5,  # 50% compute requirement
                'memory': 0.3,   # 30% memory requirement
                'network': 0.2   # 20% network requirement
            },
            'department': request.department,
            'medical_specialty': request.department,
            'urgency_score': request.priority / 10.0
        }
        
        # Add task to quantum planner
        await self.quantum_planner.add_task(task_data)
        
        # Handle different quantum routing strategies
        if self.routing_strategy == RoutingStrategy.SUPERPOSITION_BASED:
            return await self._handle_superposition_routing(request, quantum_task_id)
        elif self.routing_strategy == RoutingStrategy.ENTANGLEMENT_AWARE:
            return await self._handle_entanglement_routing(request, quantum_task_id)  
        elif self.routing_strategy == RoutingStrategy.INTERFERENCE_BALANCED:
            return await self._handle_interference_routing(request, quantum_task_id)
        else:  # QUANTUM_OPTIMIZED (default)
            return await self._handle_quantum_optimized_routing(request, quantum_task_id)
    
    async def _handle_superposition_routing(self, request: InferenceRequest, quantum_task_id: str) -> InferenceResponse:
        """Handle routing using quantum superposition principles."""
        
        # Get available nodes
        available_nodes = [node.node_id for node in self.nodes.values() 
                          if node.is_healthy and self.model_name in node.supported_models]
        
        if not available_nodes:
            raise RuntimeError("No suitable nodes available for superposition")
        
        # Create time preferences (immediate execution)
        current_time = time.time()
        time_preferences = [(current_time, current_time + request.timeout)]
        
        # Initialize superposition
        superposition = await self.superposition_scheduler.initialize_superposition(
            task_id=quantum_task_id,
            potential_nodes=available_nodes,
            time_preferences=time_preferences,
            resource_requirements=request.__dict__.get('resource_requirements', {})
        )
        
        # Measure optimal assignment
        assignment = await self.superposition_scheduler.measure_optimal_assignment(
            quantum_task_id, "maximum_probability"
        )
        
        if not assignment:
            raise RuntimeError("Failed to measure quantum assignment")
        
        selected_node_id, assignment_details = assignment
        
        # Execute inference on selected node
        response_text = await self._simulate_inference(selected_node_id, request)
        
        # Spend privacy budget
        privacy_spent = min(request.max_privacy_budget, self.privacy_accountant.config.epsilon_per_query)
        self.privacy_accountant.spend_budget(request.user_id, privacy_spent, "quantum_superposition_inference")
        
        return InferenceResponse(
            request_id=request.request_id,
            text=response_text,
            privacy_cost=privacy_spent,
            remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id),
            processing_nodes=[selected_node_id],
            latency=0.0,  # Will be set by caller
            confidence_score=assignment_details.get('selected_probability', 0.8)
        )
    
    async def _handle_entanglement_routing(self, request: InferenceRequest, quantum_task_id: str) -> InferenceResponse:
        """Handle routing using quantum entanglement optimization."""
        
        # Find entangled resources for optimization
        available_nodes = [node.node_id for node in self.nodes.values() 
                          if node.is_healthy and self.model_name in node.supported_models]
        
        if len(available_nodes) < 2:
            # Fall back to single node if insufficient nodes for entanglement
            return await self._handle_single_node_request(request)
        
        # Create entanglement for related tasks (same user or department)
        entanglement_id = await self.entanglement_optimizer.create_resource_entanglement(
            resource_pairs=[(node_id, 'task_processing') for node_id in available_nodes[:4]],
            entanglement_type=EntanglementType.USER if request.user_id else EntanglementType.TEMPORAL,
            target_correlation=0.7
        )
        
        # Measure entangled correlations
        optimized_allocations = await self.entanglement_optimizer.measure_entangled_correlations(entanglement_id)
        
        # Select node with best optimized allocation
        if optimized_allocations:
            best_node_resource = max(optimized_allocations.items(), key=lambda x: x[1])
            selected_node_id = best_node_resource[0].split('_')[0]  # Extract node_id from resource_id
        else:
            # Fall back to best available node
            selected_node = self._select_best_node(request)
            selected_node_id = selected_node.node_id if selected_node else available_nodes[0]
        
        # Execute inference
        response_text = await self._simulate_inference(selected_node_id, request)
        
        # Spend privacy budget (higher cost for entangled optimization)
        privacy_spent = min(request.max_privacy_budget, self.privacy_accountant.config.epsilon_per_query * 1.2)
        self.privacy_accountant.spend_budget(request.user_id, privacy_spent, "quantum_entanglement_inference")
        
        return InferenceResponse(
            request_id=request.request_id,
            text=response_text,
            privacy_cost=privacy_spent,
            remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id),
            processing_nodes=[selected_node_id],
            latency=0.0,
            confidence_score=0.9  # Higher confidence for entangled optimization
        )
    
    async def _handle_interference_routing(self, request: InferenceRequest, quantum_task_id: str) -> InferenceResponse:
        """Handle routing using quantum interference load balancing."""
        
        # Get current load distribution
        current_loads = {node_id: self.load_tracker.node_loads.get(node_id, 0.0) 
                        for node_id in self.nodes.keys()}
        
        # Define target load (balanced)
        target_load = 0.5  # 50% target utilization
        target_loads = {node_id: target_load for node_id in self.nodes.keys()}
        
        # Create interference pattern for load balancing
        available_nodes = [node.node_id for node in self.nodes.values() 
                          if node.is_healthy and self.model_name in node.supported_models]
        
        if len(available_nodes) >= 2:
            interference_id = await self.interference_balancer.create_task_interference(
                task_ids=[quantum_task_id],
                target_nodes=available_nodes,
                interference_type=InterferenceType.CONSTRUCTIVE
            )
            
            # Optimize load distribution
            optimized_loads = await self.interference_balancer.optimize_load_distribution(
                current_loads, target_loads
            )
            
            # Select node with best interference-optimized load
            best_node_id = min(optimized_loads.items(), key=lambda x: x[1])[0]
        else:
            # Fall back to least loaded node
            best_node_id = min(current_loads.items(), key=lambda x: x[1])[0]
        
        # Execute inference
        response_text = await self._simulate_inference(best_node_id, request)
        
        # Spend privacy budget
        privacy_spent = min(request.max_privacy_budget, self.privacy_accountant.config.epsilon_per_query * 1.1)
        self.privacy_accountant.spend_budget(request.user_id, privacy_spent, "quantum_interference_inference")
        
        return InferenceResponse(
            request_id=request.request_id,
            text=response_text,
            privacy_cost=privacy_spent,
            remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id),
            processing_nodes=[best_node_id],
            latency=0.0,
            confidence_score=0.85
        )
    
    async def _handle_quantum_optimized_routing(self, request: InferenceRequest, quantum_task_id: str) -> InferenceResponse:
        """Handle routing using comprehensive quantum optimization."""
        
        # Generate optimal assignments using quantum planner
        assignments = await self.quantum_planner.plan_optimal_assignments()
        
        # Find assignment for this task
        task_assignment = None
        for assignment in assignments:
            if assignment.get('task_id') == quantum_task_id:
                task_assignment = assignment
                break
        
        if task_assignment:
            selected_node_id = task_assignment['node_id']
        else:
            # Fall back to traditional routing
            selected_node = self._select_best_node(request)
            selected_node_id = selected_node.node_id if selected_node else list(self.nodes.keys())[0]
        
        # Execute inference
        response_text = await self._simulate_inference(selected_node_id, request)
        
        # Spend privacy budget
        privacy_spent = min(request.max_privacy_budget, self.privacy_accountant.config.epsilon_per_query)
        self.privacy_accountant.spend_budget(request.user_id, privacy_spent, "quantum_optimized_inference")
        
        return InferenceResponse(
            request_id=request.request_id,
            text=response_text,
            privacy_cost=privacy_spent,
            remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id),
            processing_nodes=[selected_node_id],
            latency=0.0,
            confidence_score=task_assignment.get('assignment_probability', 0.8) if task_assignment else 0.7
        )
    
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
        """Get comprehensive routing statistics including quantum metrics."""
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
        
        # Gather quantum statistics
        quantum_stats = {}
        try:
            quantum_stats["quantum_planner"] = self.quantum_planner.get_quantum_statistics()
            quantum_stats["superposition_scheduler"] = self.superposition_scheduler.get_superposition_status()
            quantum_stats["entanglement_optimizer"] = self.entanglement_optimizer.get_entanglement_statistics()  
            quantum_stats["interference_balancer"] = self.interference_balancer.get_interference_statistics()
        except Exception as e:
            quantum_stats["error"] = f"Failed to gather quantum stats: {str(e)}"
        
        return {
            "total_requests": total_requests,
            "success_rate": successful_requests / max(total_requests, 1),
            "active_requests": len(self.active_requests),
            "registered_nodes": len(self.nodes),
            "healthy_nodes": sum(1 for node in self.nodes.values() if node.is_healthy),
            "routing_strategy": self.routing_strategy.value,
            "node_statistics": node_stats,
            "quantum_statistics": quantum_stats,
            "quantum_task_counter": self.quantum_task_counter
        }