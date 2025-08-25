"""
Quantum Entanglement-Based Resource Optimizer

Implements quantum entanglement principles for correlated optimization
of resource allocation across federated nodes, maintaining coherent
states between related tasks and resources.
"""

import asyncio
import time
from .numpy_fallback import get_numpy_backend

HAS_NUMPY, np = get_numpy_backend()
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class EntanglementType(Enum):
    """Types of quantum entanglement in resource optimization."""
    TEMPORAL = "temporal"           # Time-correlated tasks
    SPATIAL = "spatial"             # Location/node-correlated tasks  
    RESOURCE = "resource"           # Resource-sharing tasks
    PRIORITY = "priority"           # Priority-level correlated tasks
    USER = "user"                   # User-session correlated tasks
    MEDICAL = "medical"             # Medical-specialty correlated tasks


@dataclass
class ResourceEntanglement:
    """
    Represents quantum entanglement between resources across federated nodes.
    """
    entanglement_id: str
    resource_pairs: List[Tuple[str, str]]  # (node_id, resource_type) pairs
    entanglement_strength: float  # 0.0 to 1.0
    entanglement_type: EntanglementType
    creation_time: float
    
    # Quantum state information
    bell_state: str = "phi_plus"  # phi_plus, phi_minus, psi_plus, psi_minus
    correlation_matrix: List = field(default_factory=lambda: np.array([]))
    shared_measurements: Dict[str, float] = field(default_factory=dict)
    
    # Decoherence properties
    decoherence_rate: float = 0.01
    coherence_time: float = 600.0  # 10 minutes
    last_measurement: float = field(default_factory=time.time)
    
    # Optimization parameters
    min_correlation_threshold: float = 0.6
    max_entangled_resources: int = 8
    measurement_history: List[Tuple[float, Dict[str, float]]] = field(default_factory=list)


@dataclass
class EntangledResourceState:
    """State of an entangled resource pair."""
    resource_id: str
    node_id: str
    current_allocation: float
    entangled_with: Set[str]
    correlation_strengths: Dict[str, float]
    
    # Quantum properties
    spin_state: complex = complex(1.0, 0.0)  # |0⟩ or |1⟩ or superposition
    phase_relationship: float = 0.0
    measurement_basis: str = "computational"  # computational, hadamard, etc.
    
    # Resource-specific properties
    capacity_limit: float = 100.0
    utilization_rate: float = 0.0
    quality_of_service: float = 1.0
    last_correlation_update: float = field(default_factory=time.time)


class EntanglementOptimizer:
    """
    Quantum entanglement-based optimizer for federated resource management.
    
    Uses quantum entanglement principles to:
    - Correlate resource allocations across nodes
    - Maintain coherent optimization states
    - Enable non-local optimization effects
    - Implement Bell inequality-based resource sharing
    """
    
    def __init__(self, 
                 max_entanglement_distance: float = 1000.0,  # km
                 bell_inequality_threshold: float = 2.0,
                 decoherence_mitigation_enabled: bool = True):
        self.max_entanglement_distance = max_entanglement_distance
        self.bell_inequality_threshold = bell_inequality_threshold
        self.decoherence_mitigation_enabled = decoherence_mitigation_enabled
        
        # Entanglement registry
        self.resource_entanglements: Dict[str, ResourceEntanglement] = {}
        self.entangled_resource_states: Dict[str, EntangledResourceState] = {}
        self.entanglement_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Bell state templates
        self.bell_states = {
            "phi_plus": (1/math.sqrt(2)) * np.array([1, 0, 0, 1]),    # (|00⟩ + |11⟩)/√2
            "phi_minus": (1/math.sqrt(2)) * np.array([1, 0, 0, -1]),   # (|00⟩ - |11⟩)/√2  
            "psi_plus": (1/math.sqrt(2)) * np.array([0, 1, 1, 0]),     # (|01⟩ + |10⟩)/√2
            "psi_minus": (1/math.sqrt(2)) * np.array([0, 1, -1, 0])    # (|01⟩ - |10⟩)/√2
        }
        
        # Optimization state
        self.global_entanglement_strength = 0.0
        self.total_entangled_pairs = 0
        self.correlation_violations = 0  # Bell inequality violations
        
        # Performance tracking
        self.optimization_metrics = {
            "entanglement_creation_events": 0,
            "correlation_measurements": 0,
            "bell_inequality_violations": 0,
            "resource_utilization_improvement": 0.0,
            "decoherence_events": 0,
            "average_correlation_strength": 0.0
        }
        
        # History and monitoring
        self.entanglement_history: List[Dict[str, Any]] = []
        self.correlation_matrix_snapshots: List[Tuple[float, List]] = []
        
    async def create_resource_entanglement(self,
                                         resource_pairs: List[Tuple[str, str]],
                                         entanglement_type: EntanglementType,
                                         target_correlation: float = 0.8) -> str:
        """Create quantum entanglement between resource pairs."""
        
        if len(resource_pairs) < 2:
            raise ValueError("Need at least 2 resources for entanglement")
        
        entanglement_id = f"ent_{int(time.time())}_{len(self.resource_entanglements)}"
        
        # Determine optimal Bell state based on entanglement type and correlation target
        bell_state = self._select_optimal_bell_state(entanglement_type, target_correlation)
        
        # Calculate initial correlation matrix
        n_resources = len(resource_pairs)
        correlation_matrix = np.eye(n_resources) * target_correlation
        
        # Add off-diagonal correlations based on entanglement type
        for i in range(n_resources):
            for j in range(i + 1, n_resources):
                # Distance-based correlation decay
                distance_factor = self._calculate_resource_distance(resource_pairs[i], resource_pairs[j])
                correlation_strength = target_correlation * math.exp(-distance_factor / 100.0)
                
                correlation_matrix[i, j] = correlation_strength
                correlation_matrix[j, i] = correlation_strength
        
        # Create entanglement object
        entanglement = ResourceEntanglement(
            entanglement_id=entanglement_id,
            resource_pairs=resource_pairs,
            entanglement_strength=target_correlation,
            entanglement_type=entanglement_type,
            creation_time=time.time(),
            bell_state=bell_state,
            correlation_matrix=correlation_matrix
        )
        
        # Register entanglement
        self.resource_entanglements[entanglement_id] = entanglement
        
        # Create entangled resource states
        for node_id, resource_type in resource_pairs:
            resource_id = f"{node_id}_{resource_type}"
            
            if resource_id not in self.entangled_resource_states:
                self.entangled_resource_states[resource_id] = EntangledResourceState(
                    resource_id=resource_id,
                    node_id=node_id,
                    current_allocation=0.0,
                    entangled_with=set(),
                    correlation_strengths={}
                )
            
            # Add entanglement relationships
            state = self.entangled_resource_states[resource_id]
            other_resources = [f"{n}_{r}" for n, r in resource_pairs if f"{n}_{r}" != resource_id]
            
            for other_resource in other_resources:
                state.entangled_with.add(other_resource)
                state.correlation_strengths[other_resource] = target_correlation
                
                # Update entanglement graph
                self.entanglement_graph[resource_id].add(other_resource)
                self.entanglement_graph[other_resource].add(resource_id)
        
        # Initialize Bell state for the entangled system
        await self._initialize_bell_state(entanglement_id)
        
        # Update global metrics
        self.total_entangled_pairs += len(resource_pairs)
        self.optimization_metrics["entanglement_creation_events"] += 1
        self._update_global_entanglement_strength()
        
        logger.info(f"Created {entanglement_type.value} entanglement {entanglement_id} "
                   f"between {len(resource_pairs)} resources with strength {target_correlation:.3f}")
        
        return entanglement_id
    
    def _select_optimal_bell_state(self, entanglement_type: EntanglementType, target_correlation: float) -> str:
        """Select optimal Bell state based on entanglement characteristics."""
        
        if entanglement_type in [EntanglementType.TEMPORAL, EntanglementType.PRIORITY]:
            # Time and priority correlations benefit from phi_plus (maximum correlation)
            return "phi_plus"
        elif entanglement_type == EntanglementType.RESOURCE:
            # Resource sharing might benefit from anti-correlation
            return "phi_minus" if target_correlation < 0 else "phi_plus"
        elif entanglement_type == EntanglementType.SPATIAL:
            # Spatial correlations use psi states for distributed optimization
            return "psi_plus"
        else:
            # Default to phi_plus for maximum correlation
            return "phi_plus"
    
    def _calculate_resource_distance(self, resource1: Tuple[str, str], resource2: Tuple[str, str]) -> float:
        """Calculate logical distance between resources for correlation calculation."""
        node1, type1 = resource1
        node2, type2 = resource2
        
        # Same node resources have zero distance
        if node1 == node2:
            return 0.0 if type1 == type2 else 1.0
        
        # Different nodes - use hash-based distance approximation
        node_distance = abs(hash(node1) - hash(node2)) % 1000
        type_distance = 0.0 if type1 == type2 else 10.0
        
        return node_distance + type_distance
    
    async def _initialize_bell_state(self, entanglement_id: str):
        """Initialize quantum Bell state for entangled resources."""
        
        entanglement = self.resource_entanglements[entanglement_id]
        bell_state_vector = self.bell_states[entanglement.bell_state]
        
        # Set initial quantum states for entangled resources
        for i, (node_id, resource_type) in enumerate(entanglement.resource_pairs):
            resource_id = f"{node_id}_{resource_type}"
            state = self.entangled_resource_states.get(resource_id)
            
            if state:
                # Set spin state based on Bell state
                if i % 2 == 0:  # Even index resources
                    state.spin_state = complex(bell_state_vector[0], bell_state_vector[2])
                else:  # Odd index resources  
                    state.spin_state = complex(bell_state_vector[1], bell_state_vector[3])
                
                # Set phase relationship
                state.phase_relationship = np.angle(state.spin_state)
    
    async def measure_entangled_correlations(self, entanglement_id: str) -> Dict[str, float]:
        """
        Perform quantum measurement on entangled resources to determine
        optimal resource allocations based on correlations.
        """
        
        if entanglement_id not in self.resource_entanglements:
            return {}
        
        entanglement = self.resource_entanglements[entanglement_id]
        measurement_results = {}
        measurement_time = time.time()
        
        # Check Bell inequality before measurement
        bell_violation = await self._check_bell_inequality(entanglement_id)
        
        # Perform correlated measurements on all entangled resources
        resource_measurements = []
        for node_id, resource_type in entanglement.resource_pairs:
            resource_id = f"{node_id}_{resource_type}"
            state = self.entangled_resource_states.get(resource_id)
            
            if state:
                # Quantum measurement collapses superposition
                measurement_value = self._perform_quantum_measurement(state)
                resource_measurements.append((resource_id, measurement_value))
                measurement_results[resource_id] = measurement_value
        
        # Update correlation matrix based on measurements
        await self._update_correlation_matrix(entanglement, resource_measurements)
        
        # Apply correlation-based resource optimization
        optimized_allocations = await self._optimize_correlated_allocations(
            entanglement, resource_measurements
        )
        
        # Store measurement in history
        entanglement.measurement_history.append((measurement_time, measurement_results.copy()))
        entanglement.shared_measurements = measurement_results
        entanglement.last_measurement = measurement_time
        
        # Update global metrics
        self.optimization_metrics["correlation_measurements"] += 1
        if bell_violation:
            self.optimization_metrics["bell_inequality_violations"] += 1
        
        # Record in optimization history
        self.entanglement_history.append({
            "entanglement_id": entanglement_id,
            "measurement_time": measurement_time,
            "bell_violation": bell_violation,
            "resource_measurements": resource_measurements,
            "optimized_allocations": optimized_allocations
        })
        
        logger.debug(f"Measured correlations for entanglement {entanglement_id}: "
                    f"{len(resource_measurements)} resources, "
                    f"Bell violation: {bell_violation}")
        
        return optimized_allocations
    
    def _perform_quantum_measurement(self, resource_state: EntangledResourceState) -> float:
        """Perform quantum measurement on individual resource state."""
        
        # Measurement collapses superposition to eigenstate
        spin_magnitude = abs(resource_state.spin_state)
        spin_phase = resource_state.phase_relationship
        
        # Measurement outcome based on quantum probabilities
        probability_0 = spin_magnitude ** 2
        probability_1 = 1.0 - probability_0
        
        # Quantum measurement outcome
        if np.random.random() < probability_0:
            # Measured in |0⟩ state - low resource allocation
            measurement_outcome = 0.2 + 0.3 * np.random.random()
        else:
            # Measured in |1⟩ state - high resource allocation  
            measurement_outcome = 0.7 + 0.3 * np.random.random()
        
        # Apply phase-based modulation
        phase_factor = 1.0 + 0.2 * math.cos(spin_phase)
        measurement_outcome *= phase_factor
        
        # Update resource state after measurement (collapse)
        if measurement_outcome < 0.5:
            resource_state.spin_state = complex(1.0, 0.0)  # |0⟩
        else:
            resource_state.spin_state = complex(0.0, 1.0)  # |1⟩
        
        resource_state.current_allocation = min(1.0, measurement_outcome)
        return resource_state.current_allocation
    
    async def _check_bell_inequality(self, entanglement_id: str) -> bool:
        """Check for Bell inequality violations (quantum correlations)."""
        
        entanglement = self.resource_entanglements[entanglement_id]
        
        if len(entanglement.resource_pairs) < 2:
            return False
        
        # CHSH Bell inequality: |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2
        # For quantum systems, this can be violated up to 2√2 ≈ 2.828
        
        # Calculate correlation expectations for different measurement bases
        correlations = []
        
        for i in range(len(entanglement.resource_pairs)):
            for j in range(i + 1, len(entanglement.resource_pairs)):
                resource1_id = f"{entanglement.resource_pairs[i][0]}_{entanglement.resource_pairs[i][1]}"
                resource2_id = f"{entanglement.resource_pairs[j][0]}_{entanglement.resource_pairs[j][1]}"
                
                state1 = self.entangled_resource_states.get(resource1_id)
                state2 = self.entangled_resource_states.get(resource2_id)
                
                if state1 and state2:
                    # Calculate correlation expectation
                    correlation = self._calculate_expectation_value(state1, state2)
                    correlations.append(correlation)
        
        if len(correlations) >= 4:
            # CHSH inequality test
            chsh_value = abs(correlations[0] - correlations[1] + correlations[2] + correlations[3])
            return chsh_value > self.bell_inequality_threshold
        
        return False
    
    def _calculate_expectation_value(self, state1: EntangledResourceState, state2: EntangledResourceState) -> float:
        """Calculate correlation expectation value between two entangled resources."""
        
        # Quantum correlation based on spin states
        spin1_magnitude = abs(state1.spin_state)
        spin2_magnitude = abs(state2.spin_state)
        
        phase_difference = state1.phase_relationship - state2.phase_relationship
        
        # Correlation expectation: ⟨σ₁σ₂⟩ = cos(θ) for Bell states
        correlation = spin1_magnitude * spin2_magnitude * math.cos(phase_difference)
        
        return correlation
    
    async def _update_correlation_matrix(self, 
                                       entanglement: ResourceEntanglement,
                                       measurements: List[Tuple[str, float]]):
        """Update correlation matrix based on measurement outcomes."""
        
        n_resources = len(measurements)
        if n_resources < 2:
            return
        
        # Calculate empirical correlations
        measurement_values = [value for _, value in measurements]
        
        for i in range(n_resources):
            for j in range(i + 1, n_resources):
                # Pearson correlation coefficient
                val1, val2 = measurement_values[i], measurement_values[j]
                
                # Update correlation with exponential moving average
                current_correlation = entanglement.correlation_matrix[i, j]
                alpha = 0.3  # Learning rate
                
                # Simple correlation update (in practice, would use full covariance)
                new_correlation = val1 * val2 * entanglement.entanglement_strength
                updated_correlation = alpha * new_correlation + (1 - alpha) * current_correlation
                
                entanglement.correlation_matrix[i, j] = updated_correlation
                entanglement.correlation_matrix[j, i] = updated_correlation
        
        # Store correlation matrix snapshot
        self.correlation_matrix_snapshots.append((time.time(), entanglement.correlation_matrix.copy()))
        
        # Keep limited history
        if len(self.correlation_matrix_snapshots) > 100:
            self.correlation_matrix_snapshots = self.correlation_matrix_snapshots[-50:]
    
    async def _optimize_correlated_allocations(self,
                                             entanglement: ResourceEntanglement,
                                             measurements: List[Tuple[str, float]]) -> Dict[str, float]:
        """Optimize resource allocations based on entangled correlations."""
        
        optimized_allocations = {}
        
        # Use correlation matrix for optimization
        correlation_matrix = entanglement.correlation_matrix
        measurement_values = np.array([value for _, value in measurements])
        
        # Apply correlation-based adjustment
        if correlation_matrix.size > 0 and len(measurement_values) > 0:
            # Matrix multiplication to apply correlations
            n_resources = min(len(measurements), correlation_matrix.shape[0])
            correlated_values = np.dot(correlation_matrix[:n_resources, :n_resources], 
                                     measurement_values[:n_resources])
            
            # Normalize to valid allocation range [0, 1]
            if np.max(correlated_values) > 0:
                correlated_values = correlated_values / np.max(correlated_values)
            
            # Apply entanglement-based optimization
            for i, (resource_id, original_value) in enumerate(measurements[:n_resources]):
                # Blend original measurement with correlation-optimized value
                correlation_weight = entanglement.entanglement_strength
                optimized_value = (
                    correlation_weight * correlated_values[i] + 
                    (1 - correlation_weight) * original_value
                )
                
                optimized_allocations[resource_id] = min(1.0, max(0.0, optimized_value))
        else:
            # Fallback to original measurements
            optimized_allocations = {resource_id: value for resource_id, value in measurements}
        
        return optimized_allocations
    
    async def evolve_entangled_states(self, time_step: float = 1.0):
        """Evolve entangled quantum states over time."""
        
        current_time = time.time()
        
        for entanglement_id, entanglement in self.resource_entanglements.items():
            # Check for decoherence
            time_since_creation = current_time - entanglement.creation_time
            
            if time_since_creation > entanglement.coherence_time:
                await self._handle_entanglement_decoherence(entanglement_id)
                continue
            
            # Apply time evolution to entangled states
            for node_id, resource_type in entanglement.resource_pairs:
                resource_id = f"{node_id}_{resource_type}"
                state = self.entangled_resource_states.get(resource_id)
                
                if state:
                    # Quantum time evolution: |ψ(t)⟩ = e^(-iHt/ℏ)|ψ(0)⟩
                    hamiltonian_factor = 0.1 * time_step  # Simplified Hamiltonian
                    phase_evolution = np.exp(-1j * hamiltonian_factor)
                    
                    state.spin_state *= phase_evolution
                    state.phase_relationship = np.angle(state.spin_state)
                    state.last_correlation_update = current_time
            
            # Apply gradual decoherence
            if self.decoherence_mitigation_enabled:
                decoherence_factor = math.exp(-entanglement.decoherence_rate * time_step)
                entanglement.entanglement_strength *= decoherence_factor
                
                # Update correlation matrix
                entanglement.correlation_matrix *= decoherence_factor
    
    async def _handle_entanglement_decoherence(self, entanglement_id: str):
        """Handle decoherence of quantum entanglement."""
        
        entanglement = self.resource_entanglements.get(entanglement_id)
        if not entanglement:
            return
        
        # Break entanglement relationships
        for node_id, resource_type in entanglement.resource_pairs:
            resource_id = f"{node_id}_{resource_type}"
            state = self.entangled_resource_states.get(resource_id)
            
            if state:
                # Remove entanglement relationships
                other_resources = state.entangled_with.copy()
                for other_resource in other_resources:
                    state.entangled_with.discard(other_resource)
                    state.correlation_strengths.pop(other_resource, None)
                    
                    # Update graph
                    self.entanglement_graph[resource_id].discard(other_resource)
                    self.entanglement_graph[other_resource].discard(resource_id)
                
                # Collapse to mixed state
                state.spin_state = complex(0.5, 0.5)  # Maximally mixed state
        
        # Remove entanglement
        del self.resource_entanglements[entanglement_id]
        self.optimization_metrics["decoherence_events"] += 1
        
        logger.warning(f"Entanglement {entanglement_id} has decoherent")
    
    def _update_global_entanglement_strength(self):
        """Update global entanglement strength metric."""
        
        if not self.resource_entanglements:
            self.global_entanglement_strength = 0.0
            return
        
        total_strength = sum(ent.entanglement_strength for ent in self.resource_entanglements.values())
        self.global_entanglement_strength = total_strength / len(self.resource_entanglements)
        
        # Update average correlation strength
        all_correlations = []
        for entanglement in self.resource_entanglements.values():
            if entanglement.correlation_matrix.size > 0:
                # Get off-diagonal elements (actual correlations)
                n = entanglement.correlation_matrix.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        all_correlations.append(abs(entanglement.correlation_matrix[i, j]))
        
        if all_correlations:
            self.optimization_metrics["average_correlation_strength"] = np.mean(all_correlations)
    
    def get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive entanglement optimization statistics."""
        
        active_entanglements = len(self.resource_entanglements)
        total_entangled_resources = len(self.entangled_resource_states)
        
        # Calculate network properties
        max_entanglement_depth = 0
        if self.entanglement_graph:
            for resource_id, connected_resources in self.entanglement_graph.items():
                max_entanglement_depth = max(max_entanglement_depth, len(connected_resources))
        
        # Bell inequality violation rate
        total_measurements = self.optimization_metrics["correlation_measurements"]
        bell_violations = self.optimization_metrics["bell_inequality_violations"]
        violation_rate = bell_violations / max(total_measurements, 1)
        
        # Recent correlation trends
        recent_correlations = []
        if len(self.correlation_matrix_snapshots) >= 2:
            latest_matrix = self.correlation_matrix_snapshots[-1][1]
            if latest_matrix.size > 0:
                recent_correlations = [abs(latest_matrix[i, j]) 
                                     for i in range(latest_matrix.shape[0])
                                     for j in range(i + 1, latest_matrix.shape[1])]
        
        return {
            "active_entanglements": active_entanglements,
            "total_entangled_resources": total_entangled_resources,
            "global_entanglement_strength": self.global_entanglement_strength,
            "max_entanglement_depth": max_entanglement_depth,
            "bell_inequality_violation_rate": violation_rate,
            "recent_correlation_distribution": {
                "mean": np.mean(recent_correlations) if recent_correlations else 0.0,
                "std": np.std(recent_correlations) if recent_correlations else 0.0,
                "max": np.max(recent_correlations) if recent_correlations else 0.0
            },
            "optimization_metrics": self.optimization_metrics,
            "entanglement_types_distribution": self._get_entanglement_type_distribution(),
            "decoherence_rate": np.mean([ent.decoherence_rate for ent in self.resource_entanglements.values()]) if self.resource_entanglements else 0.0
        }
    
    def _get_entanglement_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entanglement types."""
        
        type_counts = defaultdict(int)
        for entanglement in self.resource_entanglements.values():
            type_counts[entanglement.entanglement_type.value] += 1
        
        return dict(type_counts)
    
    async def suggest_new_entanglements(self, 
                                      resource_usage_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Suggest new entanglements based on resource usage patterns."""
        
        suggestions = []
        
        # Analyze resource usage patterns to find correlation opportunities
        resource_pairs = []
        for node1, resources1 in resource_usage_data.items():
            for resource_type1, usage1 in resources1.items():
                for node2, resources2 in resource_usage_data.items():
                    if node1 >= node2:  # Avoid duplicates
                        continue
                    for resource_type2, usage2 in resources2.items():
                        if resource_type1 == resource_type2:  # Same resource type
                            correlation = abs(usage1 - usage2) / max(usage1 + usage2, 1.0)
                            if correlation > 0.6:  # High similarity suggests entanglement potential
                                resource_pairs.append({
                                    "resources": [(node1, resource_type1), (node2, resource_type2)],
                                    "correlation_potential": 1.0 - correlation,
                                    "suggested_type": EntanglementType.RESOURCE,
                                    "expected_benefit": correlation * 0.3  # Estimated efficiency gain
                                })
        
        # Sort by potential benefit
        resource_pairs.sort(key=lambda x: x["expected_benefit"], reverse=True)
        
        # Return top suggestions
        return resource_pairs[:5]