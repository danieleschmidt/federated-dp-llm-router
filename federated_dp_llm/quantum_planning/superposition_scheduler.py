"""
Superposition-Based Task Scheduler

Implements quantum superposition principles for scheduling tasks across
federated nodes, maintaining multiple potential execution paths until
optimal measurement/assignment occurs.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)


class SchedulingPhase(Enum):
    """Phases of quantum-inspired scheduling process."""
    INITIALIZATION = "initialization"
    SUPERPOSITION = "superposition" 
    INTERFERENCE = "interference"
    MEASUREMENT = "measurement"
    EXECUTION = "execution"


@dataclass
class TaskSuperposition:
    """
    Represents a task existing in superposition across multiple potential
    execution contexts (nodes, times, resource allocations).
    """
    task_id: str
    amplitude_distribution: Dict[str, complex]  # Node -> complex amplitude
    phase_angles: Dict[str, float]  # Phase information for interference
    measurement_probability: Dict[str, float]  # |amplitude|^2
    coherence_lifetime: float = 300.0  # Seconds before decoherence
    interference_susceptibility: float = 0.7
    
    # Temporal superposition
    time_slots: List[Tuple[float, float]] = field(default_factory=list)  # (start, end) pairs
    temporal_amplitudes: List[complex] = field(default_factory=list)
    
    # Resource superposition  
    resource_configurations: List[Dict[str, float]] = field(default_factory=list)
    resource_amplitudes: List[complex] = field(default_factory=list)
    
    created_at: float = field(default_factory=time.time)
    last_interference: float = field(default_factory=time.time)


@dataclass
class SchedulingWaveFunction:
    """Global wave function for the scheduling system."""
    superposed_tasks: Dict[str, TaskSuperposition]
    global_phase: float = 0.0
    entanglement_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    total_amplitude: complex = complex(1.0, 0.0)
    normalization_constant: float = 1.0


class SuperpositionScheduler:
    """
    Quantum-inspired scheduler that maintains tasks in superposition
    until optimal measurement occurs.
    """
    
    def __init__(self,
                 max_superposition_time: float = 300.0,
                 interference_strength: float = 0.5,
                 decoherence_rate: float = 0.01):
        self.max_superposition_time = max_superposition_time
        self.interference_strength = interference_strength
        self.decoherence_rate = decoherence_rate
        
        # Core quantum scheduling state
        self.wave_function = SchedulingWaveFunction({})
        self.scheduling_phases: Dict[str, SchedulingPhase] = {}
        self.interference_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Node and resource tracking
        self.node_availability: Dict[str, List[Tuple[float, float]]] = {}
        self.resource_pools: Dict[str, float] = {}
        self.node_characteristics: Dict[str, Dict[str, float]] = {}
        
        # Performance metrics
        self.scheduler_metrics = {
            "total_scheduled": 0,
            "average_superposition_time": 0.0,
            "interference_events": 0,
            "measurement_accuracy": 0.0,
            "resource_efficiency": 0.0
        }
        
        # Scheduling history
        self.scheduling_history: deque = deque(maxlen=1000)
        self.active_superpositions: Dict[str, float] = {}  # task_id -> creation_time
        
    async def initialize_superposition(self, 
                                     task_id: str,
                                     potential_nodes: List[str],
                                     time_preferences: List[Tuple[float, float]],
                                     resource_requirements: Dict[str, float]) -> TaskSuperposition:
        """Initialize task in quantum superposition across execution contexts."""
        
        # Calculate initial amplitudes based on node suitability
        node_amplitudes = {}
        total_amplitude_squared = 0.0
        
        for node_id in potential_nodes:
            # Base suitability from node characteristics
            node_chars = self.node_characteristics.get(node_id, {})
            base_suitability = node_chars.get('performance_score', 0.5)
            
            # Availability factor
            availability = self._calculate_node_availability(node_id, time_preferences)
            
            # Resource compatibility
            resource_compat = self._calculate_resource_compatibility(node_id, resource_requirements)
            
            # Combined amplitude (complex number with phase)
            amplitude_magnitude = math.sqrt(base_suitability * availability * resource_compat)
            phase = np.random.uniform(0, 2 * math.pi)  # Random initial phase
            
            amplitude = amplitude_magnitude * np.exp(1j * phase)
            node_amplitudes[node_id] = amplitude
            total_amplitude_squared += amplitude_magnitude ** 2
        
        # Normalize amplitudes
        normalization = math.sqrt(total_amplitude_squared) if total_amplitude_squared > 0 else 1.0
        normalized_amplitudes = {
            node_id: amp / normalization for node_id, amp in node_amplitudes.items()
        }
        
        # Extract phase information
        phase_angles = {
            node_id: np.angle(amp) for node_id, amp in normalized_amplitudes.items()
        }
        
        # Calculate measurement probabilities
        measurement_probs = {
            node_id: abs(amp) ** 2 for node_id, amp in normalized_amplitudes.items()
        }
        
        # Create temporal superposition
        temporal_amps = []
        for start_time, end_time in time_preferences:
            # Amplitude based on time slot preference
            time_factor = self._calculate_temporal_preference(start_time, end_time)
            temporal_amps.append(complex(math.sqrt(time_factor), 0))
        
        # Create resource superposition
        resource_configs = self._generate_resource_configurations(resource_requirements)
        resource_amps = [complex(1.0/len(resource_configs), 0) for _ in resource_configs]
        
        superposition = TaskSuperposition(
            task_id=task_id,
            amplitude_distribution=normalized_amplitudes,
            phase_angles=phase_angles,
            measurement_probability=measurement_probs,
            time_slots=time_preferences,
            temporal_amplitudes=temporal_amps,
            resource_configurations=resource_configs,
            resource_amplitudes=resource_amps
        )
        
        # Register in global wave function
        self.wave_function.superposed_tasks[task_id] = superposition
        self.scheduling_phases[task_id] = SchedulingPhase.SUPERPOSITION
        self.active_superpositions[task_id] = time.time()
        
        logger.info(f"Initialized superposition for task {task_id} across {len(potential_nodes)} nodes")
        return superposition
    
    def _calculate_node_availability(self, node_id: str, time_preferences: List[Tuple[float, float]]) -> float:
        """Calculate node availability factor for given time preferences."""
        if node_id not in self.node_availability:
            return 1.0  # Assume fully available if no info
        
        available_slots = self.node_availability[node_id]
        total_preference_time = sum(end - start for start, end in time_preferences)
        total_available_overlap = 0.0
        
        for pref_start, pref_end in time_preferences:
            for avail_start, avail_end in available_slots:
                overlap_start = max(pref_start, avail_start)
                overlap_end = min(pref_end, avail_end)
                if overlap_start < overlap_end:
                    total_available_overlap += overlap_end - overlap_start
        
        return min(1.0, total_available_overlap / max(total_preference_time, 1.0))
    
    def _calculate_resource_compatibility(self, node_id: str, requirements: Dict[str, float]) -> float:
        """Calculate how well node resources match requirements."""
        if not requirements:
            return 1.0
        
        compatibility_scores = []
        for resource, required_amount in requirements.items():
            available = self.resource_pools.get(f"{node_id}_{resource}", 0.0)
            if available >= required_amount:
                compatibility_scores.append(1.0)
            elif available > 0:
                compatibility_scores.append(available / required_amount)
            else:
                compatibility_scores.append(0.0)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.0
    
    def _calculate_temporal_preference(self, start_time: float, end_time: float) -> float:
        """Calculate preference factor for temporal slot."""
        current_time = time.time()
        duration = end_time - start_time
        
        # Prefer sooner start times and reasonable durations
        time_factor = max(0.1, 1.0 - (start_time - current_time) / 3600.0)  # Decay over hour
        duration_factor = min(1.0, duration / 1800.0)  # Optimal around 30 minutes
        
        return time_factor * duration_factor
    
    def _generate_resource_configurations(self, base_requirements: Dict[str, float]) -> List[Dict[str, float]]:
        """Generate multiple resource configuration options."""
        configs = [base_requirements.copy()]  # Base configuration
        
        # Generate variations (trade-offs between resources)
        if 'cpu' in base_requirements and 'memory' in base_requirements:
            # High CPU, lower memory
            high_cpu_config = base_requirements.copy()
            high_cpu_config['cpu'] *= 1.5
            high_cpu_config['memory'] *= 0.8
            configs.append(high_cpu_config)
            
            # High memory, lower CPU  
            high_mem_config = base_requirements.copy()
            high_mem_config['cpu'] *= 0.8
            high_mem_config['memory'] *= 1.5
            configs.append(high_mem_config)
        
        return configs
    
    async def evolve_superposition(self, time_step: float = 1.0):
        """Evolve quantum superpositions over time."""
        current_time = time.time()
        
        tasks_to_remove = []
        
        for task_id, superposition in self.wave_function.superposed_tasks.items():
            # Check for decoherence
            time_in_superposition = current_time - superposition.created_at
            
            if time_in_superposition > superposition.coherence_lifetime:
                # Task has decoherent - force measurement
                await self.force_measurement(task_id, "decoherence")
                tasks_to_remove.append(task_id)
                continue
            
            # Apply decoherence gradually
            decoherence_factor = math.exp(-self.decoherence_rate * time_step)
            
            # Update amplitudes with decoherence
            for node_id in superposition.amplitude_distribution:
                current_amp = superposition.amplitude_distribution[node_id]
                superposition.amplitude_distribution[node_id] = current_amp * decoherence_factor
            
            # Renormalize
            total_prob = sum(abs(amp)**2 for amp in superposition.amplitude_distribution.values())
            if total_prob > 0:
                norm_factor = math.sqrt(total_prob)
                for node_id in superposition.amplitude_distribution:
                    superposition.amplitude_distribution[node_id] /= norm_factor
                
                # Update measurement probabilities
                superposition.measurement_probability = {
                    node_id: abs(amp)**2 
                    for node_id, amp in superposition.amplitude_distribution.items()
                }
        
        # Remove decoherent tasks
        for task_id in tasks_to_remove:
            self.wave_function.superposed_tasks.pop(task_id, None)
            self.active_superpositions.pop(task_id, None)
    
    async def apply_interference(self, interfering_task_ids: List[str]):
        """Apply quantum interference between tasks in superposition."""
        if len(interfering_task_ids) < 2:
            return
        
        interference_start = time.time()
        
        # Get superpositions for interfering tasks
        superpositions = []
        for task_id in interfering_task_ids:
            if task_id in self.wave_function.superposed_tasks:
                superpositions.append(self.wave_function.superposed_tasks[task_id])
        
        if len(superpositions) < 2:
            return
        
        # Find common nodes
        common_nodes = set(superpositions[0].amplitude_distribution.keys())
        for superposition in superpositions[1:]:
            common_nodes &= set(superposition.amplitude_distribution.keys())
        
        # Apply interference for each common node
        for node_id in common_nodes:
            # Constructive/destructive interference
            combined_amplitude = 0j
            
            for superposition in superpositions:
                amplitude = superposition.amplitude_distribution[node_id]
                phase = superposition.phase_angles[node_id]
                
                # Apply phase evolution
                time_evolution_phase = 2 * math.pi * (time.time() - superposition.last_interference) / 60.0
                evolved_phase = phase + time_evolution_phase
                
                combined_amplitude += abs(amplitude) * np.exp(1j * evolved_phase)
            
            # Distribute combined amplitude back to tasks
            combined_magnitude = abs(combined_amplitude)
            combined_phase = np.angle(combined_amplitude)
            
            for i, superposition in enumerate(superpositions):
                # Apply interference effect
                interference_factor = self.interference_strength * (1.0 + 0.5 * math.cos(combined_phase))
                
                old_amplitude = superposition.amplitude_distribution[node_id]
                new_magnitude = abs(old_amplitude) * interference_factor
                
                superposition.amplitude_distribution[node_id] = new_magnitude * np.exp(1j * combined_phase)
                superposition.phase_angles[node_id] = combined_phase
                superposition.last_interference = time.time()
        
        # Renormalize all affected superpositions
        for superposition in superpositions:
            total_prob = sum(abs(amp)**2 for amp in superposition.amplitude_distribution.values())
            if total_prob > 0:
                norm_factor = math.sqrt(total_prob)
                for node_id in superposition.amplitude_distribution:
                    superposition.amplitude_distribution[node_id] /= norm_factor
                
                # Update measurement probabilities
                superposition.measurement_probability = {
                    node_id: abs(amp)**2 
                    for node_id, amp in superposition.amplitude_distribution.items()
                }
        
        # Record interference event
        self.scheduler_metrics["interference_events"] += 1
        
        # Store interference pattern
        for task_id in interfering_task_ids:
            if task_id in self.wave_function.superposed_tasks:
                pattern_strength = max(self.wave_function.superposed_tasks[task_id].measurement_probability.values())
                self.interference_patterns[task_id].append(pattern_strength)
        
        logger.debug(f"Applied interference to {len(interfering_task_ids)} tasks in {time.time() - interference_start:.3f}s")
    
    async def measure_optimal_assignment(self, task_id: str, 
                                       measurement_strategy: str = "maximum_probability") -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Collapse superposition through quantum measurement to get optimal assignment.
        """
        if task_id not in self.wave_function.superposed_tasks:
            return None
        
        superposition = self.wave_function.superposed_tasks[task_id]
        measurement_time = time.time()
        
        # Choose measurement strategy
        if measurement_strategy == "maximum_probability":
            # Select node with highest probability
            selected_node = max(superposition.measurement_probability.items(), 
                              key=lambda x: x[1])[0]
            
        elif measurement_strategy == "weighted_random":
            # Probabilistic selection based on amplitudes
            nodes = list(superposition.measurement_probability.keys())
            probabilities = list(superposition.measurement_probability.values())
            
            # Ensure probabilities sum to 1
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p/prob_sum for p in probabilities]
                selected_node = np.random.choice(nodes, p=probabilities)
            else:
                selected_node = nodes[0] if nodes else None
                
        elif measurement_strategy == "interference_optimized":
            # Consider interference patterns
            best_node = None
            best_score = -float('inf')
            
            for node_id, probability in superposition.measurement_probability.items():
                interference_bonus = 0.0
                if task_id in self.interference_patterns:
                    recent_patterns = self.interference_patterns[task_id][-5:]  # Last 5 patterns
                    if recent_patterns:
                        interference_bonus = np.mean(recent_patterns) * 0.2
                
                score = probability + interference_bonus
                if score > best_score:
                    best_score = score
                    best_node = node_id
            
            selected_node = best_node
        else:
            # Default to first available
            selected_node = list(superposition.measurement_probability.keys())[0] if superposition.measurement_probability else None
        
        if not selected_node:
            return None
        
        # Select optimal time slot and resource configuration
        temporal_selection = self._measure_temporal_slot(superposition)
        resource_selection = self._measure_resource_configuration(superposition)
        
        # Create assignment details
        assignment_details = {
            "node_id": selected_node,
            "selected_probability": superposition.measurement_probability[selected_node],
            "amplitude_magnitude": abs(superposition.amplitude_distribution[selected_node]),
            "phase_angle": superposition.phase_angles[selected_node],
            "time_slot": temporal_selection,
            "resource_configuration": resource_selection,
            "measurement_time": measurement_time,
            "superposition_duration": measurement_time - superposition.created_at,
            "measurement_strategy": measurement_strategy,
            "interference_events": len(self.interference_patterns.get(task_id, []))
        }
        
        # Collapse wave function (remove from superposition)
        self.wave_function.superposed_tasks.pop(task_id, None)
        self.scheduling_phases[task_id] = SchedulingPhase.MEASUREMENT
        self.active_superpositions.pop(task_id, None)
        
        # Update metrics
        self._update_measurement_metrics(assignment_details)
        
        # Record in history
        self.scheduling_history.append({
            "task_id": task_id,
            "selected_node": selected_node,
            "measurement_time": measurement_time,
            "strategy": measurement_strategy,
            **assignment_details
        })
        
        logger.info(f"Measured assignment for task {task_id}: node {selected_node} "
                   f"with probability {assignment_details['selected_probability']:.3f}")
        
        return selected_node, assignment_details
    
    def _measure_temporal_slot(self, superposition: TaskSuperposition) -> Optional[Tuple[float, float]]:
        """Measure optimal temporal slot from superposition."""
        if not superposition.time_slots or not superposition.temporal_amplitudes:
            return None
        
        # Calculate probabilities for each time slot
        probabilities = [abs(amp)**2 for amp in superposition.temporal_amplitudes]
        prob_sum = sum(probabilities)
        
        if prob_sum > 0:
            probabilities = [p/prob_sum for p in probabilities]
            selected_idx = np.random.choice(len(probabilities), p=probabilities)
            return superposition.time_slots[selected_idx]
        
        return superposition.time_slots[0] if superposition.time_slots else None
    
    def _measure_resource_configuration(self, superposition: TaskSuperposition) -> Optional[Dict[str, float]]:
        """Measure optimal resource configuration from superposition."""
        if not superposition.resource_configurations or not superposition.resource_amplitudes:
            return None
        
        # Calculate probabilities for each resource config
        probabilities = [abs(amp)**2 for amp in superposition.resource_amplitudes]
        prob_sum = sum(probabilities)
        
        if prob_sum > 0:
            probabilities = [p/prob_sum for p in probabilities]
            selected_idx = np.random.choice(len(probabilities), p=probabilities)
            return superposition.resource_configurations[selected_idx]
        
        return superposition.resource_configurations[0] if superposition.resource_configurations else None
    
    async def force_measurement(self, task_id: str, reason: str = "timeout"):
        """Force measurement of a task (e.g., due to timeout or external event)."""
        if task_id in self.wave_function.superposed_tasks:
            assignment = await self.measure_optimal_assignment(task_id, "maximum_probability")
            if assignment:
                logger.warning(f"Forced measurement of task {task_id} due to {reason}")
            return assignment
        return None
    
    def _update_measurement_metrics(self, assignment_details: Dict[str, Any]):
        """Update performance metrics after measurement."""
        self.scheduler_metrics["total_scheduled"] += 1
        
        # Update average superposition time
        duration = assignment_details["superposition_duration"]
        total_scheduled = self.scheduler_metrics["total_scheduled"]
        current_avg = self.scheduler_metrics["average_superposition_time"]
        
        self.scheduler_metrics["average_superposition_time"] = (
            (current_avg * (total_scheduled - 1) + duration) / total_scheduled
        )
        
        # Update measurement accuracy (based on selected probability)
        selected_prob = assignment_details["selected_probability"]
        current_accuracy = self.scheduler_metrics["measurement_accuracy"]
        
        self.scheduler_metrics["measurement_accuracy"] = (
            (current_accuracy * (total_scheduled - 1) + selected_prob) / total_scheduled
        )
    
    def update_node_availability(self, node_id: str, available_slots: List[Tuple[float, float]]):
        """Update node availability information."""
        self.node_availability[node_id] = available_slots
    
    def update_resource_pools(self, resource_updates: Dict[str, float]):
        """Update resource pool availability."""
        self.resource_pools.update(resource_updates)
    
    def update_node_characteristics(self, node_id: str, characteristics: Dict[str, float]):
        """Update node performance characteristics."""
        self.node_characteristics[node_id] = characteristics
    
    def get_superposition_status(self) -> Dict[str, Any]:
        """Get current status of all superpositions."""
        current_time = time.time()
        
        superposition_info = {}
        for task_id, superposition in self.wave_function.superposed_tasks.items():
            superposition_info[task_id] = {
                "nodes_in_superposition": list(superposition.amplitude_distribution.keys()),
                "max_probability": max(superposition.measurement_probability.values()) if superposition.measurement_probability else 0,
                "time_in_superposition": current_time - superposition.created_at,
                "coherence_remaining": max(0, superposition.coherence_lifetime - (current_time - superposition.created_at)),
                "interference_events": len(self.interference_patterns.get(task_id, [])),
                "phase": self.scheduling_phases.get(task_id, "unknown").value
            }
        
        return {
            "active_superpositions": len(self.wave_function.superposed_tasks),
            "global_wave_function_phase": self.wave_function.global_phase,
            "total_amplitude": abs(self.wave_function.total_amplitude),
            "superposition_details": superposition_info,
            "scheduler_metrics": self.scheduler_metrics,
            "recent_measurements": list(self.scheduling_history)[-10:]  # Last 10 measurements
        }