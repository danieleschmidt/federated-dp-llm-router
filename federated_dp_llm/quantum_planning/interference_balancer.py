"""
Quantum Interference-Based Load Balancer

Implements quantum interference principles for optimal load balancing
across federated nodes, using constructive and destructive interference
patterns to optimize resource distribution and minimize conflicts.
"""

import asyncio
import time
from .numpy_fallback import get_numpy_backend

HAS_NUMPY, np = get_numpy_backend()
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import math
import cmath

logger = logging.getLogger(__name__)


class InterferenceType(Enum):
    """Types of quantum interference patterns."""
    CONSTRUCTIVE = "constructive"    # Amplifies resource allocation
    DESTRUCTIVE = "destructive"      # Reduces resource conflicts
    MIXED = "mixed"                  # Complex interference pattern
    COHERENT = "coherent"           # Phase-locked interference
    INCOHERENT = "incoherent"       # Random phase interference


class WavePhase(Enum):
    """Quantum wave phases for interference calculations."""
    IN_PHASE = 0.0                  # 0 radians - maximum constructive
    QUADRATURE = 1.5707963267949    # π/2 radians - 90° phase shift
    OUT_OF_PHASE = 3.1415926535898  # π radians - maximum destructive  
    ANTI_QUADRATURE = 4.7123889803847  # 3π/2 radians - 270° phase shift


@dataclass
class TaskInterference:
    """
    Represents quantum interference between computational tasks
    across federated nodes.
    """
    interference_id: str
    interfering_tasks: List[str]
    interference_type: InterferenceType
    
    # Wave properties
    amplitude_pattern: Dict[str, complex]  # Node -> complex amplitude
    phase_relationships: Dict[str, float]  # Node -> phase angle
    frequency_spectrum: Dict[str, float]   # Node -> frequency
    wavelength: float = 100.0              # Interference wavelength
    
    # Interference characteristics
    visibility: float = 1.0                # Fringe visibility (0-1)
    coherence_length: float = 1000.0       # Spatial coherence
    coherence_time: float = 300.0          # Temporal coherence  
    
    # Pattern evolution
    phase_velocity: float = 1.0            # Phase propagation speed
    group_velocity: float = 0.8            # Information propagation speed
    dispersion_coefficient: float = 0.01   # Phase dispersion rate
    
    # Measurement data
    interference_strength: float = 0.0     # Current interference magnitude
    last_measurement: float = field(default_factory=time.time)
    measurement_history: List[Tuple[float, float]] = field(default_factory=list)
    
    creation_time: float = field(default_factory=time.time)


@dataclass 
class NodeWaveState:
    """Quantum wave state of a federated node for interference calculations."""
    node_id: str
    
    # Wave function components
    psi_real: float = 1.0              # Real part of wave function
    psi_imaginary: float = 0.0         # Imaginary part of wave function  
    phase: float = 0.0                 # Current phase
    amplitude: float = 1.0             # Wave amplitude
    frequency: float = 1.0             # Oscillation frequency
    
    # Node characteristics affecting wave behavior
    refractive_index: float = 1.0      # Processing "refractive index"
    impedance: float = 1.0             # Resource impedance
    quality_factor: float = 10.0       # Resonance quality factor
    
    # Dynamic properties
    load_induced_phase_shift: float = 0.0
    thermal_noise_level: float = 0.01
    quantum_decoherence_rate: float = 0.001
    
    last_update: float = field(default_factory=time.time)


class InterferenceBalancer:
    """
    Quantum interference-based load balancer for federated systems.
    
    Uses quantum interference principles to:
    - Create constructive interference for optimal resource allocation
    - Generate destructive interference to cancel resource conflicts
    - Implement wave-based load distribution algorithms
    - Optimize phase relationships between computational tasks
    """
    
    def __init__(self,
                 interference_resolution: float = 0.1,
                 coherence_threshold: float = 0.7,
                 phase_locked_loop_enabled: bool = True):
        self.interference_resolution = interference_resolution
        self.coherence_threshold = coherence_threshold
        self.phase_locked_loop_enabled = phase_locked_loop_enabled
        
        # Core interference state
        self.task_interferences: Dict[str, TaskInterference] = {}
        self.node_wave_states: Dict[str, NodeWaveState] = {}
        self.global_wave_function: complex = complex(1.0, 0.0)
        
        # Interference pattern tracking
        self.interference_patterns: Dict[str, List] = {}
        self.standing_wave_nodes: Dict[str, List[float]] = {}
        self.phase_synchronization_matrix: List = np.array([])
        
        # Load balancing state
        self.optimal_phase_configurations: Dict[str, float] = {}
        self.load_distribution_amplitudes: Dict[str, complex] = {}
        self.interference_history: deque = deque(maxlen=1000)
        
        # Performance metrics
        self.balancer_metrics = {
            "total_interferences": 0,
            "constructive_events": 0,
            "destructive_events": 0,
            "load_balancing_efficiency": 0.0,
            "average_coherence": 0.0,
            "phase_lock_stability": 0.0,
            "wave_interference_gain": 0.0
        }
        
        # Advanced interference parameters
        self.beam_splitter_ratios: Dict[str, float] = {}
        self.polarization_states: Dict[str, Tuple[float, float]] = {}
        self.nonlinear_coefficients: Dict[str, float] = {}
        
    async def initialize_node_wave_state(self, 
                                       node_id: str,
                                       node_characteristics: Dict[str, float]):
        """Initialize quantum wave state for a federated node."""
        
        # Extract wave parameters from node characteristics
        base_frequency = node_characteristics.get('processing_frequency', 1.0)
        load_capacity = node_characteristics.get('load_capacity', 1.0)
        network_latency = node_characteristics.get('network_latency', 0.1)
        
        # Calculate wave properties based on node characteristics
        refractive_index = 1.0 + 0.1 * (1.0 / max(load_capacity, 0.1))  # Higher load = higher refractive index
        impedance = math.sqrt(network_latency * load_capacity)  # Impedance matching
        quality_factor = 10.0 * load_capacity  # Higher capacity = better quality factor
        
        # Initialize with random but coherent phase
        initial_phase = np.random.uniform(0, 2 * math.pi)
        
        wave_state = NodeWaveState(
            node_id=node_id,
            phase=initial_phase,
            amplitude=math.sqrt(load_capacity),  # Amplitude proportional to capacity
            frequency=base_frequency,
            refractive_index=refractive_index,
            impedance=impedance,
            quality_factor=quality_factor
        )
        
        self.node_wave_states[node_id] = wave_state
        
        # Initialize beam splitter ratio for this node
        self.beam_splitter_ratios[node_id] = 0.5  # 50/50 split initially
        
        # Initialize polarization state (computational basis)
        self.polarization_states[node_id] = (1.0, 0.0)  # |H⟩ polarization
        
        logger.info(f"Initialized wave state for node {node_id}: "
                   f"frequency={base_frequency:.3f}, amplitude={wave_state.amplitude:.3f}, "
                   f"phase={initial_phase:.3f}")
    
    async def create_task_interference(self,
                                     task_ids: List[str],
                                     target_nodes: List[str],
                                     interference_type: InterferenceType = InterferenceType.CONSTRUCTIVE) -> str:
        """Create quantum interference pattern for load balancing optimization."""
        
        if len(task_ids) < 2 or len(target_nodes) < 2:
            raise ValueError("Need at least 2 tasks and 2 nodes for interference")
        
        interference_id = f"interference_{int(time.time())}_{len(self.task_interferences)}"
        
        # Calculate optimal interference pattern
        amplitude_pattern = {}
        phase_relationships = {}
        frequency_spectrum = {}
        
        for node_id in target_nodes:
            node_state = self.node_wave_states.get(node_id)
            if not node_state:
                continue
            
            # Calculate complex amplitude for this node
            base_amplitude = node_state.amplitude
            
            if interference_type == InterferenceType.CONSTRUCTIVE:
                # All phases aligned for maximum constructive interference
                phase_offset = 0.0
                amplitude_factor = 1.0
            elif interference_type == InterferenceType.DESTRUCTIVE:
                # Alternating phases for destructive interference
                phase_offset = math.pi if target_nodes.index(node_id) % 2 == 1 else 0.0
                amplitude_factor = 0.8  # Slightly reduced for partial cancellation
            else:  # MIXED interference
                # Complex pattern with varied phases
                phase_offset = (2 * math.pi * target_nodes.index(node_id)) / len(target_nodes)
                amplitude_factor = 0.9
            
            # Account for node's current phase and load
            total_phase = node_state.phase + phase_offset + node_state.load_induced_phase_shift
            amplitude = base_amplitude * amplitude_factor
            
            amplitude_pattern[node_id] = amplitude * cmath.exp(1j * total_phase)
            phase_relationships[node_id] = total_phase
            frequency_spectrum[node_id] = node_state.frequency
        
        # Calculate optimal wavelength for interference pattern
        avg_frequency = np.mean(list(frequency_spectrum.values())) if frequency_spectrum else 1.0
        wavelength = 2 * math.pi / avg_frequency  # λ = 2π/ω
        
        # Create interference object
        interference = TaskInterference(
            interference_id=interference_id,
            interfering_tasks=task_ids.copy(),
            interference_type=interference_type,
            amplitude_pattern=amplitude_pattern,
            phase_relationships=phase_relationships,
            frequency_spectrum=frequency_spectrum,
            wavelength=wavelength,
            visibility=self._calculate_fringe_visibility(amplitude_pattern),
            coherence_length=self._calculate_coherence_length(target_nodes),
            phase_velocity=avg_frequency * wavelength
        )
        
        self.task_interferences[interference_id] = interference
        
        # Generate initial interference pattern
        await self._generate_interference_pattern(interference_id)
        
        # Update global metrics
        self.balancer_metrics["total_interferences"] += 1
        if interference_type == InterferenceType.CONSTRUCTIVE:
            self.balancer_metrics["constructive_events"] += 1
        elif interference_type == InterferenceType.DESTRUCTIVE:
            self.balancer_metrics["destructive_events"] += 1
        
        logger.info(f"Created {interference_type.value} interference {interference_id} "
                   f"for {len(task_ids)} tasks across {len(target_nodes)} nodes")
        
        return interference_id
    
    def _calculate_fringe_visibility(self, amplitude_pattern: Dict[str, complex]) -> float:
        """Calculate fringe visibility of interference pattern."""
        
        if len(amplitude_pattern) < 2:
            return 0.0
        
        amplitudes = [abs(amp) for amp in amplitude_pattern.values()]
        max_amp = max(amplitudes)
        min_amp = min(amplitudes)
        
        # Visibility V = (Imax - Imin) / (Imax + Imin)
        intensity_max = max_amp ** 2
        intensity_min = min_amp ** 2
        
        if intensity_max + intensity_min == 0:
            return 0.0
        
        visibility = (intensity_max - intensity_min) / (intensity_max + intensity_min)
        return min(1.0, max(0.0, visibility))
    
    def _calculate_coherence_length(self, target_nodes: List[str]) -> float:
        """Calculate spatial coherence length based on node distribution."""
        
        if len(target_nodes) < 2:
            return 1000.0  # Default coherence length
        
        # Estimate based on node network topology (simplified)
        max_distance = 0.0
        for i, node1 in enumerate(target_nodes):
            for j, node2 in enumerate(target_nodes[i+1:], i+1):
                # Hash-based distance approximation
                distance = abs(hash(node1) - hash(node2)) % 1000
                max_distance = max(max_distance, distance)
        
        # Coherence length inversely related to maximum distance
        coherence_length = 1000.0 / (1.0 + max_distance / 500.0)
        return coherence_length
    
    async def _generate_interference_pattern(self, interference_id: str):
        """Generate spatial interference pattern for load distribution."""
        
        interference = self.task_interferences.get(interference_id)
        if not interference:
            return
        
        nodes = list(interference.amplitude_pattern.keys())
        n_points = max(100, len(nodes) * 10)  # Resolution for pattern calculation
        
        # Create spatial grid for interference calculation
        x_positions = np.linspace(0, len(nodes) * interference.wavelength, n_points)
        interference_pattern = np.zeros(n_points, dtype=complex)
        
        # Calculate superposition of waves from each node
        for i, node_id in enumerate(nodes):
            amplitude = interference.amplitude_pattern[node_id]
            phase = interference.phase_relationships[node_id]
            frequency = interference.frequency_spectrum.get(node_id, 1.0)
            
            # Node position in interference space
            node_position = i * interference.wavelength / len(nodes)
            
            # Calculate wave contribution at each point
            for j, x in enumerate(x_positions):
                # Distance from node to point
                distance = abs(x - node_position)
                
                # Wave amplitude with distance decay
                distance_factor = 1.0 / (1.0 + distance / interference.coherence_length)
                
                # Phase shift due to propagation
                phase_shift = 2 * math.pi * distance / interference.wavelength
                
                # Add wave contribution
                wave_contribution = (amplitude * distance_factor * 
                                   cmath.exp(1j * (phase + phase_shift)))
                interference_pattern[j] += wave_contribution
        
        # Calculate intensity pattern |ψ|²
        intensity_pattern = np.abs(interference_pattern) ** 2
        
        # Store pattern
        self.interference_patterns[interference_id] = intensity_pattern
        
        # Find standing wave nodes (zero intensity points)
        nodes_indices = []
        for i in range(1, len(intensity_pattern) - 1):
            if (intensity_pattern[i] < 0.1 * np.max(intensity_pattern) and
                intensity_pattern[i] < intensity_pattern[i-1] and
                intensity_pattern[i] < intensity_pattern[i+1]):
                nodes_indices.append(i)
        
        # Convert indices to positions
        standing_wave_positions = [x_positions[i] for i in nodes_indices]
        self.standing_wave_nodes[interference_id] = standing_wave_positions
        
        # Calculate interference strength
        max_intensity = np.max(intensity_pattern)
        min_intensity = np.min(intensity_pattern)
        interference.interference_strength = (max_intensity - min_intensity) / (max_intensity + min_intensity)
    
    async def optimize_load_distribution(self, 
                                       current_loads: Dict[str, float],
                                       target_loads: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize load distribution using quantum interference principles.
        """
        
        optimization_start = time.time()
        
        # Update node wave states based on current loads
        await self._update_wave_states_from_loads(current_loads)
        
        # Find optimal phase configuration
        optimal_phases = await self._find_optimal_phase_configuration(current_loads, target_loads)
        
        # Apply interference-based load redistribution
        optimized_loads = await self._apply_interference_redistribution(
            current_loads, target_loads, optimal_phases
        )
        
        # Update phase synchronization if enabled
        if self.phase_locked_loop_enabled:
            await self._update_phase_locked_loop()
        
        # Calculate efficiency improvement
        efficiency_improvement = self._calculate_load_balancing_efficiency(
            current_loads, optimized_loads, target_loads
        )
        
        # Update metrics
        self.balancer_metrics["load_balancing_efficiency"] = efficiency_improvement
        self.balancer_metrics["wave_interference_gain"] = self._calculate_interference_gain()
        
        # Store optimization event in history
        self.interference_history.append({
            "timestamp": optimization_start,
            "current_loads": current_loads.copy(),
            "target_loads": target_loads.copy(),
            "optimized_loads": optimized_loads.copy(),
            "efficiency_improvement": efficiency_improvement,
            "optimization_time": time.time() - optimization_start,
            "active_interferences": len(self.task_interferences)
        })
        
        logger.debug(f"Optimized load distribution in {time.time() - optimization_start:.3f}s, "
                    f"efficiency improvement: {efficiency_improvement:.3f}")
        
        return optimized_loads
    
    async def _update_wave_states_from_loads(self, current_loads: Dict[str, float]):
        """Update node wave states based on current computational loads."""
        
        for node_id, load in current_loads.items():
            node_state = self.node_wave_states.get(node_id)
            if not node_state:
                continue
            
            # Load affects wave properties
            # Higher load increases phase shift (analogous to optical path length)
            node_state.load_induced_phase_shift = 2 * math.pi * load * 0.1
            
            # Load affects amplitude (analogous to attenuation)
            base_amplitude = math.sqrt(max(0.1, 1.0 - load))  # Amplitude decreases with load
            node_state.amplitude = base_amplitude
            
            # Update wave function components
            total_phase = node_state.phase + node_state.load_induced_phase_shift
            node_state.psi_real = node_state.amplitude * math.cos(total_phase)
            node_state.psi_imaginary = node_state.amplitude * math.sin(total_phase)
            
            node_state.last_update = time.time()
    
    async def _find_optimal_phase_configuration(self, 
                                              current_loads: Dict[str, float],
                                              target_loads: Dict[str, float]) -> Dict[str, float]:
        """Find optimal phase relationships for constructive interference."""
        
        optimal_phases = {}
        
        # Use gradient-based optimization to find phase configuration
        # that maximizes constructive interference for load balancing
        
        for node_id in current_loads.keys():
            node_state = self.node_wave_states.get(node_id)
            if not node_state:
                continue
            
            current_load = current_loads[node_id]
            target_load = target_loads.get(node_id, current_load)
            load_error = target_load - current_load
            
            # Calculate optimal phase for this node
            # Phase adjustment proportional to load error
            phase_adjustment = math.pi * load_error  # π phase shift for maximum effect
            
            # Constrain phase to [0, 2π]
            optimal_phase = (node_state.phase + phase_adjustment) % (2 * math.pi)
            optimal_phases[node_id] = optimal_phase
        
        # Store optimal configuration
        self.optimal_phase_configurations.update(optimal_phases)
        
        return optimal_phases
    
    async def _apply_interference_redistribution(self,
                                               current_loads: Dict[str, float],
                                               target_loads: Dict[str, float],
                                               optimal_phases: Dict[str, float]) -> Dict[str, float]:
        """Apply interference-based load redistribution."""
        
        optimized_loads = current_loads.copy()
        
        # Calculate load redistribution using interference patterns
        for interference_id, interference in self.task_interferences.items():
            if interference_id not in self.interference_patterns:
                continue
            
            pattern = self.interference_patterns[interference_id]
            nodes = list(interference.amplitude_pattern.keys())
            
            # Calculate load redistribution weights based on interference pattern
            for i, node_id in enumerate(nodes):
                if node_id not in current_loads:
                    continue
                
                # Get interference intensity at node position
                pattern_index = int((i / len(nodes)) * len(pattern))
                pattern_index = min(pattern_index, len(pattern) - 1)
                
                interference_weight = pattern[pattern_index].real
                
                # Normalize interference weight
                max_pattern_value = np.max(np.real(pattern))
                if max_pattern_value > 0:
                    normalized_weight = interference_weight / max_pattern_value
                else:
                    normalized_weight = 1.0 / len(nodes)
                
                # Apply phase-based load adjustment
                phase_factor = math.cos(optimal_phases.get(node_id, 0.0))
                load_adjustment_factor = 1.0 + 0.2 * normalized_weight * phase_factor
                
                # Calculate new load
                current_load = optimized_loads[node_id]
                target_load = target_loads.get(node_id, current_load)
                
                # Blend current and target loads using interference
                blend_factor = min(1.0, normalized_weight)
                new_load = (current_load * (1 - blend_factor) + 
                           target_load * blend_factor * load_adjustment_factor)
                
                optimized_loads[node_id] = max(0.0, min(1.0, new_load))
        
        return optimized_loads
    
    async def _update_phase_locked_loop(self):
        """Update phase-locked loop for phase synchronization."""
        
        if len(self.node_wave_states) < 2:
            return
        
        # Calculate phase synchronization matrix
        nodes = list(self.node_wave_states.keys())
        n_nodes = len(nodes)
        
        if self.phase_synchronization_matrix.size != (n_nodes, n_nodes):
            self.phase_synchronization_matrix = np.zeros((n_nodes, n_nodes))
        
        # Update phase relationships
        for i, node1 in enumerate(nodes):
            state1 = self.node_wave_states[node1]
            for j, node2 in enumerate(nodes):
                if i == j:
                    self.phase_synchronization_matrix[i, j] = 1.0
                    continue
                
                state2 = self.node_wave_states[node2]
                
                # Calculate phase difference
                phase_diff = state1.phase - state2.phase
                phase_diff = ((phase_diff + math.pi) % (2 * math.pi)) - math.pi  # Wrap to [-π, π]
                
                # Synchronization strength based on phase difference
                sync_strength = math.cos(phase_diff)
                self.phase_synchronization_matrix[i, j] = sync_strength
        
        # Calculate average coherence
        off_diagonal_elements = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                off_diagonal_elements.append(abs(self.phase_synchronization_matrix[i, j]))
        
        if off_diagonal_elements:
            self.balancer_metrics["average_coherence"] = np.mean(off_diagonal_elements)
        
        # Phase lock stability (variance of phase differences)
        phase_diffs = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                state1 = self.node_wave_states[nodes[i]]
                state2 = self.node_wave_states[nodes[j]]
                phase_diff = abs(state1.phase - state2.phase)
                phase_diffs.append(phase_diff)
        
        if phase_diffs:
            phase_stability = 1.0 / (1.0 + np.var(phase_diffs))
            self.balancer_metrics["phase_lock_stability"] = phase_stability
    
    def _calculate_load_balancing_efficiency(self,
                                           current_loads: Dict[str, float],
                                           optimized_loads: Dict[str, float],
                                           target_loads: Dict[str, float]) -> float:
        """Calculate efficiency improvement from interference-based optimization."""
        
        # Calculate variance before and after optimization
        current_variance = np.var(list(current_loads.values())) if current_loads else 0
        optimized_variance = np.var(list(optimized_loads.values())) if optimized_loads else 0
        
        # Calculate distance to target before and after
        current_distance = 0.0
        optimized_distance = 0.0
        
        for node_id in current_loads:
            target = target_loads.get(node_id, current_loads[node_id])
            current_distance += (current_loads[node_id] - target) ** 2
            optimized_distance += (optimized_loads[node_id] - target) ** 2
        
        # Efficiency improvement (reduction in distance to target)
        if current_distance > 0:
            distance_improvement = (current_distance - optimized_distance) / current_distance
        else:
            distance_improvement = 0.0
        
        # Variance improvement (reduction in load imbalance)
        if current_variance > 0:
            variance_improvement = (current_variance - optimized_variance) / current_variance
        else:
            variance_improvement = 0.0
        
        # Combined efficiency metric
        efficiency = 0.6 * distance_improvement + 0.4 * variance_improvement
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_interference_gain(self) -> float:
        """Calculate overall interference gain from all patterns."""
        
        if not self.task_interferences:
            return 0.0
        
        total_gain = 0.0
        for interference in self.task_interferences.values():
            # Gain based on interference strength and visibility
            gain = interference.interference_strength * interference.visibility
            total_gain += gain
        
        return total_gain / len(self.task_interferences)
    
    async def evolve_interference_patterns(self, time_step: float = 1.0):
        """Evolve interference patterns over time."""
        
        current_time = time.time()
        
        for interference_id, interference in self.task_interferences.items():
            # Time evolution of phase relationships
            for node_id in interference.phase_relationships:
                frequency = interference.frequency_spectrum.get(node_id, 1.0)
                
                # Phase evolution: φ(t) = φ₀ + ωt
                phase_evolution = frequency * time_step
                interference.phase_relationships[node_id] += phase_evolution
                
                # Apply dispersion
                dispersion_shift = interference.dispersion_coefficient * time_step * frequency
                interference.phase_relationships[node_id] += dispersion_shift
                
                # Keep phase in [0, 2π]
                interference.phase_relationships[node_id] %= (2 * math.pi)
            
            # Update amplitude pattern based on new phases
            for node_id in interference.amplitude_pattern:
                amplitude_magnitude = abs(interference.amplitude_pattern[node_id])
                new_phase = interference.phase_relationships[node_id]
                interference.amplitude_pattern[node_id] = amplitude_magnitude * cmath.exp(1j * new_phase)
            
            # Update interference strength measurement
            if interference_id in self.interference_patterns:
                pattern = self.interference_patterns[interference_id]
                max_intensity = np.max(np.abs(pattern) ** 2)
                min_intensity = np.min(np.abs(pattern) ** 2)
                
                if max_intensity + min_intensity > 0:
                    new_strength = (max_intensity - min_intensity) / (max_intensity + min_intensity)
                    interference.interference_strength = new_strength
                    
                    # Record measurement
                    interference.measurement_history.append((current_time, new_strength))
                    interference.last_measurement = current_time
                    
                    # Limit history size
                    if len(interference.measurement_history) > 100:
                        interference.measurement_history = interference.measurement_history[-50:]
    
    def get_interference_statistics(self) -> Dict[str, Any]:
        """Get comprehensive interference balancing statistics."""
        
        active_interferences = len(self.task_interferences)
        total_nodes_in_interference = len(self.node_wave_states)
        
        # Calculate interference strength distribution
        interference_strengths = [intf.interference_strength for intf in self.task_interferences.values()]
        
        # Phase coherence statistics
        phase_spreads = []
        for interference in self.task_interferences.values():
            phases = list(interference.phase_relationships.values())
            if len(phases) > 1:
                phase_spread = max(phases) - min(phases)
                phase_spreads.append(phase_spread)
        
        # Standing wave analysis
        total_standing_wave_nodes = sum(len(nodes) for nodes in self.standing_wave_nodes.values())
        
        return {
            "active_interferences": active_interferences,
            "nodes_in_interference": total_nodes_in_interference,
            "interference_strength_stats": {
                "mean": np.mean(interference_strengths) if interference_strengths else 0.0,
                "std": np.std(interference_strengths) if interference_strengths else 0.0,
                "max": np.max(interference_strengths) if interference_strengths else 0.0
            },
            "phase_coherence_stats": {
                "mean_phase_spread": np.mean(phase_spreads) if phase_spreads else 0.0,
                "phase_synchronization_quality": self.balancer_metrics["average_coherence"]
            },
            "standing_wave_nodes": total_standing_wave_nodes,
            "balancer_metrics": self.balancer_metrics,
            "recent_optimizations": len(self.interference_history),
            "global_wave_function_magnitude": abs(self.global_wave_function),
            "global_wave_function_phase": cmath.phase(self.global_wave_function)
        }