"""
Quantum-Inspired Task Planner for Federated Healthcare LLM Systems

Implements quantum-inspired algorithms for optimal task distribution across
federated nodes while maintaining privacy constraints and maximizing efficiency.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels inspired by quantum energy states."""
    CRITICAL = 0      # Ground state - highest priority (emergency care)
    HIGH = 1          # First excited state (urgent diagnostics) 
    MEDIUM = 2        # Second excited state (routine analysis)
    LOW = 3           # Third excited state (research queries)
    BACKGROUND = 4    # Highest excited state (maintenance tasks)


class QuantumState(Enum):
    """Quantum-inspired task states."""
    SUPERPOSITION = "superposition"    # Task can be assigned to multiple nodes
    ENTANGLED = "entangled"           # Task dependent on other tasks
    COLLAPSED = "collapsed"           # Task assigned to specific node
    DECOHERENT = "decoherent"         # Task failed/cancelled


@dataclass
class QuantumTask:
    """Represents a task in the quantum-inspired planning system."""
    task_id: str
    user_id: str
    prompt: str
    priority: TaskPriority
    privacy_budget: float
    estimated_duration: float
    resource_requirements: Dict[str, float]
    dependencies: Set[str] = field(default_factory=set)
    
    # Quantum properties
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    entangled_tasks: Set[str] = field(default_factory=set)
    coherence_time: float = 300.0  # 5 minutes default
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    department: Optional[str] = None
    medical_specialty: Optional[str] = None
    urgency_score: float = 1.0


@dataclass 
class NodeQuantumState:
    """Quantum state representation of a federated node."""
    node_id: str
    current_load: float
    privacy_budget_available: float
    computational_capacity: Dict[str, float]
    task_affinity_scores: Dict[TaskPriority, float]
    quantum_coherence: float = 1.0  # Node stability measure
    last_measurement: float = field(default_factory=time.time)


class QuantumTaskPlanner:
    """
    Quantum-inspired task planner for federated healthcare LLM systems.
    
    Uses principles from quantum mechanics to optimize task distribution:
    - Superposition: Tasks exist in multiple potential node assignments
    - Entanglement: Related tasks are processed together optimally
    - Interference: Task scheduling considers constructive/destructive patterns
    - Measurement: Task assignment collapses superposition to definite state
    """
    
    def __init__(self, 
                 privacy_accountant,
                 coherence_threshold: float = 0.8,
                 max_entangled_tasks: int = 5):
        self.privacy_accountant = privacy_accountant
        self.coherence_threshold = coherence_threshold
        self.max_entangled_tasks = max_entangled_tasks
        
        # Core data structures
        self.quantum_tasks: Dict[str, QuantumTask] = {}
        self.node_states: Dict[str, NodeQuantumState] = {}
        self.task_queue = deque()
        self.entanglement_matrix: Dict[Tuple[str, str], float] = {}
        
        # Performance tracking
        self.planning_history: List[Dict[str, Any]] = []
        self.optimization_metrics = {
            "total_tasks_planned": 0,
            "average_planning_time": 0.0,
            "resource_utilization": 0.0,
            "privacy_efficiency": 0.0
        }
        
        # Quantum algorithm parameters
        self.wave_function_decay_rate = 0.1  # How fast superposition decays
        self.entanglement_strength = 0.7     # Coupling strength between tasks
        self.interference_threshold = 0.5    # Minimum interference for optimization
        
    def register_node(self, node_id: str, capabilities: Dict[str, Any]):
        """Register a federated node with quantum state tracking."""
        self.node_states[node_id] = NodeQuantumState(
            node_id=node_id,
            current_load=capabilities.get('current_load', 0.0),
            privacy_budget_available=capabilities.get('privacy_budget', 100.0),
            computational_capacity=capabilities.get('compute_capacity', {}),
            task_affinity_scores={
                TaskPriority.CRITICAL: capabilities.get('critical_affinity', 1.0),
                TaskPriority.HIGH: capabilities.get('high_affinity', 0.9),
                TaskPriority.MEDIUM: capabilities.get('medium_affinity', 0.8),
                TaskPriority.LOW: capabilities.get('low_affinity', 0.7),
                TaskPriority.BACKGROUND: capabilities.get('background_affinity', 0.5),
            }
        )
        logger.info(f"Registered quantum node: {node_id}")
    
    async def add_task(self, task_data: Dict[str, Any]) -> str:
        """Add a new task to the quantum planning system."""
        task = QuantumTask(
            task_id=task_data['task_id'],
            user_id=task_data['user_id'],
            prompt=task_data['prompt'],
            priority=TaskPriority(task_data.get('priority', 2)),
            privacy_budget=task_data.get('privacy_budget', 1.0),
            estimated_duration=task_data.get('estimated_duration', 30.0),
            resource_requirements=task_data.get('resource_requirements', {}),
            dependencies=set(task_data.get('dependencies', [])),
            department=task_data.get('department'),
            medical_specialty=task_data.get('medical_specialty'),
            urgency_score=task_data.get('urgency_score', 1.0)
        )
        
        # Initialize superposition over available nodes
        await self._initialize_superposition(task)
        
        # Check for entanglement opportunities
        await self._detect_entanglements(task)
        
        self.quantum_tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        
        logger.info(f"Added quantum task {task.task_id} in {task.quantum_state.value} state")
        return task.task_id
    
    async def _initialize_superposition(self, task: QuantumTask):
        """Initialize task superposition over available nodes."""
        if not self.node_states:
            logger.warning("No nodes available for superposition")
            return
            
        total_suitability = 0.0
        node_suitabilities = {}
        
        for node_id, node_state in self.node_states.items():
            # Calculate suitability based on multiple factors
            priority_affinity = node_state.task_affinity_scores.get(task.priority, 0.5)
            load_factor = 1.0 - node_state.current_load
            privacy_factor = min(1.0, node_state.privacy_budget_available / task.privacy_budget)
            coherence_factor = node_state.quantum_coherence
            
            suitability = (priority_affinity * 0.3 + 
                          load_factor * 0.3 + 
                          privacy_factor * 0.25 + 
                          coherence_factor * 0.15)
            
            if suitability > 0.1:  # Minimum threshold for inclusion
                node_suitabilities[node_id] = suitability
                total_suitability += suitability
        
        # Normalize to probability distribution
        if total_suitability > 0:
            task.probability_distribution = {
                node_id: suitability / total_suitability
                for node_id, suitability in node_suitabilities.items()
            }
        else:
            # Fallback: uniform distribution
            num_nodes = len(self.node_states)
            task.probability_distribution = {
                node_id: 1.0 / num_nodes
                for node_id in self.node_states.keys()
            }
    
    async def _detect_entanglements(self, new_task: QuantumTask):
        """Detect potential task entanglements for optimization."""
        for task_id, existing_task in self.quantum_tasks.items():
            if existing_task.quantum_state == QuantumState.COLLAPSED:
                continue
                
            entanglement_score = self._calculate_entanglement_score(new_task, existing_task)
            
            if entanglement_score > self.entanglement_strength:
                # Create entanglement
                new_task.entangled_tasks.add(task_id)
                existing_task.entangled_tasks.add(new_task.task_id)
                self.entanglement_matrix[(new_task.task_id, task_id)] = entanglement_score
                self.entanglement_matrix[(task_id, new_task.task_id)] = entanglement_score
                
                logger.debug(f"Entangled tasks {new_task.task_id} and {task_id} "
                           f"with strength {entanglement_score:.3f}")
    
    def _calculate_entanglement_score(self, task1: QuantumTask, task2: QuantumTask) -> float:
        """Calculate entanglement score between two tasks."""
        score = 0.0
        
        # Same user
        if task1.user_id == task2.user_id:
            score += 0.3
            
        # Same department
        if task1.department and task1.department == task2.department:
            score += 0.2
            
        # Similar priority levels
        priority_diff = abs(task1.priority.value - task2.priority.value)
        score += 0.2 * (1.0 - priority_diff / 4.0)
        
        # Similar resource requirements
        resource_similarity = self._calculate_resource_similarity(
            task1.resource_requirements, task2.resource_requirements
        )
        score += 0.3 * resource_similarity
        
        return min(1.0, score)
    
    def _calculate_resource_similarity(self, req1: Dict[str, float], req2: Dict[str, float]) -> float:
        """Calculate similarity between resource requirements."""
        if not req1 or not req2:
            return 0.0
            
        all_resources = set(req1.keys()) | set(req2.keys())
        if not all_resources:
            return 1.0
            
        similarity = 0.0
        for resource in all_resources:
            val1 = req1.get(resource, 0.0)
            val2 = req2.get(resource, 0.0)
            max_val = max(val1, val2)
            if max_val > 0:
                similarity += 1.0 - abs(val1 - val2) / max_val
                
        return similarity / len(all_resources)
    
    async def plan_optimal_assignments(self) -> List[Dict[str, Any]]:
        """Generate optimal task assignments using quantum-inspired algorithms."""
        planning_start = time.time()
        
        if not self.task_queue:
            return []
            
        # Update node quantum states
        await self._update_node_coherence()
        
        # Process tasks in quantum batches
        assignments = []
        processed_tasks = set()
        
        while self.task_queue and len(processed_tasks) < 100:  # Batch limit
            task_id = self.task_queue.popleft()
            
            if task_id in processed_tasks or task_id not in self.quantum_tasks:
                continue
                
            task = self.quantum_tasks[task_id]
            
            # Check if task has lost coherence
            if self._has_lost_coherence(task):
                await self._handle_decoherence(task)
                continue
            
            # Process entangled group together
            if task.entangled_tasks and task.quantum_state == QuantumState.SUPERPOSITION:
                group_assignments = await self._process_entangled_group(task)
                assignments.extend(group_assignments)
                processed_tasks.update([a['task_id'] for a in group_assignments])
            else:
                # Process individual task
                assignment = await self._collapse_wave_function(task)
                if assignment:
                    assignments.append(assignment)
                    processed_tasks.add(task_id)
        
        # Update metrics
        planning_time = time.time() - planning_start
        self._update_planning_metrics(assignments, planning_time)
        
        logger.info(f"Generated {len(assignments)} optimal assignments in {planning_time:.3f}s")
        return assignments
    
    async def _update_node_coherence(self):
        """Update quantum coherence of all nodes."""
        current_time = time.time()
        
        for node_state in self.node_states.values():
            time_since_measurement = current_time - node_state.last_measurement
            
            # Coherence decays exponentially over time
            decay_factor = np.exp(-self.wave_function_decay_rate * time_since_measurement)
            node_state.quantum_coherence *= decay_factor
            
            # Clamp coherence to reasonable bounds
            node_state.quantum_coherence = max(0.1, min(1.0, node_state.quantum_coherence))
    
    def _has_lost_coherence(self, task: QuantumTask) -> bool:
        """Check if task has lost quantum coherence."""
        if task.quantum_state == QuantumState.DECOHERENT:
            return True
            
        time_elapsed = time.time() - task.created_at
        return time_elapsed > task.coherence_time
    
    async def _handle_decoherence(self, task: QuantumTask):
        """Handle task that has lost coherence."""
        task.quantum_state = QuantumState.DECOHERENT
        
        # Break entanglements
        for entangled_id in task.entangled_tasks:
            if entangled_id in self.quantum_tasks:
                self.quantum_tasks[entangled_id].entangled_tasks.discard(task.task_id)
        
        task.entangled_tasks.clear()
        logger.warning(f"Task {task.task_id} has lost coherence and become decoherent")
    
    async def _process_entangled_group(self, anchor_task: QuantumTask) -> List[Dict[str, Any]]:
        """Process a group of entangled tasks together for optimal assignment."""
        entangled_group = [anchor_task.task_id] + list(anchor_task.entangled_tasks)
        entangled_group = [tid for tid in entangled_group if tid in self.quantum_tasks]
        
        if len(entangled_group) > self.max_entangled_tasks:
            # Limit group size to prevent exponential complexity
            entangled_group = entangled_group[:self.max_entangled_tasks]
        
        # Find optimal node assignment for the entire group
        best_assignment = await self._optimize_group_assignment(entangled_group)
        
        assignments = []
        for task_id, node_id in best_assignment.items():
            task = self.quantum_tasks[task_id]
            
            assignment = await self._create_assignment(task, node_id)
            if assignment:
                assignments.append(assignment)
                task.quantum_state = QuantumState.COLLAPSED
        
        return assignments
    
    async def _optimize_group_assignment(self, task_ids: List[str]) -> Dict[str, str]:
        """Optimize node assignment for a group of entangled tasks."""
        if not task_ids:
            return {}
        
        # Use simulated annealing for optimization
        current_assignment = self._generate_random_assignment(task_ids)
        current_cost = await self._calculate_assignment_cost(current_assignment)
        
        best_assignment = current_assignment.copy()
        best_cost = current_cost
        
        temperature = 1.0
        cooling_rate = 0.95
        min_temperature = 0.01
        
        while temperature > min_temperature:
            # Generate neighbor solution
            neighbor_assignment = self._generate_neighbor_assignment(current_assignment)
            neighbor_cost = await self._calculate_assignment_cost(neighbor_assignment)
            
            # Accept or reject based on simulated annealing criterion
            if (neighbor_cost < current_cost or 
                np.random.random() < np.exp(-(neighbor_cost - current_cost) / temperature)):
                current_assignment = neighbor_assignment
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_assignment = current_assignment.copy()
                    best_cost = current_cost
            
            temperature *= cooling_rate
        
        return best_assignment
    
    def _generate_random_assignment(self, task_ids: List[str]) -> Dict[str, str]:
        """Generate random valid assignment for tasks."""
        assignment = {}
        available_nodes = list(self.node_states.keys())
        
        for task_id in task_ids:
            if task_id in self.quantum_tasks:
                task = self.quantum_tasks[task_id]
                # Choose from nodes with non-zero probability
                valid_nodes = [node for node, prob in task.probability_distribution.items() if prob > 0]
                if valid_nodes:
                    assignment[task_id] = np.random.choice(valid_nodes)
                elif available_nodes:
                    assignment[task_id] = np.random.choice(available_nodes)
        
        return assignment
    
    def _generate_neighbor_assignment(self, current_assignment: Dict[str, str]) -> Dict[str, str]:
        """Generate neighbor assignment by changing one task's node."""
        neighbor = current_assignment.copy()
        
        if not neighbor:
            return neighbor
            
        # Pick random task to reassign
        task_id = np.random.choice(list(neighbor.keys()))
        task = self.quantum_tasks.get(task_id)
        
        if task:
            valid_nodes = [node for node, prob in task.probability_distribution.items() if prob > 0]
            if valid_nodes:
                neighbor[task_id] = np.random.choice(valid_nodes)
        
        return neighbor
    
    async def _calculate_assignment_cost(self, assignment: Dict[str, str]) -> float:
        """Calculate cost of a task assignment."""
        total_cost = 0.0
        node_loads = defaultdict(float)
        
        for task_id, node_id in assignment.items():
            task = self.quantum_tasks.get(task_id)
            node = self.node_states.get(node_id)
            
            if not task or not node:
                total_cost += 1000.0  # Heavy penalty for invalid assignment
                continue
            
            # Load balancing cost
            node_loads[node_id] += task.estimated_duration
            load_cost = node_loads[node_id] ** 2  # Quadratic penalty for overloading
            
            # Privacy budget cost
            privacy_cost = max(0, task.privacy_budget - node.privacy_budget_available) * 10
            
            # Priority cost (higher priority tasks should get better nodes)
            priority_weight = 5 - task.priority.value  # Critical=5, Background=1
            node_suitability = task.probability_distribution.get(node_id, 0.1)
            priority_cost = priority_weight * (1.0 - node_suitability)
            
            total_cost += load_cost + privacy_cost + priority_cost
        
        # Add entanglement violation cost
        entanglement_cost = self._calculate_entanglement_cost(assignment)
        total_cost += entanglement_cost
        
        return total_cost
    
    def _calculate_entanglement_cost(self, assignment: Dict[str, str]) -> float:
        """Calculate cost of violating entanglement constraints."""
        cost = 0.0
        
        for (task1_id, task2_id), strength in self.entanglement_matrix.items():
            if task1_id in assignment and task2_id in assignment:
                node1 = assignment[task1_id]
                node2 = assignment[task2_id]
                
                # Penalty for separating entangled tasks
                if node1 != node2:
                    cost += strength * 5.0  # Penalty proportional to entanglement strength
        
        return cost
    
    async def _collapse_wave_function(self, task: QuantumTask) -> Optional[Dict[str, Any]]:
        """Collapse task superposition to definite node assignment."""
        if not task.probability_distribution:
            return None
        
        # Weighted random selection based on probability distribution
        nodes = list(task.probability_distribution.keys())
        probabilities = list(task.probability_distribution.values())
        
        selected_node = np.random.choice(nodes, p=probabilities)
        
        return await self._create_assignment(task, selected_node)
    
    async def _create_assignment(self, task: QuantumTask, node_id: str) -> Optional[Dict[str, Any]]:
        """Create final task assignment."""
        node = self.node_states.get(node_id)
        if not node:
            return None
        
        # Check constraints
        if not self.privacy_accountant.check_budget(task.user_id, task.privacy_budget):
            logger.warning(f"Insufficient privacy budget for task {task.task_id}")
            return None
        
        # Update node state
        node.current_load += task.estimated_duration / 100.0  # Normalize
        node.privacy_budget_available -= task.privacy_budget
        node.last_measurement = time.time()
        
        # Mark task as collapsed
        task.quantum_state = QuantumState.COLLAPSED
        
        assignment = {
            "task_id": task.task_id,
            "node_id": node_id,
            "user_id": task.user_id,
            "priority": task.priority.value,
            "estimated_duration": task.estimated_duration,
            "privacy_budget": task.privacy_budget,
            "assignment_probability": task.probability_distribution.get(node_id, 0.0),
            "entangled_tasks": list(task.entangled_tasks),
            "assignment_timestamp": time.time()
        }
        
        return assignment
    
    def _update_planning_metrics(self, assignments: List[Dict[str, Any]], planning_time: float):
        """Update performance metrics."""
        self.optimization_metrics["total_tasks_planned"] += len(assignments)
        
        # Update average planning time
        total_tasks = self.optimization_metrics["total_tasks_planned"]
        current_avg = self.optimization_metrics["average_planning_time"]
        self.optimization_metrics["average_planning_time"] = (
            (current_avg * (total_tasks - len(assignments)) + planning_time * len(assignments)) / total_tasks
        )
        
        # Calculate resource utilization
        if self.node_states:
            total_load = sum(node.current_load for node in self.node_states.values())
            self.optimization_metrics["resource_utilization"] = total_load / len(self.node_states)
        
        # Store planning event
        self.planning_history.append({
            "timestamp": time.time(),
            "assignments_count": len(assignments),
            "planning_time": planning_time,
            "average_assignment_probability": np.mean([a.get("assignment_probability", 0) for a in assignments]) if assignments else 0
        })
        
        # Keep history size manageable
        if len(self.planning_history) > 1000:
            self.planning_history = self.planning_history[-500:]
    
    def get_quantum_statistics(self) -> Dict[str, Any]:
        """Get comprehensive quantum planning statistics."""
        active_tasks = len([t for t in self.quantum_tasks.values() 
                           if t.quantum_state != QuantumState.DECOHERENT])
        
        superposition_tasks = len([t for t in self.quantum_tasks.values() 
                                  if t.quantum_state == QuantumState.SUPERPOSITION])
        
        entangled_tasks = len([t for t in self.quantum_tasks.values() 
                              if t.entangled_tasks])
        
        collapsed_tasks = len([t for t in self.quantum_tasks.values() 
                              if t.quantum_state == QuantumState.COLLAPSED])
        
        return {
            "active_tasks": active_tasks,
            "superposition_tasks": superposition_tasks,
            "entangled_tasks": entangled_tasks,
            "collapsed_tasks": collapsed_tasks,
            "decoherent_tasks": len(self.quantum_tasks) - active_tasks,
            "entanglement_pairs": len(self.entanglement_matrix) // 2,
            "average_node_coherence": np.mean([node.quantum_coherence for node in self.node_states.values()]) if self.node_states else 0,
            "total_planning_events": len(self.planning_history),
            **self.optimization_metrics
        }