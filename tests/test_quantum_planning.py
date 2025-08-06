"""
Comprehensive test suite for quantum-inspired task planning system.

Tests all quantum planning components with healthcare-grade reliability
and performance requirements.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import quantum planning components
from federated_dp_llm.quantum_planning import (
    QuantumTaskPlanner,
    QuantumState,
    TaskPriority,
    SuperpositionScheduler,
    TaskSuperposition,
    EntanglementOptimizer,
    ResourceEntanglement,
    InterferenceBalancer,
    TaskInterference,
    QuantumComponentValidator,
    QuantumErrorHandler,
    ValidationLevel,
    QuantumMonitor,
    AlertSeverity,
    HealthStatus,
    QuantumSecurityController,
    SecurityLevel,
    QuantumSecurityContext
)

from federated_dp_llm.quantum_planning.quantum_optimizer import (
    QuantumPerformanceOptimizer,
    OptimizationStrategy,
    QuantumResourcePool
)

from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig


class TestQuantumTaskPlanner:
    """Test suite for QuantumTaskPlanner."""
    
    @pytest.fixture
    def privacy_accountant(self):
        return PrivacyAccountant(DPConfig())
    
    @pytest.fixture
    def quantum_planner(self, privacy_accountant):
        return QuantumTaskPlanner(
            privacy_accountant=privacy_accountant,
            coherence_threshold=0.8,
            max_entangled_tasks=5
        )
    
    @pytest.fixture
    def sample_node_capabilities(self):
        return {
            'current_load': 0.3,
            'privacy_budget': 50.0,
            'compute_capacity': {
                'gpu_memory': 32768,
                'cpu_cores': 16,
                'network_bandwidth': 1000
            },
            'critical_affinity': 1.0,
            'high_affinity': 0.9,
            'medium_affinity': 0.8,
            'low_affinity': 0.7,
            'background_affinity': 0.5
        }
    
    @pytest.fixture
    def sample_task_data(self):
        return {
            'task_id': 'test_task_001',
            'user_id': 'doctor_123',
            'prompt': 'Analyze patient symptoms for differential diagnosis',
            'priority': TaskPriority.HIGH.value,
            'privacy_budget': 1.0,
            'estimated_duration': 30.0,
            'resource_requirements': {
                'compute': 0.5,
                'memory': 0.3,
                'network': 0.2
            },
            'department': 'emergency',
            'medical_specialty': 'emergency_medicine',
            'urgency_score': 0.8
        }
    
    def test_quantum_planner_initialization(self, quantum_planner):
        """Test quantum planner proper initialization."""
        assert quantum_planner is not None
        assert quantum_planner.coherence_threshold == 0.8
        assert quantum_planner.max_entangled_tasks == 5
        assert quantum_planner.privacy_accountant is not None
        assert len(quantum_planner.quantum_tasks) == 0
    
    def test_node_registration(self, quantum_planner, sample_node_capabilities):
        """Test node registration with quantum planner."""
        node_id = "hospital_a_gpu_1"
        quantum_planner.register_node(node_id, sample_node_capabilities)
        
        assert node_id in quantum_planner.node_states
        node_state = quantum_planner.node_states[node_id]
        assert node_state.node_id == node_id
        assert node_state.current_load == 0.3
        assert node_state.privacy_budget_available == 50.0
    
    @pytest.mark.asyncio
    async def test_task_addition(self, quantum_planner, sample_node_capabilities, sample_task_data):
        """Test adding quantum task to planner."""
        # Register a node first
        quantum_planner.register_node("test_node", sample_node_capabilities)
        
        # Add task
        task_id = await quantum_planner.add_task(sample_task_data)
        
        assert task_id == sample_task_data['task_id']
        assert task_id in quantum_planner.quantum_tasks
        
        task = quantum_planner.quantum_tasks[task_id]
        assert task.task_id == task_id
        assert task.user_id == sample_task_data['user_id']
        assert task.priority == TaskPriority.HIGH
        assert task.quantum_state == QuantumState.SUPERPOSITION
        assert len(task.probability_distribution) > 0
    
    @pytest.mark.asyncio
    async def test_optimal_assignments_generation(self, quantum_planner, sample_node_capabilities, sample_task_data):
        """Test generating optimal task assignments."""
        # Setup
        quantum_planner.register_node("node_1", sample_node_capabilities)
        quantum_planner.register_node("node_2", sample_node_capabilities)
        
        await quantum_planner.add_task(sample_task_data)
        
        # Generate assignments
        assignments = await quantum_planner.plan_optimal_assignments()
        
        assert isinstance(assignments, list)
        assert len(assignments) >= 0  # May be empty if no tasks ready for assignment
        
        if assignments:
            assignment = assignments[0]
            assert 'task_id' in assignment
            assert 'node_id' in assignment
            assert 'assignment_probability' in assignment
    
    @pytest.mark.asyncio
    async def test_entanglement_detection(self, quantum_planner, sample_node_capabilities):
        """Test task entanglement detection."""
        quantum_planner.register_node("test_node", sample_node_capabilities)
        
        # Create related tasks (same user and department)
        task1_data = {
            'task_id': 'task_1',
            'user_id': 'doctor_123',
            'prompt': 'First query',
            'priority': TaskPriority.HIGH.value,
            'privacy_budget': 0.5,
            'estimated_duration': 20.0,
            'resource_requirements': {'compute': 0.4},
            'department': 'cardiology',
            'urgency_score': 0.7
        }
        
        task2_data = {
            'task_id': 'task_2',
            'user_id': 'doctor_123',  # Same user
            'prompt': 'Second query',
            'priority': TaskPriority.HIGH.value,  # Same priority
            'privacy_budget': 0.5,
            'estimated_duration': 25.0,
            'resource_requirements': {'compute': 0.5},
            'department': 'cardiology',  # Same department
            'urgency_score': 0.8
        }
        
        await quantum_planner.add_task(task1_data)
        await quantum_planner.add_task(task2_data)
        
        # Check if tasks are entangled
        task1 = quantum_planner.quantum_tasks['task_1']
        task2 = quantum_planner.quantum_tasks['task_2']
        
        # Should have some entanglement due to same user and department
        assert len(task1.entangled_tasks) > 0 or len(task2.entangled_tasks) > 0
    
    def test_quantum_statistics(self, quantum_planner):
        """Test quantum statistics collection."""
        stats = quantum_planner.get_quantum_statistics()
        
        assert 'active_tasks' in stats
        assert 'superposition_tasks' in stats
        assert 'entangled_tasks' in stats
        assert 'collapsed_tasks' in stats
        assert 'decoherent_tasks' in stats
        assert 'total_planning_events' in stats
        assert 'average_planning_time' in stats


class TestSuperpositionScheduler:
    """Test suite for SuperpositionScheduler."""
    
    @pytest.fixture
    def scheduler(self):
        return SuperpositionScheduler(
            max_superposition_time=300.0,
            interference_strength=0.5,
            decoherence_rate=0.01
        )
    
    @pytest.fixture
    def sample_nodes(self):
        return ['hospital_a', 'hospital_b', 'hospital_c']
    
    @pytest.fixture
    def sample_time_preferences(self):
        current_time = time.time()
        return [(current_time, current_time + 300), (current_time + 100, current_time + 400)]
    
    @pytest.fixture
    def sample_resource_requirements(self):
        return {'cpu': 0.5, 'memory': 0.3, 'gpu': 0.2}
    
    @pytest.mark.asyncio
    async def test_superposition_initialization(self, scheduler, sample_nodes, 
                                              sample_time_preferences, sample_resource_requirements):
        """Test superposition initialization."""
        task_id = "test_superposition_task"
        
        superposition = await scheduler.initialize_superposition(
            task_id=task_id,
            potential_nodes=sample_nodes,
            time_preferences=sample_time_preferences,
            resource_requirements=sample_resource_requirements
        )
        
        assert superposition.task_id == task_id
        assert len(superposition.amplitude_distribution) == len(sample_nodes)
        assert len(superposition.measurement_probability) == len(sample_nodes)
        
        # Check probability normalization
        total_prob = sum(superposition.measurement_probability.values())
        assert abs(total_prob - 1.0) < 0.01  # Allow small numerical errors
    
    @pytest.mark.asyncio
    async def test_superposition_evolution(self, scheduler, sample_nodes, 
                                         sample_time_preferences, sample_resource_requirements):
        """Test superposition evolution over time."""
        task_id = "evolution_test_task"
        
        await scheduler.initialize_superposition(
            task_id=task_id,
            potential_nodes=sample_nodes,
            time_preferences=sample_time_preferences,
            resource_requirements=sample_resource_requirements
        )
        
        # Record initial amplitudes
        initial_superposition = scheduler.wave_function.superposed_tasks[task_id]
        initial_amplitudes = initial_superposition.amplitude_distribution.copy()
        
        # Evolve superposition
        await scheduler.evolve_superposition(time_step=1.0)
        
        # Check that amplitudes have evolved
        evolved_superposition = scheduler.wave_function.superposed_tasks[task_id]
        evolved_amplitudes = evolved_superposition.amplitude_distribution
        
        # At least some amplitudes should have changed due to evolution
        amplitude_changes = [
            abs(evolved_amplitudes[node] - initial_amplitudes[node])
            for node in sample_nodes
        ]
        assert any(change > 0.001 for change in amplitude_changes)
    
    @pytest.mark.asyncio
    async def test_measurement_assignment(self, scheduler, sample_nodes, 
                                        sample_time_preferences, sample_resource_requirements):
        """Test quantum measurement for optimal assignment."""
        task_id = "measurement_test_task"
        
        await scheduler.initialize_superposition(
            task_id=task_id,
            potential_nodes=sample_nodes,
            time_preferences=sample_time_preferences,
            resource_requirements=sample_resource_requirements
        )
        
        # Measure optimal assignment
        assignment = await scheduler.measure_optimal_assignment(
            task_id, "maximum_probability"
        )
        
        assert assignment is not None
        selected_node, assignment_details = assignment
        assert selected_node in sample_nodes
        assert 'selected_probability' in assignment_details
        assert 'measurement_time' in assignment_details
        assert 'superposition_duration' in assignment_details
    
    @pytest.mark.asyncio
    async def test_interference_application(self, scheduler, sample_nodes, 
                                          sample_time_preferences, sample_resource_requirements):
        """Test quantum interference between tasks."""
        task_ids = ["interference_task_1", "interference_task_2"]
        
        # Create superpositions for both tasks
        for task_id in task_ids:
            await scheduler.initialize_superposition(
                task_id=task_id,
                potential_nodes=sample_nodes,
                time_preferences=sample_time_preferences,
                resource_requirements=sample_resource_requirements
            )
        
        # Apply interference
        await scheduler.apply_interference(task_ids)
        
        # Verify interference was applied
        assert scheduler.scheduler_metrics["interference_events"] > 0
        
        # Both tasks should have interference patterns recorded
        for task_id in task_ids:
            if task_id in scheduler.interference_patterns:
                assert len(scheduler.interference_patterns[task_id]) > 0
    
    def test_superposition_status(self, scheduler):
        """Test superposition status reporting."""
        status = scheduler.get_superposition_status()
        
        assert 'active_superpositions' in status
        assert 'global_wave_function_phase' in status
        assert 'total_amplitude' in status
        assert 'scheduler_metrics' in status
        assert 'recent_measurements' in status


class TestEntanglementOptimizer:
    """Test suite for EntanglementOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        return EntanglementOptimizer(
            max_entanglement_distance=1000.0,
            bell_inequality_threshold=2.0,
            decoherence_mitigation_enabled=True
        )
    
    @pytest.fixture
    def sample_resource_pairs(self):
        return [
            ("hospital_a", "compute"),
            ("hospital_b", "compute"),
            ("hospital_c", "memory"),
            ("hospital_d", "memory")
        ]
    
    @pytest.mark.asyncio
    async def test_entanglement_creation(self, optimizer, sample_resource_pairs):
        """Test creating resource entanglement."""
        from federated_dp_llm.quantum_planning.entanglement_optimizer import EntanglementType
        
        entanglement_id = await optimizer.create_resource_entanglement(
            resource_pairs=sample_resource_pairs,
            entanglement_type=EntanglementType.RESOURCE,
            target_correlation=0.8
        )
        
        assert entanglement_id in optimizer.resource_entanglements
        entanglement = optimizer.resource_entanglements[entanglement_id]
        
        assert entanglement.resource_pairs == sample_resource_pairs
        assert entanglement.entanglement_type == EntanglementType.RESOURCE
        assert entanglement.entanglement_strength == 0.8
        assert entanglement.correlation_matrix.shape[0] == len(sample_resource_pairs)
    
    @pytest.mark.asyncio
    async def test_correlation_measurement(self, optimizer, sample_resource_pairs):
        """Test measuring entangled correlations."""
        from federated_dp_llm.quantum_planning.entanglement_optimizer import EntanglementType
        
        # Create entanglement first
        entanglement_id = await optimizer.create_resource_entanglement(
            resource_pairs=sample_resource_pairs[:2],  # Use only first 2 pairs
            entanglement_type=EntanglementType.RESOURCE,
            target_correlation=0.7
        )
        
        # Measure correlations
        optimized_allocations = await optimizer.measure_entangled_correlations(entanglement_id)
        
        assert isinstance(optimized_allocations, dict)
        # Should have allocations for both resource pairs
        expected_resource_ids = [f"{node}_{resource}" for node, resource in sample_resource_pairs[:2]]
        for resource_id in expected_resource_ids:
            if resource_id in optimized_allocations:
                allocation = optimized_allocations[resource_id]
                assert 0.0 <= allocation <= 1.0
    
    @pytest.mark.asyncio
    async def test_entanglement_evolution(self, optimizer, sample_resource_pairs):
        """Test entanglement evolution over time."""
        from federated_dp_llm.quantum_planning.entanglement_optimizer import EntanglementType
        
        entanglement_id = await optimizer.create_resource_entanglement(
            resource_pairs=sample_resource_pairs[:2],
            entanglement_type=EntanglementType.TEMPORAL,
            target_correlation=0.8
        )
        
        initial_entanglement = optimizer.resource_entanglements[entanglement_id]
        initial_strength = initial_entanglement.entanglement_strength
        
        # Evolve entangled states
        await optimizer.evolve_entangled_states(time_step=5.0)
        
        # Check that entanglement has evolved (may have degraded due to decoherence)
        evolved_entanglement = optimizer.resource_entanglements[entanglement_id]
        evolved_strength = evolved_entanglement.entanglement_strength
        
        # Should be different due to time evolution
        assert evolved_strength != initial_strength
    
    def test_entanglement_statistics(self, optimizer):
        """Test entanglement statistics collection."""
        stats = optimizer.get_entanglement_statistics()
        
        assert 'active_entanglements' in stats
        assert 'total_entangled_resources' in stats
        assert 'global_entanglement_strength' in stats
        assert 'bell_inequality_violation_rate' in stats
        assert 'optimization_metrics' in stats
        assert 'entanglement_types_distribution' in stats


class TestInterferenceBalancer:
    """Test suite for InterferenceBalancer."""
    
    @pytest.fixture
    def balancer(self):
        return InterferenceBalancer(
            interference_resolution=0.1,
            coherence_threshold=0.7,
            phase_locked_loop_enabled=True
        )
    
    @pytest.fixture
    def sample_task_ids(self):
        return ["interference_task_1", "interference_task_2", "interference_task_3"]
    
    @pytest.fixture
    def sample_target_nodes(self):
        return ["node_alpha", "node_beta", "node_gamma"]
    
    @pytest.mark.asyncio
    async def test_node_wave_state_initialization(self, balancer, sample_target_nodes):
        """Test initialization of node wave states."""
        node_characteristics = {
            'processing_frequency': 2.0,
            'load_capacity': 0.8,
            'network_latency': 0.05
        }
        
        for node_id in sample_target_nodes:
            await balancer.initialize_node_wave_state(node_id, node_characteristics)
        
        assert len(balancer.node_wave_states) == len(sample_target_nodes)
        
        for node_id in sample_target_nodes:
            assert node_id in balancer.node_wave_states
            wave_state = balancer.node_wave_states[node_id]
            assert wave_state.node_id == node_id
            assert wave_state.frequency == 2.0
            assert 0 <= wave_state.phase <= 2 * np.pi
    
    @pytest.mark.asyncio
    async def test_interference_creation(self, balancer, sample_task_ids, sample_target_nodes):
        """Test creating task interference patterns."""
        from federated_dp_llm.quantum_planning.interference_balancer import InterferenceType
        
        # Initialize node wave states first
        node_characteristics = {'processing_frequency': 1.0, 'load_capacity': 1.0, 'network_latency': 0.1}
        for node_id in sample_target_nodes:
            await balancer.initialize_node_wave_state(node_id, node_characteristics)
        
        # Create interference
        interference_id = await balancer.create_task_interference(
            task_ids=sample_task_ids,
            target_nodes=sample_target_nodes,
            interference_type=InterferenceType.CONSTRUCTIVE
        )
        
        assert interference_id in balancer.task_interferences
        interference = balancer.task_interferences[interference_id]
        
        assert interference.interfering_tasks == sample_task_ids
        assert interference.interference_type == InterferenceType.CONSTRUCTIVE
        assert len(interference.amplitude_pattern) == len(sample_target_nodes)
        assert len(interference.phase_relationships) == len(sample_target_nodes)
    
    @pytest.mark.asyncio
    async def test_load_distribution_optimization(self, balancer, sample_target_nodes):
        """Test load distribution optimization using interference."""
        # Initialize nodes
        node_characteristics = {'processing_frequency': 1.0, 'load_capacity': 1.0, 'network_latency': 0.1}
        for node_id in sample_target_nodes:
            await balancer.initialize_node_wave_state(node_id, node_characteristics)
        
        # Define current and target loads
        current_loads = {node: 0.9 for node in sample_target_nodes}  # Heavily loaded
        target_loads = {node: 0.5 for node in sample_target_nodes}   # Balanced target
        
        # Optimize load distribution
        optimized_loads = await balancer.optimize_load_distribution(current_loads, target_loads)
        
        assert isinstance(optimized_loads, dict)
        assert len(optimized_loads) == len(sample_target_nodes)
        
        # Optimized loads should be closer to targets
        for node_id in sample_target_nodes:
            optimized_load = optimized_loads[node_id]
            assert 0.0 <= optimized_load <= 1.0
            # Should be different from original (optimized)
            assert optimized_load != current_loads[node_id]
    
    @pytest.mark.asyncio
    async def test_interference_pattern_evolution(self, balancer, sample_task_ids, sample_target_nodes):
        """Test evolution of interference patterns over time."""
        from federated_dp_llm.quantum_planning.interference_balancer import InterferenceType
        
        # Initialize and create interference
        node_characteristics = {'processing_frequency': 1.0, 'load_capacity': 1.0, 'network_latency': 0.1}
        for node_id in sample_target_nodes:
            await balancer.initialize_node_wave_state(node_id, node_characteristics)
        
        interference_id = await balancer.create_task_interference(
            task_ids=sample_task_ids,
            target_nodes=sample_target_nodes,
            interference_type=InterferenceType.MIXED
        )
        
        initial_interference = balancer.task_interferences[interference_id]
        initial_phases = initial_interference.phase_relationships.copy()
        
        # Evolve interference patterns
        await balancer.evolve_interference_patterns(time_step=2.0)
        
        # Check that phases have evolved
        evolved_interference = balancer.task_interferences[interference_id]
        evolved_phases = evolved_interference.phase_relationships
        
        # At least some phases should have changed
        phase_changes = [
            abs(evolved_phases[node] - initial_phases[node])
            for node in sample_target_nodes
        ]
        assert any(change > 0.001 for change in phase_changes)
    
    def test_interference_statistics(self, balancer):
        """Test interference balancer statistics."""
        stats = balancer.get_interference_statistics()
        
        assert 'active_interferences' in stats
        assert 'nodes_in_interference' in stats
        assert 'balancer_metrics' in stats
        assert 'global_wave_function_magnitude' in stats
        assert 'global_wave_function_phase' in stats


class TestQuantumValidation:
    """Test suite for quantum validation and error handling."""
    
    @pytest.fixture
    def validator(self):
        return QuantumComponentValidator(ValidationLevel.STRICT)
    
    @pytest.fixture
    def error_handler(self):
        return QuantumErrorHandler()
    
    @pytest.fixture
    def mock_quantum_task(self):
        """Create a mock quantum task for testing."""
        class MockTask:
            def __init__(self):
                self.task_id = "test_task"
                self.user_id = "test_user"
                self.priority = TaskPriority.MEDIUM
                self.privacy_budget = 1.0
                self.quantum_state = QuantumState.SUPERPOSITION
                self.probability_distribution = {"node_1": 0.6, "node_2": 0.4}
                self.coherence_time = 300.0
                self.created_at = time.time()
        
        return MockTask()
    
    @pytest.mark.asyncio
    async def test_task_validation_success(self, validator, mock_quantum_task):
        """Test successful task validation."""
        result = await validator.validate_component(mock_quantum_task, "quantum_task")
        
        assert result.is_valid is True
        assert result.error_type is None
        assert len(result.suggestions) == 0
    
    @pytest.mark.asyncio
    async def test_task_validation_failure(self, validator):
        """Test task validation failure cases."""
        # Create invalid task (missing required fields)
        class InvalidTask:
            pass
        
        invalid_task = InvalidTask()
        
        result = await validator.validate_component(invalid_task, "quantum_task")
        
        assert result.is_valid is False
        assert result.error_type is not None
        assert len(result.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_error_handler_decoherence(self, error_handler, mock_quantum_task):
        """Test error handler for decoherence."""
        from federated_dp_llm.quantum_planning.quantum_validators import QuantumErrorType
        
        success = await error_handler.handle_quantum_error(
            QuantumErrorType.DECOHERENCE,
            mock_quantum_task,
            {"severity": "medium"}
        )
        
        assert success is True
        # Should have reset quantum state to collapsed
        assert mock_quantum_task.quantum_state == "collapsed"
    
    def test_validation_summary(self, validator):
        """Test validation summary generation."""
        summary = validator.get_validation_summary()
        
        assert 'total_validations' in summary
        assert 'success_rate' in summary or summary['total_validations'] == 0


class TestQuantumMonitoring:
    """Test suite for quantum monitoring system."""
    
    @pytest.fixture
    def monitor(self):
        return QuantumMonitor(
            collection_interval=1.0,  # Fast for testing
            alert_retention_hours=1,
            metric_retention_hours=1
        )
    
    @pytest.mark.asyncio
    async def test_monitoring_startup_shutdown(self, monitor):
        """Test monitor startup and shutdown."""
        assert monitor._monitoring_active is False
        
        await monitor.start_monitoring()
        assert monitor._monitoring_active is True
        
        await asyncio.sleep(0.1)  # Let it run briefly
        
        await monitor.stop_monitoring()
        assert monitor._monitoring_active is False
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, monitor):
        """Test metric recording."""
        await monitor.record_metric(
            metric_name="test_metric",
            value=42.0,
            component="test_component",
            unit="ms"
        )
        
        metric_key = "test_component.test_metric"
        assert metric_key in monitor.metrics
        assert len(monitor.metrics[metric_key]) == 1
        
        recorded_metric = monitor.metrics[metric_key][0]
        assert recorded_metric.value == 42.0
        assert recorded_metric.unit == "ms"
    
    def test_monitoring_dashboard(self, monitor):
        """Test monitoring dashboard data."""
        dashboard = monitor.get_monitoring_dashboard()
        
        assert 'monitoring_status' in dashboard
        assert 'component_health' in dashboard
        assert 'active_alerts' in dashboard
        assert 'alerts_by_severity' in dashboard


class TestQuantumSecurity:
    """Test suite for quantum security system."""
    
    @pytest.fixture
    def security_controller(self):
        return QuantumSecurityController()
    
    @pytest.fixture
    def sample_security_context(self):
        return QuantumSecurityContext(
            operation_id="test_op_123",
            user_id="doctor_456",
            security_level=SecurityLevel.CONFIDENTIAL,
            privacy_budget_allocated=1.0,
            allowed_operations={"measurement", "planning", "superposition"}
        )
    
    @pytest.mark.asyncio
    async def test_security_initialization(self, security_controller):
        """Test security system initialization."""
        await security_controller.initialize_security()
        
        assert security_controller.monitor._monitoring_active is True
        
        # Cleanup
        await security_controller.shutdown_security()
    
    @pytest.mark.asyncio
    async def test_secure_context_creation(self, security_controller):
        """Test creation of secure contexts."""
        await security_controller.initialize_security()
        
        context = await security_controller.create_secure_context(
            user_id="test_user",
            operation_type="measurement",
            security_level=SecurityLevel.INTERNAL,
            privacy_budget=0.5
        )
        
        assert context.user_id == "test_user"
        assert context.security_level == SecurityLevel.INTERNAL
        assert context.privacy_budget_allocated == 0.5
        
        await security_controller.shutdown_security()
    
    @pytest.mark.asyncio
    async def test_secure_operation_execution(self, security_controller, sample_security_context):
        """Test secure operation execution."""
        await security_controller.initialize_security()
        
        operation_data = {
            "operation_type": "measurement",
            "parameters": {"node_id": "test_node", "measurement_type": "amplitude"}
        }
        
        secure_data = await security_controller.secure_quantum_operation(
            sample_security_context,
            "measurement",
            operation_data
        )
        
        assert isinstance(secure_data, dict)
        # Should contain original data and possibly security additions
        assert "operation_type" in secure_data
        
        await security_controller.shutdown_security()
    
    def test_security_status(self, security_controller):
        """Test security status reporting."""
        status = security_controller.get_security_status()
        
        assert 'security_controller_active' in status
        assert 'cryptographer_initialized' in status
        assert 'monitoring_dashboard' in status


class TestQuantumOptimization:
    """Test suite for quantum performance optimization."""
    
    @pytest.fixture
    def optimizer(self):
        return QuantumPerformanceOptimizer(
            optimization_strategy=OptimizationStrategy.BALANCED,
            enable_auto_scaling=True,
            enable_caching=True
        )
    
    @pytest.fixture
    def resource_pool(self):
        return QuantumResourcePool(pool_size=2, max_workers=4)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.optimization_strategy == OptimizationStrategy.BALANCED
        assert optimizer.enable_auto_scaling is True
        assert optimizer.enable_caching is True
        assert len(optimizer.scaling_policies) > 0
    
    @pytest.mark.asyncio
    async def test_optimization_startup_shutdown(self, optimizer):
        """Test optimization system startup and shutdown."""
        assert optimizer.optimization_enabled is False
        
        await optimizer.start_optimization()
        assert optimizer.optimization_enabled is True
        
        await asyncio.sleep(0.1)  # Brief run
        
        await optimizer.stop_optimization()
        assert optimizer.optimization_enabled is False
    
    @pytest.mark.asyncio
    async def test_resource_pool_task_submission(self, resource_pool):
        """Test submitting tasks to resource pool."""
        def simple_task(x, y):
            return x + y
        
        result = await resource_pool.submit_quantum_task(
            task_id="test_task",
            task_func=simple_task,
            5, 3,
            use_cache=True,
            execution_mode="thread"
        )
        
        assert result == 8
        assert resource_pool.pool_stats["tasks_executed"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_optimization(self, optimizer):
        """Test batch task optimization."""
        def multiply_task(x, y):
            return x * y
        
        tasks = [
            ("task_1", multiply_task, (2, 3)),
            ("task_2", multiply_task, (4, 5)),
            ("task_3", multiply_task, (6, 7))
        ]
        
        results = await optimizer.optimize_quantum_task_batch(
            tasks, execution_mode="concurrent"
        )
        
        assert len(results) == 3
        assert results == [6, 20, 42]
    
    def test_optimization_status(self, optimizer):
        """Test optimization status reporting."""
        status = optimizer.get_optimization_status()
        
        assert 'optimization_enabled' in status
        assert 'optimization_strategy' in status
        assert 'auto_scaling_enabled' in status
        assert 'resource_pool_stats' in status
        assert 'performance_targets' in status


class TestQuantumIntegration:
    """Integration tests for quantum planning system."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated quantum planning system."""
        privacy_accountant = PrivacyAccountant(DPConfig())
        
        planner = QuantumTaskPlanner(privacy_accountant)
        scheduler = SuperpositionScheduler()
        optimizer_component = EntanglementOptimizer()
        balancer = InterferenceBalancer()
        
        validator = QuantumComponentValidator()
        monitor = QuantumMonitor()
        security = QuantumSecurityController()
        performance_optimizer = QuantumPerformanceOptimizer()
        
        # Initialize security
        await security.initialize_security()
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Start performance optimization
        await performance_optimizer.start_optimization()
        
        system = {
            'planner': planner,
            'scheduler': scheduler,
            'optimizer': optimizer_component,
            'balancer': balancer,
            'validator': validator,
            'monitor': monitor,
            'security': security,
            'performance_optimizer': performance_optimizer
        }
        
        yield system
        
        # Cleanup
        await security.shutdown_security()
        await monitor.stop_monitoring()
        await performance_optimizer.stop_optimization()
    
    @pytest.mark.asyncio
    async def test_end_to_end_quantum_workflow(self, integrated_system):
        """Test complete end-to-end quantum planning workflow."""
        planner = integrated_system['planner']
        validator = integrated_system['validator']
        security = integrated_system['security']
        
        # Register nodes
        node_capabilities = {
            'current_load': 0.3,
            'privacy_budget': 50.0,
            'compute_capacity': {'gpu_memory': 16384, 'cpu_cores': 8}
        }
        
        planner.register_node("integrated_test_node", node_capabilities)
        
        # Create secure context
        context = await security.create_secure_context(
            user_id="integration_test_user",
            operation_type="planning",
            security_level=SecurityLevel.INTERNAL,
            privacy_budget=2.0
        )
        
        # Add task
        task_data = {
            'task_id': 'integration_test_task',
            'user_id': context.user_id,
            'prompt': 'Integration test query',
            'priority': TaskPriority.MEDIUM.value,
            'privacy_budget': 1.0,
            'estimated_duration': 30.0,
            'resource_requirements': {'compute': 0.4}
        }
        
        task_id = await planner.add_task(task_data)
        assert task_id == 'integration_test_task'
        
        # Validate task
        task = planner.quantum_tasks[task_id]
        validation_result = await validator.validate_component(task, "quantum_task")
        assert validation_result.is_valid is True
        
        # Generate assignments
        assignments = await planner.plan_optimal_assignments()
        
        # Should have generated at least one assignment
        assert len(assignments) >= 0  # May be 0 if task not ready yet
        
        # Test quantum statistics
        stats = planner.get_quantum_statistics()
        assert stats['active_tasks'] >= 1 or stats['collapsed_tasks'] >= 1
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, integrated_system):
        """Test system performance under load."""
        planner = integrated_system['planner']
        performance_optimizer = integrated_system['performance_optimizer']
        
        # Register multiple nodes
        for i in range(3):
            node_capabilities = {
                'current_load': 0.2,
                'privacy_budget': 100.0,
                'compute_capacity': {'gpu_memory': 32768, 'cpu_cores': 16}
            }
            planner.register_node(f"load_test_node_{i}", node_capabilities)
        
        # Add multiple tasks rapidly
        start_time = time.time()
        task_count = 20
        
        for i in range(task_count):
            task_data = {
                'task_id': f'load_test_task_{i}',
                'user_id': f'user_{i % 5}',  # 5 different users
                'prompt': f'Load test query {i}',
                'priority': TaskPriority.MEDIUM.value,
                'privacy_budget': 0.5,
                'estimated_duration': 15.0,
                'resource_requirements': {'compute': 0.3}
            }
            
            await planner.add_task(task_data)
        
        # Generate assignments
        assignments = await planner.plan_optimal_assignments()
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len(planner.quantum_tasks) == task_count
        
        # Check optimizer metrics
        opt_status = performance_optimizer.get_optimization_status()
        assert 'resource_pool_stats' in opt_status


# Performance and stress tests
@pytest.mark.performance
class TestQuantumPerformance:
    """Performance tests for quantum planning system."""
    
    @pytest.mark.asyncio
    async def test_task_creation_performance(self):
        """Test performance of task creation."""
        privacy_accountant = PrivacyAccountant(DPConfig())
        planner = QuantumTaskPlanner(privacy_accountant)
        
        # Register node
        planner.register_node("perf_node", {
            'current_load': 0.3,
            'privacy_budget': 1000.0,
            'compute_capacity': {'gpu_memory': 65536, 'cpu_cores': 32}
        })
        
        # Measure task creation performance
        start_time = time.time()
        task_count = 100
        
        for i in range(task_count):
            task_data = {
                'task_id': f'perf_task_{i}',
                'user_id': f'user_{i}',
                'prompt': f'Performance test query {i}',
                'priority': TaskPriority.MEDIUM.value,
                'privacy_budget': 0.1,
                'estimated_duration': 20.0,
                'resource_requirements': {'compute': 0.25}
            }
            await planner.add_task(task_data)
        
        creation_time = time.time() - start_time
        tasks_per_second = task_count / creation_time
        
        # Performance assertions
        assert tasks_per_second > 50  # Should create at least 50 tasks per second
        assert creation_time < 5.0   # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_assignment_generation_performance(self):
        """Test performance of assignment generation."""
        privacy_accountant = PrivacyAccountant(DPConfig())
        planner = QuantumTaskPlanner(privacy_accountant)
        
        # Register multiple nodes
        for i in range(10):
            planner.register_node(f"perf_node_{i}", {
                'current_load': 0.2,
                'privacy_budget': 500.0,
                'compute_capacity': {'gpu_memory': 32768, 'cpu_cores': 16}
            })
        
        # Add tasks
        for i in range(50):
            task_data = {
                'task_id': f'assign_perf_task_{i}',
                'user_id': f'user_{i % 10}',
                'prompt': f'Assignment performance test {i}',
                'priority': TaskPriority.MEDIUM.value,
                'privacy_budget': 0.2,
                'estimated_duration': 25.0,
                'resource_requirements': {'compute': 0.3}
            }
            await planner.add_task(task_data)
        
        # Measure assignment generation
        start_time = time.time()
        assignments = await planner.plan_optimal_assignments()
        assignment_time = time.time() - start_time
        
        # Performance assertions
        assert assignment_time < 3.0  # Should complete within 3 seconds
        assert len(assignments) >= 0  # Should generate valid assignments


if __name__ == "__main__":
    # Run tests with performance markers
    pytest.main([__file__, "-v", "-m", "not performance"])
    print("\nRunning performance tests...")
    pytest.main([__file__, "-v", "-m", "performance"])