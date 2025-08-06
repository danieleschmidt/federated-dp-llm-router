"""
Quantum Planning Validators and Error Handlers

Comprehensive validation and error handling for quantum-inspired
task planning components with healthcare-grade reliability.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import asynccontextmanager
import traceback

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"        # Healthcare/production grade
    MODERATE = "moderate"    # Development/testing  
    PERMISSIVE = "permissive"  # Research/experimental


class QuantumErrorType(Enum):
    """Types of quantum planning errors."""
    DECOHERENCE = "decoherence"
    ENTANGLEMENT_VIOLATION = "entanglement_violation"
    SUPERPOSITION_COLLAPSE = "superposition_collapse"
    INTERFERENCE_DISRUPTION = "interference_disruption"
    PRIVACY_VIOLATION = "privacy_violation"
    RESOURCE_CONSTRAINT = "resource_constraint"
    QUANTUM_STATE_INVALID = "quantum_state_invalid"
    MEASUREMENT_FAILURE = "measurement_failure"


@dataclass
class ValidationResult:
    """Result of quantum component validation."""
    is_valid: bool
    error_type: Optional[QuantumErrorType] = None
    error_message: str = ""
    suggestions: List[str] = None
    severity: str = "low"  # low, medium, high, critical
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class QuantumComponentValidator:
    """Base validator for quantum planning components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.validation_history: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, int] = {}
        
    async def validate_component(self, component: Any, component_type: str) -> ValidationResult:
        """Validate quantum planning component."""
        validation_start = time.time()
        
        try:
            # Dispatch to specific validation method
            if component_type == "quantum_task":
                result = await self._validate_quantum_task(component)
            elif component_type == "superposition":
                result = await self._validate_superposition(component)
            elif component_type == "entanglement":
                result = await self._validate_entanglement(component)
            elif component_type == "interference":
                result = await self._validate_interference(component)
            elif component_type == "node_state":
                result = await self._validate_node_state(component)
            else:
                result = ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                    error_message=f"Unknown component type: {component_type}",
                    severity="high"
                )
            
            # Record validation
            self.validation_history.append({
                "timestamp": validation_start,
                "component_type": component_type,
                "validation_time": time.time() - validation_start,
                "result": result.is_valid,
                "error_type": result.error_type.value if result.error_type else None,
                "severity": result.severity
            })
            
            # Track error patterns
            if not result.is_valid and result.error_type:
                error_key = f"{component_type}_{result.error_type.value}"
                self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {component_type}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                error_message=f"Validation exception: {str(e)}",
                severity="critical"
            )
    
    async def _validate_quantum_task(self, task) -> ValidationResult:
        """Validate quantum task structure and properties."""
        
        # Check required fields
        required_fields = ['task_id', 'user_id', 'priority', 'privacy_budget']
        missing_fields = []
        
        for field in required_fields:
            if not hasattr(task, field) or getattr(task, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                error_message=f"Missing required fields: {missing_fields}",
                suggestions=[f"Set {field} to valid value" for field in missing_fields],
                severity="high"
            )
        
        # Validate privacy budget
        if hasattr(task, 'privacy_budget') and task.privacy_budget <= 0:
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.PRIVACY_VIOLATION,
                error_message="Privacy budget must be positive",
                suggestions=["Set privacy_budget to positive value"],
                severity="critical"
            )
        
        # Check quantum state consistency
        if hasattr(task, 'quantum_state') and hasattr(task, 'probability_distribution'):
            if task.quantum_state.value == 'superposition' and not task.probability_distribution:
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.SUPERPOSITION_COLLAPSE,
                    error_message="Superposition task missing probability distribution",
                    suggestions=["Initialize probability distribution for superposition"],
                    severity="medium"
                )
        
        # Validate coherence time
        if hasattr(task, 'coherence_time'):
            current_time = time.time()
            time_elapsed = current_time - getattr(task, 'created_at', current_time)
            
            if time_elapsed > task.coherence_time:
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.DECOHERENCE,
                    error_message="Task has exceeded coherence time",
                    suggestions=["Handle decoherence or extend coherence time"],
                    severity="medium"
                )
        
        return ValidationResult(is_valid=True)
    
    async def _validate_superposition(self, superposition) -> ValidationResult:
        """Validate quantum superposition state."""
        
        if not hasattr(superposition, 'amplitude_distribution'):
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.SUPERPOSITION_COLLAPSE,
                error_message="Superposition missing amplitude distribution",
                severity="high"
            )
        
        # Check probability normalization
        if hasattr(superposition, 'measurement_probability'):
            total_prob = sum(superposition.measurement_probability.values())
            if abs(total_prob - 1.0) > 0.01:  # Allow small numerical errors
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                    error_message=f"Probabilities not normalized: sum={total_prob:.3f}",
                    suggestions=["Renormalize probability distribution"],
                    severity="medium"
                )
        
        # Check amplitude consistency
        if (hasattr(superposition, 'amplitude_distribution') and 
            hasattr(superposition, 'measurement_probability')):
            
            for node_id in superposition.amplitude_distribution.keys():
                amplitude = superposition.amplitude_distribution[node_id]
                expected_prob = abs(amplitude) ** 2
                actual_prob = superposition.measurement_probability.get(node_id, 0)
                
                if abs(expected_prob - actual_prob) > 0.01:
                    return ValidationResult(
                        is_valid=False,
                        error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                        error_message=f"Amplitude-probability mismatch for node {node_id}",
                        suggestions=["Recalculate measurement probabilities from amplitudes"],
                        severity="medium"
                    )
        
        return ValidationResult(is_valid=True)
    
    async def _validate_entanglement(self, entanglement) -> ValidationResult:
        """Validate quantum entanglement state."""
        
        if not hasattr(entanglement, 'resource_pairs') or len(entanglement.resource_pairs) < 2:
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.ENTANGLEMENT_VIOLATION,
                error_message="Entanglement requires at least 2 resource pairs",
                severity="high"
            )
        
        # Check entanglement strength bounds
        if hasattr(entanglement, 'entanglement_strength'):
            strength = entanglement.entanglement_strength
            if not 0.0 <= strength <= 1.0:
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.ENTANGLEMENT_VIOLATION,
                    error_message=f"Entanglement strength {strength} out of bounds [0,1]",
                    suggestions=["Clamp entanglement strength to [0,1]"],
                    severity="medium"
                )
        
        # Validate correlation matrix
        if hasattr(entanglement, 'correlation_matrix'):
            matrix = entanglement.correlation_matrix
            if matrix.size > 0:
                # Check if matrix is symmetric
                if not np.allclose(matrix, matrix.T, atol=1e-6):
                    return ValidationResult(
                        is_valid=False,
                        error_type=QuantumErrorType.ENTANGLEMENT_VIOLATION,
                        error_message="Correlation matrix not symmetric",
                        suggestions=["Ensure correlation matrix symmetry"],
                        severity="medium"
                    )
                
                # Check diagonal elements (should be 1.0 or close)
                diag_elements = np.diag(matrix)
                if not np.allclose(diag_elements, 1.0, atol=0.1):
                    return ValidationResult(
                        is_valid=False,
                        error_type=QuantumErrorType.ENTANGLEMENT_VIOLATION,
                        error_message="Correlation matrix diagonal not normalized",
                        suggestions=["Normalize correlation matrix diagonal"],
                        severity="low"
                    )
        
        # Check coherence time
        if hasattr(entanglement, 'coherence_time') and hasattr(entanglement, 'creation_time'):
            current_time = time.time()
            age = current_time - entanglement.creation_time
            
            if age > entanglement.coherence_time:
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.DECOHERENCE,
                    error_message="Entanglement has exceeded coherence time",
                    suggestions=["Handle entanglement decoherence"],
                    severity="medium"
                )
        
        return ValidationResult(is_valid=True)
    
    async def _validate_interference(self, interference) -> ValidationResult:
        """Validate quantum interference pattern."""
        
        if not hasattr(interference, 'amplitude_pattern') or not interference.amplitude_pattern:
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.INTERFERENCE_DISRUPTION,
                error_message="Interference missing amplitude pattern",
                severity="high"
            )
        
        # Check interference strength bounds
        if hasattr(interference, 'interference_strength'):
            strength = interference.interference_strength
            if not -1.0 <= strength <= 1.0:
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.INTERFERENCE_DISRUPTION,
                    error_message=f"Interference strength {strength} out of bounds [-1,1]",
                    severity="medium"
                )
        
        # Validate phase relationships
        if hasattr(interference, 'phase_relationships'):
            for node_id, phase in interference.phase_relationships.items():
                # Phase should be in [0, 2π]
                if not 0 <= phase <= 2 * np.pi:
                    normalized_phase = phase % (2 * np.pi)
                    return ValidationResult(
                        is_valid=True,  # Auto-correctable
                        error_message=f"Phase {phase} for node {node_id} normalized to {normalized_phase}",
                        suggestions=[f"Normalize phase for node {node_id}"],
                        severity="low"
                    )
        
        # Check wavelength validity
        if hasattr(interference, 'wavelength') and interference.wavelength <= 0:
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.INTERFERENCE_DISRUPTION,
                error_message="Wavelength must be positive",
                suggestions=["Set positive wavelength value"],
                severity="medium"
            )
        
        return ValidationResult(is_valid=True)
    
    async def _validate_node_state(self, node_state) -> ValidationResult:
        """Validate quantum node state."""
        
        # Check wave function normalization
        if hasattr(node_state, 'psi_real') and hasattr(node_state, 'psi_imaginary'):
            amplitude_squared = node_state.psi_real**2 + node_state.psi_imaginary**2
            
            if abs(amplitude_squared - 1.0) > 0.1:  # Allow some deviation
                return ValidationResult(
                    is_valid=False,
                    error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                    error_message=f"Wave function not normalized: |ψ|² = {amplitude_squared:.3f}",
                    suggestions=["Normalize wave function"],
                    severity="medium"
                )
        
        # Validate physical bounds
        if hasattr(node_state, 'quality_factor') and node_state.quality_factor <= 0:
            return ValidationResult(
                is_valid=False,
                error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                error_message="Quality factor must be positive",
                suggestions=["Set positive quality factor"],
                severity="low"
            )
        
        # Check for NaN/Inf values
        numeric_fields = ['psi_real', 'psi_imaginary', 'phase', 'amplitude', 'frequency']
        for field in numeric_fields:
            if hasattr(node_state, field):
                value = getattr(node_state, field)
                if np.isnan(value) or np.isinf(value):
                    return ValidationResult(
                        is_valid=False,
                        error_type=QuantumErrorType.QUANTUM_STATE_INVALID,
                        error_message=f"Invalid numeric value for {field}: {value}",
                        suggestions=[f"Reset {field} to valid numeric value"],
                        severity="high"
                    )
        
        return ValidationResult(is_valid=True)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities."""
        
        total_validations = len(self.validation_history)
        if total_validations == 0:
            return {"total_validations": 0}
        
        successful_validations = sum(1 for v in self.validation_history if v["result"])
        success_rate = successful_validations / total_validations
        
        # Error distribution
        error_distribution = {}
        for error_pattern, count in self.error_patterns.items():
            error_distribution[error_pattern] = count
        
        # Severity distribution
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for validation in self.validation_history:
            if not validation["result"]:
                severity_counts[validation["severity"]] += 1
        
        return {
            "total_validations": total_validations,
            "success_rate": success_rate,
            "error_patterns": error_distribution,
            "severity_distribution": severity_counts,
            "recent_validations": self.validation_history[-10:],
            "validation_level": self.validation_level.value
        }


class QuantumErrorHandler:
    """Error handler for quantum planning system failures."""
    
    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[QuantumErrorType, Callable] = {
            QuantumErrorType.DECOHERENCE: self._handle_decoherence,
            QuantumErrorType.ENTANGLEMENT_VIOLATION: self._handle_entanglement_violation,
            QuantumErrorType.SUPERPOSITION_COLLAPSE: self._handle_superposition_collapse,
            QuantumErrorType.INTERFERENCE_DISRUPTION: self._handle_interference_disruption,
            QuantumErrorType.PRIVACY_VIOLATION: self._handle_privacy_violation,
            QuantumErrorType.RESOURCE_CONSTRAINT: self._handle_resource_constraint,
            QuantumErrorType.QUANTUM_STATE_INVALID: self._handle_invalid_state,
            QuantumErrorType.MEASUREMENT_FAILURE: self._handle_measurement_failure,
        }
        
    @asynccontextmanager
    async def error_boundary(self, operation_name: str, fallback_action=None):
        """Async context manager for error boundary with automatic recovery."""
        try:
            yield
        except Exception as e:
            # Log error
            error_info = {
                "timestamp": time.time(),
                "operation": operation_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            self.error_history.append(error_info)
            logger.error(f"Quantum error in {operation_name}: {str(e)}")
            
            # Attempt recovery
            if fallback_action:
                try:
                    await fallback_action()
                    logger.info(f"Fallback action succeeded for {operation_name}")
                except Exception as fallback_error:
                    logger.error(f"Fallback action failed: {str(fallback_error)}")
                    raise e
            else:
                raise e
    
    async def handle_quantum_error(self, 
                                 error_type: QuantumErrorType,
                                 component: Any,
                                 context: Dict[str, Any] = None) -> bool:
        """Handle specific quantum error type."""
        
        if context is None:
            context = {}
        
        handler = self.recovery_strategies.get(error_type)
        if not handler:
            logger.error(f"No recovery strategy for error type: {error_type}")
            return False
        
        try:
            recovery_result = await handler(component, context)
            
            # Log recovery attempt
            self.error_history.append({
                "timestamp": time.time(),
                "error_type": error_type.value,
                "recovery_attempted": True,
                "recovery_successful": recovery_result,
                "context": context
            })
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"Recovery failed for {error_type}: {str(e)}")
            return False
    
    async def _handle_decoherence(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle quantum decoherence."""
        try:
            # Reset quantum state to ground state
            if hasattr(component, 'quantum_state'):
                component.quantum_state = "collapsed"  # Safe fallback state
            
            # Clear entanglements
            if hasattr(component, 'entangled_tasks'):
                component.entangled_tasks.clear()
            
            # Reset coherence time
            if hasattr(component, 'coherence_time'):
                component.coherence_time = 300.0  # Reset to default
                
            return True
        except Exception:
            return False
    
    async def _handle_entanglement_violation(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle entanglement violations."""
        try:
            # Break problematic entanglements
            if hasattr(component, 'entanglement_strength'):
                component.entanglement_strength *= 0.5  # Reduce strength
            
            # Reset correlation matrix to identity
            if hasattr(component, 'correlation_matrix'):
                n = component.correlation_matrix.shape[0] if component.correlation_matrix.size > 0 else 2
                component.correlation_matrix = np.eye(n)
            
            return True
        except Exception:
            return False
    
    async def _handle_superposition_collapse(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle unexpected superposition collapse."""
        try:
            # Reinitialize superposition if possible
            if hasattr(component, 'probability_distribution'):
                # Create uniform distribution as fallback
                if hasattr(component, 'amplitude_distribution'):
                    nodes = list(component.amplitude_distribution.keys())
                    uniform_prob = 1.0 / len(nodes) if nodes else 1.0
                    component.probability_distribution = {node: uniform_prob for node in nodes}
            
            return True
        except Exception:
            return False
    
    async def _handle_interference_disruption(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle interference pattern disruption."""
        try:
            # Reset interference parameters
            if hasattr(component, 'interference_strength'):
                component.interference_strength = 0.0  # Disable interference
            
            # Reset phase relationships
            if hasattr(component, 'phase_relationships'):
                for node_id in component.phase_relationships:
                    component.phase_relationships[node_id] = 0.0
            
            return True
        except Exception:
            return False
    
    async def _handle_privacy_violation(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle privacy budget violations."""
        try:
            # This is critical - cannot recover, only log and fail safely
            logger.critical("Privacy violation detected - failing safely")
            return False
        except Exception:
            return False
    
    async def _handle_resource_constraint(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle resource constraint violations."""
        try:
            # Reduce resource requirements
            if hasattr(component, 'resource_requirements'):
                for resource, requirement in component.resource_requirements.items():
                    component.resource_requirements[resource] = requirement * 0.8
            
            return True
        except Exception:
            return False
    
    async def _handle_invalid_state(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle invalid quantum states."""
        try:
            # Reset to known good state
            if hasattr(component, 'psi_real') and hasattr(component, 'psi_imaginary'):
                # Reset to |0⟩ state
                component.psi_real = 1.0
                component.psi_imaginary = 0.0
                component.phase = 0.0
                component.amplitude = 1.0
            
            return True
        except Exception:
            return False
    
    async def _handle_measurement_failure(self, component: Any, context: Dict[str, Any]) -> bool:
        """Handle quantum measurement failures."""
        try:
            # Force measurement to ground state
            if hasattr(component, 'quantum_state'):
                component.quantum_state = "collapsed"
            
            # Clear superposition
            if hasattr(component, 'probability_distribution'):
                component.probability_distribution.clear()
            
            return True
        except Exception:
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0}
        
        # Error type distribution
        error_types = {}
        recovery_attempts = 0
        successful_recoveries = 0
        
        for error in self.error_history:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error.get("recovery_attempted"):
                recovery_attempts += 1
                if error.get("recovery_successful"):
                    successful_recoveries += 1
        
        recovery_rate = successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0
        
        return {
            "total_errors": total_errors,
            "error_type_distribution": error_types,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": recovery_rate,
            "recent_errors": self.error_history[-5:],
            "available_recovery_strategies": list(self.recovery_strategies.keys())
        }