#!/usr/bin/env python3
"""
Enhanced Privacy Parameter Validator
Comprehensive validation for differential privacy parameters in healthcare context
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
try:
    from ..quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()
except ImportError:
    # For files outside quantum_planning module
    from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
    HAS_NUMPY, np = get_numpy_backend()

logger = logging.getLogger(__name__)

class HealthcareDataSensitivity(Enum):
    """Healthcare data sensitivity classifications"""
    PUBLIC = "public"              # No PHI - ε up to 10.0
    LOW_SENSITIVITY = "low"        # Aggregated data - ε up to 5.0  
    MEDIUM_SENSITIVITY = "medium"  # Diagnostic codes - ε up to 2.0
    HIGH_SENSITIVITY = "high"      # Patient records - ε up to 1.0
    CRITICAL = "critical"          # Genetic/mental health - ε ≤ 0.1

class DepartmentType(Enum):
    """Medical department classifications with risk profiles"""
    EMERGENCY = "emergency"        # Higher budget for critical care
    ICU = "icu"                   # Critical care high priority
    CARDIOLOGY = "cardiology"     # Moderate sensitivity
    ONCOLOGY = "oncology"         # High sensitivity
    PSYCHIATRY = "psychiatry"     # Critical sensitivity
    GENETICS = "genetics"         # Critical sensitivity
    GENERAL = "general"           # Standard sensitivity
    RESEARCH = "research"         # Strict limits
    ADMIN = "admin"               # Low sensitivity

@dataclass
class ValidationResult:
    """Privacy parameter validation result"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    recommended_params: Optional[Dict[str, float]] = None

@dataclass 
class HealthcarePrivacyLimits:
    """Healthcare-specific privacy parameter limits"""
    max_epsilon_per_query: float
    max_epsilon_daily: float
    max_epsilon_weekly: float
    max_delta: float
    min_noise_multiplier: float
    max_queries_per_hour: int

class EnhancedPrivacyValidator:
    """
    Comprehensive privacy parameter validator for healthcare environments
    Implements strict validation beyond basic DP parameter checks
    """
    
    def __init__(self):
        # Healthcare compliance limits per department
        self.department_limits = {
            DepartmentType.EMERGENCY: HealthcarePrivacyLimits(
                max_epsilon_per_query=2.0,
                max_epsilon_daily=50.0,
                max_epsilon_weekly=300.0,
                max_delta=1e-5,
                min_noise_multiplier=0.8,
                max_queries_per_hour=200
            ),
            DepartmentType.ICU: HealthcarePrivacyLimits(
                max_epsilon_per_query=1.5,
                max_epsilon_daily=30.0,
                max_epsilon_weekly=180.0,
                max_delta=1e-6,
                min_noise_multiplier=1.0,
                max_queries_per_hour=150
            ),
            DepartmentType.CARDIOLOGY: HealthcarePrivacyLimits(
                max_epsilon_per_query=1.0,
                max_epsilon_daily=20.0,
                max_epsilon_weekly=120.0,
                max_delta=1e-6,
                min_noise_multiplier=1.1,
                max_queries_per_hour=100
            ),
            DepartmentType.ONCOLOGY: HealthcarePrivacyLimits(
                max_epsilon_per_query=0.5,
                max_epsilon_daily=10.0,
                max_epsilon_weekly=60.0,
                max_delta=1e-7,
                min_noise_multiplier=1.5,
                max_queries_per_hour=50
            ),
            DepartmentType.PSYCHIATRY: HealthcarePrivacyLimits(
                max_epsilon_per_query=0.1,
                max_epsilon_daily=2.0,
                max_epsilon_weekly=10.0,
                max_delta=1e-8,
                min_noise_multiplier=2.0,
                max_queries_per_hour=20
            ),
            DepartmentType.GENETICS: HealthcarePrivacyLimits(
                max_epsilon_per_query=0.05,
                max_epsilon_daily=1.0,
                max_epsilon_weekly=5.0,
                max_delta=1e-9,
                min_noise_multiplier=3.0,
                max_queries_per_hour=10
            ),
            DepartmentType.GENERAL: HealthcarePrivacyLimits(
                max_epsilon_per_query=1.0,
                max_epsilon_daily=15.0,
                max_epsilon_weekly=90.0,
                max_delta=1e-6,
                min_noise_multiplier=1.2,
                max_queries_per_hour=80
            ),
            DepartmentType.RESEARCH: HealthcarePrivacyLimits(
                max_epsilon_per_query=0.2,
                max_epsilon_daily=3.0,
                max_epsilon_weekly=15.0,
                max_delta=1e-7,
                min_noise_multiplier=2.0,
                max_queries_per_hour=30
            ),
            DepartmentType.ADMIN: HealthcarePrivacyLimits(
                max_epsilon_per_query=5.0,
                max_epsilon_daily=100.0,
                max_epsilon_weekly=500.0,
                max_delta=1e-5,
                min_noise_multiplier=0.5,
                max_queries_per_hour=500
            )
        }
        
        # Data sensitivity limits
        self.sensitivity_limits = {
            HealthcareDataSensitivity.PUBLIC: 10.0,
            HealthcareDataSensitivity.LOW_SENSITIVITY: 5.0,
            HealthcareDataSensitivity.MEDIUM_SENSITIVITY: 2.0,
            HealthcareDataSensitivity.HIGH_SENSITIVITY: 1.0,
            HealthcareDataSensitivity.CRITICAL: 0.1
        }
    
    def validate_privacy_parameters(self, 
                                  epsilon: float,
                                  delta: float,
                                  noise_multiplier: float,
                                  department: str = "general",
                                  data_sensitivity: str = "medium",
                                  user_role: str = "doctor",
                                  query_type: str = "inference") -> ValidationResult:
        """
        Comprehensive validation of privacy parameters for healthcare context
        """
        issues = []
        warnings = []
        
        try:
            dept_type = DepartmentType(department.lower())
        except ValueError:
            dept_type = DepartmentType.GENERAL
            warnings.append(f"Unknown department '{department}', using GENERAL limits")
        
        try:
            data_sens = HealthcareDataSensitivity(data_sensitivity.lower())
        except ValueError:
            data_sens = HealthcareDataSensitivity.MEDIUM_SENSITIVITY
            warnings.append(f"Unknown data sensitivity '{data_sensitivity}', using MEDIUM")
        
        limits = self.department_limits[dept_type]
        
        # 1. Basic parameter validation
        basic_validation = self._validate_basic_parameters(epsilon, delta, noise_multiplier)
        issues.extend(basic_validation['issues'])
        warnings.extend(basic_validation['warnings'])
        
        # 2. Department-specific limits
        dept_validation = self._validate_department_limits(epsilon, delta, noise_multiplier, limits)
        issues.extend(dept_validation['issues'])
        warnings.extend(dept_validation['warnings'])
        
        # 3. Data sensitivity validation
        sens_validation = self._validate_data_sensitivity(epsilon, data_sens)
        issues.extend(sens_validation['issues'])
        warnings.extend(sens_validation['warnings'])
        
        # 4. Role-based validation
        role_validation = self._validate_user_role(epsilon, user_role, dept_type)
        issues.extend(role_validation['issues'])
        warnings.extend(role_validation['warnings'])
        
        # 5. Query type validation
        query_validation = self._validate_query_type(epsilon, delta, query_type)
        issues.extend(query_validation['issues'])
        warnings.extend(query_validation['warnings'])
        
        # 6. Healthcare compliance validation
        compliance_validation = self._validate_healthcare_compliance(epsilon, delta, dept_type)
        issues.extend(compliance_validation['issues'])
        warnings.extend(compliance_validation['warnings'])
        
        # Generate recommended parameters if current ones are invalid
        recommended_params = None
        if issues:
            recommended_params = self._get_recommended_parameters(dept_type, data_sens, user_role)
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            recommended_params=recommended_params
        )
    
    def _validate_basic_parameters(self, epsilon: float, delta: float, noise_multiplier: float) -> Dict[str, List[str]]:
        """Basic mathematical validation of DP parameters"""
        issues = []
        warnings = []
        
        # Epsilon validation
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon):
            issues.append("Epsilon must be a finite number")
        elif epsilon <= 0:
            issues.append("Epsilon must be positive")
        elif epsilon > 100:
            issues.append("Epsilon is unreasonably large (>100), no meaningful privacy")
        elif epsilon > 10:
            warnings.append("High epsilon value (>10) provides limited privacy protection")
        
        # Delta validation  
        if not isinstance(delta, (int, float)) or not math.isfinite(delta):
            issues.append("Delta must be a finite number")
        elif delta < 0:
            issues.append("Delta must be non-negative")
        elif delta > 0.1:
            issues.append("Delta is too large (>0.1), violates typical DP assumptions")
        elif delta > 1e-5:
            warnings.append("Delta is relatively large (>1e-5), consider smaller value")
        
        # Delta vs dataset size check
        if delta > 0 and delta > 1e-6:
            warnings.append("Delta should typically be much smaller than 1/dataset_size")
        
        # Noise multiplier validation
        if not isinstance(noise_multiplier, (int, float)) or not math.isfinite(noise_multiplier):
            issues.append("Noise multiplier must be a finite number")
        elif noise_multiplier <= 0:
            issues.append("Noise multiplier must be positive")
        elif noise_multiplier > 10:
            warnings.append("Very high noise multiplier may degrade utility significantly")
        
        # Consistency check between epsilon and noise multiplier
        if epsilon > 0 and noise_multiplier > 0:
            implied_epsilon = 1.0 / noise_multiplier  # Rough approximation
            if abs(epsilon - implied_epsilon) > epsilon * 0.5:
                warnings.append("Epsilon and noise multiplier may be inconsistent")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_department_limits(self, epsilon: float, delta: float, 
                                  noise_multiplier: float, limits: HealthcarePrivacyLimits) -> Dict[str, List[str]]:
        """Validate parameters against department-specific limits"""
        issues = []
        warnings = []
        
        if epsilon > limits.max_epsilon_per_query:
            issues.append(f"Epsilon {epsilon} exceeds department limit {limits.max_epsilon_per_query}")
        
        if delta > limits.max_delta:
            issues.append(f"Delta {delta} exceeds department limit {limits.max_delta}")
        
        if noise_multiplier < limits.min_noise_multiplier:
            issues.append(f"Noise multiplier {noise_multiplier} below department minimum {limits.min_noise_multiplier}")
        
        # Warning for approaching limits
        if epsilon > limits.max_epsilon_per_query * 0.8:
            warnings.append(f"Epsilon approaching department limit ({limits.max_epsilon_per_query})")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_data_sensitivity(self, epsilon: float, data_sensitivity: HealthcareDataSensitivity) -> Dict[str, List[str]]:
        """Validate epsilon against data sensitivity requirements"""
        issues = []
        warnings = []
        
        max_epsilon = self.sensitivity_limits[data_sensitivity]
        
        if epsilon > max_epsilon:
            issues.append(f"Epsilon {epsilon} too high for {data_sensitivity.value} data (max: {max_epsilon})")
        
        # Special validation for critical data
        if data_sensitivity in [HealthcareDataSensitivity.CRITICAL, HealthcareDataSensitivity.HIGH_SENSITIVITY]:
            if epsilon > max_epsilon * 0.5:
                warnings.append(f"High epsilon for sensitive {data_sensitivity.value} data")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_user_role(self, epsilon: float, user_role: str, department: DepartmentType) -> Dict[str, List[str]]:
        """Validate parameters based on user role and department"""
        issues = []
        warnings = []
        
        # Role-based epsilon limits
        role_multipliers = {
            'admin': 1.5,      # Admin can use higher epsilon
            'doctor': 1.0,     # Standard limits
            'nurse': 0.8,      # Slightly restricted
            'researcher': 0.5,  # More restricted for research
            'student': 0.3,    # Highly restricted
            'guest': 0.1       # Very restricted
        }
        
        multiplier = role_multipliers.get(user_role.lower(), 1.0)
        limits = self.department_limits[department]
        max_role_epsilon = limits.max_epsilon_per_query * multiplier
        
        if epsilon > max_role_epsilon:
            issues.append(f"Epsilon {epsilon} exceeds role limit for {user_role} ({max_role_epsilon})")
        
        # Warnings for roles with special considerations
        if user_role.lower() in ['student', 'guest'] and epsilon > 0.1:
            warnings.append(f"High epsilon for {user_role} role, consider further restrictions")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_query_type(self, epsilon: float, delta: float, query_type: str) -> Dict[str, List[str]]:
        """Validate parameters based on query type"""
        issues = []
        warnings = []
        
        # Query-specific limits
        query_limits = {
            'count': {'max_epsilon': 0.5, 'max_delta': 1e-6},
            'sum': {'max_epsilon': 1.0, 'max_delta': 1e-6},
            'mean': {'max_epsilon': 1.0, 'max_delta': 1e-6},
            'inference': {'max_epsilon': 2.0, 'max_delta': 1e-5},
            'training': {'max_epsilon': 5.0, 'max_delta': 1e-5},
            'analysis': {'max_epsilon': 3.0, 'max_delta': 1e-6}
        }
        
        if query_type.lower() in query_limits:
            limits = query_limits[query_type.lower()]
            
            if epsilon > limits['max_epsilon']:
                issues.append(f"Epsilon {epsilon} too high for {query_type} queries (max: {limits['max_epsilon']})")
            
            if delta > limits['max_delta']:
                issues.append(f"Delta {delta} too high for {query_type} queries (max: {limits['max_delta']})")
        else:
            warnings.append(f"Unknown query type '{query_type}', using default validation")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _validate_healthcare_compliance(self, epsilon: float, delta: float, department: DepartmentType) -> Dict[str, List[str]]:
        """Validate healthcare regulatory compliance requirements"""
        issues = []
        warnings = []
        
        # HIPAA Safe Harbor method considerations
        if department in [DepartmentType.PSYCHIATRY, DepartmentType.GENETICS]:
            if epsilon > 0.1:
                warnings.append("High epsilon for sensitive medical data may not meet HIPAA Safe Harbor requirements")
        
        # Clinical research compliance (21 CFR Part 11)
        if department == DepartmentType.RESEARCH:
            if delta > 1e-7:
                warnings.append("Research queries should use very small delta for regulatory compliance")
        
        # Emergency care balance between privacy and utility
        if department == DepartmentType.EMERGENCY:
            if epsilon < 0.5:
                warnings.append("Very low epsilon in emergency care may impede critical decision-making")
        
        # Audit trail requirements
        if epsilon > 1.0:
            warnings.append("High epsilon queries require enhanced audit trails for compliance")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _get_recommended_parameters(self, department: DepartmentType, 
                                  data_sensitivity: HealthcareDataSensitivity,
                                  user_role: str) -> Dict[str, float]:
        """Generate recommended privacy parameters"""
        limits = self.department_limits[department]
        sensitivity_limit = self.sensitivity_limits[data_sensitivity]
        
        role_multipliers = {
            'admin': 0.8, 'doctor': 0.7, 'nurse': 0.6,
            'researcher': 0.4, 'student': 0.2, 'guest': 0.1
        }
        
        multiplier = role_multipliers.get(user_role.lower(), 0.7)
        
        # Conservative recommendations
        recommended_epsilon = min(
            limits.max_epsilon_per_query * multiplier,
            sensitivity_limit * 0.8,
            1.0  # Never recommend more than 1.0
        )
        
        recommended_delta = min(limits.max_delta, 1e-6)
        recommended_noise_multiplier = max(limits.min_noise_multiplier, 1.1)
        
        return {
            'epsilon': recommended_epsilon,
            'delta': recommended_delta,
            'noise_multiplier': recommended_noise_multiplier
        }
    
    def validate_budget_request(self, user_id: str, requested_epsilon: float,
                              daily_spent: float, weekly_spent: float,
                              department: str = "general") -> ValidationResult:
        """Validate privacy budget request against spending limits"""
        issues = []
        warnings = []
        
        try:
            dept_type = DepartmentType(department.lower())
        except ValueError:
            dept_type = DepartmentType.GENERAL
            warnings.append(f"Unknown department '{department}', using GENERAL limits")
        
        limits = self.department_limits[dept_type]
        
        # Daily limit check
        if daily_spent + requested_epsilon > limits.max_epsilon_daily:
            issues.append(f"Request would exceed daily epsilon limit ({limits.max_epsilon_daily})")
        elif daily_spent + requested_epsilon > limits.max_epsilon_daily * 0.8:
            warnings.append("Approaching daily epsilon limit")
        
        # Weekly limit check
        if weekly_spent + requested_epsilon > limits.max_epsilon_weekly:
            issues.append(f"Request would exceed weekly epsilon limit ({limits.max_epsilon_weekly})")
        elif weekly_spent + requested_epsilon > limits.max_epsilon_weekly * 0.8:
            warnings.append("Approaching weekly epsilon limit")
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
    
    def get_department_info(self, department: str) -> Dict[str, any]:
        """Get information about department limits and requirements"""
        try:
            dept_type = DepartmentType(department.lower())
            limits = self.department_limits[dept_type]
            
            return {
                'department': dept_type.value,
                'limits': {
                    'max_epsilon_per_query': limits.max_epsilon_per_query,
                    'max_epsilon_daily': limits.max_epsilon_daily,
                    'max_epsilon_weekly': limits.max_epsilon_weekly,
                    'max_delta': limits.max_delta,
                    'min_noise_multiplier': limits.min_noise_multiplier,
                    'max_queries_per_hour': limits.max_queries_per_hour
                },
                'description': f"Healthcare department: {dept_type.value}"
            }
        except ValueError:
            return {
                'department': 'unknown',
                'error': f"Unknown department: {department}"
            }

# Global validator instance
_privacy_validator = None

def get_privacy_validator() -> EnhancedPrivacyValidator:
    """Get global privacy validator instance"""
    global _privacy_validator
    if _privacy_validator is None:
        _privacy_validator = EnhancedPrivacyValidator()
    return _privacy_validator

def validate_privacy_parameters(epsilon: float, delta: float, noise_multiplier: float,
                              department: str = "general", data_sensitivity: str = "medium",
                              user_role: str = "doctor", query_type: str = "inference") -> ValidationResult:
    """Convenience function for privacy parameter validation"""
    validator = get_privacy_validator()
    return validator.validate_privacy_parameters(
        epsilon, delta, noise_multiplier, department, 
        data_sensitivity, user_role, query_type
    )