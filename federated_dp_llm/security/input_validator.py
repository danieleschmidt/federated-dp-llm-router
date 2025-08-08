"""
Input Validation and Sanitization

Enhanced security module for validating and sanitizing all inputs to prevent
injection attacks and ensure healthcare data compliance.
"""

import re
import html
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Input validation security levels."""
    BASIC = "basic"
    HEALTHCARE = "healthcare" 
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    violations: List[str]
    risk_score: float


class HealthcareInputValidator:
    """Healthcare-grade input validator with PHI protection."""
    
    # Patterns for potentially sensitive healthcare data
    PHI_PATTERNS = {
        'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
        'mrn': re.compile(r'\b(?:MRN|mrn|patient\s*(?:id|number))[\s:]\s*\d+\b', re.IGNORECASE),
        'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'dob': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'),
        'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    }
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"(union|select|insert|update|delete|drop|create|alter|exec|execute)", re.IGNORECASE),
        re.compile(r"[';\"]\s*(or|and)\s+['\"]?\d+['\"]?\s*[=<>]", re.IGNORECASE),
        re.compile(r"--\s*$", re.MULTILINE),
        re.compile(r"/\*.*?\*/", re.DOTALL),
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
    ]
    
    # Command injection patterns
    COMMAND_PATTERNS = [
        re.compile(r"[;&|`$(){}[\]\\]"),
        re.compile(r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|rm|mv|cp)\b"),
    ]
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.HEALTHCARE):
        self.validation_level = validation_level
        self.blocked_patterns: Set[str] = set()
        
        # Configure logging for security violations
        logging.basicConfig(level=logging.INFO)
        
    def validate_medical_prompt(self, prompt: str, user_id: str = None) -> ValidationResult:
        """Validate medical inference prompts with PHI detection."""
        violations = []
        risk_score = 0.0
        
        # Check for PHI (Protected Health Information)
        phi_violations = self._detect_phi(prompt)
        if phi_violations:
            violations.extend([f"PHI detected: {v}" for v in phi_violations])
            risk_score += len(phi_violations) * 0.3
            
            # Log security violation
            logger.warning(f"PHI detected in prompt from user {user_id}: {phi_violations}")
        
        # Check for injection attacks
        injection_violations = self._detect_injection_attacks(prompt)
        if injection_violations:
            violations.extend(injection_violations)
            risk_score += len(injection_violations) * 0.4
            
            logger.error(f"Injection attack detected from user {user_id}: {injection_violations}")
        
        # Sanitize input
        sanitized = self._sanitize_input(prompt)
        
        # Length validation
        if len(prompt) > 8192:
            violations.append("Input exceeds maximum length (8192 characters)")
            risk_score += 0.2
        
        is_valid = len(violations) == 0 or risk_score < 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            violations=violations,
            risk_score=risk_score
        )
    
    def _detect_phi(self, text: str) -> List[str]:
        """Detect potential PHI in text."""
        violations = []
        
        for phi_type, pattern in self.PHI_PATTERNS.items():
            if pattern.search(text):
                violations.append(phi_type)
        
        return violations
    
    def _detect_injection_attacks(self, text: str) -> List[str]:
        """Detect potential injection attacks."""
        violations = []
        
        # SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                violations.append("SQL injection pattern detected")
                break
        
        # XSS
        for pattern in self.XSS_PATTERNS:
            if pattern.search(text):
                violations.append("XSS pattern detected")
                break
        
        # Command injection
        for pattern in self.COMMAND_PATTERNS:
            if pattern.search(text):
                violations.append("Command injection pattern detected")
                break
        
        return violations
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def validate_user_credentials(self, credentials: Dict[str, Any]) -> ValidationResult:
        """Validate user authentication credentials."""
        violations = []
        risk_score = 0.0
        
        # Check required fields
        required_fields = ['user_id', 'department', 'role']
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                violations.append(f"Missing required field: {field}")
                risk_score += 0.2
        
        # Validate user_id format
        if 'user_id' in credentials:
            user_id = credentials['user_id']
            if not re.match(r'^[a-zA-Z0-9_.-]+$', user_id):
                violations.append("Invalid user_id format")
                risk_score += 0.3
        
        # Validate department
        if 'department' in credentials:
            dept = credentials['department']
            valid_departments = {
                'emergency', 'radiology', 'oncology', 'cardiology', 
                'neurology', 'pediatrics', 'surgery', 'icu', 'research'
            }
            if dept.lower() not in valid_departments:
                violations.append(f"Invalid department: {dept}")
                risk_score += 0.2
        
        is_valid = risk_score < 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=str(credentials),
            violations=violations,
            risk_score=risk_score
        )
    
    def validate_model_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate model inference parameters."""
        violations = []
        risk_score = 0.0
        
        # Privacy budget validation
        if 'max_privacy_budget' in params:
            budget = params['max_privacy_budget']
            if not isinstance(budget, (int, float)) or budget <= 0 or budget > 10.0:
                violations.append("Invalid privacy budget (must be 0 < budget <= 10.0)")
                risk_score += 0.3
        
        # Timeout validation
        if 'timeout' in params:
            timeout = params['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 300:
                violations.append("Invalid timeout (must be 0 < timeout <= 300)")
                risk_score += 0.2
        
        # Priority validation
        if 'priority' in params:
            priority = params['priority']
            if not isinstance(priority, int) or priority < 1 or priority > 10:
                violations.append("Invalid priority (must be 1-10)")
                risk_score += 0.1
        
        is_valid = risk_score < 0.3
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=str(params),
            violations=violations,
            risk_score=risk_score
        )


class SecurityAuditLogger:
    """Enhanced security event logging for compliance."""
    
    def __init__(self):
        self.logger = logging.getLogger('security_audit')
        handler = logging.FileHandler('/var/log/federated-dp-llm/security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_validation_failure(self, user_id: str, violation_type: str, details: Dict[str, Any]):
        """Log input validation failures."""
        self.logger.error(f"VALIDATION_FAILURE user={user_id} type={violation_type} details={details}")
    
    def log_privacy_violation(self, user_id: str, phi_types: List[str]):
        """Log PHI detection events."""
        self.logger.error(f"PHI_DETECTED user={user_id} types={','.join(phi_types)}")
    
    def log_injection_attempt(self, user_id: str, attack_type: str, source_ip: str = None):
        """Log injection attack attempts."""
        self.logger.critical(f"INJECTION_ATTACK user={user_id} type={attack_type} ip={source_ip}")
    
    def log_budget_exhaustion(self, user_id: str, requested_budget: float, remaining: float):
        """Log privacy budget exhaustion."""
        self.logger.warning(f"BUDGET_EXHAUSTED user={user_id} requested={requested_budget} remaining={remaining}")