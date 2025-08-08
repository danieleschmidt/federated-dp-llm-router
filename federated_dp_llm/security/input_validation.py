"""
Comprehensive Input Validation and Sanitization

Implements robust input validation, sanitization, and security checks
for all user inputs and API requests to prevent injection attacks.
"""

import re
import html
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union, Set, Callable
from dataclasses import dataclass
from enum import Enum
import ipaddress
from urllib.parse import urlparse
import base64


class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class SanitizationMode(Enum):
    """Sanitization modes."""
    STRICT = "strict"        # Remove all potentially dangerous content
    PRESERVE = "preserve"    # Escape dangerous content but preserve structure
    MEDICAL = "medical"      # Special handling for medical text
    RESEARCH = "research"    # Preserve scientific notation and symbols


@dataclass
class ValidationRule:
    """Validation rule configuration."""
    field_name: str
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[Set[str]] = None
    custom_validator: Optional[Callable] = None
    sanitization_mode: SanitizationMode = SanitizationMode.STRICT


class InputValidator:
    """Comprehensive input validation and sanitization system."""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.sql_injection_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+",
            r"(?i)(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
            r"(?i)'\s*or\s*'1'\s*=\s*'1",
            r"(?i)--\s*",
            r"(?i)/\*.*?\*/",
            r"(?i)\bexec\s*\(",
            r"(?i)\bsp_\w+",
            r"(?i)\bxp_\w+"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<form[^>]*>",
            r"eval\s*\(",
            r"expression\s*\("
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$]",
            r"\$\([^)]*\)",
            r"`[^`]*`",
            r"\\x[0-9a-fA-F]{2}",
            r"\\[0-7]{1,3}",
            r"\.\./",
            r"\\\\",
            r"/bin/",
            r"/usr/bin/",
            r"cmd\.exe",
            r"powershell"
        ]
        
        self.privacy_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}-\d{3}-\d{4}\b",  # Phone number
            r"\b\d{5}(-\d{4})?\b",  # ZIP code
        ]
        
        # Medical term preservation for healthcare context
        self.medical_terms = {
            "mg", "ml", "kg", "bp", "hr", "temp", "bpm", "mmhg", "iu", "mcg",
            "diabetes", "hypertension", "medication", "dosage", "symptom",
            "diagnosis", "treatment", "patient", "clinical", "therapy"
        }
        
        # Common validation rules
        self.predefined_rules = self._create_predefined_rules()
    
    def _create_predefined_rules(self) -> Dict[str, ValidationRule]:
        """Create predefined validation rules for common fields."""
        return {
            "user_id": ValidationRule(
                field_name="user_id",
                required=True,
                pattern=r"^[a-zA-Z0-9_-]{1,64}$",
                max_length=64
            ),
            "email": ValidationRule(
                field_name="email",
                required=True,
                pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                max_length=254
            ),
            "password": ValidationRule(
                field_name="password",
                required=True,
                min_length=12,
                max_length=128,
                pattern=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$"
            ),
            "username": ValidationRule(
                field_name="username",
                required=True,
                pattern=r"^[a-zA-Z0-9_]{3,32}$",
                min_length=3,
                max_length=32
            ),
            "department": ValidationRule(
                field_name="department",
                required=True,
                allowed_values={
                    "emergency", "cardiology", "neurology", "radiology",
                    "oncology", "pediatrics", "surgery", "general", "admin"
                }
            ),
            "prompt": ValidationRule(
                field_name="prompt",
                required=True,
                min_length=1,
                max_length=8192,
                sanitization_mode=SanitizationMode.MEDICAL
            ),
            "model_name": ValidationRule(
                field_name="model_name",
                required=True,
                pattern=r"^[a-zA-Z0-9_-]{1,64}$",
                max_length=64
            ),
            "privacy_budget": ValidationRule(
                field_name="privacy_budget",
                required=True,
                custom_validator=lambda x: 0.0 < float(x) <= 10.0
            ),
            "ip_address": ValidationRule(
                field_name="ip_address",
                required=False,
                custom_validator=self._validate_ip_address
            ),
            "file_path": ValidationRule(
                field_name="file_path",
                required=False,
                pattern=r"^[a-zA-Z0-9/._-]+$",
                max_length=256
            ),
            "url": ValidationRule(
                field_name="url",
                required=False,
                custom_validator=self._validate_url,
                max_length=2048
            ),
            "json_data": ValidationRule(
                field_name="json_data",
                required=False,
                custom_validator=self._validate_json
            )
        }
    
    def validate_field(
        self,
        field_name: str,
        value: Any,
        rule: Optional[ValidationRule] = None
    ) -> Any:
        """Validate a single field."""
        # Get rule
        if rule is None:
            rule = self.predefined_rules.get(field_name)
            if rule is None:
                raise ValidationError(f"No validation rule found for field: {field_name}")
        
        # Check required
        if rule.required and (value is None or value == ""):
            raise ValidationError(f"Field {field_name} is required", field_name, value)
        
        # Skip validation if value is None and not required
        if value is None and not rule.required:
            return None
        
        # Convert to string for most validations
        str_value = str(value)
        
        # Length validation
        if rule.min_length is not None and len(str_value) < rule.min_length:
            raise ValidationError(
                f"Field {field_name} must be at least {rule.min_length} characters",
                field_name, value
            )
        
        if rule.max_length is not None and len(str_value) > rule.max_length:
            raise ValidationError(
                f"Field {field_name} must be at most {rule.max_length} characters",
                field_name, value
            )
        
        # Pattern validation
        if rule.pattern is not None:
            if not re.match(rule.pattern, str_value):
                raise ValidationError(
                    f"Field {field_name} format is invalid",
                    field_name, value
                )
        
        # Allowed values validation
        if rule.allowed_values is not None:
            if str_value not in rule.allowed_values:
                raise ValidationError(
                    f"Field {field_name} must be one of: {', '.join(rule.allowed_values)}",
                    field_name, value
                )
        
        # Custom validator
        if rule.custom_validator is not None:
            try:
                if not rule.custom_validator(value):
                    raise ValidationError(
                        f"Field {field_name} failed custom validation",
                        field_name, value
                    )
            except Exception as e:
                raise ValidationError(
                    f"Field {field_name} validation error: {str(e)}",
                    field_name, value
                )
        
        # Sanitize based on mode
        sanitized_value = self.sanitize_input(str_value, rule.sanitization_mode)
        
        return sanitized_value
    
    def validate_dict(
        self,
        data: Dict[str, Any],
        rules: Optional[Dict[str, ValidationRule]] = None
    ) -> Dict[str, Any]:
        """Validate a dictionary of data."""
        if rules is None:
            rules = self.predefined_rules
        
        validated_data = {}
        
        # Validate each field
        for field_name, value in data.items():
            if field_name in rules:
                validated_data[field_name] = self.validate_field(
                    field_name, value, rules[field_name]
                )
            else:
                # No rule found - apply basic sanitization
                if isinstance(value, str):
                    validated_data[field_name] = self.sanitize_input(value)
                else:
                    validated_data[field_name] = value
        
        # Check for missing required fields
        for field_name, rule in rules.items():
            if rule.required and field_name not in data:
                raise ValidationError(f"Required field {field_name} is missing", field_name)
        
        return validated_data
    
    def sanitize_input(
        self,
        text: str,
        mode: SanitizationMode = SanitizationMode.STRICT
    ) -> str:
        """Sanitize input text based on mode."""
        if not isinstance(text, str):
            text = str(text)
        
        # Check for dangerous patterns first
        self._check_dangerous_patterns(text)
        
        if mode == SanitizationMode.STRICT:
            return self._strict_sanitization(text)
        elif mode == SanitizationMode.PRESERVE:
            return self._preserve_sanitization(text)
        elif mode == SanitizationMode.MEDICAL:
            return self._medical_sanitization(text)
        elif mode == SanitizationMode.RESEARCH:
            return self._research_sanitization(text)
        else:
            return self._strict_sanitization(text)
    
    def _check_dangerous_patterns(self, text: str):
        """Check for dangerous injection patterns."""
        # SQL injection check
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValidationError("Potential SQL injection detected")
        
        # XSS check
        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValidationError("Potential XSS attack detected")
        
        # Command injection check
        for pattern in self.command_injection_patterns:
            if re.search(pattern, text):
                raise ValidationError("Potential command injection detected")
    
    def _strict_sanitization(self, text: str) -> str:
        """Strict sanitization - remove/escape all potentially dangerous content."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Escape HTML entities
        text = html.escape(text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\';\\&$|`]', '', text)
        
        return text
    
    def _preserve_sanitization(self, text: str) -> str:
        """Preserve structure but escape dangerous content."""
        # Escape HTML entities
        text = html.escape(text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _medical_sanitization(self, text: str) -> str:
        """Medical text sanitization - preserve medical terms and notation."""
        # Escape HTML but preserve medical notation
        text = html.escape(text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Preserve medical units and notation
        # This is basic - would need more sophisticated medical NLP
        text = re.sub(r'(\d+)\s*(mg|ml|kg|mcg|iu)\b', r'\1 \2', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _research_sanitization(self, text: str) -> str:
        """Research text sanitization - preserve scientific notation."""
        # Escape HTML
        text = html.escape(text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Preserve scientific notation
        text = re.sub(r'(\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)', r'\1e\2', text)
        
        # Preserve mathematical symbols (basic set)
        # In production, would use a more comprehensive mathematical parser
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _validate_ip_address(self, value: Any) -> bool:
        """Validate IP address."""
        try:
            ip = ipaddress.ip_address(str(value))
            # Reject private/local addresses in production
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False
            return True
        except ValueError:
            return False
    
    def _validate_url(self, value: Any) -> bool:
        """Validate URL."""
        try:
            result = urlparse(str(value))
            # Check for valid scheme and netloc
            if not result.scheme or not result.netloc:
                return False
            
            # Only allow specific schemes
            allowed_schemes = {'http', 'https'}
            if result.scheme not in allowed_schemes:
                return False
            
            # Check for suspicious patterns
            url_str = str(value).lower()
            suspicious_patterns = [
                'javascript:', 'data:', 'vbscript:', 'file:',
                'about:', 'chrome:', 'chrome-extension:'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in url_str:
                    return False
            
            return True
        except Exception:
            return False
    
    def _validate_json(self, value: Any) -> bool:
        """Validate JSON data."""
        if isinstance(value, dict):
            return True
        
        if isinstance(value, str):
            try:
                json.loads(value)
                return True
            except json.JSONDecodeError:
                return False
        
        return False
    
    def validate_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete API request."""
        # Basic structure validation
        if not isinstance(request_data, dict):
            raise ValidationError("Request data must be a dictionary")
        
        # Check for required API fields
        api_rules = {
            "prompt": self.predefined_rules["prompt"],
            "user_id": self.predefined_rules["user_id"],
            "model_name": self.predefined_rules["model_name"]
        }
        
        # Add optional fields with validation
        if "max_privacy_budget" in request_data:
            api_rules["max_privacy_budget"] = self.predefined_rules["privacy_budget"]
        
        if "department" in request_data:
            api_rules["department"] = self.predefined_rules["department"]
        
        # Validate against rules
        validated_data = self.validate_dict(request_data, api_rules)
        
        # Additional API-specific validation
        self._validate_api_specific_rules(validated_data)
        
        return validated_data
    
    def _validate_api_specific_rules(self, data: Dict[str, Any]):
        """Additional API-specific validation rules."""
        # Check prompt length for privacy considerations
        prompt = data.get("prompt", "")
        if len(prompt) > 4096:  # Reasonable limit for medical queries
            raise ValidationError("Prompt too long for privacy-preserving processing")
        
        # Check for potential privacy violations in prompt
        self._check_privacy_patterns(prompt)
        
        # Validate privacy budget range
        if "max_privacy_budget" in data:
            budget = float(data["max_privacy_budget"])
            if budget <= 0 or budget > 10.0:
                raise ValidationError("Privacy budget must be between 0 and 10.0")
    
    def _check_privacy_patterns(self, text: str):
        """Check for potential privacy-sensitive patterns."""
        for pattern in self.privacy_patterns:
            if re.search(pattern, text):
                raise ValidationError("Prompt may contain sensitive personal information")
    
    def sanitize_for_logging(self, data: Any) -> Any:
        """Sanitize data for safe logging."""
        if isinstance(data, str):
            # Remove potential PII patterns
            sanitized = data
            for pattern in self.privacy_patterns:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
            
            # Truncate if too long
            if len(sanitized) > 1000:
                sanitized = sanitized[:997] + "..."
            
            return sanitized
        
        elif isinstance(data, dict):
            sanitized_dict = {}
            for key, value in data.items():
                # Redact sensitive keys
                if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'token', 'key']):
                    sanitized_dict[key] = '[REDACTED]'
                else:
                    sanitized_dict[key] = self.sanitize_for_logging(value)
            return sanitized_dict
        
        elif isinstance(data, list):
            return [self.sanitize_for_logging(item) for item in data[:10]]  # Limit list size
        
        else:
            return data
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, token: str, expected_token: str) -> bool:
        """Validate CSRF token."""
        if not token or not expected_token:
            return False
        return secrets.compare_digest(token, expected_token)
    
    def hash_for_rate_limiting(self, identifier: str) -> str:
        """Create hash for rate limiting."""
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    def create_validation_rules(
        self,
        field_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ValidationRule]:
        """Create custom validation rules from configuration."""
        rules = {}
        
        for field_name, config in field_configs.items():
            rule = ValidationRule(
                field_name=field_name,
                required=config.get("required", True),
                min_length=config.get("min_length"),
                max_length=config.get("max_length"),
                pattern=config.get("pattern"),
                allowed_values=set(config.get("allowed_values", [])) if config.get("allowed_values") else None,
                sanitization_mode=SanitizationMode(config.get("sanitization_mode", "strict"))
            )
            
            # Add custom validator if specified
            if "custom_validator" in config:
                validator_name = config["custom_validator"]
                if hasattr(self, f"_validate_{validator_name}"):
                    rule.custom_validator = getattr(self, f"_validate_{validator_name}")
            
            rules[field_name] = rule
        
        return rules


# Global validator instance
_global_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator


def validate_input(field_name: str, value: Any, rule: Optional[ValidationRule] = None) -> Any:
    """Validate input using global validator."""
    return get_validator().validate_field(field_name, value, rule)


def sanitize_text(text: str, mode: SanitizationMode = SanitizationMode.STRICT) -> str:
    """Sanitize text using global validator."""
    return get_validator().sanitize_input(text, mode)


def validate_api_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate API request data."""
    return get_validator().validate_api_request(data)