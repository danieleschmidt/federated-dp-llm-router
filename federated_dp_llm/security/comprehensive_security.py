"""
Comprehensive Security Framework for Federated DP-LLM System

Implements multi-layer security controls including input validation,
threat detection, secure communications, and compliance monitoring.
"""

import asyncio
import hashlib
import hmac
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVACY_VIOLATION = "privacy_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: float
    user_id: Optional[str]
    source_ip: Optional[str]
    details: Dict[str, Any]
    blocked: bool = False
    investigated: bool = False


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    def __init__(self):
        # Common injection patterns
        self.sql_injection_patterns = [
            r"(union|select|insert|update|delete|drop|create|alter)\s+",
            r"(--|#|/\*|\*/)",
            r"(exec|execute|sp_|xp_)",
            r"(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
            r"(char|ascii|substring|concat)\s*\(",
        ]
        
        self.nosql_injection_patterns = [
            r"\$where\s*:",
            r"\$regex\s*:",
            r"\$gt\s*:",
            r"\$lt\s*:",
            r"\$ne\s*:",
        ]
        
        self.script_injection_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"vbscript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"document\.",
            r"window\.",
        ]
        
        self.prompt_injection_patterns = [
            r"ignore\s+(previous|all)\s+(instructions|prompts)",
            r"forget\s+(everything|all)",
            r"you\s+are\s+now\s+",
            r"system\s*:\s*",
            r"jailbreak",
            r"DAN\s+(mode|prompt)",
        ]
    
    def sanitize_prompt(self, prompt: str, max_length: int = 10000) -> Tuple[str, List[str]]:
        """Sanitize user prompt with comprehensive checks."""
        violations = []
        
        # Length check
        if len(prompt) > max_length:
            violations.append(f"Prompt too long: {len(prompt)} > {max_length}")
            prompt = prompt[:max_length]
        
        # Check for injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(f"SQL injection pattern detected: {pattern}")
        
        for pattern in self.nosql_injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(f"NoSQL injection pattern detected: {pattern}")
        
        for pattern in self.script_injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(f"Script injection pattern detected: {pattern}")
        
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                violations.append(f"Prompt injection pattern detected: {pattern}")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', prompt)
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)  # Control characters
        
        return sanitized, violations
    
    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate request parameters."""
        violations = []
        
        for key, value in params.items():
            # Key validation
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                violations.append(f"Invalid parameter name: {key}")
            
            # Value type checking
            if isinstance(value, str):
                _, param_violations = self.sanitize_prompt(value, max_length=1000)
                violations.extend([f"Parameter {key}: {v}" for v in param_violations])
            
            elif isinstance(value, (int, float)):
                if not (-1e6 <= value <= 1e6):  # Reasonable bounds
                    violations.append(f"Parameter {key} out of bounds: {value}")
        
        return violations


class ThreatDetector:
    """Detects security threats and anomalous behavior."""
    
    def __init__(self):
        self.user_activity: Dict[str, List[float]] = {}
        self.ip_activity: Dict[str, List[float]] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.rate_limits = {
            "requests_per_minute": 60,
            "failed_auth_per_hour": 5,
            "privacy_budget_per_hour": 10.0
        }
    
    def detect_rate_limiting_violation(self, user_id: str, ip_address: str) -> Optional[SecurityEvent]:
        """Detect rate limiting violations."""
        current_time = time.time()
        
        # Track user activity
        if user_id not in self.user_activity:
            self.user_activity[user_id] = []
        
        self.user_activity[user_id].append(current_time)
        
        # Clean old entries (last minute)
        cutoff = current_time - 60
        self.user_activity[user_id] = [t for t in self.user_activity[user_id] if t > cutoff]
        
        # Check rate limit
        if len(self.user_activity[user_id]) > self.rate_limits["requests_per_minute"]:
            return SecurityEvent(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=current_time,
                user_id=user_id,
                source_ip=ip_address,
                details={
                    "requests_in_minute": len(self.user_activity[user_id]),
                    "limit": self.rate_limits["requests_per_minute"]
                },
                blocked=True
            )
        
        return None
    
    def detect_anomalous_privacy_usage(self, user_id: str, epsilon_spent: float) -> Optional[SecurityEvent]:
        """Detect anomalous privacy budget usage."""
        # Simple anomaly detection based on sudden high usage
        if epsilon_spent > 1.0:  # Unusually high single request
            return SecurityEvent(
                event_type=SecurityEventType.PRIVACY_VIOLATION,
                threat_level=ThreatLevel.HIGH,
                timestamp=time.time(),
                user_id=user_id,
                source_ip=None,
                details={
                    "epsilon_spent": epsilon_spent,
                    "threshold": 1.0,
                    "reason": "unusually_high_single_request"
                }
            )
        
        return None
    
    def detect_data_exfiltration(self, user_id: str, query: str, response_length: int) -> Optional[SecurityEvent]:
        """Detect potential data exfiltration attempts."""
        # Check for patterns that might indicate data exfiltration
        exfiltration_keywords = [
            "list all", "show all", "dump", "export", "extract",
            "patient id", "social security", "credit card", "password"
        ]
        
        for keyword in exfiltration_keywords:
            if keyword.lower() in query.lower():
                return SecurityEvent(
                    event_type=SecurityEventType.DATA_EXFILTRATION,
                    threat_level=ThreatLevel.HIGH,
                    timestamp=time.time(),
                    user_id=user_id,
                    source_ip=None,
                    details={
                        "keyword": keyword,
                        "query_length": len(query),
                        "response_length": response_length
                    }
                )
        
        # Check for unusually long responses (potential data dump)
        if response_length > 50000:  # 50KB threshold
            return SecurityEvent(
                event_type=SecurityEventType.DATA_EXFILTRATION,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=time.time(),
                user_id=user_id,
                source_ip=None,
                details={
                    "response_length": response_length,
                    "threshold": 50000,
                    "reason": "unusually_large_response"
                }
            )
        
        return None


class SecureCommunication:
    """Handles secure communication between federated nodes."""
    
    def __init__(self):
        self.encryption_key = self._generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.message_timestamps: Dict[str, float] = {}
        self.nonces: Set[str] = set()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key from environment or create new."""
        key_env = os.environ.get('FEDERATED_ENCRYPTION_KEY')
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())
        else:
            return Fernet.generate_key()
    
    def encrypt_message(self, message: str, node_id: str) -> Dict[str, str]:
        """Encrypt message for secure transmission."""
        timestamp = str(time.time())
        nonce = base64.urlsafe_b64encode(os.urandom(16)).decode()
        
        # Create message with metadata
        full_message = f"{timestamp}|{node_id}|{nonce}|{message}"
        
        # Encrypt
        encrypted = self.cipher_suite.encrypt(full_message.encode())
        
        return {
            "encrypted_data": base64.urlsafe_b64encode(encrypted).decode(),
            "timestamp": timestamp,
            "nonce": nonce
        }
    
    def decrypt_message(self, encrypted_data: str, expected_node_id: str) -> Tuple[str, bool]:
        """Decrypt and validate message."""
        try:
            # Decode and decrypt
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            full_message = decrypted.decode()
            
            # Parse components
            parts = full_message.split('|', 3)
            if len(parts) != 4:
                return "", False
            
            timestamp_str, node_id, nonce, message = parts
            
            # Validate timestamp (within 5 minutes)
            timestamp = float(timestamp_str)
            if time.time() - timestamp > 300:
                logger.warning("Message timestamp too old")
                return "", False
            
            # Validate node ID
            if node_id != expected_node_id:
                logger.warning(f"Node ID mismatch: expected {expected_node_id}, got {node_id}")
                return "", False
            
            # Check nonce replay
            if nonce in self.nonces:
                logger.warning("Nonce replay detected")
                return "", False
            
            self.nonces.add(nonce)
            
            # Clean old nonces (keep last 1000)
            if len(self.nonces) > 1000:
                self.nonces = set(list(self.nonces)[-1000:])
            
            return message, True
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return "", False


class ComplianceMonitor:
    """Monitors system compliance with healthcare regulations."""
    
    def __init__(self):
        self.audit_events: List[Dict[str, Any]] = []
        self.compliance_violations: List[Dict[str, Any]] = []
        
    def log_data_access(self, user_id: str, data_type: str, action: str, 
                       patient_id: Optional[str] = None):
        """Log data access for HIPAA compliance."""
        event = {
            "timestamp": time.time(),
            "user_id": user_id,
            "data_type": data_type,
            "action": action,
            "patient_id": patient_id,
            "ip_address": None,  # Would be filled by request handler
            "session_id": None   # Would be filled by auth system
        }
        
        self.audit_events.append(event)
        
        # Keep only recent events (last 30 days)
        cutoff = time.time() - (30 * 24 * 3600)
        self.audit_events = [e for e in self.audit_events if e["timestamp"] > cutoff]
    
    def check_minimum_necessary(self, user_role: str, requested_data: List[str]) -> List[str]:
        """Check minimum necessary principle for data access."""
        # Define role-based data access rules
        role_permissions = {
            "doctor": ["patient_data", "medical_history", "diagnosis", "treatment"],
            "nurse": ["patient_data", "vital_signs", "medications"],
            "researcher": ["anonymized_data", "statistical_summaries"],
            "admin": ["system_logs", "audit_trails"]
        }
        
        allowed_data = role_permissions.get(user_role, [])
        violations = [data for data in requested_data if data not in allowed_data]
        
        if violations:
            self.compliance_violations.append({
                "timestamp": time.time(),
                "violation_type": "minimum_necessary",
                "user_role": user_role,
                "requested_data": requested_data,
                "violations": violations
            })
        
        return violations
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        total_accesses = len(self.audit_events)
        unique_users = len(set(e["user_id"] for e in self.audit_events))
        
        # Analyze access patterns
        access_by_type = {}
        for event in self.audit_events:
            data_type = event["data_type"]
            access_by_type[data_type] = access_by_type.get(data_type, 0) + 1
        
        return {
            "total_data_accesses": total_accesses,
            "unique_users": unique_users,
            "access_by_data_type": access_by_type,
            "compliance_violations": len(self.compliance_violations),
            "recent_violations": self.compliance_violations[-10:],  # Last 10
            "report_generated": time.time()
        }


class SecurityOrchestrator:
    """Main security orchestrator coordinating all security components."""
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.threat_detector = ThreatDetector()
        self.secure_comm = SecureCommunication()
        self.compliance_monitor = ComplianceMonitor()
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
    
    async def validate_request(self, prompt: str, user_id: str, ip_address: str,
                             parameters: Dict[str, Any]) -> Tuple[bool, List[str], Optional[str]]:
        """Comprehensive request validation."""
        violations = []
        sanitized_prompt = prompt
        
        # Input sanitization
        sanitized_prompt, sanitization_violations = self.input_sanitizer.sanitize_prompt(prompt)
        violations.extend(sanitization_violations)
        
        # Parameter validation
        param_violations = self.input_sanitizer.validate_parameters(parameters)
        violations.extend(param_violations)
        
        # Rate limiting check
        rate_event = self.threat_detector.detect_rate_limiting_violation(user_id, ip_address)
        if rate_event:
            self.security_events.append(rate_event)
            violations.append("Rate limit exceeded")
        
        # Check blocked lists
        if user_id in self.blocked_users:
            violations.append("User blocked due to security violations")
        
        if ip_address in self.blocked_ips:
            violations.append("IP address blocked due to security violations")
        
        # Determine if request should be allowed
        critical_violations = [v for v in violations if "injection" in v.lower()]
        allow_request = len(critical_violations) == 0 and len(violations) < 3
        
        # Block users/IPs with severe violations
        if critical_violations:
            self.blocked_users.add(user_id)
            self.blocked_ips.add(ip_address)
        
        return allow_request, violations, sanitized_prompt if allow_request else None
    
    async def monitor_response(self, user_id: str, query: str, response: str,
                             privacy_cost: float) -> List[SecurityEvent]:
        """Monitor response for security issues."""
        events = []
        
        # Privacy usage monitoring
        privacy_event = self.threat_detector.detect_anomalous_privacy_usage(user_id, privacy_cost)
        if privacy_event:
            events.append(privacy_event)
        
        # Data exfiltration detection
        exfiltration_event = self.threat_detector.detect_data_exfiltration(
            user_id, query, len(response)
        )
        if exfiltration_event:
            events.append(exfiltration_event)
        
        # Store events
        self.security_events.extend(events)
        
        return events
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard."""
        recent_events = [
            {
                "type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "timestamp": event.timestamp,
                "user_id": event.user_id,
                "blocked": event.blocked
            }
            for event in self.security_events[-50:]  # Last 50 events
        ]
        
        threat_distribution = {}
        for event in self.security_events:
            threat_distribution[event.threat_level.value] = threat_distribution.get(event.threat_level.value, 0) + 1
        
        return {
            "total_security_events": len(self.security_events),
            "recent_events": recent_events,
            "threat_distribution": threat_distribution,
            "blocked_users": len(self.blocked_users),
            "blocked_ips": len(self.blocked_ips),
            "compliance_status": self.compliance_monitor.get_compliance_report(),
            "dashboard_generated": time.time()
        }
    
    async def secure_node_communication(self, message: str, target_node: str) -> str:
        """Securely communicate with federated node."""
        encrypted = self.secure_comm.encrypt_message(message, target_node)
        return encrypted["encrypted_data"]
    
    async def receive_node_communication(self, encrypted_data: str, source_node: str) -> Optional[str]:
        """Receive and decrypt communication from federated node."""
        message, valid = self.secure_comm.decrypt_message(encrypted_data, source_node)
        return message if valid else None


# Global security orchestrator
global_security = SecurityOrchestrator()


def get_security_orchestrator() -> SecurityOrchestrator:
    """Get the global security orchestrator instance."""
    return global_security