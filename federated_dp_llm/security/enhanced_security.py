"""
Enhanced Security System

Provides advanced security features including rate limiting, intrusion detection,
secure session management, and threat monitoring.
"""

import asyncio
import hashlib
import hmac
import time
import uuid
import secrets
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import ipaddress
import re
from collections import defaultdict

from ..monitoring.logging_config import get_logger
from ..core.error_handling import get_error_handler, ErrorCategory, ErrorSeverity


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_REQUEST = "suspicious_request"
    INVALID_TOKEN = "invalid_token"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests_per_window: int
    window_seconds: int
    burst_limit: int = None  # Allow bursts up to this limit
    
    def __post_init__(self):
        if self.burst_limit is None:
            self.burst_limit = self.requests_per_window * 2


@dataclass
class SecurityEvent:
    """Security event information."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    timestamp: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[str] = None


@dataclass
class RequestPattern:
    """Pattern for suspicious request detection."""
    pattern_id: str
    name: str
    regex_patterns: List[str]
    threat_level: ThreatLevel
    description: str


class IPWhitelist:
    """IP address whitelist management."""
    
    def __init__(self):
        self.whitelisted_ips: Set[ipaddress.IPv4Address] = set()
        self.whitelisted_networks: Set[ipaddress.IPv4Network] = set()
        self.logger = get_logger("ip_whitelist")
        
        # Add common safe networks
        self._add_default_whitelist()
    
    def _add_default_whitelist(self):
        """Add default safe IP ranges."""
        safe_networks = [
            "127.0.0.0/8",    # Localhost
            "10.0.0.0/8",     # Private network
            "172.16.0.0/12",  # Private network
            "192.168.0.0/16", # Private network
        ]
        
        for network_str in safe_networks:
            try:
                network = ipaddress.IPv4Network(network_str, strict=False)
                self.whitelisted_networks.add(network)
            except ipaddress.AddressValueError as e:
                self.logger.warning(f"Invalid default network {network_str}: {e}")
    
    def add_ip(self, ip_str: str):
        """Add IP address to whitelist."""
        try:
            ip = ipaddress.IPv4Address(ip_str)
            self.whitelisted_ips.add(ip)
            self.logger.info(f"Added IP to whitelist: {ip_str}")
        except ipaddress.AddressValueError as e:
            self.logger.error(f"Invalid IP address {ip_str}: {e}")
            raise ValueError(f"Invalid IP address: {ip_str}")
    
    def add_network(self, network_str: str):
        """Add network to whitelist."""
        try:
            network = ipaddress.IPv4Network(network_str, strict=False)
            self.whitelisted_networks.add(network)
            self.logger.info(f"Added network to whitelist: {network_str}")
        except ipaddress.AddressValueError as e:
            self.logger.error(f"Invalid network {network_str}: {e}")
            raise ValueError(f"Invalid network: {network_str}")
    
    def is_whitelisted(self, ip_str: str) -> bool:
        """Check if IP is whitelisted."""
        try:
            ip = ipaddress.IPv4Address(ip_str)
            
            # Check direct IP matches
            if ip in self.whitelisted_ips:
                return True
            
            # Check network matches
            for network in self.whitelisted_networks:
                if ip in network:
                    return True
            
            return False
            
        except ipaddress.AddressValueError:
            self.logger.warning(f"Invalid IP address for whitelist check: {ip_str}")
            return False


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.request_logs: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        self.rules: Dict[str, RateLimitRule] = {}
        self.logger = get_logger("rate_limiter")
        
        # Default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules."""
        self.rules = {
            "default": RateLimitRule(requests_per_window=100, window_seconds=60),
            "auth": RateLimitRule(requests_per_window=10, window_seconds=300),
            "inference": RateLimitRule(requests_per_window=50, window_seconds=60),
            "admin": RateLimitRule(requests_per_window=20, window_seconds=60),
        }
    
    def add_rule(self, name: str, rule: RateLimitRule):
        """Add or update a rate limiting rule."""
        self.rules[name] = rule
        self.logger.info(f"Added rate limit rule '{name}': {rule.requests_per_window}/{rule.window_seconds}s")
    
    def is_allowed(self, identifier: str, rule_name: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if identifier in self.blocked_ips:
            unblock_time = self.blocked_ips[identifier]
            if current_time < unblock_time:
                return False, {
                    "blocked": True,
                    "unblock_time": unblock_time,
                    "reason": "temporarily_blocked"
                }
            else:
                # Unblock expired
                del self.blocked_ips[identifier]
        
        rule = self.rules.get(rule_name, self.rules["default"])
        
        # Get request history for identifier
        request_times = self.request_logs[identifier]
        
        # Remove old requests outside window
        window_start = current_time - rule.window_seconds
        request_times[:] = [t for t in request_times if t >= window_start]
        
        # Check limits
        current_count = len(request_times)
        
        if current_count >= rule.requests_per_window:
            # Check burst limit
            recent_requests = [t for t in request_times if t >= current_time - 10]  # Last 10 seconds
            
            if len(recent_requests) > rule.burst_limit:
                # Temporarily block aggressive users
                self.blocked_ips[identifier] = current_time + 300  # 5 minute block
                self.logger.warning(f"Temporarily blocked {identifier} for exceeding burst limit")
                
                return False, {
                    "blocked": True,
                    "burst_limit_exceeded": True,
                    "reason": "burst_limit"
                }
            
            return False, {
                "rate_limited": True,
                "requests_in_window": current_count,
                "limit": rule.requests_per_window,
                "window_seconds": rule.window_seconds,
                "retry_after": min(request_times) + rule.window_seconds - current_time,
                "reason": "rate_limit"
            }
        
        # Record this request
        request_times.append(current_time)
        
        return True, {
            "allowed": True,
            "requests_in_window": current_count + 1,
            "limit": rule.requests_per_window,
            "remaining": rule.requests_per_window - current_count - 1
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        current_time = time.time()
        active_users = len([
            identifier for identifier, times in self.request_logs.items()
            if any(t >= current_time - 300 for t in times)  # Active in last 5 minutes
        ])
        
        blocked_count = len([
            ip for ip, unblock_time in self.blocked_ips.items()
            if unblock_time > current_time
        ])
        
        return {
            "active_users": active_users,
            "blocked_ips": blocked_count,
            "total_tracked": len(self.request_logs),
            "rules": {name: rule.__dict__ for name, rule in self.rules.items()}
        }


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.logger = get_logger("threat_detector")
        self.patterns = self._create_threat_patterns()
        self.user_behavior: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
    def _create_threat_patterns(self) -> List[RequestPattern]:
        """Create threat detection patterns."""
        return [
            RequestPattern(
                pattern_id="sql_injection",
                name="SQL Injection Attempt",
                regex_patterns=[
                    r"(?i)(union\s+select|drop\s+table|delete\s+from)",
                    r"(?i)(';\s*(drop|delete|update|insert))",
                    r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)",
                    r"(?i)(exec\s*\(|execute\s*\()",
                ],
                threat_level=ThreatLevel.HIGH,
                description="Potential SQL injection attack"
            ),
            RequestPattern(
                pattern_id="xss_attempt",
                name="Cross-Site Scripting",
                regex_patterns=[
                    r"(?i)<script[^>]*>",
                    r"(?i)javascript:",
                    r"(?i)onload\s*=",
                    r"(?i)onerror\s*=",
                ],
                threat_level=ThreatLevel.MEDIUM,
                description="Potential XSS attack"
            ),
            RequestPattern(
                pattern_id="path_traversal",
                name="Path Traversal",
                regex_patterns=[
                    r"\.\.\/",
                    r"\.\.\x5c",
                    r"\/etc\/passwd",
                    r"\/windows\/system32",
                ],
                threat_level=ThreatLevel.HIGH,
                description="Potential path traversal attack"
            ),
            RequestPattern(
                pattern_id="command_injection",
                name="Command Injection",
                regex_patterns=[
                    r"(?i);(cat|ls|ps|whoami|id)\s",
                    r"(?i)\|\s*(cat|ls|ps|whoami|id)",
                    r"(?i)&\s*(cat|ls|ps|whoami|id)",
                ],
                threat_level=ThreatLevel.CRITICAL,
                description="Potential command injection attack"
            )
        ]
    
    def analyze_request(self, request_data: str, source_ip: str, 
                       user_id: Optional[str] = None) -> List[SecurityEvent]:
        """Analyze request for threats."""
        threats = []
        
        # Pattern-based detection
        for pattern in self.patterns:
            for regex_pattern in pattern.regex_patterns:
                if re.search(regex_pattern, request_data):
                    event = SecurityEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                        threat_level=pattern.threat_level,
                        source_ip=source_ip,
                        user_id=user_id,
                        timestamp=time.time(),
                        description=f"{pattern.name}: {pattern.description}",
                        metadata={
                            "pattern_id": pattern.pattern_id,
                            "matched_pattern": regex_pattern,
                            "request_sample": request_data[:200]
                        }
                    )
                    threats.append(event)
                    break  # One match per pattern is enough
        
        # Behavioral analysis
        if user_id:
            behavioral_threats = self._analyze_user_behavior(user_id, source_ip, request_data)
            threats.extend(behavioral_threats)
        
        return threats
    
    def _analyze_user_behavior(self, user_id: str, source_ip: str, 
                             request_data: str) -> List[SecurityEvent]:
        """Analyze user behavior for anomalies."""
        threats = []
        current_time = time.time()
        
        # Get or create user behavior profile
        behavior = self.user_behavior[user_id]
        
        # Initialize behavior tracking
        if "first_seen" not in behavior:
            behavior["first_seen"] = current_time
            behavior["ip_addresses"] = set()
            behavior["request_patterns"] = []
            behavior["last_activity"] = current_time
            behavior["total_requests"] = 0
        
        behavior["ip_addresses"].add(source_ip)
        behavior["last_activity"] = current_time
        behavior["total_requests"] += 1
        
        # Detect IP address anomalies
        if len(behavior["ip_addresses"]) > 10:  # Too many IPs for one user
            threats.append(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                user_id=user_id,
                timestamp=current_time,
                description="User accessing from unusual number of IP addresses",
                metadata={"ip_count": len(behavior["ip_addresses"])}
            ))
        
        # Detect unusual request frequency
        recent_window = 60  # 1 minute
        recent_requests = [
            t for t in behavior["request_patterns"]
            if t >= current_time - recent_window
        ]
        
        if len(recent_requests) > 50:  # More than 50 requests per minute
            threats.append(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.HIGH,
                source_ip=source_ip,
                user_id=user_id,
                timestamp=current_time,
                description="Unusually high request frequency",
                metadata={"requests_per_minute": len(recent_requests)}
            ))
        
        # Update behavior pattern
        behavior["request_patterns"].append(current_time)
        
        # Keep only recent patterns to prevent memory bloat
        behavior["request_patterns"] = [
            t for t in behavior["request_patterns"]
            if t >= current_time - 3600  # Keep last hour
        ]
        
        return threats


class SecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self):
        self.logger = get_logger("security_manager")
        self.rate_limiter = RateLimiter()
        self.threat_detector = ThreatDetector()
        self.ip_whitelist = IPWhitelist()
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
        
        # Security callbacks
        self.event_callbacks: List[Callable[[SecurityEvent], None]] = []
        
    def add_event_callback(self, callback):
        """Add callback for security events."""
        self.event_callbacks.append(callback)
    
    async def validate_request(self, request_data: str, source_ip: str,
                              user_id: Optional[str] = None,
                              endpoint: str = None) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive request validation."""
        
        # Check IP whitelist first
        if not self.ip_whitelist.is_whitelisted(source_ip):
            # Check if IP is blocked
            if source_ip in self.blocked_ips:
                return False, {
                    "blocked": True,
                    "reason": "ip_blocked",
                    "message": "IP address is blocked due to security violations"
                }
        
        # Check if user is blocked
        if user_id and user_id in self.blocked_users:
            return False, {
                "blocked": True,
                "reason": "user_blocked",
                "message": "User account is blocked due to security violations"
            }
        
        # Rate limiting
        rule_name = self._get_rate_limit_rule(endpoint)
        identifier = user_id or source_ip
        
        allowed, rate_info = self.rate_limiter.is_allowed(identifier, rule_name)
        if not allowed:
            # Log rate limit event
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                user_id=user_id,
                timestamp=time.time(),
                description="Rate limit exceeded",
                metadata=rate_info
            )
            
            await self._handle_security_event(event)
            
            return False, {
                "rate_limited": True,
                **rate_info
            }
        
        # Threat detection
        threats = self.threat_detector.analyze_request(request_data, source_ip, user_id)
        
        if threats:
            for threat in threats:
                await self._handle_security_event(threat)
            
            # Block critical threats immediately
            critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
            if critical_threats:
                if source_ip not in self.ip_whitelist.whitelisted_ips:
                    self.blocked_ips.add(source_ip)
                
                return False, {
                    "blocked": True,
                    "reason": "security_threat",
                    "threat_level": "critical",
                    "threats": [t.description for t in critical_threats]
                }
        
        return True, {
            "allowed": True,
            "rate_info": rate_info,
            "threats_detected": len(threats)
        }
    
    def _get_rate_limit_rule(self, endpoint: Optional[str]) -> str:
        """Determine rate limit rule based on endpoint."""
        if not endpoint:
            return "default"
        
        if "/auth" in endpoint:
            return "auth"
        elif "/inference" in endpoint:
            return "inference"
        elif "/admin" in endpoint:
            return "admin"
        else:
            return "default"
    
    async def _handle_security_event(self, event: SecurityEvent):
        """Handle security event."""
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        # Log event
        log_level = {
            ThreatLevel.LOW: "info",
            ThreatLevel.MEDIUM: "warning",
            ThreatLevel.HIGH: "error",
            ThreatLevel.CRITICAL: "critical"
        }.get(event.threat_level, "warning")
        
        getattr(self.logger, log_level)(
            f"Security event: {event.description}",
            extra={
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "metadata": event.metadata
            }
        )
        
        # Report to error handler
        error_handler = get_error_handler()
        try:
            await error_handler.handle_error(
                Exception(f"Security event: {event.description}"),
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.HIGH if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] else ErrorSeverity.MEDIUM
            )
        except Exception as e:
            self.logger.error(f"Failed to report security event to error handler: {e}")
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in security event callback: {e}")
    
    def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block IP address."""
        self.blocked_ips.add(ip_address)
        self.logger.warning(f"Blocked IP {ip_address}: {reason}")
    
    def block_user(self, user_id: str, reason: str = "Security violation"):
        """Block user."""
        self.blocked_users.add(user_id)
        self.logger.warning(f"Blocked user {user_id}: {reason}")
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address."""
        self.blocked_ips.discard(ip_address)
        self.logger.info(f"Unblocked IP {ip_address}")
    
    def unblock_user(self, user_id: str):
        """Unblock user."""
        self.blocked_users.discard(user_id)
        self.logger.info(f"Unblocked user {user_id}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        current_time = time.time()
        recent_events = [
            e for e in self.security_events
            if current_time - e.timestamp <= 3600  # Last hour
        ]
        
        event_counts = {}
        for event in recent_events:
            key = f"{event.event_type.value}:{event.threat_level.value}"
            event_counts[key] = event_counts.get(key, 0) + 1
        
        return {
            "total_events": len(self.security_events),
            "recent_events": len(recent_events),
            "event_counts": event_counts,
            "blocked_ips": len(self.blocked_ips),
            "blocked_users": len(self.blocked_users),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "patterns_count": len(self.threat_detector.patterns)
        }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create the global security manager."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


class SecurityLevel(Enum):
    """Security level configurations."""
    BASIC = "basic"
    HEALTHCARE_HIPAA = "healthcare_hipaa"
    GOVERNMENT = "government"
    MILITARY = "military"


@dataclass 
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    risk_score: float = 0.0


class SecurityValidator:
    """Security validation for requests."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.BASIC,
                 enable_threat_detection: bool = True,
                 enable_access_logging: bool = True):
        self.security_level = security_level
        self.enable_threat_detection = enable_threat_detection
        self.enable_access_logging = enable_access_logging
        
    def validate_inference_request(self, inputs: Dict[str, Any]) -> ValidationResult:
        """Validate inference request inputs."""
        issues = []
        risk_score = 0.0
        
        # Basic validation
        if not inputs.get('user_id'):
            issues.append("Missing user_id")
            risk_score += 0.3
            
        if not inputs.get('user_prompt'):
            issues.append("Missing user_prompt")
            risk_score += 0.2
            
        # Privacy budget validation
        privacy_budget = inputs.get('privacy_budget', 0)
        if privacy_budget <= 0 or privacy_budget > 10:
            issues.append(f"Invalid privacy budget: {privacy_budget}")
            risk_score += 0.4
            
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            risk_score=risk_score
        )


class ThreatDetector:
    """Threat detection system."""
    
    def __init__(self):
        self.patterns = []
        
    def analyze_request_pattern(self, user_id: str, request_frequency: int,
                               unusual_access_patterns: bool) -> Optional[str]:
        """Analyze request patterns for threats."""
        if request_frequency > 100:  # More than 100 requests per minute
            return f"High request frequency detected for user {user_id}"
            
        if unusual_access_patterns:
            return f"Unusual access patterns detected for user {user_id}"
            
        return None


class AccessController:
    """Access control system."""
    
    def __init__(self):
        self.role_permissions = {
            "physician": ["inference_request", "patient_query", "diagnostic_assistance"],
            "nurse": ["basic_query", "patient_lookup"],
            "researcher": ["anonymized_query", "statistical_analysis"],
            "admin": ["system_management", "user_administration"]
        }
        
    def check_role_permissions(self, user_id: str, role: str, 
                              requested_operation: str, 
                              resource_sensitivity: str) -> bool:
        """Check if user role has permission for operation."""
        allowed_operations = self.role_permissions.get(role, [])
        
        if requested_operation not in allowed_operations:
            return False
            
        # Additional sensitivity checks
        if resource_sensitivity == "high" and role not in ["physician", "admin"]:
            return False
            
        return True