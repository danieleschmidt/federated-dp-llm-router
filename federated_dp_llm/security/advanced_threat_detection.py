"""
Advanced Threat Detection for Federated DP-LLM Router

Implements real-time threat detection, anomaly analysis, and automated response
for healthcare federated learning environments with enhanced privacy protection.
"""

import asyncio
import time
import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import ipaddress


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat severity levels."""
    INFO = auto()        # Informational, no immediate action
    LOW = auto()         # Minor concern, log and monitor
    MEDIUM = auto()      # Moderate threat, apply countermeasures
    HIGH = auto()        # Serious threat, immediate action required
    CRITICAL = auto()    # System compromise, emergency response


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVACY_VIOLATION = "privacy_violation"
    DDOS_ATTACK = "ddos_attack"
    MALFORMED_REQUEST = "malformed_request"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"


@dataclass
class SecurityEvent:
    """Represents a security event."""
    event_id: str
    timestamp: float
    source_ip: str
    user_id: Optional[str]
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatPattern:
    """Pattern for threat detection."""
    pattern_id: str
    name: str
    threat_type: ThreatType
    regex_patterns: List[str] = field(default_factory=list)
    keyword_patterns: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    severity_multiplier: float = 1.0


class AdvancedThreatDetector:
    """Advanced threat detection and response system."""
    
    def __init__(self, max_events: int = 10000):
        self.security_events = deque(maxlen=max_events)
        self.threat_patterns = self._initialize_threat_patterns()
        self.ip_tracking = defaultdict(lambda: {"requests": 0, "last_reset": time.time()})
        self.user_tracking = defaultdict(lambda: {"failed_attempts": 0, "last_attempt": 0})
        self.blocked_ips = set()
        self.suspicious_patterns_cache = {}
        
    def _initialize_threat_patterns(self) -> List[ThreatPattern]:
        """Initialize threat detection patterns."""
        return [
            # SQL Injection Patterns
            ThreatPattern(
                pattern_id="sql_injection_1",
                name="SQL Injection - Basic",
                threat_type=ThreatType.SQL_INJECTION,
                regex_patterns=[
                    r"('|(\\'))(union|select|insert|delete|drop|create|alter|exec)",
                    r"(union\s+select|select\s+\*\s+from|drop\s+table)",
                    r"(\bor\b\s+\d+\s*=\s*\d+|\band\b\s+\d+\s*=\s*\d+)",
                ],
                severity_multiplier=2.0
            ),
            
            # XSS Attack Patterns
            ThreatPattern(
                pattern_id="xss_attack_1",
                name="Cross-Site Scripting",
                threat_type=ThreatType.XSS_ATTACK,
                regex_patterns=[
                    r"<script[\s\S]*?>[\s\S]*?</script>",
                    r"javascript:\s*[^;]*",
                    r"on\w+\s*=\s*['\"][^'\"]*['\"]",
                ],
                severity_multiplier=1.5
            ),
            
            # Data Exfiltration Patterns
            ThreatPattern(
                pattern_id="data_exfiltration_1",
                name="Potential Data Exfiltration",
                threat_type=ThreatType.DATA_EXFILTRATION,
                keyword_patterns=[
                    "patient_data", "medical_records", "phi", "ssn", "credit_card",
                    "password", "token", "secret", "private_key"
                ],
                severity_multiplier=3.0
            ),
            
            # Privacy Violation Patterns
            ThreatPattern(
                pattern_id="privacy_violation_1",
                name="Privacy Budget Manipulation",
                threat_type=ThreatType.PRIVACY_VIOLATION,
                regex_patterns=[
                    r"epsilon\s*=\s*[0-9]+(\.[0-9]+)?",
                    r"privacy_budget\s*=\s*[0-9]+",
                    r"bypass.*privacy",
                ],
                severity_multiplier=2.5
            ),
            
            # Rate Limiting Violations
            ThreatPattern(
                pattern_id="rate_limit_1",
                name="Rate Limit Violations",
                threat_type=ThreatType.RATE_LIMIT_VIOLATION,
                rate_limits={"per_minute": 60, "per_hour": 1000, "per_day": 10000},
                severity_multiplier=1.2
            ),
            
            # Malformed Request Patterns
            ThreatPattern(
                pattern_id="malformed_request_1",
                name="Malformed Request Detection",
                threat_type=ThreatType.MALFORMED_REQUEST,
                regex_patterns=[
                    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",  # Control characters
                    r"(\\x[0-9a-f]{2}){5,}",              # Excessive URL encoding
                    r"(%[0-9a-f]{2}){10,}",                # Excessive percent encoding
                ],
                severity_multiplier=1.3
            )
        ]
    
    async def analyze_request(self, 
                            source_ip: str, 
                            user_id: Optional[str], 
                            request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Analyze incoming request for security threats."""
        events = []
        current_time = time.time()
        
        # Check if IP is blocked
        if source_ip in self.blocked_ips:
            events.append(self._create_event(
                source_ip, user_id, ThreatType.UNAUTHORIZED_ACCESS,
                ThreatLevel.HIGH, "Request from blocked IP address",
                request_data
            ))
            return events
        
        # Rate limiting check
        rate_events = await self._check_rate_limits(source_ip, current_time)
        events.extend(rate_events)
        
        # Pattern matching
        pattern_events = await self._check_threat_patterns(source_ip, user_id, request_data)
        events.extend(pattern_events)
        
        # Behavioral analysis
        behavior_events = await self._analyze_behavior(source_ip, user_id, request_data)
        events.extend(behavior_events)
        
        # Store events
        for event in events:
            self.security_events.append(event)
            
        return events
    
    async def _check_rate_limits(self, source_ip: str, current_time: float) -> List[SecurityEvent]:
        """Check rate limiting violations."""
        events = []
        tracking = self.ip_tracking[source_ip]
        
        # Reset counters if needed (per minute window)
        if current_time - tracking["last_reset"] > 60:
            tracking["requests"] = 0
            tracking["last_reset"] = current_time
        
        tracking["requests"] += 1
        
        # Find rate limit patterns
        for pattern in self.threat_patterns:
            if pattern.threat_type == ThreatType.RATE_LIMIT_VIOLATION:
                if "per_minute" in pattern.rate_limits:
                    if tracking["requests"] > pattern.rate_limits["per_minute"]:
                        events.append(self._create_event(
                            source_ip, None, ThreatType.RATE_LIMIT_VIOLATION,
                            ThreatLevel.MEDIUM,
                            f"Rate limit exceeded: {tracking['requests']} requests/minute"
                        ))
                        
                        # Auto-block aggressive IPs
                        if tracking["requests"] > pattern.rate_limits["per_minute"] * 2:
                            self.blocked_ips.add(source_ip)
                            logger.warning(f"Auto-blocked IP {source_ip} for rate limit violations")
        
        return events
    
    async def _check_threat_patterns(self, 
                                   source_ip: str, 
                                   user_id: Optional[str], 
                                   request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Check request data against threat patterns."""
        events = []
        
        # Convert request data to searchable text
        request_text = json.dumps(request_data, default=str).lower()
        
        for pattern in self.threat_patterns:
            if pattern.threat_type == ThreatType.RATE_LIMIT_VIOLATION:
                continue  # Handled separately
            
            threat_detected = False
            match_details = []
            
            # Check regex patterns
            for regex_pattern in pattern.regex_patterns:
                try:
                    if re.search(regex_pattern, request_text, re.IGNORECASE):
                        threat_detected = True
                        match_details.append(f"Regex: {regex_pattern[:50]}")
                except re.error:
                    logger.warning(f"Invalid regex pattern: {regex_pattern}")
            
            # Check keyword patterns
            for keyword in pattern.keyword_patterns:
                if keyword.lower() in request_text:
                    threat_detected = True
                    match_details.append(f"Keyword: {keyword}")
            
            if threat_detected:
                severity = self._calculate_severity(pattern, match_details)
                events.append(self._create_event(
                    source_ip, user_id, pattern.threat_type, severity,
                    f"{pattern.name}: {', '.join(match_details[:3])}",
                    request_data, {"pattern_id": pattern.pattern_id}
                ))
        
        return events
    
    async def _analyze_behavior(self, 
                              source_ip: str, 
                              user_id: Optional[str], 
                              request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Analyze behavioral patterns for anomalies."""
        events = []
        
        # Check for suspicious IP patterns
        if await self._is_suspicious_ip(source_ip):
            events.append(self._create_event(
                source_ip, user_id, ThreatType.SUSPICIOUS_PATTERN,
                ThreatLevel.LOW, "Suspicious IP address pattern detected"
            ))
        
        # Check for privacy budget manipulation attempts
        if user_id and await self._check_privacy_manipulation(user_id, request_data):
            events.append(self._create_event(
                source_ip, user_id, ThreatType.PRIVACY_VIOLATION,
                ThreatLevel.HIGH, "Potential privacy budget manipulation"
            ))
        
        # Check for unusual request patterns
        if await self._check_unusual_patterns(source_ip, request_data):
            events.append(self._create_event(
                source_ip, user_id, ThreatType.SUSPICIOUS_PATTERN,
                ThreatLevel.MEDIUM, "Unusual request pattern detected"
            ))
        
        return events
    
    async def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP address shows suspicious patterns."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check if it's a private IP from unexpected ranges
            if ip_obj.is_private and not str(ip_obj).startswith(('10.', '172.', '192.168.')):
                return True
            
            # Check against known malicious patterns
            suspicious_patterns = [
                r'^169\.254\.',  # Link-local addresses
                r'^224\.',       # Multicast addresses
                r'^0\.',         # Invalid addresses
            ]
            
            for pattern in suspicious_patterns:
                if re.match(pattern, ip):
                    return True
            
        except ValueError:
            return True  # Invalid IP format
        
        return False
    
    async def _check_privacy_manipulation(self, user_id: str, request_data: Dict[str, Any]) -> bool:
        """Check for privacy budget manipulation attempts."""
        request_str = json.dumps(request_data, default=str).lower()
        
        suspicious_terms = [
            "bypass_privacy", "infinite_budget", "reset_epsilon",
            "modify_delta", "fake_noise", "zero_privacy"
        ]
        
        return any(term in request_str for term in suspicious_terms)
    
    async def _check_unusual_patterns(self, source_ip: str, request_data: Dict[str, Any]) -> bool:
        """Check for unusual request patterns."""
        # Check request size (unusually large requests)
        request_size = len(json.dumps(request_data, default=str))
        if request_size > 100000:  # 100KB
            return True
        
        # Check for repeated identical requests
        request_hash = hashlib.sha256(
            json.dumps(request_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        cache_key = f"{source_ip}:{request_hash}"
        current_time = time.time()
        
        if cache_key in self.suspicious_patterns_cache:
            last_time, count = self.suspicious_patterns_cache[cache_key]
            if current_time - last_time < 60:  # Within last minute
                self.suspicious_patterns_cache[cache_key] = (current_time, count + 1)
                return count > 5  # More than 5 identical requests
        
        self.suspicious_patterns_cache[cache_key] = (current_time, 1)
        return False
    
    def _calculate_severity(self, pattern: ThreatPattern, match_details: List[str]) -> ThreatLevel:
        """Calculate threat severity based on pattern and matches."""
        base_severity = len(match_details) * pattern.severity_multiplier
        
        if base_severity >= 5.0:
            return ThreatLevel.CRITICAL
        elif base_severity >= 3.0:
            return ThreatLevel.HIGH
        elif base_severity >= 2.0:
            return ThreatLevel.MEDIUM
        elif base_severity >= 1.0:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO
    
    def _create_event(self, 
                     source_ip: str, 
                     user_id: Optional[str], 
                     threat_type: ThreatType,
                     threat_level: ThreatLevel, 
                     description: str,
                     request_data: Dict[str, Any] = None,
                     metadata: Dict[str, Any] = None) -> SecurityEvent:
        """Create a security event."""
        event_id = hashlib.sha256(
            f"{time.time()}:{source_ip}:{threat_type.value}:{description}".encode()
        ).hexdigest()[:16]
        
        return SecurityEvent(
            event_id=event_id,
            timestamp=time.time(),
            source_ip=source_ip,
            user_id=user_id,
            threat_type=threat_type,
            threat_level=threat_level,
            description=description,
            request_data=request_data or {},
            metadata=metadata or {}
        )
    
    def get_security_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get security summary for the specified time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_events = [
            event for event in self.security_events 
            if event.timestamp >= cutoff_time
        ]
        
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        source_ips = set()
        
        for event in recent_events:
            threat_counts[event.threat_type.value] += 1
            severity_counts[event.threat_level.name] += 1
            source_ips.add(event.source_ip)
        
        return {
            "time_window_hours": time_window / 3600,
            "total_events": len(recent_events),
            "unique_source_ips": len(source_ips),
            "blocked_ips": len(self.blocked_ips),
            "threat_breakdown": dict(threat_counts),
            "severity_breakdown": dict(severity_counts),
            "top_threats": sorted(threat_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        }
    
    def get_critical_events(self, limit: int = 10) -> List[SecurityEvent]:
        """Get most recent critical security events."""
        critical_events = [
            event for event in reversed(self.security_events)
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        return critical_events[:limit]
    
    def unblock_ip(self, ip: str) -> bool:
        """Manually unblock an IP address."""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked IP address: {ip}")
            return True
        return False
    
    def add_custom_pattern(self, pattern: ThreatPattern):
        """Add a custom threat detection pattern."""
        self.threat_patterns.append(pattern)
        logger.info(f"Added custom threat pattern: {pattern.name}")
    
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        summary = self.get_security_summary()
        critical_events = self.get_critical_events()
        
        return {
            "report_timestamp": time.time(),
            "summary": summary,
            "critical_events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "source_ip": event.source_ip,
                    "threat_type": event.threat_type.value,
                    "threat_level": event.threat_level.name,
                    "description": event.description
                } for event in critical_events
            ],
            "system_status": {
                "total_patterns": len(self.threat_patterns),
                "blocked_ips": len(self.blocked_ips),
                "active_tracking": len(self.ip_tracking)
            }
        }


# Global threat detector instance
threat_detector = AdvancedThreatDetector()


# Middleware function for FastAPI integration
async def security_middleware(request, call_next):
    """Security middleware for threat detection."""
    start_time = time.time()
    
    # Extract request info
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("user-id")
    
    # Build request data for analysis
    request_data = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "query_params": dict(request.query_params)
    }
    
    # Analyze request
    security_events = await threat_detector.analyze_request(client_ip, user_id, request_data)
    
    # Check for blocking conditions
    should_block = any(
        event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] 
        for event in security_events
    )
    
    if should_block:
        logger.warning(f"Blocking request from {client_ip} due to security threats")
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Access denied due to security policy")
    
    # Process request
    response = await call_next(request)
    
    # Log security events
    if security_events:
        for event in security_events:
            logger.info(f"Security event: {event.threat_type.value} from {client_ip}")
    
    return response