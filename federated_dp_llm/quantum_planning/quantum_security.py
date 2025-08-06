"""
Quantum Planning Security Framework

Advanced security controls for quantum-inspired task planning
with healthcare-grade privacy protection and threat mitigation.
"""

import asyncio
import time
import hashlib
import hmac
import secrets
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import json
import base64

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVACY_VIOLATION = "privacy_violation"
    DATA_EXFILTRATION = "data_exfiltration"
    QUANTUM_STATE_TAMPERING = "quantum_state_tampering"
    MEASUREMENT_INTERFERENCE = "measurement_interference"
    ENTANGLEMENT_ATTACK = "entanglement_attack"
    SUPERPOSITION_COLLAPSE_ATTACK = "superposition_collapse_attack"
    SIDE_CHANNEL_ATTACK = "side_channel_attack"
    TIMING_ATTACK = "timing_attack"


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: float
    component: str
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    attack_vector: Optional[str] = None
    mitigation_applied: bool = False
    resolved: bool = False
    impact_assessment: str = ""


@dataclass
class QuantumSecurityContext:
    """Security context for quantum operations."""
    operation_id: str
    user_id: str
    security_level: SecurityLevel
    privacy_budget_allocated: float
    allowed_operations: Set[str]
    encryption_required: bool = True
    audit_trail_required: bool = True
    
    # Quantum-specific security parameters
    measurement_authentication: bool = True
    entanglement_verification: bool = True
    superposition_integrity_check: bool = True
    quantum_signature_required: bool = False


class QuantumCryptographer:
    """Quantum-safe cryptographic operations."""
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        self.public_key = self.private_key.public_key()
        
        # Quantum-safe key derivation
        self.master_secret = secrets.token_bytes(64)
        self.quantum_nonce_counter = 0
    
    def generate_quantum_signature(self, data: bytes, context: QuantumSecurityContext) -> bytes:
        """Generate quantum-resistant digital signature."""
        
        # Create signature payload
        payload = {
            "operation_id": context.operation_id,
            "user_id": context.user_id,
            "security_level": context.security_level.value,
            "timestamp": time.time(),
            "data_hash": hashlib.sha3_256(data).hexdigest()
        }
        
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        
        # Sign with private key
        signature = self.private_key.sign(
            payload_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature)
    
    def verify_quantum_signature(self, data: bytes, signature: bytes, context: QuantumSecurityContext) -> bool:
        """Verify quantum signature authenticity."""
        
        try:
            signature_bytes = base64.b64decode(signature)
            
            # Reconstruct payload
            payload = {
                "operation_id": context.operation_id,
                "user_id": context.user_id,
                "security_level": context.security_level.value,
                "timestamp": time.time(),
                "data_hash": hashlib.sha3_256(data).hexdigest()
            }
            
            payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            
            # Verify signature
            self.public_key.verify(
                signature_bytes,
                payload_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Signature verification failed: {str(e)}")
            return False
    
    def encrypt_quantum_state(self, quantum_state: Dict[str, Any], context: QuantumSecurityContext) -> bytes:
        """Encrypt quantum state data."""
        
        # Generate unique nonce for this operation
        nonce = secrets.token_bytes(16)
        
        # Derive encryption key from master secret and context
        key_material = self.master_secret + context.operation_id.encode('utf-8')
        encryption_key = hashlib.sha3_256(key_material).digest()
        
        # Encrypt quantum state
        cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        
        plaintext = json.dumps(quantum_state).encode('utf-8')
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Return encrypted data with nonce and auth tag
        encrypted_data = {
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "auth_tag": base64.b64encode(encryptor.tag).decode('utf-8'),
            "operation_id": context.operation_id
        }
        
        return json.dumps(encrypted_data).encode('utf-8')
    
    def decrypt_quantum_state(self, encrypted_data: bytes, context: QuantumSecurityContext) -> Dict[str, Any]:
        """Decrypt quantum state data."""
        
        try:
            data_dict = json.loads(encrypted_data.decode('utf-8'))
            
            # Verify operation ID matches
            if data_dict["operation_id"] != context.operation_id:
                raise ValueError("Operation ID mismatch")
            
            # Extract components
            nonce = base64.b64decode(data_dict["nonce"])
            ciphertext = base64.b64decode(data_dict["ciphertext"])
            auth_tag = base64.b64decode(data_dict["auth_tag"])
            
            # Derive decryption key
            key_material = self.master_secret + context.operation_id.encode('utf-8')
            encryption_key = hashlib.sha3_256(key_material).digest()
            
            # Decrypt
            cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(nonce, auth_tag))
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return json.loads(plaintext.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Failed to decrypt quantum state")


class QuantumSecurityMonitor:
    """Security monitoring and threat detection for quantum systems."""
    
    def __init__(self):
        self.security_incidents: List[SecurityIncident] = []
        self.threat_indicators: Dict[str, List[float]] = {}
        self.security_policies: Dict[str, Any] = {}
        self.access_control_matrix: Dict[str, Dict[str, bool]] = {}
        
        # Initialize default security policies
        self._initialize_security_policies()
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_task = None
        
        # Threat detection thresholds
        self.threat_thresholds = {
            "measurement_frequency_anomaly": 10.0,  # measurements per second
            "entanglement_creation_rate": 5.0,     # entanglements per minute
            "superposition_collapse_rate": 0.3,    # unusual collapse rate
            "privacy_budget_consumption_rate": 1.0, # budget per minute
            "quantum_state_modification_rate": 20.0 # modifications per minute
        }
    
    def _initialize_security_policies(self):
        """Initialize default security policies."""
        
        self.security_policies = {
            "privacy_budget_limits": {
                SecurityLevel.PUBLIC.value: 0.1,
                SecurityLevel.INTERNAL.value: 0.5,
                SecurityLevel.CONFIDENTIAL.value: 1.0,
                SecurityLevel.RESTRICTED.value: 2.0,
                SecurityLevel.TOP_SECRET.value: 5.0
            },
            "allowed_quantum_operations": {
                SecurityLevel.PUBLIC.value: ["measurement", "basic_planning"],
                SecurityLevel.INTERNAL.value: ["measurement", "planning", "superposition"],
                SecurityLevel.CONFIDENTIAL.value: ["measurement", "planning", "superposition", "entanglement"],
                SecurityLevel.RESTRICTED.value: ["measurement", "planning", "superposition", "entanglement", "interference"],
                SecurityLevel.TOP_SECRET.value: ["all"]
            },
            "encryption_requirements": {
                SecurityLevel.PUBLIC.value: False,
                SecurityLevel.INTERNAL.value: True,
                SecurityLevel.CONFIDENTIAL.value: True,
                SecurityLevel.RESTRICTED.value: True,
                SecurityLevel.TOP_SECRET.value: True
            },
            "audit_requirements": {
                SecurityLevel.PUBLIC.value: False,
                SecurityLevel.INTERNAL.value: True,
                SecurityLevel.CONFIDENTIAL.value: True,
                SecurityLevel.RESTRICTED.value: True,
                SecurityLevel.TOP_SECRET.value: True
            }
        }
    
    async def start_security_monitoring(self):
        """Start security monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._security_monitoring_loop())
        logger.info("Quantum security monitoring started")
    
    async def stop_security_monitoring(self):
        """Stop security monitoring."""
        self._monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Quantum security monitoring stopped")
    
    async def _security_monitoring_loop(self):
        """Main security monitoring loop."""
        while self._monitoring_active:
            try:
                # Analyze threat indicators
                await self._analyze_threat_indicators()
                
                # Check for security policy violations
                await self._check_policy_violations()
                
                # Detect anomalous quantum behavior
                await self._detect_quantum_anomalies()
                
                # Clean up old incidents
                await self._cleanup_old_incidents()
                
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Security monitoring error: {str(e)}")
                await asyncio.sleep(30.0)
    
    async def validate_security_context(self, context: QuantumSecurityContext, operation: str) -> bool:
        """Validate security context for quantum operation."""
        
        # Check if operation is allowed for this security level
        allowed_ops = self.security_policies["allowed_quantum_operations"].get(
            context.security_level.value, []
        )
        
        if "all" not in allowed_ops and operation not in allowed_ops:
            await self._create_security_incident(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                ThreatLevel.HIGH,
                f"Unauthorized operation {operation} for security level {context.security_level.value}",
                component="quantum_access_control",
                user_id=context.user_id
            )
            return False
        
        # Check privacy budget limits
        budget_limit = self.security_policies["privacy_budget_limits"].get(
            context.security_level.value, 0.1
        )
        
        if context.privacy_budget_allocated > budget_limit:
            await self._create_security_incident(
                SecurityEvent.PRIVACY_VIOLATION,
                ThreatLevel.CRITICAL,
                f"Privacy budget {context.privacy_budget_allocated} exceeds limit {budget_limit}",
                component="privacy_enforcement",
                user_id=context.user_id
            )
            return False
        
        return True
    
    async def monitor_quantum_operation(self, 
                                       operation_type: str,
                                       context: QuantumSecurityContext,
                                       operation_data: Dict[str, Any]):
        """Monitor quantum operation for security threats."""
        
        # Record operation metrics
        current_time = time.time()
        
        if operation_type not in self.threat_indicators:
            self.threat_indicators[operation_type] = []
        
        self.threat_indicators[operation_type].append(current_time)
        
        # Keep only recent indicators (last 10 minutes)
        cutoff_time = current_time - 600
        self.threat_indicators[operation_type] = [
            t for t in self.threat_indicators[operation_type] if t >= cutoff_time
        ]
        
        # Analyze for threats
        await self._analyze_operation_threats(operation_type, context, operation_data)
    
    async def _analyze_operation_threats(self,
                                        operation_type: str,
                                        context: QuantumSecurityContext,
                                        operation_data: Dict[str, Any]):
        """Analyze specific operation for security threats."""
        
        current_time = time.time()
        recent_ops = self.threat_indicators.get(operation_type, [])
        
        # Check operation frequency
        if operation_type == "measurement":
            ops_per_second = len([t for t in recent_ops if current_time - t <= 1.0])
            if ops_per_second > self.threat_thresholds["measurement_frequency_anomaly"]:
                await self._create_security_incident(
                    SecurityEvent.TIMING_ATTACK,
                    ThreatLevel.HIGH,
                    f"Abnormal measurement frequency: {ops_per_second} ops/sec",
                    component="measurement_security",
                    user_id=context.user_id
                )
        
        # Check for quantum state tampering indicators
        if "quantum_state" in operation_data:
            await self._check_quantum_state_integrity(operation_data["quantum_state"], context)
        
        # Check for side-channel attack indicators
        if "timing_data" in operation_data:
            await self._analyze_timing_patterns(operation_data["timing_data"], context)
    
    async def _check_quantum_state_integrity(self,
                                           quantum_state: Dict[str, Any],
                                           context: QuantumSecurityContext):
        """Check quantum state for tampering indicators."""
        
        # Check probability normalization
        if "probability_distribution" in quantum_state:
            probs = list(quantum_state["probability_distribution"].values())
            total_prob = sum(probs)
            
            if abs(total_prob - 1.0) > 0.01:  # Allow small numerical errors
                await self._create_security_incident(
                    SecurityEvent.QUANTUM_STATE_TAMPERING,
                    ThreatLevel.MEDIUM,
                    f"Quantum state probability not normalized: {total_prob}",
                    component="state_integrity",
                    user_id=context.user_id
                )
        
        # Check for impossible quantum states
        if "entanglement_strength" in quantum_state:
            strength = quantum_state["entanglement_strength"]
            if not 0.0 <= strength <= 1.0:
                await self._create_security_incident(
                    SecurityEvent.QUANTUM_STATE_TAMPERING,
                    ThreatLevel.HIGH,
                    f"Invalid entanglement strength: {strength}",
                    component="entanglement_security",
                    user_id=context.user_id
                )
    
    async def _analyze_timing_patterns(self,
                                      timing_data: List[float],
                                      context: QuantumSecurityContext):
        """Analyze timing patterns for side-channel attacks."""
        
        if len(timing_data) < 10:
            return
        
        # Statistical analysis of timing data
        mean_time = np.mean(timing_data)
        std_time = np.std(timing_data)
        
        # Check for timing attack patterns
        if std_time / mean_time > 0.5:  # High variance might indicate probing
            await self._create_security_incident(
                SecurityEvent.SIDE_CHANNEL_ATTACK,
                ThreatLevel.MEDIUM,
                f"Suspicious timing variance: {std_time/mean_time:.3f}",
                component="timing_security",
                user_id=context.user_id
            )
    
    async def _analyze_threat_indicators(self):
        """Analyze collected threat indicators."""
        
        current_time = time.time()
        
        for operation_type, timestamps in self.threat_indicators.items():
            if not timestamps:
                continue
            
            # Check for burst patterns (possible attack)
            recent_ops = [t for t in timestamps if current_time - t <= 60.0]  # Last minute
            if len(recent_ops) > 100:  # More than 100 operations per minute
                await self._create_security_incident(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    ThreatLevel.HIGH,
                    f"Burst pattern detected in {operation_type}: {len(recent_ops)} ops/min",
                    component="threat_analysis"
                )
    
    async def _check_policy_violations(self):
        """Check for security policy violations."""
        
        # This would check against current system state
        # Implementation depends on integration with main system
        pass
    
    async def _detect_quantum_anomalies(self):
        """Detect anomalies in quantum system behavior."""
        
        # Check for quantum decoherence attacks
        # Check for entanglement breaking attempts
        # Check for measurement interference
        # Implementation depends on quantum system integration
        pass
    
    async def _create_security_incident(self,
                                       event_type: SecurityEvent,
                                       threat_level: ThreatLevel,
                                       description: str,
                                       component: str,
                                       user_id: str = None,
                                       source_ip: str = None):
        """Create security incident record."""
        
        incident_id = f"sec_{int(time.time())}_{len(self.security_incidents)}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            event_type=event_type,
            threat_level=threat_level,
            timestamp=time.time(),
            component=component,
            description=description,
            user_id=user_id,
            source_ip=source_ip
        )
        
        self.security_incidents.append(incident)
        
        # Apply automatic mitigation if configured
        await self._apply_incident_mitigation(incident)
        
        logger.warning(f"Security incident created: {incident.incident_id} - {description}")
    
    async def _apply_incident_mitigation(self, incident: SecurityIncident):
        """Apply automatic mitigation for security incidents."""
        
        mitigation_applied = False
        
        if incident.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Apply rate limiting
            if incident.user_id:
                await self._apply_rate_limiting(incident.user_id)
                mitigation_applied = True
            
            # Block suspicious operations
            if incident.event_type in [SecurityEvent.QUANTUM_STATE_TAMPERING, SecurityEvent.MEASUREMENT_INTERFERENCE]:
                await self._block_quantum_operations(incident.user_id, 300)  # 5 minutes
                mitigation_applied = True
        
        incident.mitigation_applied = mitigation_applied
    
    async def _apply_rate_limiting(self, user_id: str):
        """Apply rate limiting to user."""
        # Implementation would depend on main system integration
        logger.info(f"Rate limiting applied to user: {user_id}")
    
    async def _block_quantum_operations(self, user_id: str, duration_seconds: int):
        """Temporarily block quantum operations for user."""
        # Implementation would depend on main system integration
        logger.warning(f"Quantum operations blocked for user {user_id} for {duration_seconds} seconds")
    
    async def _cleanup_old_incidents(self):
        """Clean up old security incidents."""
        
        current_time = time.time()
        retention_period = 7 * 24 * 3600  # 7 days
        
        self.security_incidents = [
            incident for incident in self.security_incidents
            if current_time - incident.timestamp < retention_period
        ]
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data."""
        
        current_time = time.time()
        
        # Incident statistics
        total_incidents = len(self.security_incidents)
        recent_incidents = [i for i in self.security_incidents if current_time - i.timestamp < 3600]
        
        incidents_by_severity = {level.value: 0 for level in ThreatLevel}
        incidents_by_type = {event.value: 0 for event in SecurityEvent}
        
        for incident in self.security_incidents:
            incidents_by_severity[incident.threat_level.value] += 1
            incidents_by_type[incident.event_type.value] += 1
        
        # Threat indicator summary
        threat_summary = {}
        for operation_type, timestamps in self.threat_indicators.items():
            recent_count = len([t for t in timestamps if current_time - t < 3600])
            threat_summary[operation_type] = {
                "recent_operations": recent_count,
                "total_operations": len(timestamps)
            }
        
        return {
            "security_monitoring_active": self._monitoring_active,
            "total_incidents": total_incidents,
            "recent_incidents": len(recent_incidents),
            "incidents_by_severity": incidents_by_severity,
            "incidents_by_type": incidents_by_type,
            "threat_indicators": threat_summary,
            "security_policies_active": len(self.security_policies),
            "recent_incident_details": [
                {
                    "incident_id": i.incident_id,
                    "event_type": i.event_type.value,
                    "threat_level": i.threat_level.value,
                    "description": i.description,
                    "timestamp": i.timestamp,
                    "resolved": i.resolved
                } for i in recent_incidents[-10:]  # Last 10 recent incidents
            ]
        }


class QuantumSecurityController:
    """Main controller for quantum planning security."""
    
    def __init__(self):
        self.cryptographer = QuantumCryptographer()
        self.monitor = QuantumSecurityMonitor()
        self.access_control = {}
        self.audit_trail = []
        
    async def initialize_security(self):
        """Initialize quantum security systems."""
        await self.monitor.start_security_monitoring()
        logger.info("Quantum security initialized")
    
    async def create_secure_context(self, 
                                   user_id: str,
                                   operation_type: str,
                                   security_level: SecurityLevel,
                                   privacy_budget: float) -> QuantumSecurityContext:
        """Create secure context for quantum operations."""
        
        operation_id = f"op_{int(time.time())}_{secrets.token_hex(8)}"
        
        # Determine allowed operations based on security level
        allowed_ops = self.monitor.security_policies["allowed_quantum_operations"].get(
            security_level.value, ["measurement"]
        )
        
        context = QuantumSecurityContext(
            operation_id=operation_id,
            user_id=user_id,
            security_level=security_level,
            privacy_budget_allocated=privacy_budget,
            allowed_operations=set(allowed_ops),
            encryption_required=self.monitor.security_policies["encryption_requirements"].get(
                security_level.value, True
            ),
            audit_trail_required=self.monitor.security_policies["audit_requirements"].get(
                security_level.value, True
            )
        )
        
        # Validate context
        if not await self.monitor.validate_security_context(context, operation_type):
            raise ValueError("Security context validation failed")
        
        return context
    
    async def secure_quantum_operation(self,
                                      context: QuantumSecurityContext,
                                      operation_type: str,
                                      operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum operation with security controls."""
        
        start_time = time.time()
        
        try:
            # Pre-operation security checks
            if not await self.monitor.validate_security_context(context, operation_type):
                raise ValueError("Security validation failed")
            
            # Monitor operation
            await self.monitor.monitor_quantum_operation(operation_type, context, operation_data)
            
            # Encrypt sensitive data if required
            if context.encryption_required and "quantum_state" in operation_data:
                encrypted_state = self.cryptographer.encrypt_quantum_state(
                    operation_data["quantum_state"], context
                )
                operation_data["quantum_state"] = encrypted_state
            
            # Add quantum signature if required
            if context.quantum_signature_required:
                operation_bytes = json.dumps(operation_data, sort_keys=True).encode('utf-8')
                signature = self.cryptographer.generate_quantum_signature(operation_bytes, context)
                operation_data["quantum_signature"] = signature.decode('utf-8')
            
            # Record audit trail
            if context.audit_trail_required:
                await self._record_audit_event(context, operation_type, operation_data, start_time)
            
            return operation_data
            
        except Exception as e:
            # Log security exception
            logger.error(f"Secure quantum operation failed: {str(e)}")
            
            # Create security incident
            await self.monitor._create_security_incident(
                SecurityEvent.UNAUTHORIZED_ACCESS,
                ThreatLevel.HIGH,
                f"Quantum operation failed: {str(e)}",
                component="quantum_security_controller",
                user_id=context.user_id
            )
            
            raise e
    
    async def _record_audit_event(self,
                                 context: QuantumSecurityContext,
                                 operation_type: str,
                                 operation_data: Dict[str, Any],
                                 start_time: float):
        """Record audit event for quantum operation."""
        
        audit_event = {
            "timestamp": time.time(),
            "operation_id": context.operation_id,
            "user_id": context.user_id,
            "operation_type": operation_type,
            "security_level": context.security_level.value,
            "privacy_budget_used": context.privacy_budget_allocated,
            "execution_time": time.time() - start_time,
            "data_hash": hashlib.sha3_256(
                json.dumps(operation_data, sort_keys=True).encode('utf-8')
            ).hexdigest()
        }
        
        self.audit_trail.append(audit_event)
        
        # Limit audit trail size
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-5000:]
    
    async def shutdown_security(self):
        """Shutdown security systems."""
        await self.monitor.stop_security_monitoring()
        logger.info("Quantum security shutdown")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        
        return {
            "security_controller_active": True,
            "cryptographer_initialized": self.cryptographer is not None,
            "monitoring_dashboard": self.monitor.get_security_dashboard(),
            "audit_events_count": len(self.audit_trail),
            "recent_audit_events": self.audit_trail[-10:] if self.audit_trail else []
        }