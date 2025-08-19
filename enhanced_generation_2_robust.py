#!/usr/bin/env python3
"""
Enhanced Generation 2: Robust & Reliable Implementation
Adds comprehensive error handling, validation, logging, monitoring, and security measures.
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import traceback

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NodeStatus(Enum):
    """Node status enumeration."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"

class ErrorType(Enum):
    """Error type classification."""
    PRIVACY_VIOLATION = "privacy_violation"
    NODE_FAILURE = "node_failure"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

@dataclass
class RobustPrivacyConfig:
    """Enhanced privacy configuration with validation."""
    epsilon_per_query: float = 0.1
    max_budget_per_user: float = 5.0
    delta: float = 1e-5
    budget_refresh_hours: int = 24
    max_queries_per_minute: int = 10
    require_authentication: bool = True
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate privacy configuration."""
        if self.epsilon_per_query <= 0:
            raise ValueError("epsilon_per_query must be positive")
        if self.max_budget_per_user <= 0:
            raise ValueError("max_budget_per_user must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if self.budget_refresh_hours <= 0:
            raise ValueError("budget_refresh_hours must be positive")

@dataclass
class RobustNode:
    """Enhanced node with health monitoring and security."""
    node_id: str
    endpoint: str
    status: NodeStatus = NodeStatus.ACTIVE
    load: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    max_errors: int = 5
    capabilities: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    auth_token: Optional[str] = None
    
    def __post_init__(self):
        if not self.node_id:
            raise ValueError("node_id cannot be empty")
        if not self.endpoint:
            raise ValueError("endpoint cannot be empty")
    
    def validate(self):
        """Validate node configuration."""
        if not self.node_id:
            raise ValueError("node_id cannot be empty")
        if not self.endpoint:
            raise ValueError("endpoint cannot be empty")

@dataclass
class RobustRequest:
    """Enhanced request with comprehensive validation."""
    user_id: str
    prompt: str
    privacy_budget: float = 0.1
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 5
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    department: Optional[str] = None
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate request parameters."""
        if not self.user_id:
            raise ValueError("user_id cannot be empty")
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if self.privacy_budget <= 0:
            raise ValueError("privacy_budget must be positive")
        if self.priority < 0 or self.priority > 10:
            raise ValueError("priority must be between 0 and 10")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

class SecurityManager:
    """Handles authentication and authorization."""
    
    def __init__(self, secret_key: str = "default_secret_key"):
        self.secret_key = secret_key.encode()
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_duration = timedelta(minutes=15)
        logger.info("🔐 Security Manager initialized")
    
    def generate_token(self, user_id: str, department: str = None) -> str:
        """Generate secure authentication token."""
        payload = {
            "user_id": user_id,
            "department": department,
            "timestamp": datetime.now().isoformat(),
            "expires": (datetime.now() + timedelta(hours=8)).isoformat()
        }
        
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(self.secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
        
        token = f"{payload_str.encode().hex()}.{signature}"
        
        self.active_sessions[user_id] = {
            "token": token,
            "created": datetime.now(),
            "last_access": datetime.now()
        }
        
        logger.info(f"🔑 Token generated for user: {user_id}")
        return token
    
    def validate_token(self, token: str, user_id: str) -> bool:
        """Validate authentication token."""
        try:
            if user_id in self.failed_attempts and self.failed_attempts[user_id] >= 5:
                logger.warning(f"🚫 User {user_id} is locked out")
                return False
            
            parts = token.split('.')
            if len(parts) != 2:
                self._record_failed_attempt(user_id)
                return False
            
            payload_hex, signature = parts
            payload_str = bytes.fromhex(payload_hex).decode()
            expected_signature = hmac.new(self.secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                self._record_failed_attempt(user_id)
                return False
            
            payload = json.loads(payload_str)
            expires = datetime.fromisoformat(payload["expires"])
            
            if datetime.now() > expires:
                logger.warning(f"⏰ Token expired for user: {user_id}")
                self._record_failed_attempt(user_id)
                return False
            
            # Update last access
            if user_id in self.active_sessions:
                self.active_sessions[user_id]["last_access"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Token validation error: {e}")
            self._record_failed_attempt(user_id)
            return False
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt."""
        self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
        logger.warning(f"⚠️ Failed attempt #{self.failed_attempts[user_id]} for user: {user_id}")

class AdvancedHealthMonitor:
    """Advanced health monitoring with alerting."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "request_count": 0,
            "error_count": 0,
            "average_response_time": 0.0,
            "privacy_violations": 0,
            "node_failures": 0
        }
        self.alerts: List[Dict] = []
        self.thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "response_time": 5.0,  # 5 seconds
            "node_failure_rate": 0.2  # 20% node failure rate
        }
        logger.info("📊 Advanced Health Monitor initialized")
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics."""
        self.metrics["request_count"] += 1
        if not success:
            self.metrics["error_count"] += 1
        
        # Update rolling average response time
        count = self.metrics["request_count"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = ((current_avg * (count - 1)) + response_time) / count
        
        self._check_thresholds()
    
    def record_privacy_violation(self):
        """Record privacy violation."""
        self.metrics["privacy_violations"] += 1
        self._create_alert("CRITICAL", "Privacy violation detected")
    
    def record_node_failure(self, node_id: str):
        """Record node failure."""
        self.metrics["node_failures"] += 1
        self._create_alert("HIGH", f"Node failure: {node_id}")
    
    def _check_thresholds(self):
        """Check if metrics exceed thresholds."""
        if self.metrics["request_count"] > 0:
            error_rate = self.metrics["error_count"] / self.metrics["request_count"]
            if error_rate > self.thresholds["error_rate"]:
                self._create_alert("MEDIUM", f"High error rate: {error_rate:.2%}")
        
        if self.metrics["average_response_time"] > self.thresholds["response_time"]:
            self._create_alert("MEDIUM", f"High response time: {self.metrics['average_response_time']:.2f}s")
    
    def _create_alert(self, severity: str, message: str):
        """Create alert."""
        alert = {
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4())
        }
        self.alerts.append(alert)
        logger.warning(f"🚨 Alert [{severity}]: {message}")
    
    def get_health_status(self) -> Dict:
        """Get comprehensive health status."""
        status = "healthy"
        recent_alerts = [a for a in self.alerts if 
                        datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(minutes=5)]
        
        if any(a["severity"] == "CRITICAL" for a in recent_alerts):
            status = "critical"
        elif any(a["severity"] == "HIGH" for a in recent_alerts):
            status = "unhealthy"
        elif any(a["severity"] == "MEDIUM" for a in recent_alerts):
            status = "degraded"
        
        return {
            "status": status,
            "metrics": self.metrics,
            "recent_alerts": recent_alerts[-5:],  # Last 5 alerts
            "timestamp": datetime.now().isoformat()
        }

class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Sanitize input prompt."""
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Remove potential injection patterns
        dangerous_patterns = ['<script>', 'javascript:', 'data:', 'vbscript:']
        prompt_lower = prompt.lower()
        
        for pattern in dangerous_patterns:
            if pattern in prompt_lower:
                raise ValueError(f"Potentially dangerous content detected: {pattern}")
        
        # Limit length
        if len(prompt) > 10000:
            raise ValueError("Prompt too long (max 10000 characters)")
        
        return prompt.strip()
    
    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate user ID format."""
        if not user_id:
            raise ValueError("User ID cannot be empty")
        
        if len(user_id) > 100:
            raise ValueError("User ID too long")
        
        # Allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValueError("Invalid user ID format")
        
        return user_id

class RobustFederatedRouter:
    """
    Generation 2: Robust and reliable federated router.
    Comprehensive error handling, validation, logging, monitoring, and security.
    """
    
    def __init__(self, privacy_config: RobustPrivacyConfig = None):
        self.privacy_config = privacy_config or RobustPrivacyConfig()
        self.nodes: Dict[str, RobustNode] = {}
        self.user_budgets: Dict[str, Dict] = {}
        self.request_history: List[Dict] = []
        self.security_manager = SecurityManager()
        self.health_monitor = AdvancedHealthMonitor()
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        logger.info("🚀 Generation 2: Robust Federated Router initialized")
        logger.info(f"🔧 Privacy config: ε={self.privacy_config.epsilon_per_query}, "
                   f"max_budget={self.privacy_config.max_budget_per_user}")
    
    def register_node(self, node: RobustNode) -> bool:
        """Register node with comprehensive validation."""
        try:
            node.validate()  # Validate node configuration
            
            # Check for duplicate node IDs
            if node.node_id in self.nodes:
                logger.warning(f"⚠️ Node {node.node_id} already registered, updating...")
            
            self.nodes[node.node_id] = node
            logger.info(f"✅ Node registered: {node.node_id} ({node.endpoint})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Node registration failed: {e}")
            return False
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Remove old requests
        self.rate_limits[user_id] = [req_time for req_time in self.rate_limits[user_id] 
                                    if req_time > minute_ago]
        
        # Check limit
        if len(self.rate_limits[user_id]) >= self.privacy_config.max_queries_per_minute:
            logger.warning(f"⚠️ Rate limit exceeded for user: {user_id}")
            return False
        
        return True
    
    def _refresh_budget_if_needed(self, user_id: str):
        """Refresh user budget if refresh period has passed."""
        if user_id not in self.user_budgets:
            self.user_budgets[user_id] = {
                "consumed": 0.0,
                "last_refresh": datetime.now()
            }
            return
        
        last_refresh = self.user_budgets[user_id]["last_refresh"]
        refresh_period = timedelta(hours=self.privacy_config.budget_refresh_hours)
        
        if datetime.now() - last_refresh > refresh_period:
            self.user_budgets[user_id] = {
                "consumed": 0.0,
                "last_refresh": datetime.now()
            }
            logger.info(f"🔄 Budget refreshed for user: {user_id}")
    
    def check_privacy_budget(self, user_id: str, requested_budget: float) -> tuple[bool, float]:
        """Enhanced privacy budget checking with refresh logic."""
        self._refresh_budget_if_needed(user_id)
        
        consumed = self.user_budgets[user_id]["consumed"]
        remaining = self.privacy_config.max_budget_per_user - consumed
        
        if requested_budget <= remaining:
            return True, remaining
        
        logger.warning(f"⚠️ Privacy budget exceeded for user {user_id}: "
                      f"requested={requested_budget}, remaining={remaining}")
        self.health_monitor.record_privacy_violation()
        return False, remaining
    
    def select_optimal_node(self, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> Optional[RobustNode]:
        """Enhanced node selection with health and security considerations."""
        if not self.nodes:
            return None
        
        # Filter nodes by status and security level
        suitable_nodes = []
        for node in self.nodes.values():
            if node.status not in [NodeStatus.ACTIVE, NodeStatus.DEGRADED]:
                continue
            
            # Check security level compatibility
            if node.security_level.value < security_level.value:
                continue
            
            # Check error threshold
            if node.error_count >= node.max_errors:
                continue
            
            suitable_nodes.append(node)
        
        if not suitable_nodes:
            logger.error("❌ No suitable nodes available")
            return None
        
        # Select node with lowest load, penalizing degraded nodes
        def node_score(node):
            base_score = node.load
            if node.status == NodeStatus.DEGRADED:
                base_score += 0.5  # Penalty for degraded nodes
            return base_score
        
        optimal_node = min(suitable_nodes, key=node_score)
        return optimal_node
    
    async def process_request_with_retry(self, request: RobustRequest, auth_token: str = None) -> Dict:
        """Process request with comprehensive retry logic."""
        for attempt in range(request.max_retries + 1):
            try:
                result = await self._process_single_request(request, auth_token)
                
                if result["success"]:
                    return result
                
                # If not success and we have retries left, wait and retry
                if attempt < request.max_retries:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    logger.info(f"🔄 Retrying request {request.request_id} in {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    request.retry_count = attempt + 1
                else:
                    return result
                    
            except Exception as e:
                logger.error(f"❌ Request processing error (attempt {attempt + 1}): {e}")
                
                if attempt < request.max_retries:
                    wait_time = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": f"Request failed after {request.max_retries + 1} attempts: {str(e)}",
                        "error_type": ErrorType.TIMEOUT.value
                    }
    
    async def _process_single_request(self, request: RobustRequest, auth_token: str = None) -> Dict:
        """Process single request with comprehensive validation and monitoring."""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Processing request {request.request_id} for user: {request.user_id}")
            
            # Input validation
            request.user_id = InputValidator.validate_user_id(request.user_id)
            request.prompt = InputValidator.sanitize_prompt(request.prompt)
            
            # Authentication check
            if self.privacy_config.require_authentication and auth_token:
                if not self.security_manager.validate_token(auth_token, request.user_id):
                    return {
                        "success": False,
                        "error": "Authentication failed",
                        "error_type": ErrorType.AUTHENTICATION.value
                    }
            
            # Rate limiting
            if not self._check_rate_limit(request.user_id):
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "error_type": ErrorType.RESOURCE_EXHAUSTION.value
                }
            
            # Privacy budget check
            budget_ok, remaining = self.check_privacy_budget(request.user_id, request.privacy_budget)
            if not budget_ok:
                return {
                    "success": False,
                    "error": "Privacy budget exceeded",
                    "error_type": ErrorType.PRIVACY_VIOLATION.value,
                    "remaining_budget": remaining
                }
            
            # Node selection
            node = self.select_optimal_node(request.security_level)
            if not node:
                self.health_monitor.record_node_failure("all_nodes")
                return {
                    "success": False,
                    "error": "No suitable nodes available",
                    "error_type": ErrorType.NODE_FAILURE.value
                }
            
            # Process with timeout
            try:
                response = await asyncio.wait_for(
                    self._simulate_node_processing(node, request),
                    timeout=request.timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"⏰ Request {request.request_id} timed out")
                node.error_count += 1
                return {
                    "success": False,
                    "error": f"Request timed out after {request.timeout}s",
                    "error_type": ErrorType.TIMEOUT.value
                }
            
            # Success path
            self._consume_privacy_budget(request.user_id, request.privacy_budget)
            self._update_node_metrics(node, success=True)
            self._record_request(request, node.node_id, True)
            
            # Add to rate limit tracking
            self.rate_limits.setdefault(request.user_id, []).append(datetime.now())
            
            response_time = time.time() - start_time
            self.health_monitor.record_request(response_time, True)
            
            return {
                "success": True,
                "response": response,
                "node_id": node.node_id,
                "privacy_cost": request.privacy_budget,
                "remaining_budget": remaining - request.privacy_budget,
                "processing_time": response_time,
                "request_id": request.request_id
            }
            
        except Exception as e:
            # Error path
            logger.error(f"❌ Request processing failed: {e}\n{traceback.format_exc()}")
            
            response_time = time.time() - start_time
            self.health_monitor.record_request(response_time, False)
            
            return {
                "success": False,
                "error": str(e),
                "error_type": ErrorType.VALIDATION.value,
                "processing_time": response_time,
                "request_id": request.request_id
            }
    
    async def _simulate_node_processing(self, node: RobustNode, request: RobustRequest) -> str:
        """Simulate node processing with potential failures."""
        # Simulate network delay
        await asyncio.sleep(0.1 + (node.load * 0.05))
        
        # Simulate occasional node failures
        import random
        if random.random() < 0.02:  # 2% failure rate
            node.error_count += 1
            raise Exception(f"Simulated node failure on {node.node_id}")
        
        return f"Processed: {request.prompt[:50]}... (via {node.node_id}, security: {request.security_level.value})"
    
    def _consume_privacy_budget(self, user_id: str, amount: float):
        """Consume privacy budget with logging."""
        self.user_budgets[user_id]["consumed"] += amount
        remaining = self.privacy_config.max_budget_per_user - self.user_budgets[user_id]["consumed"]
        logger.info(f"💰 Privacy budget consumed: {amount:.3f}, remaining: {remaining:.3f}")
    
    def _update_node_metrics(self, node: RobustNode, success: bool):
        """Update node performance metrics."""
        if success:
            node.load += 0.1
            node.last_heartbeat = datetime.now()
        else:
            node.error_count += 1
            if node.error_count >= node.max_errors:
                node.status = NodeStatus.DEGRADED
                logger.warning(f"⚠️ Node {node.node_id} marked as degraded due to errors")
    
    def _record_request(self, request: RobustRequest, node_id: str, success: bool):
        """Record request in history."""
        self.request_history.append({
            "timestamp": request.timestamp.isoformat(),
            "request_id": request.request_id,
            "user_id": request.user_id,
            "node_id": node_id,
            "privacy_cost": request.privacy_budget,
            "success": success,
            "retry_count": request.retry_count,
            "security_level": request.security_level.value
        })
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status."""
        active_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.ACTIVE)
        degraded_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.DEGRADED)
        
        return {
            "nodes": {
                "total": len(self.nodes),
                "active": active_nodes,
                "degraded": degraded_nodes,
                "inactive": len(self.nodes) - active_nodes - degraded_nodes
            },
            "requests": {
                "total": len(self.request_history),
                "successful": sum(1 for r in self.request_history if r["success"]),
                "failed": sum(1 for r in self.request_history if not r["success"])
            },
            "users": {
                "total": len(self.user_budgets),
                "active_sessions": len(self.security_manager.active_sessions)
            },
            "privacy": {
                "config": asdict(self.privacy_config),
                "violations": self.health_monitor.metrics["privacy_violations"]
            },
            "health": self.health_monitor.get_health_status()
        }

async def demo_generation_2():
    """Demonstrate Generation 2 robust functionality."""
    print("🌟 GENERATION 2 DEMO: Robust & Reliable Implementation")
    print("=" * 70)
    
    # Initialize enhanced router
    config = RobustPrivacyConfig(
        epsilon_per_query=0.1,
        max_budget_per_user=3.0,
        max_queries_per_minute=5,
        require_authentication=True
    )
    router = RobustFederatedRouter(config)
    
    # Register nodes with different security levels
    nodes = [
        RobustNode("hospital_a", "https://hospital-a.local:8443", security_level=SecurityLevel.HIGH),
        RobustNode("hospital_b", "https://hospital-b.local:8443", security_level=SecurityLevel.MEDIUM),
        RobustNode("hospital_c", "https://hospital-c.local:8443", security_level=SecurityLevel.HIGH),
        RobustNode("hospital_d", "https://hospital-d.local:8443", security_level=SecurityLevel.LOW)
    ]
    
    for node in nodes:
        router.register_node(node)
    
    # Generate authentication tokens
    doctor1_token = router.security_manager.generate_token("doctor_001", "emergency")
    doctor2_token = router.security_manager.generate_token("doctor_002", "oncology")
    
    # Process enhanced requests with different security levels
    requests = [
        RobustRequest("doctor_001", "Critical patient assessment needed urgently", 
                     privacy_budget=0.2, security_level=SecurityLevel.HIGH, priority=1),
        RobustRequest("doctor_002", "Review oncology case with sensitive genetic data", 
                     privacy_budget=0.15, security_level=SecurityLevel.CRITICAL, priority=2),
        RobustRequest("doctor_001", "Follow-up consultation for chronic condition", 
                     privacy_budget=0.1, security_level=SecurityLevel.MEDIUM, priority=5),
        RobustRequest("doctor_003", "Unauthorized access attempt", 
                     privacy_budget=0.3, security_level=SecurityLevel.HIGH, priority=3),  # No token
        RobustRequest("doctor_001", "Routine health check analysis", 
                     privacy_budget=0.05, security_level=SecurityLevel.LOW, priority=7)
    ]
    
    print("\n📋 Processing Enhanced Requests:")
    tokens = {"doctor_001": doctor1_token, "doctor_002": doctor2_token}
    
    for i, request in enumerate(requests, 1):
        print(f"\n{i}. Request {request.request_id[:8]}... from {request.user_id}")
        print(f"   🔒 Security Level: {request.security_level.value}")
        print(f"   ⚡ Priority: {request.priority}")
        
        auth_token = tokens.get(request.user_id)
        result = await router.process_request_with_retry(request, auth_token)
        
        if result["success"]:
            print(f"   ✅ Success: {result['response'][:60]}...")
            print(f"   💰 Privacy cost: {result['privacy_cost']}")
            print(f"   🏦 Node: {result['node_id']}")
            print(f"   💳 Remaining budget: {result['remaining_budget']:.3f}")
            print(f"   ⏱️ Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"   ❌ Error: {result['error']}")
            print(f"   🏷️ Error type: {result.get('error_type', 'unknown')}")
    
    # Demonstrate rate limiting
    print(f"\n🚦 Testing Rate Limiting:")
    rapid_requests = [
        RobustRequest("doctor_001", f"Rapid query #{i}", privacy_budget=0.01)
        for i in range(8)
    ]
    
    for request in rapid_requests:
        result = await router.process_request_with_retry(request, doctor1_token)
        status = "✅" if result["success"] else "❌"
        print(f"   {status} Query result: {result.get('error', 'Success')}")
    
    # Show comprehensive status
    print(f"\n📊 Comprehensive System Status:")
    status = router.get_comprehensive_status()
    
    print(f"   Nodes: {status['nodes']['active']}/{status['nodes']['total']} active, "
          f"{status['nodes']['degraded']} degraded")
    print(f"   Requests: {status['requests']['successful']}/{status['requests']['total']} successful")
    print(f"   Users: {status['users']['total']} total, {status['users']['active_sessions']} active sessions")
    print(f"   Privacy violations: {status['privacy']['violations']}")
    print(f"   System health: {status['health']['status']}")
    
    if status['health']['recent_alerts']:
        print(f"   Recent alerts: {len(status['health']['recent_alerts'])}")
        for alert in status['health']['recent_alerts'][-3:]:
            print(f"     🚨 [{alert['severity']}] {alert['message']}")
    
    print(f"\n✅ Generation 2 Complete: Robust system with comprehensive error handling!")
    return router

if __name__ == "__main__":
    asyncio.run(demo_generation_2())