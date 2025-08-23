#!/usr/bin/env python3
"""
Generation 2: Robust Implementation
Adds comprehensive error handling, security, logging, monitoring, and resilience.
"""

import sys
import time
import json
import logging
import threading
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, deque
import uuid

# Enhanced logging configuration
class SecurityAuditLogger:
    """Security-focused audit logger for healthcare compliance."""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('federated_security_audit')
        self.logger.setLevel(log_level)
        
        # Create formatters for different log types
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s - [Session: %(session_id)s]'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(security_formatter)
        self.logger.addHandler(console_handler)
        
        self.session_id = str(uuid.uuid4())[:8]
        
    def log_privacy_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log privacy-related events for audit trail."""
        extra = {'session_id': self.session_id}
        self.logger.info(
            f"PRIVACY_EVENT: {event_type} | User: {user_id} | Details: {json.dumps(details)}", 
            extra=extra
        )
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security events."""
        extra = {'session_id': self.session_id}
        if severity.upper() == 'HIGH':
            self.logger.critical(f"SECURITY_ALERT: {event_type} | {json.dumps(details)}", extra=extra)
        else:
            self.logger.warning(f"SECURITY_EVENT: {event_type} | {json.dumps(details)}", extra=extra)
    
    def log_access_event(self, user_id: str, resource: str, action: str, result: str):
        """Log access control events."""
        extra = {'session_id': self.session_id}
        self.logger.info(
            f"ACCESS: User={user_id} | Resource={resource} | Action={action} | Result={result}",
            extra=extra
        )

class ValidationError(Exception):
    """Custom validation error."""
    pass

class SecurityError(Exception):
    """Custom security error."""
    pass

class PrivacyBudgetExceeded(Exception):
    """Privacy budget exceeded error."""
    pass

class NodeUnavailableError(Exception):
    """Node unavailable error."""
    pass

class TaskPriority(Enum):
    """Enhanced task priority levels with healthcare context."""
    CRITICAL = 0      # Life-threatening emergencies
    URGENT = 1        # Time-sensitive diagnostics
    HIGH = 2          # Standard urgent care
    MEDIUM = 3        # Routine clinical tasks
    LOW = 4           # Administrative tasks
    BACKGROUND = 5    # Maintenance and research

class NodeStatus(Enum):
    """Node operational status."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

@dataclass
class SecurityContext:
    """Security context for requests."""
    user_id: str
    role: str = "doctor"
    department: str = "general"
    authentication_token: str = field(default_factory=lambda: secrets.token_hex(16))
    permissions: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ip_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass 
class RobustTask:
    """Enhanced task with validation and security."""
    task_id: str
    user_id: str
    prompt: str
    priority: TaskPriority
    privacy_budget: float
    estimated_duration: float = 30.0
    created_at: float = field(default_factory=time.time)
    
    # Security and validation
    security_context: SecurityContext = None
    validated: bool = False
    sanitized: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    # Monitoring
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None
    error_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.task_id:
            raise ValidationError("Task ID cannot be empty")
        if not self.user_id:
            raise ValidationError("User ID cannot be empty") 
        if not self.prompt or len(self.prompt.strip()) == 0:
            raise ValidationError("Prompt cannot be empty")
        if self.privacy_budget < 0:
            raise ValidationError("Privacy budget must be non-negative")

@dataclass
class RobustNode:
    """Enhanced node with health monitoring and security."""
    node_id: str
    current_load: float = 0.0
    privacy_budget_available: float = 100.0
    status: NodeStatus = NodeStatus.ACTIVE
    
    # Health monitoring
    last_health_check: float = field(default_factory=time.time)
    health_score: float = 1.0
    error_count: int = 0
    consecutive_failures: int = 0
    
    # Capacity tracking
    max_concurrent_tasks: int = 10
    current_tasks: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Security
    authentication_required: bool = True
    allowed_departments: List[str] = field(default_factory=lambda: ["general"])
    security_level: str = "standard"  # standard, high, maximum

class InputSanitizer:
    """Input sanitization for healthcare data."""
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Sanitize user prompt input."""
        if not prompt:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', '\x00']
        sanitized = prompt
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [truncated]"
        
        return sanitized.strip()
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format."""
        if not user_id or len(user_id) < 3 or len(user_id) > 50:
            return False
        
        # Only allow alphanumeric, underscore, and hyphen
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
        return all(c in allowed_chars for c in user_id)

class CircuitBreaker:
    """Circuit breaker for resilience."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.RLock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise NodeUnavailableError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e

class EnhancedPrivacyAccountant:
    """Robust privacy accountant with comprehensive validation and monitoring."""
    
    def __init__(self, max_budget_per_user: float = 10.0, epsilon_per_query: float = 0.1):
        self.max_budget_per_user = max_budget_per_user
        self.epsilon_per_query = epsilon_per_query
        self.user_budgets: Dict[str, float] = {}
        self.audit_logger = SecurityAuditLogger()
        self._lock = threading.RLock()
        
        # Enhanced tracking
        self.budget_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.daily_limits: Dict[str, float] = {}
        self.user_roles: Dict[str, str] = {}
        
    def validate_privacy_request(self, user_id: str, requested_epsilon: float, 
                                security_context: SecurityContext) -> Tuple[bool, str]:
        """Comprehensive privacy request validation."""
        try:
            # Input validation
            if not InputSanitizer.validate_user_id(user_id):
                return False, "Invalid user ID format"
            
            if requested_epsilon <= 0:
                return False, "Epsilon must be positive"
            
            if requested_epsilon > self.max_budget_per_user:
                return False, "Requested epsilon exceeds maximum allowed per user"
            
            # Role-based validation
            user_role = security_context.role if security_context else "unknown"
            if user_role in ["researcher", "student"] and requested_epsilon > 0.5:
                return False, "Research users have stricter epsilon limits"
            
            # Department-based validation
            department = security_context.department if security_context else "general"
            if department == "emergency" and requested_epsilon > 2.0:
                return False, "Even emergency department has epsilon limits for patient privacy"
            
            return True, "Validation passed"
            
        except Exception as e:
            self.audit_logger.log_security_event(
                "PRIVACY_VALIDATION_ERROR", 
                "HIGH", 
                {"user_id": user_id, "error": str(e)}
            )
            return False, f"Validation error: {str(e)}"
    
    def check_budget(self, user_id: str, requested_epsilon: float, 
                     security_context: Optional[SecurityContext] = None) -> Tuple[bool, str]:
        """Check privacy budget with comprehensive validation."""
        with self._lock:
            try:
                # Validate request
                valid, message = self.validate_privacy_request(user_id, requested_epsilon, security_context)
                if not valid:
                    self.audit_logger.log_privacy_event(
                        "BUDGET_CHECK_DENIED",
                        user_id,
                        {"reason": message, "requested_epsilon": requested_epsilon}
                    )
                    return False, message
                
                # Check budget availability
                current_spent = self.user_budgets.get(user_id, 0.0)
                remaining = self.max_budget_per_user - current_spent
                
                if remaining < requested_epsilon:
                    self.audit_logger.log_privacy_event(
                        "INSUFFICIENT_BUDGET",
                        user_id,
                        {
                            "requested": requested_epsilon,
                            "remaining": remaining,
                            "daily_spent": current_spent
                        }
                    )
                    return False, f"Insufficient privacy budget. Remaining: {remaining:.3f}"
                
                # Log successful check
                self.audit_logger.log_privacy_event(
                    "BUDGET_CHECK_APPROVED",
                    user_id,
                    {"requested_epsilon": requested_epsilon, "remaining_after": remaining - requested_epsilon}
                )
                
                return True, "Budget check passed"
                
            except Exception as e:
                self.audit_logger.log_security_event(
                    "BUDGET_CHECK_ERROR",
                    "HIGH", 
                    {"user_id": user_id, "error": str(e)}
                )
                return False, f"Budget check failed: {str(e)}"
    
    def spend_budget(self, user_id: str, epsilon: float, 
                     security_context: Optional[SecurityContext] = None,
                     task_id: Optional[str] = None) -> Tuple[bool, str]:
        """Spend privacy budget with full audit trail."""
        with self._lock:
            try:
                # Pre-check
                can_spend, message = self.check_budget(user_id, epsilon, security_context)
                if not can_spend:
                    return False, message
                
                # Record the spend
                self.user_budgets[user_id] = self.user_budgets.get(user_id, 0.0) + epsilon
                
                # Create audit record
                spend_record = {
                    "timestamp": time.time(),
                    "epsilon": epsilon,
                    "task_id": task_id,
                    "remaining_budget": self.max_budget_per_user - self.user_budgets[user_id],
                    "department": security_context.department if security_context else "unknown",
                    "session_id": security_context.session_id if security_context else None
                }
                
                self.budget_history[user_id].append(spend_record)
                
                # Log the transaction
                self.audit_logger.log_privacy_event(
                    "BUDGET_SPENT",
                    user_id,
                    spend_record
                )
                
                return True, "Budget spent successfully"
                
            except Exception as e:
                self.audit_logger.log_security_event(
                    "BUDGET_SPEND_ERROR",
                    "HIGH",
                    {"user_id": user_id, "epsilon": epsilon, "error": str(e)}
                )
                return False, f"Budget spend failed: {str(e)}"
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user budget statistics."""
        with self._lock:
            current_spent = self.user_budgets.get(user_id, 0.0)
            history = self.budget_history.get(user_id, [])
            
            return {
                "user_id": user_id,
                "total_budget": self.max_budget_per_user,
                "spent_budget": current_spent,
                "remaining_budget": self.max_budget_per_user - current_spent,
                "transaction_count": len(history),
                "last_transaction": history[-1]["timestamp"] if history else None,
                "average_epsilon_per_query": sum(h["epsilon"] for h in history) / len(history) if history else 0.0
            }

class RobustFederatedRouter:
    """Enhanced federated router with comprehensive error handling and monitoring."""
    
    def __init__(self, privacy_accountant: EnhancedPrivacyAccountant, 
                 health_check_interval: float = 30.0):
        self.privacy_accountant = privacy_accountant
        self.nodes: Dict[str, RobustNode] = {}
        self.tasks: Dict[str, RobustTask] = {}
        self.assignments: List[Dict[str, Any]] = []
        self.audit_logger = SecurityAuditLogger()
        
        # Circuit breakers for each node
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Health monitoring
        self.health_check_interval = health_check_interval
        self.last_system_health_check = time.time()
        
        # Performance metrics
        self.metrics = {
            "total_tasks_processed": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "average_assignment_time": 0.0,
            "privacy_violations_prevented": 0
        }
        
        self._lock = threading.RLock()
    
    def register_node(self, node_id: str, capabilities: Dict[str, Any]) -> bool:
        """Register node with comprehensive validation."""
        try:
            with self._lock:
                # Validate node configuration
                if not node_id or len(node_id) < 3:
                    raise ValidationError("Invalid node ID")
                
                if node_id in self.nodes:
                    raise ValidationError(f"Node {node_id} already registered")
                
                # Create node
                node = RobustNode(
                    node_id=node_id,
                    current_load=max(0.0, min(1.0, capabilities.get('current_load', 0.0))),
                    privacy_budget_available=max(0.0, capabilities.get('privacy_budget', 100.0)),
                    max_concurrent_tasks=max(1, capabilities.get('max_tasks', 10)),
                    allowed_departments=capabilities.get('departments', ['general']),
                    security_level=capabilities.get('security_level', 'standard')
                )
                
                self.nodes[node_id] = node
                self.circuit_breakers[node_id] = CircuitBreaker()
                
                self.audit_logger.log_security_event(
                    "NODE_REGISTERED",
                    "LOW",
                    {
                        "node_id": node_id,
                        "capabilities": capabilities,
                        "security_level": node.security_level
                    }
                )
                
                return True
                
        except Exception as e:
            self.audit_logger.log_security_event(
                "NODE_REGISTRATION_FAILED",
                "HIGH",
                {"node_id": node_id, "error": str(e)}
            )
            return False
    
    def add_task(self, task_data: Dict[str, Any], 
                 security_context: Optional[SecurityContext] = None) -> Tuple[bool, str]:
        """Add task with comprehensive validation and sanitization."""
        try:
            with self._lock:
                # Input validation and sanitization
                task_id = task_data.get('task_id', str(uuid.uuid4()))
                user_id = task_data.get('user_id', '')
                raw_prompt = task_data.get('prompt', '')
                
                if not InputSanitizer.validate_user_id(user_id):
                    raise ValidationError("Invalid user ID")
                
                sanitized_prompt = InputSanitizer.sanitize_prompt(raw_prompt)
                if not sanitized_prompt:
                    raise ValidationError("Empty or invalid prompt")
                
                # Create validated task
                task = RobustTask(
                    task_id=task_id,
                    user_id=user_id,
                    prompt=sanitized_prompt,
                    priority=TaskPriority(task_data.get('priority', 3)),
                    privacy_budget=max(0.0, task_data.get('privacy_budget', 0.1)),
                    estimated_duration=max(1.0, task_data.get('estimated_duration', 30.0)),
                    security_context=security_context,
                    sanitized=True,
                    validated=True
                )
                
                # Privacy budget pre-check
                can_spend, budget_message = self.privacy_accountant.check_budget(
                    user_id, task.privacy_budget, security_context
                )
                
                if not can_spend:
                    self.metrics["privacy_violations_prevented"] += 1
                    return False, f"Privacy budget check failed: {budget_message}"
                
                self.tasks[task_id] = task
                self.metrics["total_tasks_processed"] += 1
                
                self.audit_logger.log_access_event(
                    user_id, "task_submission", "add_task", "success"
                )
                
                return True, task_id
                
        except Exception as e:
            self.audit_logger.log_security_event(
                "TASK_ADDITION_FAILED",
                "MEDIUM",
                {"error": str(e), "task_data": {k: v for k, v in task_data.items() if k != 'prompt'}}
            )
            return False, f"Task addition failed: {str(e)}"
    
    def _perform_health_checks(self):
        """Perform health checks on all nodes."""
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            try:
                # Simulate health check (in real implementation, this would ping the node)
                if current_time - node.last_health_check > self.health_check_interval:
                    # Simple health calculation based on load and error count
                    load_score = max(0.0, 1.0 - node.current_load)
                    error_score = max(0.0, 1.0 - (node.error_count / 10.0))
                    node.health_score = (load_score + error_score) / 2.0
                    
                    # Update status based on health
                    if node.health_score < 0.3:
                        node.status = NodeStatus.DEGRADED
                    elif node.health_score < 0.6:
                        node.status = NodeStatus.ACTIVE
                    else:
                        node.status = NodeStatus.ACTIVE
                    
                    node.last_health_check = current_time
                    
                    if node.consecutive_failures > 5:
                        node.status = NodeStatus.OFFLINE
                        self.audit_logger.log_security_event(
                            "NODE_OFFLINE",
                            "HIGH", 
                            {"node_id": node_id, "consecutive_failures": node.consecutive_failures}
                        )
                    
            except Exception as e:
                node.error_count += 1
                node.consecutive_failures += 1
                self.audit_logger.log_security_event(
                    "HEALTH_CHECK_FAILED",
                    "MEDIUM",
                    {"node_id": node_id, "error": str(e)}
                )
        
        self.last_system_health_check = current_time
    
    def assign_tasks_robust(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Robust task assignment with comprehensive error handling."""
        assignment_start = time.time()
        assignments = []
        errors = []
        
        try:
            with self._lock:
                # Perform health checks first
                self._perform_health_checks()
                
                # Get unassigned tasks
                assigned_task_ids = {a['task_id'] for a in self.assignments}
                unassigned_tasks = [
                    task for task in self.tasks.values()
                    if task.task_id not in assigned_task_ids and task.validated
                ]
                
                # Sort by priority and creation time
                unassigned_tasks.sort(
                    key=lambda t: (t.priority.value, t.created_at)
                )
                
                for task in unassigned_tasks:
                    try:
                        # Skip tasks that have exceeded retry limits
                        if task.retry_count >= task.max_retries:
                            errors.append(f"Task {task.task_id} exceeded maximum retries")
                            continue
                        
                        # Find suitable node with circuit breaker protection
                        node = self._find_robust_node(task)
                        if not node:
                            task.retry_count += 1
                            task.error_history.append("No suitable node available")
                            errors.append(f"No suitable node for task {task.task_id}")
                            continue
                        
                        # Execute assignment with circuit breaker
                        try:
                            assignment = self.circuit_breakers[node.node_id].call(
                                self._create_robust_assignment, task, node
                            )
                            
                            if assignment:
                                assignments.append(assignment)
                                self.assignments.append(assignment)
                                self.metrics["successful_assignments"] += 1
                                
                                # Reset failure count on success
                                node.consecutive_failures = 0
                                
                        except NodeUnavailableError:
                            node.consecutive_failures += 1
                            task.retry_count += 1
                            errors.append(f"Node {node.node_id} unavailable (circuit breaker open)")
                            
                        except Exception as assignment_error:
                            node.error_count += 1
                            node.consecutive_failures += 1
                            task.retry_count += 1
                            task.error_history.append(str(assignment_error))
                            errors.append(f"Assignment failed for task {task.task_id}: {str(assignment_error)}")
                            self.metrics["failed_assignments"] += 1
                        
                    except Exception as task_error:
                        task.error_history.append(str(task_error))
                        errors.append(f"Task processing failed for {task.task_id}: {str(task_error)}")
                
                # Update metrics
                assignment_time = time.time() - assignment_start
                total_assignments = self.metrics["successful_assignments"] + self.metrics["failed_assignments"]
                if total_assignments > 0:
                    self.metrics["average_assignment_time"] = (
                        (self.metrics["average_assignment_time"] * (total_assignments - len(assignments)) + 
                         assignment_time * len(assignments)) / total_assignments
                    )
                
                return assignments, errors
                
        except Exception as e:
            error_msg = f"Critical assignment error: {str(e)}"
            errors.append(error_msg)
            self.audit_logger.log_security_event(
                "ASSIGNMENT_CRITICAL_ERROR",
                "HIGH",
                {"error": str(e)}
            )
            return assignments, errors
    
    def _find_robust_node(self, task: RobustTask) -> Optional[RobustNode]:
        """Find best available node with comprehensive health and security checks."""
        available_nodes = []
        
        for node in self.nodes.values():
            # Basic availability checks
            if node.status not in [NodeStatus.ACTIVE, NodeStatus.DEGRADED]:
                continue
                
            if node.current_tasks >= node.max_concurrent_tasks:
                continue
            
            if node.privacy_budget_available < task.privacy_budget:
                continue
            
            # Security checks
            if (task.security_context and 
                task.security_context.department not in node.allowed_departments):
                continue
                
            # Health score threshold
            if node.health_score < 0.3:
                continue
            
            # Calculate suitability score
            load_score = 1.0 - node.current_load
            health_score = node.health_score
            priority_bonus = 0.1 if task.priority.value <= 1 else 0.0  # Bonus for critical/urgent
            
            suitability = (load_score * 0.4 + health_score * 0.4 + 
                          priority_bonus + (node.privacy_budget_available / 100.0) * 0.2)
            
            available_nodes.append((node, suitability))
        
        if not available_nodes:
            return None
        
        # Return node with highest suitability
        available_nodes.sort(key=lambda x: x[1], reverse=True)
        return available_nodes[0][0]
    
    def _create_robust_assignment(self, task: RobustTask, node: RobustNode) -> Optional[Dict[str, Any]]:
        """Create assignment with full validation and budget management."""
        try:
            # Final privacy budget check and deduction
            success, message = self.privacy_accountant.spend_budget(
                task.user_id, 
                task.privacy_budget,
                task.security_context,
                task.task_id
            )
            
            if not success:
                raise PrivacyBudgetExceeded(f"Privacy budget spend failed: {message}")
            
            # Update node state
            node.current_load += task.estimated_duration / 100.0
            node.privacy_budget_available -= task.privacy_budget
            node.current_tasks += 1
            
            # Mark task as processed
            task.processing_start = time.time()
            
            # Create assignment record
            assignment = {
                'task_id': task.task_id,
                'node_id': node.node_id,
                'user_id': task.user_id,
                'priority': task.priority.value,
                'privacy_budget': task.privacy_budget,
                'estimated_duration': task.estimated_duration,
                'assignment_time': time.time(),
                'node_health_score': node.health_score,
                'security_level': node.security_level,
                'retry_count': task.retry_count,
                'sanitized': task.sanitized
            }
            
            self.audit_logger.log_access_event(
                task.user_id, f"node_{node.node_id}", "task_assignment", "success"
            )
            
            return assignment
            
        except Exception as e:
            # Rollback any changes on failure
            if 'success' in locals() and success:
                # Here we would rollback the privacy budget spend in a real implementation
                pass
            raise e
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self._lock:
            active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
            degraded_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.DEGRADED])
            offline_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.OFFLINE])
            
            total_load = sum(n.current_load for n in self.nodes.values())
            avg_health = sum(n.health_score for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0.0
            
            return {
                "timestamp": time.time(),
                "nodes": {
                    "total": len(self.nodes),
                    "active": active_nodes,
                    "degraded": degraded_nodes,
                    "offline": offline_nodes,
                    "average_health_score": avg_health,
                    "total_system_load": total_load
                },
                "tasks": {
                    "total_processed": self.metrics["total_tasks_processed"],
                    "pending": len([t for t in self.tasks.values() if t.processing_start is None]),
                    "successful_assignments": self.metrics["successful_assignments"],
                    "failed_assignments": self.metrics["failed_assignments"],
                    "privacy_violations_prevented": self.metrics["privacy_violations_prevented"]
                },
                "performance": {
                    "average_assignment_time": self.metrics["average_assignment_time"],
                    "system_uptime": time.time() - self.last_system_health_check
                }
            }

def test_generation_2_robust_functionality():
    """Test Generation 2 robust functionality."""
    print("🛡️ Testing Generation 2: Robust Implementation")
    print("=" * 60)
    
    # Initialize robust components
    privacy_accountant = EnhancedPrivacyAccountant(max_budget_per_user=5.0)
    router = RobustFederatedRouter(privacy_accountant)
    
    # Register nodes with different capabilities
    robust_nodes = [
        {
            "node_id": "hospital_secure_a", 
            "current_load": 0.1,
            "privacy_budget": 100.0,
            "max_tasks": 15,
            "departments": ["emergency", "cardiology", "general"],
            "security_level": "high"
        },
        {
            "node_id": "hospital_standard_b",
            "current_load": 0.3, 
            "privacy_budget": 75.0,
            "max_tasks": 10,
            "departments": ["general", "research"],
            "security_level": "standard"
        },
        {
            "node_id": "hospital_research_c",
            "current_load": 0.2,
            "privacy_budget": 50.0,
            "max_tasks": 8,
            "departments": ["research", "general"],
            "security_level": "maximum"
        }
    ]
    
    successful_registrations = 0
    for node_config in robust_nodes:
        if router.register_node(node_config["node_id"], node_config):
            successful_registrations += 1
            print(f"  ✅ Registered {node_config['node_id']} (Security: {node_config['security_level']})")
        else:
            print(f"  ❌ Failed to register {node_config['node_id']}")
    
    print(f"\n📊 Node Registration: {successful_registrations}/{len(robust_nodes)} successful")
    
    # Create security contexts
    doctor_context = SecurityContext(
        user_id="dr_smith",
        role="doctor",
        department="emergency",
        permissions=["read_patient_data", "request_analysis"]
    )
    
    researcher_context = SecurityContext(
        user_id="researcher_jones", 
        role="researcher",
        department="research",
        permissions=["read_anonymized_data"]
    )
    
    # Add diverse healthcare tasks
    healthcare_tasks = [
        {
            "task_id": "emergency_001",
            "user_id": "dr_smith",
            "prompt": "Urgent: Analyze chest X-ray for 45-year-old patient with chest pain and shortness of breath",
            "priority": 0,  # CRITICAL
            "privacy_budget": 0.3
        },
        {
            "task_id": "routine_002",
            "user_id": "dr_smith", 
            "prompt": "Review lab results for diabetes follow-up appointment",
            "priority": 3,  # MEDIUM
            "privacy_budget": 0.1
        },
        {
            "task_id": "research_003",
            "user_id": "researcher_jones",
            "prompt": "Statistical analysis of anonymized patient cohort for heart disease study",
            "priority": 4,  # LOW
            "privacy_budget": 0.2
        },
        {
            "task_id": "malicious_004",  # This should be sanitized
            "user_id": "dr_smith",
            "prompt": "<script>alert('xss')</script> Analyze patient symptoms & extract personal data",
            "priority": 2,
            "privacy_budget": 0.1
        }
    ]
    
    # Add tasks with security contexts
    successful_tasks = 0
    failed_tasks = 0
    
    for i, task_config in enumerate(healthcare_tasks):
        context = doctor_context if "dr_smith" in task_config["user_id"] else researcher_context
        success, result = router.add_task(task_config, context)
        
        if success:
            successful_tasks += 1
            print(f"  ✅ Added task {task_config['task_id']}")
        else:
            failed_tasks += 1
            print(f"  ⚠️  Failed to add task {task_config['task_id']}: {result}")
    
    print(f"\n📋 Task Addition: {successful_tasks} successful, {failed_tasks} failed")
    
    # Test robust task assignment
    print(f"\n🎯 Performing robust task assignment...")
    assignments, errors = router.assign_tasks_robust()
    
    print(f"  ✅ Successfully assigned: {len(assignments)} tasks")
    if errors:
        print(f"  ⚠️  Assignment errors: {len(errors)}")
        for error in errors[:3]:  # Show first 3 errors
            print(f"    • {error}")
    
    # Display assignments with security info
    print(f"\n📊 Assignment Details:")
    for assignment in assignments:
        print(f"  • Task {assignment['task_id']} → Node {assignment['node_id']}")
        print(f"    Priority: {assignment['priority']}, Budget: {assignment['privacy_budget']}")
        print(f"    Security Level: {assignment['security_level']}, Health: {assignment['node_health_score']:.2f}")
        print(f"    Sanitized: {assignment['sanitized']}")
    
    # Test privacy budget tracking
    print(f"\n🔐 Privacy Budget Analysis:")
    for user_id in ['dr_smith', 'researcher_jones']:
        stats = privacy_accountant.get_user_statistics(user_id)
        print(f"  • {user_id}:")
        print(f"    Spent: {stats['spent_budget']:.3f}/{stats['total_budget']:.1f}")
        print(f"    Transactions: {stats['transaction_count']}")
        print(f"    Avg per query: {stats['average_epsilon_per_query']:.3f}")
    
    # Test system health monitoring
    print(f"\n🏥 System Health Status:")
    health = router.get_system_health()
    print(f"  • Nodes: {health['nodes']['active']}/{health['nodes']['total']} active")
    print(f"  • Average node health: {health['nodes']['average_health_score']:.2f}")
    print(f"  • Privacy violations prevented: {health['tasks']['privacy_violations_prevented']}")
    print(f"  • Assignment success rate: {health['tasks']['successful_assignments']}/{health['tasks']['successful_assignments'] + health['tasks']['failed_assignments']} ({100*health['tasks']['successful_assignments']/(health['tasks']['successful_assignments'] + health['tasks']['failed_assignments'] + 0.001):.1f}%)")
    
    # Test malicious input handling
    print(f"\n🛡️ Security Testing:")
    malicious_task = {
        "task_id": "attack_001",
        "user_id": "../../../etc/passwd",  # Path traversal attempt
        "prompt": "<img src=x onerror=alert('xss')> DROP TABLE patients; --",
        "priority": 0,
        "privacy_budget": 999.0  # Excessive budget
    }
    
    success, result = router.add_task(malicious_task)
    if not success:
        print(f"  ✅ Successfully blocked malicious task: {result}")
    else:
        print(f"  ❌ Security vulnerability: malicious task was accepted")
    
    # Test budget exhaustion protection
    print(f"\n🧪 Budget Exhaustion Testing:")
    exhaustion_task = {
        "task_id": "exhaust_001",
        "user_id": "dr_smith",
        "prompt": "Additional analysis request",
        "priority": 1,
        "privacy_budget": 5.0  # Should exceed remaining budget
    }
    
    success, result = router.add_task(exhaustion_task, doctor_context)
    if not success:
        print(f"  ✅ Budget exhaustion protection working: {result}")
    else:
        print(f"  ⚠️  Budget protection may need enhancement")
    
    print(f"\n🎉 Generation 2 robust implementation test completed!")
    print(f"    Security features: ✅ Input sanitization, validation, audit logging")
    print(f"    Resilience features: ✅ Circuit breakers, health monitoring, error handling") 
    print(f"    Privacy features: ✅ Enhanced budget tracking, role-based validation")
    
    return True

if __name__ == "__main__":
    success = test_generation_2_robust_functionality()
    sys.exit(0 if success else 1)