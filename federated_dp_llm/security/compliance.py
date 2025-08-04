"""
HIPAA/GDPR Compliance and Budget Management

Implements compliance monitoring, audit trails, and departmental privacy
budget management for healthcare environments.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"


class AuditEventType(Enum):
    """Types of audit events."""
    QUERY_SUBMITTED = "query_submitted"
    PRIVACY_BUDGET_SPENT = "privacy_budget_spent"
    MODEL_ACCESS = "model_access"
    DATA_EXPORT = "data_export"
    USER_LOGIN = "user_login"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class AuditEvent:
    """Represents an audit event for compliance tracking."""
    event_id: str
    event_type: AuditEventType
    user_id: str
    department: str
    timestamp: float
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    violation_id: str
    framework: ComplianceFramework
    rule: str
    description: str
    severity: str  # low, medium, high, critical
    user_id: str
    department: str
    timestamp: float
    resolved: bool = False
    resolution_notes: Optional[str] = None


class BudgetManager:
    """Manages privacy budgets per department with compliance tracking."""
    
    def __init__(self, department_budgets: Dict[str, float]):
        self.department_budgets = department_budgets.copy()
        self.initial_budgets = department_budgets.copy()
        self.spent_budgets: Dict[str, float] = {dept: 0.0 for dept in department_budgets}
        self.budget_history: List[Dict[str, Any]] = []
        self.reset_schedule: Dict[str, str] = {}  # department -> reset_frequency
        
        # Default reset schedules
        for dept in department_budgets:
            self.reset_schedule[dept] = "daily"  # daily, weekly, monthly
    
    def can_query(self, department: str, requested_epsilon: float) -> bool:
        """Check if department has sufficient budget for query."""
        if department not in self.department_budgets:
            return False
        
        remaining = self.get_remaining_budget(department)
        return remaining >= requested_epsilon
    
    def deduct(self, department: str, epsilon: float, user_id: str = None) -> bool:
        """Deduct privacy budget from department."""
        if not self.can_query(department, epsilon):
            return False
        
        self.spent_budgets[department] += epsilon
        
        # Record budget transaction
        transaction = {
            "department": department,
            "user_id": user_id,
            "epsilon_spent": epsilon,
            "remaining_budget": self.get_remaining_budget(department),
            "timestamp": time.time()
        }
        self.budget_history.append(transaction)
        
        return True
    
    def get_remaining_budget(self, department: str) -> float:
        """Get remaining privacy budget for department."""
        if department not in self.department_budgets:
            return 0.0
        
        return max(0.0, self.department_budgets[department] - self.spent_budgets[department])
    
    def reset_department_budget(self, department: str) -> bool:
        """Reset privacy budget for a department."""
        if department not in self.department_budgets:
            return False
        
        self.spent_budgets[department] = 0.0
        
        # Record reset event
        reset_event = {
            "department": department,
            "action": "budget_reset",
            "previous_spent": self.spent_budgets[department],
            "new_budget": self.department_budgets[department],
            "timestamp": time.time()
        }
        self.budget_history.append(reset_event)
        
        return True
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status for all departments."""
        status = {}
        
        for dept in self.department_budgets:
            remaining = self.get_remaining_budget(dept)
            utilization = (self.spent_budgets[dept] / self.department_budgets[dept]) * 100
            
            # Determine alert level
            if utilization >= 90:
                alert_level = "critical"
            elif utilization >= 75:
                alert_level = "warning"
            elif utilization >= 50:
                alert_level = "caution"
            else:
                alert_level = "normal"
            
            status[dept] = {
                "total_budget": self.department_budgets[dept],
                "spent": self.spent_budgets[dept],
                "remaining": remaining,
                "utilization_percent": utilization,
                "alert_level": alert_level,
                "reset_schedule": self.reset_schedule.get(dept, "daily")
            }
        
        return status
    
    def schedule_budget_reset(self, department: str, frequency: str) -> bool:
        """Schedule automatic budget resets."""
        valid_frequencies = ["daily", "weekly", "monthly"]
        
        if frequency not in valid_frequencies:
            return False
        
        if department not in self.department_budgets:
            return False
        
        self.reset_schedule[department] = frequency
        return True


class ComplianceMonitor:
    """Monitors compliance with healthcare regulations."""
    
    def __init__(self, 
                 frameworks: List[ComplianceFramework] = None,
                 audit_retention_days: int = 2555):  # 7 years for HIPAA
        self.frameworks = frameworks or [ComplianceFramework.HIPAA, ComplianceFramework.GDPR]
        self.audit_retention_days = audit_retention_days
        
        # Storage
        self.audit_events: List[AuditEvent] = []
        self.violations: List[ComplianceViolation] = []
        
        # Configuration
        self.sensitive_data_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        ]
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def track(self, func):
        """Decorator to automatically track function calls for compliance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract user context if available
            user_id = kwargs.get('user_id', 'unknown')
            department = kwargs.get('department', 'general')
            
            try:
                result = func(*args, **kwargs)
                
                # Create audit event
                event = AuditEvent(
                    event_id=self._generate_event_id(),
                    event_type=AuditEventType.QUERY_SUBMITTED,
                    user_id=user_id,
                    department=department,
                    timestamp=start_time,
                    details={
                        "function": func.__name__,
                        "duration": time.time() - start_time,
                        "success": True,
                        "args_hash": self._hash_args(args, kwargs)
                    }
                )
                
                self.record_event(event)
                return result
                
            except Exception as e:
                # Record failed attempt
                event = AuditEvent(
                    event_id=self._generate_event_id(),
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    user_id=user_id,
                    department=department,
                    timestamp=start_time,
                    details={
                        "function": func.__name__,
                        "duration": time.time() - start_time,
                        "success": False,
                        "error": str(e),
                        "args_hash": self._hash_args(args, kwargs)
                    },
                    risk_level="high"
                )
                
                self.record_event(event)
                raise
        
        return wrapper
    
    def record_event(self, event: AuditEvent):
        """Record an audit event."""
        self.audit_events.append(event)
        
        # Check for compliance violations
        violations = self._check_compliance_rules(event)
        self.violations.extend(violations)
        
        # Log high-risk events
        if event.risk_level in ["high", "critical"]:
            self.logger.warning(f"High-risk audit event: {event.event_type.value} by {event.user_id}")
        
        # Clean up old events
        self._cleanup_old_events()
    
    def _check_compliance_rules(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Check event against compliance rules."""
        violations = []
        
        for framework in self.frameworks:
            framework_violations = []
            
            if framework == ComplianceFramework.HIPAA:
                framework_violations.extend(self._check_hipaa_rules(event))
            elif framework == ComplianceFramework.GDPR:
                framework_violations.extend(self._check_gdpr_rules(event))
            
            violations.extend(framework_violations)
        
        return violations
    
    def _check_hipaa_rules(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Check HIPAA compliance rules."""
        violations = []
        
        # Rule: Minimum necessary standard
        if event.event_type == AuditEventType.QUERY_SUBMITTED:
            query_details = event.details.get('query', '')
            if self._contains_sensitive_data(query_details):
                violation = ComplianceViolation(
                    violation_id=self._generate_violation_id(),
                    framework=ComplianceFramework.HIPAA,
                    rule="minimum_necessary",
                    description="Query may contain unnecessary sensitive information",
                    severity="medium",
                    user_id=event.user_id,
                    department=event.department,
                    timestamp=event.timestamp
                )
                violations.append(violation)
        
        # Rule: Access control
        if event.risk_level == "critical":
            violation = ComplianceViolation(
                violation_id=self._generate_violation_id(),
                framework=ComplianceFramework.HIPAA,
                rule="access_control",
                description="Critical security event detected",
                severity="high",
                user_id=event.user_id,
                department=event.department,
                timestamp=event.timestamp
            )
            violations.append(violation)
        
        return violations
    
    def _check_gdpr_rules(self, event: AuditEvent) -> List[ComplianceViolation]:
        """Check GDPR compliance rules."""
        violations = []
        
        # Rule: Data minimization
        if event.event_type == AuditEventType.DATA_EXPORT:
            export_size = event.details.get('data_size', 0)
            if export_size > 1000000:  # 1MB threshold
                violation = ComplianceViolation(
                    violation_id=self._generate_violation_id(),
                    framework=ComplianceFramework.GDPR,
                    rule="data_minimization",
                    description="Large data export may violate data minimization principle",
                    severity="medium",
                    user_id=event.user_id,
                    department=event.department,
                    timestamp=event.timestamp
                )
                violations.append(violation)
        
        return violations
    
    def _contains_sensitive_data(self, text: str) -> bool:
        """Check if text contains sensitive data patterns."""
        import re
        
        for pattern in self.sensitive_data_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def generate_report(self, 
                       period: str = "monthly",
                       include_privacy_metrics: bool = True,
                       format: str = "json") -> Dict[str, Any]:
        """Generate compliance report."""
        
        # Calculate time period
        end_time = time.time()
        if period == "daily":
            start_time = end_time - (24 * 3600)
        elif period == "weekly":
            start_time = end_time - (7 * 24 * 3600)
        elif period == "monthly":
            start_time = end_time - (30 * 24 * 3600)
        else:
            start_time = 0  # All time
        
        # Filter events and violations by time period
        period_events = [
            event for event in self.audit_events
            if start_time <= event.timestamp <= end_time
        ]
        
        period_violations = [
            violation for violation in self.violations
            if start_time <= violation.timestamp <= end_time
        ]
        
        # Generate report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "period": period,
                "start_date": datetime.fromtimestamp(start_time).isoformat(),
                "end_date": datetime.fromtimestamp(end_time).isoformat(),
                "total_events": len(period_events),
                "total_violations": len(period_violations)
            },
            "event_summary": self._summarize_events(period_events),
            "violation_summary": self._summarize_violations(period_violations),
            "compliance_score": self._calculate_compliance_score(period_events, period_violations),
            "recommendations": self._generate_recommendations(period_violations)
        }
        
        if include_privacy_metrics:
            report["privacy_metrics"] = self._calculate_privacy_metrics(period_events)
        
        return report
    
    def _summarize_events(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Summarize audit events."""
        if not events:
            return {"total": 0}
        
        event_types = {}
        departments = {}
        risk_levels = {}
        
        for event in events:
            # Count by event type
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Count by department
            departments[event.department] = departments.get(event.department, 0) + 1
            
            # Count by risk level
            risk_levels[event.risk_level] = risk_levels.get(event.risk_level, 0) + 1
        
        return {
            "total": len(events),
            "by_event_type": event_types,
            "by_department": departments,
            "by_risk_level": risk_levels
        }
    
    def _summarize_violations(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Summarize compliance violations."""
        if not violations:
            return {"total": 0, "resolved": 0, "unresolved": 0}
        
        frameworks = {}
        severities = {}
        resolved_count = 0
        
        for violation in violations:
            # Count by framework
            framework = violation.framework.value
            frameworks[framework] = frameworks.get(framework, 0) + 1
            
            # Count by severity
            severities[violation.severity] = severities.get(violation.severity, 0) + 1
            
            # Count resolved
            if violation.resolved:
                resolved_count += 1
        
        return {
            "total": len(violations),
            "resolved": resolved_count,
            "unresolved": len(violations) - resolved_count,
            "by_framework": frameworks,
            "by_severity": severities
        }
    
    def _calculate_compliance_score(self, events: List[AuditEvent], violations: List[ComplianceViolation]) -> float:
        """Calculate overall compliance score (0-100)."""
        if not events:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Deduct points for violations
        for violation in violations:
            if violation.severity == "critical":
                score -= 20
            elif violation.severity == "high":
                score -= 10
            elif violation.severity == "medium":
                score -= 5
            elif violation.severity == "low":
                score -= 2
        
        # Deduct points for high-risk events
        high_risk_events = [e for e in events if e.risk_level in ["high", "critical"]]
        score -= len(high_risk_events) * 2
        
        return max(0.0, score)
    
    def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Count violation types
        violation_counts = {}
        for violation in violations:
            key = f"{violation.framework.value}:{violation.rule}"
            violation_counts[key] = violation_counts.get(key, 0) + 1
        
        # Generate recommendations based on common violations
        for violation_key, count in violation_counts.items():
            if count >= 3:  # Multiple occurrences
                framework, rule = violation_key.split(':')
                recommendations.append(
                    f"Review {framework.upper()} {rule} compliance - {count} violations detected"
                )
        
        # Generic recommendations
        if len(violations) > 10:
            recommendations.append("Consider additional compliance training for staff")
        
        if not recommendations:
            recommendations.append("No specific recommendations - compliance appears satisfactory")
        
        return recommendations
        
    def _calculate_privacy_metrics(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Calculate privacy-related metrics."""
        privacy_events = [
            e for e in events 
            if e.event_type in [AuditEventType.QUERY_SUBMITTED, AuditEventType.PRIVACY_BUDGET_SPENT]
        ]
        
        if not privacy_events:
            return {"total_privacy_queries": 0}
        
        total_epsilon_spent = 0.0
        for event in privacy_events:
            epsilon = event.details.get('epsilon_spent', 0.0)
            total_epsilon_spent += epsilon
        
        return {
            "total_privacy_queries": len(privacy_events),
            "total_epsilon_spent": total_epsilon_spent,
            "average_epsilon_per_query": total_epsilon_spent / len(privacy_events) if privacy_events else 0.0,
            "privacy_budget_utilization": min(100.0, (total_epsilon_spent / 100.0) * 100)  # Assuming max budget of 100
        }
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"event_{int(time.time() * 1000000)}"
    
    def _generate_violation_id(self) -> str:
        """Generate unique violation ID."""
        return f"violation_{int(time.time() * 1000000)}"
    
    def _hash_args(self, args: tuple, kwargs: dict) -> str:
        """Generate hash of function arguments for audit trail."""
        arg_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(arg_str.encode()).hexdigest()[:16]
    
    def _cleanup_old_events(self):
        """Remove old audit events based on retention policy."""
        cutoff_time = time.time() - (self.audit_retention_days * 24 * 3600)
        
        self.audit_events = [
            event for event in self.audit_events
            if event.timestamp > cutoff_time
        ]
        
        self.violations = [
            violation for violation in self.violations
            if violation.timestamp > cutoff_time
        ]