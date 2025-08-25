"""
Graceful Degradation System for Federated DP-LLM Router

Implements intelligent failover and feature degradation strategies to maintain
service availability under various failure conditions while preserving privacy.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
import traceback


logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL_SERVICE = auto()      # All features available
    REDUCED_FEATURES = auto()  # Non-critical features disabled
    ESSENTIAL_ONLY = auto()    # Only critical healthcare functions
    EMERGENCY_MODE = auto()    # Minimal viable service
    OFFLINE = auto()           # Complete service unavailable


class ServiceComponent(Enum):
    """System components that can be degraded."""
    QUANTUM_PLANNING = "quantum_planning"
    PRIVACY_ACCOUNTING = "privacy_accounting"
    FEDERATED_ROUTING = "federated_routing"
    SECURITY_VALIDATION = "security_validation"
    MONITORING = "monitoring"
    CACHING = "caching"
    LOAD_BALANCING = "load_balancing"
    GLOBAL_ROUTING = "global_routing"


@dataclass
class DegradationRule:
    """Rule for degrading a specific component."""
    component: ServiceComponent
    trigger_condition: str
    fallback_behavior: str
    degradation_level: DegradationLevel
    recovery_threshold: float = 0.8
    max_degradation_time: int = 300  # 5 minutes


@dataclass
class SystemHealth:
    """Current system health status."""
    overall_health: float = 1.0
    component_health: Dict[ServiceComponent, float] = field(default_factory=dict)
    active_degradations: List[DegradationRule] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    error_counts: Dict[str, int] = field(default_factory=dict)


class GracefulDegradationManager:
    """Manages graceful degradation of system components."""
    
    def __init__(self):
        self.health = SystemHealth()
        self.degradation_rules = self._initialize_degradation_rules()
        self.fallback_handlers = {}
        self.recovery_tasks = {}
        
    def _initialize_degradation_rules(self) -> List[DegradationRule]:
        """Initialize degradation rules for different components."""
        return [
            # Quantum Planning Degradation
            DegradationRule(
                component=ServiceComponent.QUANTUM_PLANNING,
                trigger_condition="import_error or cpu_usage > 0.9",
                fallback_behavior="switch_to_simple_round_robin",
                degradation_level=DegradationLevel.REDUCED_FEATURES
            ),
            
            # Privacy Accounting Degradation
            DegradationRule(
                component=ServiceComponent.PRIVACY_ACCOUNTING,
                trigger_condition="memory_usage > 0.85",
                fallback_behavior="use_simplified_accounting",
                degradation_level=DegradationLevel.ESSENTIAL_ONLY
            ),
            
            # Security Validation Degradation
            DegradationRule(
                component=ServiceComponent.SECURITY_VALIDATION,
                trigger_condition="validation_latency > 500ms",
                fallback_behavior="use_basic_validation_only",
                degradation_level=DegradationLevel.REDUCED_FEATURES
            ),
            
            # Global Routing Degradation
            DegradationRule(
                component=ServiceComponent.GLOBAL_ROUTING,
                trigger_condition="network_latency > 200ms",
                fallback_behavior="use_local_routing_only",
                degradation_level=DegradationLevel.REDUCED_FEATURES
            ),
            
            # Monitoring Degradation
            DegradationRule(
                component=ServiceComponent.MONITORING,
                trigger_condition="storage_usage > 0.9",
                fallback_behavior="reduce_metric_collection",
                degradation_level=DegradationLevel.REDUCED_FEATURES
            )
        ]
    
    def register_fallback_handler(self, component: ServiceComponent, handler: Callable):
        """Register a fallback handler for a component."""
        self.fallback_handlers[component] = handler
        logger.info(f"Registered fallback handler for {component.value}")
    
    def with_graceful_degradation(self, component: ServiceComponent, fallback_result: Any = None):
        """Decorator for methods that should gracefully degrade on failure."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Component {component.value} failed: {str(e)}")
                    self._record_failure(component, str(e))
                    
                    # Check if degradation is needed
                    if self._should_degrade(component):
                        await self._trigger_degradation(component)
                    
                    # Return fallback result or call fallback handler
                    if component in self.fallback_handlers:
                        try:
                            return await self.fallback_handlers[component](*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback handler for {component.value} failed: {fallback_error}")
                    
                    return fallback_result
            return wrapper
        return decorator
    
    def _record_failure(self, component: ServiceComponent, error: str):
        """Record component failure for health tracking."""
        error_key = f"{component.value}:{error[:50]}"
        self.health.error_counts[error_key] = self.health.error_counts.get(error_key, 0) + 1
        
        # Update component health
        current_health = self.health.component_health.get(component, 1.0)
        self.health.component_health[component] = max(0.0, current_health - 0.1)
        
        # Update overall health
        self.health.overall_health = min(self.health.component_health.values()) if self.health.component_health else 1.0
        self.health.last_update = time.time()
    
    def _should_degrade(self, component: ServiceComponent) -> bool:
        """Check if component should be degraded based on health and error patterns."""
        component_health = self.health.component_health.get(component, 1.0)
        error_count = sum(count for key, count in self.health.error_counts.items() 
                         if key.startswith(component.value))
        
        return component_health < 0.7 or error_count > 5
    
    async def _trigger_degradation(self, component: ServiceComponent):
        """Trigger degradation for a specific component."""
        rule = next((r for r in self.degradation_rules if r.component == component), None)
        if not rule:
            logger.warning(f"No degradation rule found for {component.value}")
            return
        
        if rule not in self.health.active_degradations:
            self.health.active_degradations.append(rule)
            logger.info(f"Triggered degradation for {component.value}: {rule.fallback_behavior}")
            
            # Schedule recovery attempt
            if component not in self.recovery_tasks:
                self.recovery_tasks[component] = asyncio.create_task(
                    self._attempt_recovery(component, rule)
                )
    
    async def _attempt_recovery(self, component: ServiceComponent, rule: DegradationRule):
        """Attempt to recover from degradation."""
        start_time = time.time()
        
        while time.time() - start_time < rule.max_degradation_time:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            component_health = self.health.component_health.get(component, 0.0)
            if component_health >= rule.recovery_threshold:
                # Remove degradation
                if rule in self.health.active_degradations:
                    self.health.active_degradations.remove(rule)
                
                logger.info(f"Recovered from degradation: {component.value}")
                break
        else:
            logger.warning(f"Failed to recover {component.value} within time limit")
        
        # Cleanup recovery task
        if component in self.recovery_tasks:
            del self.recovery_tasks[component]
    
    def get_current_degradation_level(self) -> DegradationLevel:
        """Get current system degradation level."""
        if not self.health.active_degradations:
            return DegradationLevel.FULL_SERVICE
        
        levels = [rule.degradation_level for rule in self.health.active_degradations]
        
        if DegradationLevel.OFFLINE in levels:
            return DegradationLevel.OFFLINE
        elif DegradationLevel.EMERGENCY_MODE in levels:
            return DegradationLevel.EMERGENCY_MODE
        elif DegradationLevel.ESSENTIAL_ONLY in levels:
            return DegradationLevel.ESSENTIAL_ONLY
        elif DegradationLevel.REDUCED_FEATURES in levels:
            return DegradationLevel.REDUCED_FEATURES
        else:
            return DegradationLevel.FULL_SERVICE
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "overall_health": self.health.overall_health,
            "degradation_level": self.get_current_degradation_level().name,
            "component_health": {
                comp.value: health for comp, health in self.health.component_health.items()
            },
            "active_degradations": [
                {
                    "component": rule.component.value,
                    "behavior": rule.fallback_behavior,
                    "level": rule.degradation_level.name
                } for rule in self.health.active_degradations
            ],
            "error_summary": dict(self.health.error_counts),
            "last_update": self.health.last_update
        }
    
    def is_feature_available(self, feature: ServiceComponent) -> bool:
        """Check if a specific feature is currently available."""
        degradation_level = self.get_current_degradation_level()
        component_health = self.health.component_health.get(feature, 1.0)
        
        # Feature availability based on degradation level
        if degradation_level == DegradationLevel.FULL_SERVICE:
            return component_health > 0.5
        elif degradation_level == DegradationLevel.REDUCED_FEATURES:
            critical_features = {
                ServiceComponent.PRIVACY_ACCOUNTING,
                ServiceComponent.FEDERATED_ROUTING,
                ServiceComponent.SECURITY_VALIDATION
            }
            return feature in critical_features and component_health > 0.3
        elif degradation_level == DegradationLevel.ESSENTIAL_ONLY:
            essential_features = {
                ServiceComponent.PRIVACY_ACCOUNTING,
                ServiceComponent.FEDERATED_ROUTING
            }
            return feature in essential_features and component_health > 0.1
        else:
            return False
    
    async def force_recovery(self, component: ServiceComponent):
        """Force recovery attempt for a specific component."""
        logger.info(f"Forcing recovery for {component.value}")
        
        # Reset component health
        self.health.component_health[component] = 0.8
        
        # Clear related errors
        keys_to_remove = [key for key in self.health.error_counts.keys() 
                         if key.startswith(component.value)]
        for key in keys_to_remove:
            del self.health.error_counts[key]
        
        # Remove active degradations for this component
        self.health.active_degradations = [
            rule for rule in self.health.active_degradations 
            if rule.component != component
        ]
        
        logger.info(f"Recovery completed for {component.value}")


# Global instance
graceful_degradation = GracefulDegradationManager()


# Convenience decorators
def with_quantum_fallback(fallback_result=None):
    """Decorator for quantum planning functions with fallback."""
    return graceful_degradation.with_graceful_degradation(
        ServiceComponent.QUANTUM_PLANNING, fallback_result
    )


def with_privacy_fallback(fallback_result=None):
    """Decorator for privacy accounting functions with fallback."""
    return graceful_degradation.with_graceful_degradation(
        ServiceComponent.PRIVACY_ACCOUNTING, fallback_result
    )


def with_security_fallback(fallback_result=True):
    """Decorator for security validation functions with fallback."""
    return graceful_degradation.with_graceful_degradation(
        ServiceComponent.SECURITY_VALIDATION, fallback_result
    )


def with_monitoring_fallback(fallback_result=None):
    """Decorator for monitoring functions with fallback."""
    return graceful_degradation.with_graceful_degradation(
        ServiceComponent.MONITORING, fallback_result
    )