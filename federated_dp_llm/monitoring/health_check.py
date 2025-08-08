"""
Health Check System

Implements comprehensive health checks for federated nodes, system components,
and overall system health monitoring with circuit breakers.
"""

import asyncio
import time
import httpx
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading

# Conditional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    NODE = "node"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    response_time: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ComponentConfig:
    """Configuration for a component health check."""
    component_id: str
    component_type: ComponentType
    check_function: Callable
    check_interval: int = 30  # seconds
    timeout: int = 10  # seconds
    retry_count: int = 3
    enabled: bool = True


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
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
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold
        }


class HealthChecker:
    """Main health checking system."""
    
    def __init__(self):
        self.components: Dict[str, ComponentConfig] = {}
        self.health_results: Dict[str, HealthCheckResult] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Thread-safe locks
        self._results_lock = threading.Lock()
        self._components_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        
        # Register built-in health checks
        self._register_builtin_checks()
    
    def _register_builtin_checks(self):
        """Register built-in system health checks."""
        # System resource checks
        self.register_component(ComponentConfig(
            component_id="system_memory",
            component_type=ComponentType.STORAGE,
            check_function=self._check_memory,
            check_interval=60
        ))
        
        self.register_component(ComponentConfig(
            component_id="system_cpu",
            component_type=ComponentType.STORAGE,
            check_function=self._check_cpu,
            check_interval=60
        ))
        
        self.register_component(ComponentConfig(
            component_id="system_disk",
            component_type=ComponentType.STORAGE,
            check_function=self._check_disk,
            check_interval=120
        ))
    
    def register_component(self, config: ComponentConfig):
        """Register a component for health checking."""
        with self._components_lock:
            self.components[config.component_id] = config
            
            # Create circuit breaker
            self.circuit_breakers[config.component_id] = CircuitBreaker()
        
        self.logger.info(f"Registered health check for component: {config.component_id}")
    
    def register_node_check(self, node_id: str, endpoint: str, 
                           check_interval: int = 30, timeout: int = 10):
        """Register health check for a federated node."""
        async def check_node():
            return await self._check_node_health(endpoint, timeout)
        
        config = ComponentConfig(
            component_id=node_id,
            component_type=ComponentType.NODE,
            check_function=check_node,
            check_interval=check_interval,
            timeout=timeout
        )
        
        self.register_component(config)
    
    def start_monitoring(self):
        """Start health monitoring for all components."""
        if self.running:
            return
        
        self.running = True
        
        # Start health check tasks for each component
        for component_id, config in self.components.items():
            if config.enabled:
                task = asyncio.create_task(self._health_check_loop(component_id))
                self.check_tasks[component_id] = task
        
        self.logger.info(f"Started health monitoring for {len(self.check_tasks)} components")
    
    async def stop_monitoring(self):
        """Stop health monitoring for all components."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all tasks
        for task in self.check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.check_tasks:
            await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        
        self.check_tasks.clear()
        self.logger.info("Stopped health monitoring")
    
    async def _health_check_loop(self, component_id: str):
        """Health check loop for a component."""
        config = self.components[component_id]
        circuit_breaker = self.circuit_breakers[component_id]
        
        while self.running:
            try:
                start_time = time.time()
                
                # Execute health check with circuit breaker
                if asyncio.iscoroutinefunction(config.check_function):
                    result = await circuit_breaker.call(config.check_function)
                else:
                    result = circuit_breaker.call(config.check_function)
                
                response_time = time.time() - start_time
                
                # Create health check result
                health_result = HealthCheckResult(
                    component_id=component_id,
                    component_type=config.component_type,
                    status=HealthStatus.HEALTHY,
                    response_time=response_time,
                    timestamp=time.time(),
                    details=result if isinstance(result, dict) else {"result": result}
                )
                
                with self._results_lock:
                    self.health_results[component_id] = health_result
                
                self.logger.debug(f"Health check passed for {component_id}")
                
            except Exception as e:
                # Health check failed
                health_result = HealthCheckResult(
                    component_id=component_id,
                    component_type=config.component_type,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    timestamp=time.time(),
                    error_message=str(e)
                )
                
                with self._results_lock:
                    self.health_results[component_id] = health_result
                
                self.logger.warning(f"Health check failed for {component_id}: {e}")
            
            # Wait for next check
            await asyncio.sleep(config.check_interval)
    
    async def _check_node_health(self, endpoint: str, timeout: int) -> Dict[str, Any]:
        """Check health of a federated node."""
        try:
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                response = await client.get(f"{endpoint}/health")
                response.raise_for_status()
                
                health_data = response.json()
                
                return {
                    "status": health_data.get("status", "unknown"),
                    "response_time": response.elapsed.total_seconds(),
                    "version": health_data.get("version"),
                    "node_info": health_data.get("node_info", {}),
                    "resource_usage": health_data.get("resource_usage", {})
                }
                
        except httpx.TimeoutException:
            raise Exception("Health check timeout")
        except httpx.HTTPError as e:
            raise Exception(f"HTTP error: {e}")
        except Exception as e:
            raise Exception(f"Health check failed: {e}")
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        if not PSUTIL_AVAILABLE:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "psutil not available - cannot check memory usage",
                "details": {}
            }
        
        memory = psutil.virtual_memory()
        
        # Determine status based on usage
        if memory.percent > 90:
            status = HealthStatus.UNHEALTHY
        elif memory.percent > 80:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        result = {
            "memory_percent": memory.percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_used": memory.used,
            "status": status.value
        }
        
        if status != HealthStatus.HEALTHY:
            raise Exception(f"Memory usage too high: {memory.percent}%")
        
        return result
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check system CPU usage."""
        if not PSUTIL_AVAILABLE:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "psutil not available - cannot check CPU usage",
                "details": {}
            }
        
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Determine status based on usage
        if cpu_percent > 95:
            status = HealthStatus.UNHEALTHY
        elif cpu_percent > 85:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        result = {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            "status": status.value
        }
        
        if status != HealthStatus.HEALTHY:
            raise Exception(f"CPU usage too high: {cpu_percent}%")
        
        return result
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check system disk usage."""
        if not PSUTIL_AVAILABLE:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "psutil not available - cannot check disk usage",
                "details": {}
            }
        
        disk = psutil.disk_usage('/')
        
        # Determine status based on usage
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 95:
            status = HealthStatus.UNHEALTHY
        elif usage_percent > 85:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        result = {
            "disk_percent": usage_percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_free": disk.free,
            "status": status.value
        }
        
        if status != HealthStatus.HEALTHY:
            raise Exception(f"Disk usage too high: {usage_percent:.1f}%")
        
        return result
    
    async def check_component(self, component_id: str) -> Optional[HealthCheckResult]:
        """Perform immediate health check for a specific component."""
        config = self.components.get(component_id)
        if not config:
            return None
        
        circuit_breaker = self.circuit_breakers[component_id]
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(config.check_function):
                result = await circuit_breaker.call(config.check_function)
            else:
                result = circuit_breaker.call(config.check_function)
            
            response_time = time.time() - start_time
            
            health_result = HealthCheckResult(
                component_id=component_id,
                component_type=config.component_type,
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                timestamp=time.time(),
                details=result if isinstance(result, dict) else {"result": result}
            )
            
            with self._results_lock:
                self.health_results[component_id] = health_result
            
            return health_result
            
        except Exception as e:
            health_result = HealthCheckResult(
                component_id=component_id,
                component_type=config.component_type,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                timestamp=time.time(),
                error_message=str(e)
            )
            
            with self._results_lock:
                self.health_results[component_id] = health_result
            
            return health_result
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self._results_lock:
            results = self.health_results.copy()
        
        if not results:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "total_components": 0,
                "healthy_components": 0,
                "unhealthy_components": 0,
                "degraded_components": 0
            }
        
        # Count components by status
        status_counts = {
            HealthStatus.HEALTHY.value: 0,
            HealthStatus.DEGRADED.value: 0,
            HealthStatus.UNHEALTHY.value: 0,
            HealthStatus.UNKNOWN.value: 0
        }
        
        for result in results.values():
            status_counts[result.status.value] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY.value] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED.value] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY.value] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Component details
        component_details = {}
        for component_id, result in results.items():
            component_details[component_id] = {
                "status": result.status.value,
                "type": result.component_type.value,
                "response_time": result.response_time,
                "last_check": result.timestamp,
                "error": result.error_message,
                "circuit_breaker": self.circuit_breakers[component_id].get_state()
            }
        
        return {
            "overall_status": overall_status.value,
            "total_components": len(results),
            "healthy_components": status_counts[HealthStatus.HEALTHY.value],
            "degraded_components": status_counts[HealthStatus.DEGRADED.value],
            "unhealthy_components": status_counts[HealthStatus.UNHEALTHY.value],
            "unknown_components": status_counts[HealthStatus.UNKNOWN.value],
            "components": component_details,
            "timestamp": time.time()
        }
    
    def get_component_health(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get health information for a specific component."""
        with self._results_lock:
            result = self.health_results.get(component_id)
        
        if not result:
            return None
        
        return {
            "component_id": result.component_id,
            "type": result.component_type.value,
            "status": result.status.value,
            "response_time": result.response_time,
            "last_check": result.timestamp,
            "details": result.details,
            "error": result.error_message,
            "circuit_breaker": self.circuit_breakers[component_id].get_state()
        }
    
    def enable_component(self, component_id: str):
        """Enable health checking for a component."""
        if component_id in self.components:
            self.components[component_id].enabled = True
            
            # Start task if monitoring is running
            if self.running and component_id not in self.check_tasks:
                task = asyncio.create_task(self._health_check_loop(component_id))
                self.check_tasks[component_id] = task
            
            self.logger.info(f"Enabled health checking for {component_id}")
    
    def disable_component(self, component_id: str):
        """Disable health checking for a component."""
        if component_id in self.components:
            self.components[component_id].enabled = False
            
            # Cancel task if running
            if component_id in self.check_tasks:
                self.check_tasks[component_id].cancel()
                del self.check_tasks[component_id]
            
            self.logger.info(f"Disabled health checking for {component_id}")
    
    def remove_component(self, component_id: str):
        """Remove a component from health checking."""
        if component_id in self.components:
            # Disable first
            self.disable_component(component_id)
            
            # Remove from all collections
            with self._components_lock:
                del self.components[component_id]
            
            with self._results_lock:
                self.health_results.pop(component_id, None)
            
            self.circuit_breakers.pop(component_id, None)
            
            self.logger.info(f"Removed health check for {component_id}")


class HealthCheckMiddleware:
    """FastAPI middleware for health check endpoints."""
    
    def __init__(self, health_checker: HealthChecker):
        self.health_checker = health_checker
    
    async def __call__(self, request, call_next):
        """Process health check requests."""
        if request.url.path == "/health":
            return await self._handle_health_check(request)
        elif request.url.path == "/health/ready":
            return await self._handle_readiness_check(request)
        elif request.url.path == "/health/live":
            return await self._handle_liveness_check(request)
        else:
            return await call_next(request)
    
    async def _handle_health_check(self, request):
        """Handle comprehensive health check."""
        from fastapi.responses import JSONResponse
        
        health_summary = self.health_checker.get_health_summary()
        
        status_code = 200
        if health_summary["overall_status"] == HealthStatus.UNHEALTHY.value:
            status_code = 503
        elif health_summary["overall_status"] == HealthStatus.DEGRADED.value:
            status_code = 200  # Still serving but degraded
        
        return JSONResponse(content=health_summary, status_code=status_code)
    
    async def _handle_readiness_check(self, request):
        """Handle readiness probe (ready to serve traffic)."""
        from fastapi.responses import JSONResponse
        
        health_summary = self.health_checker.get_health_summary()
        
        # Ready if not unhealthy
        ready = health_summary["overall_status"] != HealthStatus.UNHEALTHY.value
        
        return JSONResponse(
            content={"ready": ready, "timestamp": time.time()},
            status_code=200 if ready else 503
        )
    
    async def _handle_liveness_check(self, request):
        """Handle liveness probe (process is running)."""
        from fastapi.responses import JSONResponse
        
        # Simple liveness check - process is alive
        return JSONResponse(
            content={"alive": True, "timestamp": time.time()},
            status_code=200
        )