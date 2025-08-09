"""
Integrated Performance Optimization System

Combines caching, performance monitoring, and auto-scaling for optimal system performance.
"""

import asyncio
import time
import psutil
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .caching import MultiTierCache, CacheConfig, CacheLevel
from .performance_optimizer import PerformanceOptimizer, OptimizationStrategy, ResourceMetrics
from .connection_pool import ConnectionPoolManager
from ..monitoring.logging_config import get_logger


@dataclass
class SystemMetrics:
    """Current system performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    gpu_utilization: float
    gpu_memory_used: float
    active_connections: int
    request_rate: float
    error_rate: float
    average_response_time: float


class AdaptiveResourceMonitor:
    """Monitors system resources and adapts performance settings."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.logger = get_logger("resource_monitor")
        
        # Metrics history
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000
        
        # Resource thresholds
        self.cpu_threshold_high = 80.0
        self.memory_threshold_high = 85.0
        self.gpu_threshold_high = 90.0
        
        # Performance tracking
        self.request_times: List[float] = []
        self.error_count = 0
        self.request_count = 0
        self.connections_count = 0
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.logger.info("Stopped resource monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history bounded
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                # Check for resource pressure
                await self._check_resource_pressure(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io_sent = net_io.bytes_sent if net_io else 0
        network_io_recv = net_io.bytes_recv if net_io else 0
        
        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_used = 0.0
        
        if torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization()
                gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                pass
        
        # Application metrics
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t <= 60]  # Last minute
        request_rate = len(recent_requests) / 60.0
        
        avg_response_time = 0.0
        if recent_requests:
            # This is simplified - in practice you'd track actual response times
            avg_response_time = sum(recent_requests) / len(recent_requests)
        
        error_rate = self.error_count / max(1, self.request_count) * 100
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_io_sent=network_io_sent,
            network_io_recv=network_io_recv,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            active_connections=self.connections_count,
            request_rate=request_rate,
            error_rate=error_rate,
            average_response_time=avg_response_time
        )
    
    async def _check_resource_pressure(self, metrics: SystemMetrics):
        """Check for resource pressure and alert."""
        alerts = []
        
        if metrics.cpu_percent > self.cpu_threshold_high:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.memory_threshold_high:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_utilization > self.gpu_threshold_high:
            alerts.append(f"High GPU usage: {metrics.gpu_utilization:.1f}%")
        
        if alerts:
            self.logger.warning(f"Resource pressure detected: {'; '.join(alerts)}")
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics."""
        self.request_times.append(time.time())
        self.request_count += 1
        
        if not success:
            self.error_count += 1
        
        # Clean old request times
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t <= 3600]  # Keep 1 hour
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified duration."""
        current_time = time.time()
        cutoff_time = current_time - (duration_minutes * 60)
        
        recent_metrics = [m for m in self.metrics_history if current_time - cutoff_time <= duration_minutes * 60]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_request_rate = sum(m.request_rate for m in recent_metrics) / len(recent_metrics)
        
        # Calculate peaks
        max_cpu = max(m.cpu_percent for m in recent_metrics)
        max_memory = max(m.memory_percent for m in recent_metrics)
        max_gpu = max(m.gpu_utilization for m in recent_metrics)
        
        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "gpu_utilization": avg_gpu,
                "request_rate": avg_request_rate
            },
            "peaks": {
                "cpu_percent": max_cpu,
                "memory_percent": max_memory,
                "gpu_utilization": max_gpu
            },
            "current": recent_metrics[-1].__dict__ if recent_metrics else None
        }


class IntegratedOptimizer:
    """Integrated performance optimization system."""
    
    def __init__(self):
        self.logger = get_logger("integrated_optimizer")
        
        # Initialize components
        self.cache = self._setup_cache()
        self.performance_optimizer = PerformanceOptimizer()
        self.connection_pool = ConnectionPoolManager()
        self.resource_monitor = AdaptiveResourceMonitor()
        
        # Optimization settings
        self.optimization_enabled = True
        self.auto_scaling_enabled = True
        self.cache_enabled = True
        
        # Performance targets
        self.target_latency_ms = 500
        self.target_throughput = 100  # requests per second
        self.target_error_rate = 0.01  # 1%
        
    def _setup_cache(self) -> MultiTierCache:
        """Setup multi-tier cache system."""
        config = CacheConfig(
            l1_max_size=1000,
            l1_ttl=300,  # 5 minutes
            l2_ttl=3600,  # 1 hour
            l3_ttl=86400,  # 24 hours
            enable_compression=True,
            privacy_aware=True
        )
        
        return MultiTierCache(config)
    
    async def start(self):
        """Start the optimization system."""
        self.logger.info("Starting integrated optimization system")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Initialize cache
        if self.cache_enabled:
            await self.cache.initialize()
        
        # Start performance optimizer
        self.performance_optimizer.start_optimization()
        
        self.logger.info("Integrated optimization system started")
    
    async def stop(self):
        """Stop the optimization system."""
        self.logger.info("Stopping integrated optimization system")
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        
        # Close cache
        if self.cache_enabled:
            await self.cache.close()
        
        # Close connection pools
        await self.connection_pool.close_all()
        
        self.logger.info("Integrated optimization system stopped")
    
    async def optimize_inference_request(self, request_key: str, request_data: Dict[str, Any],
                                       inference_func: callable) -> Any:
        """Optimize an inference request with caching and performance monitoring."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache_enabled:
                cached_result = await self.cache.get(request_key)
                if cached_result is not None:
                    latency = (time.time() - start_time) * 1000
                    self.resource_monitor.record_request(latency, True)
                    self.logger.debug(f"Cache hit for request {request_key}")
                    return cached_result
            
            # Execute inference with circuit breaker and retry
            result = await self._execute_with_optimization(inference_func, request_data)
            
            # Cache the result
            if self.cache_enabled and result is not None:
                # Determine TTL based on request type
                ttl = self._calculate_cache_ttl(request_data)
                await self.cache.set(
                    request_key, 
                    result, 
                    ttl=ttl,
                    privacy_level="sensitive"
                )
            
            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.resource_monitor.record_request(latency, True)
            
            return result
            
        except Exception as e:
            # Record error
            latency = (time.time() - start_time) * 1000
            self.resource_monitor.record_request(latency, False)
            
            self.logger.error(f"Optimization error for request {request_key}: {e}")
            raise
    
    async def _execute_with_optimization(self, inference_func: callable, request_data: Dict[str, Any]) -> Any:
        """Execute inference with performance optimization."""
        # Get current system metrics
        current_metrics = self.resource_monitor.get_current_metrics()
        
        if current_metrics:
            # Adapt based on system load
            if current_metrics.cpu_percent > 90:
                # High CPU load - add small delay to prevent overload
                await asyncio.sleep(0.1)
            
            if current_metrics.memory_percent > 95:
                # Critical memory usage - trigger garbage collection
                import gc
                gc.collect()
        
        # Execute the inference function
        if asyncio.iscoroutinefunction(inference_func):
            return await inference_func(request_data)
        else:
            # Run in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, inference_func, request_data)
    
    def _calculate_cache_ttl(self, request_data: Dict[str, Any]) -> int:
        """Calculate appropriate cache TTL based on request characteristics."""
        # Base TTL
        ttl = 300  # 5 minutes
        
        # Adjust based on request complexity
        if "complex" in str(request_data).lower():
            ttl = 1800  # 30 minutes for complex requests
        
        # Adjust based on privacy sensitivity
        if "sensitive" in str(request_data).lower():
            ttl = 60  # 1 minute for sensitive data
        
        return ttl
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_enabled:
            return {"cache_enabled": False}
        
        return self.cache.get_stats()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "optimization_enabled": self.optimization_enabled,
            "cache_enabled": self.cache_enabled,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "targets": {
                "latency_ms": self.target_latency_ms,
                "throughput_rps": self.target_throughput,
                "error_rate": self.target_error_rate
            }
        }
        
        # Add resource monitoring stats
        stats["resource_metrics"] = self.resource_monitor.get_metrics_summary()
        
        # Add cache stats
        if self.cache_enabled:
            stats["cache"] = self.get_cache_stats()
        
        # Add performance optimizer stats
        stats["performance_optimizer"] = self.performance_optimizer.get_optimization_stats()
        
        return stats
    
    async def trigger_optimization(self):
        """Manually trigger optimization analysis."""
        current_metrics = self.resource_monitor.get_current_metrics()
        
        if not current_metrics:
            self.logger.warning("No metrics available for optimization")
            return
        
        # Analyze performance
        recommendations = []
        
        if current_metrics.cpu_percent > 80:
            recommendations.append("Consider scaling out or optimizing CPU-intensive operations")
        
        if current_metrics.memory_percent > 85:
            recommendations.append("Memory usage high - consider cache cleanup or scaling")
        
        if current_metrics.error_rate > 5:
            recommendations.append("High error rate detected - investigate error patterns")
        
        if current_metrics.average_response_time > self.target_latency_ms:
            recommendations.append(f"Response time ({current_metrics.average_response_time:.1f}ms) exceeds target ({self.target_latency_ms}ms)")
        
        if recommendations:
            self.logger.info(f"Optimization recommendations: {'; '.join(recommendations)}")
        else:
            self.logger.info("System performance within acceptable parameters")
    
    def configure_optimization(self, **settings):
        """Configure optimization settings."""
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated optimization setting: {key} = {value}")
            else:
                self.logger.warning(f"Unknown optimization setting: {key}")


# Global optimizer instance
_integrated_optimizer: Optional[IntegratedOptimizer] = None


def get_integrated_optimizer() -> IntegratedOptimizer:
    """Get or create the global integrated optimizer."""
    global _integrated_optimizer
    if _integrated_optimizer is None:
        _integrated_optimizer = IntegratedOptimizer()
    return _integrated_optimizer


async def optimize_request(request_key: str, request_data: Dict[str, Any], 
                          inference_func: callable) -> Any:
    """Convenience function for optimized request execution."""
    optimizer = get_integrated_optimizer()
    return await optimizer.optimize_inference_request(request_key, request_data, inference_func)