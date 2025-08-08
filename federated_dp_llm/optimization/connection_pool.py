"""
Connection Pool Management

Implements intelligent connection pooling with adaptive sizing, health monitoring,
and resource optimization for federated node communications.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
# Conditional import for HTTP client
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False
import logging
from collections import defaultdict, deque
import weakref


class ConnectionState(Enum):
    """Connection states."""
    IDLE = "idle"
    ACTIVE = "active"
    STALE = "stale"
    FAILED = "failed"
    CLOSING = "closing"


@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    connection_id: str
    endpoint: str
    created_at: float
    last_used: float
    use_count: int
    state: ConnectionState
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    
    def get_avg_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request metrics."""
        self.last_used = time.time()
        self.use_count += 1
        
        if success:
            self.response_times.append(response_time)
        else:
            self.error_count += 1


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 2
    max_size: int = 20
    max_idle_time: float = 300  # 5 minutes
    max_lifetime: float = 3600  # 1 hour
    health_check_interval: float = 60  # 1 minute
    request_timeout: float = 30
    connect_timeout: float = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_metrics: bool = True


class ConnectionPool:
    """Manages connections for a specific endpoint."""
    
    def __init__(self, endpoint: str, config: PoolConfig):
        self.endpoint = endpoint
        self.config = config
        self.connections: Dict[str, httpx.AsyncClient] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.idle_connections: Set[str] = set()
        self.active_connections: Set[str] = set()
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._connection_semaphore = asyncio.Semaphore(config.max_size)
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._closed = False
        
        self.logger = logging.getLogger(f"{__name__}.{endpoint}")
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def get_connection(self) -> Tuple[httpx.AsyncClient, str]:
        """Get a connection from the pool."""
        async with self._connection_semaphore:
            async with self._lock:
                # Try to get idle connection
                if self.idle_connections:
                    conn_id = self.idle_connections.pop()
                    self.active_connections.add(conn_id)
                    connection = self.connections[conn_id]
                    
                    # Update connection info
                    self.connection_info[conn_id].state = ConnectionState.ACTIVE
                    
                    return connection, conn_id
                
                # Create new connection if under limit
                if len(self.connections) < self.config.max_size:
                    conn_id = await self._create_connection()
                    if conn_id:
                        return self.connections[conn_id], conn_id
                
                # Wait for connection to become available
                # This is a simplified approach - in production would use proper waiting
                raise RuntimeError("No connections available")
    
    async def return_connection(self, connection_id: str, success: bool = True, 
                               response_time: float = 0.0):
        """Return a connection to the pool."""
        async with self._lock:
            if connection_id in self.active_connections:
                self.active_connections.remove(connection_id)
                
                # Update connection info
                if connection_id in self.connection_info:
                    info = self.connection_info[connection_id]
                    info.record_request(response_time, success)
                    info.state = ConnectionState.IDLE
                
                # Return to idle pool if connection is healthy
                if success and connection_id in self.connections:
                    self.idle_connections.add(connection_id)
                else:
                    # Remove failed connection
                    await self._remove_connection(connection_id)
    
    async def _create_connection(self) -> Optional[str]:
        """Create a new connection."""
        try:
            connection_id = f"conn_{int(time.time() * 1000000)}"
            
            # Create HTTP client with optimized settings
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.request_timeout,
                    write=self.config.request_timeout,
                    pool=self.config.request_timeout
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=300
                ),
                verify=False,  # For testing - should verify in production
                http2=True  # Enable HTTP/2
            )
            
            # Test connection
            try:
                response = await client.get(f"{self.endpoint}/health", timeout=5.0)
                response.raise_for_status()
            except Exception as e:
                await client.aclose()
                self.logger.warning(f"Connection test failed for {self.endpoint}: {e}")
                return None
            
            # Store connection
            self.connections[connection_id] = client
            self.connection_info[connection_id] = ConnectionInfo(
                connection_id=connection_id,
                endpoint=self.endpoint,
                created_at=time.time(),
                last_used=time.time(),
                use_count=0,
                state=ConnectionState.ACTIVE
            )
            self.active_connections.add(connection_id)
            
            self.logger.debug(f"Created connection {connection_id} for {self.endpoint}")
            return connection_id
            
        except Exception as e:
            self.logger.error(f"Failed to create connection for {self.endpoint}: {e}")
            return None
    
    async def _remove_connection(self, connection_id: str):
        """Remove a connection from the pool."""
        if connection_id in self.connections:
            client = self.connections[connection_id]
            await client.aclose()
            del self.connections[connection_id]
        
        if connection_id in self.connection_info:
            del self.connection_info[connection_id]
        
        self.idle_connections.discard(connection_id)
        self.active_connections.discard(connection_id)
        
        self.logger.debug(f"Removed connection {connection_id}")
    
    async def _health_check_loop(self):
        """Health check loop for connection maintenance."""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._cleanup_stale_connections()
                await self._ensure_minimum_connections()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    async def _cleanup_stale_connections(self):
        """Clean up stale and expired connections."""
        current_time = time.time()
        to_remove = []
        
        async with self._lock:
            for conn_id, info in self.connection_info.items():
                # Remove connections that exceeded lifetime
                if current_time - info.created_at > self.config.max_lifetime:
                    to_remove.append(conn_id)
                    continue
                
                # Remove idle connections that exceeded idle time
                if (conn_id in self.idle_connections and 
                    current_time - info.last_used > self.config.max_idle_time):
                    to_remove.append(conn_id)
                    continue
                
                # Remove failed connections
                if info.state == ConnectionState.FAILED:
                    to_remove.append(conn_id)
        
        # Remove stale connections
        for conn_id in to_remove:
            await self._remove_connection(conn_id)
    
    async def _ensure_minimum_connections(self):
        """Ensure minimum number of connections."""
        async with self._lock:
            current_count = len(self.connections)
            
            if current_count < self.config.min_size:
                needed = self.config.min_size - current_count
                
                for _ in range(needed):
                    conn_id = await self._create_connection()
                    if conn_id:
                        # Move to idle immediately
                        self.active_connections.remove(conn_id)
                        self.idle_connections.add(conn_id)
                        self.connection_info[conn_id].state = ConnectionState.IDLE
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_connections = len(self.connections)
        idle_count = len(self.idle_connections)
        active_count = len(self.active_connections)
        
        # Calculate average response times
        avg_response_times = []
        total_requests = 0
        total_errors = 0
        
        for info in self.connection_info.values():
            avg_response_times.append(info.get_avg_response_time())
            total_requests += info.use_count
            total_errors += info.error_count
        
        avg_response_time = sum(avg_response_times) / max(len(avg_response_times), 1)
        error_rate = total_errors / max(total_requests, 1)
        
        return {
            "endpoint": self.endpoint,
            "total_connections": total_connections,
            "idle_connections": idle_count,
            "active_connections": active_count,
            "avg_response_time": avg_response_time,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "pool_utilization": active_count / max(total_connections, 1)
        }
    
    async def close(self):
        """Close all connections in the pool."""
        self._closed = True
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for client in self.connections.values():
                await client.aclose()
            
            self.connections.clear()
            self.connection_info.clear()
            self.idle_connections.clear()
            self.active_connections.clear()


class ConnectionPoolManager:
    """Manages connection pools for multiple endpoints."""
    
    def __init__(self, default_config: PoolConfig = None):
        self.default_config = default_config or PoolConfig()
        self.pools: Dict[str, ConnectionPool] = {}
        self.pool_configs: Dict[str, PoolConfig] = {}
        self._lock = asyncio.Lock()
        
        # Adaptive sizing
        self.pool_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.auto_scaling_enabled = True
        self.auto_scaling_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Start auto-scaling task
        if self.auto_scaling_enabled:
            self.auto_scaling_task = asyncio.create_task(self._auto_scaling_loop())
    
    async def get_pool(self, endpoint: str, config: PoolConfig = None) -> ConnectionPool:
        """Get or create connection pool for endpoint."""
        async with self._lock:
            if endpoint not in self.pools:
                pool_config = config or self.default_config
                self.pools[endpoint] = ConnectionPool(endpoint, pool_config)
                self.pool_configs[endpoint] = pool_config
                
                self.logger.info(f"Created connection pool for {endpoint}")
            
            return self.pools[endpoint]
    
    async def execute_request(self, endpoint: str, method: str, path: str = "/", 
                             **kwargs) -> httpx.Response:
        """Execute HTTP request using connection pool."""
        pool = await self.get_pool(endpoint)
        
        for attempt in range(self.default_config.retry_attempts):
            try:
                start_time = time.time()
                
                # Get connection from pool
                client, conn_id = await pool.get_connection()
                
                try:
                    # Make request
                    url = f"{endpoint}{path}"
                    response = await client.request(method, url, **kwargs)
                    
                    # Record success
                    response_time = time.time() - start_time
                    await pool.return_connection(conn_id, True, response_time)
                    
                    # Record metrics for auto-scaling
                    self._record_metrics(endpoint, response_time, True)
                    
                    return response
                    
                except Exception as e:
                    # Record failure
                    response_time = time.time() - start_time
                    await pool.return_connection(conn_id, False, response_time)
                    
                    self._record_metrics(endpoint, response_time, False)
                    
                    if attempt < self.default_config.retry_attempts - 1:
                        await asyncio.sleep(self.default_config.retry_delay * (2 ** attempt))
                        continue
                    else:
                        raise
                        
            except Exception as e:
                if attempt == self.default_config.retry_attempts - 1:
                    self.logger.error(f"Request failed after {self.default_config.retry_attempts} attempts: {e}")
                    raise
                
                await asyncio.sleep(self.default_config.retry_delay * (2 ** attempt))
    
    def _record_metrics(self, endpoint: str, response_time: float, success: bool):
        """Record metrics for auto-scaling decisions."""
        self.pool_metrics[endpoint].append({
            "timestamp": time.time(),
            "response_time": response_time,
            "success": success
        })
    
    async def _auto_scaling_loop(self):
        """Auto-scaling loop for adaptive pool sizing."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._adjust_pool_sizes()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
    
    async def _adjust_pool_sizes(self):
        """Adjust pool sizes based on metrics."""
        async with self._lock:
            for endpoint, pool in self.pools.items():
                metrics = self.pool_metrics[endpoint]
                
                if len(metrics) < 10:  # Need enough data
                    continue
                
                # Calculate recent performance
                recent_metrics = list(metrics)[-20:]  # Last 20 requests
                avg_response_time = sum(m["response_time"] for m in recent_metrics) / len(recent_metrics)
                success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
                
                # Get current pool stats
                stats = pool.get_stats()
                current_size = stats["total_connections"]
                utilization = stats["pool_utilization"]
                
                # Scaling decision logic
                config = self.pool_configs[endpoint]
                
                # Scale up if high utilization and good performance
                if (utilization > 0.8 and success_rate > 0.95 and 
                    current_size < config.max_size):
                    
                    # Increase pool size
                    new_min_size = min(config.min_size + 1, config.max_size)
                    config.min_size = new_min_size
                    
                    self.logger.info(f"Scaling up pool for {endpoint} to {new_min_size}")
                
                # Scale down if low utilization
                elif (utilization < 0.3 and current_size > config.min_size and 
                      config.min_size > 2):
                    
                    # Decrease pool size
                    new_min_size = max(config.min_size - 1, 2)
                    config.min_size = new_min_size
                    
                    self.logger.info(f"Scaling down pool for {endpoint} to {new_min_size}")
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        stats = {
            "total_pools": len(self.pools),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "pools": {}
        }
        
        for endpoint, pool in self.pools.items():
            stats["pools"][endpoint] = pool.get_stats()
        
        return stats
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all pools."""
        results = {}
        
        for endpoint, pool in self.pools.items():
            try:
                # Try to get a connection and make a simple request
                client, conn_id = await pool.get_connection()
                
                try:
                    response = await client.get(f"{endpoint}/health", timeout=5.0)
                    healthy = response.status_code == 200
                    await pool.return_connection(conn_id, healthy)
                    results[endpoint] = healthy
                    
                except Exception:
                    await pool.return_connection(conn_id, False)
                    results[endpoint] = False
                    
            except Exception:
                results[endpoint] = False
        
        return results
    
    async def remove_pool(self, endpoint: str):
        """Remove and close a connection pool."""
        async with self._lock:
            if endpoint in self.pools:
                pool = self.pools[endpoint]
                await pool.close()
                
                del self.pools[endpoint]
                del self.pool_configs[endpoint]
                
                if endpoint in self.pool_metrics:
                    del self.pool_metrics[endpoint]
                
                self.logger.info(f"Removed connection pool for {endpoint}")
    
    async def close_all(self):
        """Close all connection pools."""
        if self.auto_scaling_task:
            self.auto_scaling_task.cancel()
            try:
                await self.auto_scaling_task
            except asyncio.CancelledError:
                pass
        
        # Close all pools
        for pool in self.pools.values():
            await pool.close()
        
        self.pools.clear()
        self.pool_configs.clear()
        self.pool_metrics.clear()


class HealthMonitor:
    """Monitors health of endpoints and manages connection pools."""
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        self.pool_manager = pool_manager
        self.endpoint_health: Dict[str, bool] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.unhealthy_threshold = 3  # consecutive failures
        self.recovery_threshold = 2   # consecutive successes
        
        self.logger = logging.getLogger(__name__)
    
    async def check_endpoint_health(self, endpoint: str) -> bool:
        """Check health of a specific endpoint."""
        try:
            # Use connection pool for health check
            response = await self.pool_manager.execute_request(
                endpoint, "GET", "/health", timeout=10.0
            )
            
            healthy = response.status_code == 200
            self._record_health_check(endpoint, healthy)
            
            return healthy
            
        except Exception as e:
            self.logger.warning(f"Health check failed for {endpoint}: {e}")
            self._record_health_check(endpoint, False)
            return False
    
    def _record_health_check(self, endpoint: str, healthy: bool):
        """Record health check result."""
        self.health_history[endpoint].append({
            "timestamp": time.time(),
            "healthy": healthy
        })
        
        # Update current health status based on recent history
        recent_checks = list(self.health_history[endpoint])[-5:]  # Last 5 checks
        
        if len(recent_checks) >= self.unhealthy_threshold:
            # Mark as unhealthy if recent failures exceed threshold
            recent_failures = sum(1 for check in recent_checks if not check["healthy"])
            
            if recent_failures >= self.unhealthy_threshold:
                if self.endpoint_health.get(endpoint, True):
                    self.logger.warning(f"Marking endpoint {endpoint} as unhealthy")
                self.endpoint_health[endpoint] = False
            
            # Mark as healthy if recent successes exceed threshold
            elif recent_failures == 0 and len(recent_checks) >= self.recovery_threshold:
                if not self.endpoint_health.get(endpoint, True):
                    self.logger.info(f"Marking endpoint {endpoint} as healthy")
                self.endpoint_health[endpoint] = True
    
    def is_endpoint_healthy(self, endpoint: str) -> bool:
        """Check if endpoint is currently considered healthy."""
        return self.endpoint_health.get(endpoint, True)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all monitored endpoints."""
        summary = {
            "total_endpoints": len(self.endpoint_health),
            "healthy_endpoints": sum(1 for h in self.endpoint_health.values() if h),
            "unhealthy_endpoints": sum(1 for h in self.endpoint_health.values() if not h),
            "endpoints": {}
        }
        
        for endpoint, healthy in self.endpoint_health.items():
            history = list(self.health_history[endpoint])
            recent_history = history[-10:] if history else []
            
            summary["endpoints"][endpoint] = {
                "healthy": healthy,
                "total_checks": len(history),
                "recent_success_rate": sum(1 for h in recent_history if h["healthy"]) / max(len(recent_history), 1)
            }
        
        return summary