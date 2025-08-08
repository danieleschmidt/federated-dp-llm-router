"""
Advanced Connection Pooling and Resource Management

Implements intelligent connection pooling, resource optimization,
and adaptive resource allocation for maximum performance.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import deque
import concurrent.futures
import ssl
import aiohttp
import logging


class PoolState(Enum):
    """Connection pool states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SCALING = "scaling"
    DRAINING = "draining"
    CLOSED = "closed"


class ConnectionState(Enum):
    """Individual connection states."""
    IDLE = "idle"
    ACTIVE = "active"
    VALIDATING = "validating"
    FAILED = "failed"
    CLOSING = "closing"


@dataclass
class ConnectionMetrics:
    """Metrics for a connection."""
    created_at: float
    last_used: float
    total_requests: int
    failed_requests: int
    avg_response_time: float
    bytes_sent: int = 0
    bytes_received: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used


@dataclass
class PoolConfig:
    """Configuration for connection pool."""
    min_connections: int = 5
    max_connections: int = 100
    max_idle_time: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    connection_timeout: float = 30.0
    validation_timeout: float = 10.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_health_checks: bool = True
    health_check_interval: float = 60.0
    enable_metrics: bool = True


class Connection:
    """Represents a pooled connection."""
    
    def __init__(self, connection_id: str, session: aiohttp.ClientSession):
        self.connection_id = connection_id
        self.session = session
        self.state = ConnectionState.IDLE
        self.metrics = ConnectionMetrics(
            created_at=time.time(),
            last_used=time.time(),
            total_requests=0,
            failed_requests=0,
            avg_response_time=0.0
        )
        self._lock = asyncio.Lock()
        self._in_use = False
    
    async def acquire(self) -> bool:
        """Acquire connection for use."""
        async with self._lock:
            if self.state != ConnectionState.IDLE:
                return False
            
            self.state = ConnectionState.ACTIVE
            self._in_use = True
            return True
    
    async def release(self):
        """Release connection back to pool."""
        async with self._lock:
            if self._in_use:
                self.state = ConnectionState.IDLE
                self._in_use = False
                self.metrics.last_used = time.time()
    
    async def validate(self) -> bool:
        """Validate connection health."""
        if self.state == ConnectionState.FAILED:
            return False
        
        try:
            self.state = ConnectionState.VALIDATING
            
            # Simple health check - can be customized
            async with self.session.get("http://httpbin.org/status/200", timeout=5) as response:
                healthy = response.status == 200
            
            if healthy:
                self.state = ConnectionState.IDLE
            else:
                self.state = ConnectionState.FAILED
            
            return healthy
            
        except Exception:
            self.state = ConnectionState.FAILED
            return False
    
    async def close(self):
        """Close connection."""
        async with self._lock:
            if self.state != ConnectionState.CLOSING:
                self.state = ConnectionState.CLOSING
                try:
                    await self.session.close()
                except Exception:
                    pass
    
    def record_request(self, response_time: float, success: bool, bytes_sent: int = 0, bytes_received: int = 0):
        """Record request metrics."""
        self.metrics.total_requests += 1
        if not success:
            self.metrics.failed_requests += 1
        
        # Update running average response time
        total_time = self.metrics.avg_response_time * (self.metrics.total_requests - 1)
        self.metrics.avg_response_time = (total_time + response_time) / self.metrics.total_requests
        
        self.metrics.bytes_sent += bytes_sent
        self.metrics.bytes_received += bytes_received
    
    @property
    def is_expired(self) -> bool:
        """Check if connection has expired."""
        return (self.metrics.age_seconds > 3600 or  # 1 hour max lifetime
                self.metrics.idle_time > 300)  # 5 minutes max idle


class ConnectionPool:
    """Advanced connection pool with intelligent management."""
    
    def __init__(self, pool_id: str, config: PoolConfig):
        self.pool_id = pool_id
        self.config = config
        self.state = PoolState.INITIALIZING
        
        # Connection management
        self.connections: Dict[str, Connection] = {}
        self.idle_connections: deque = deque()
        self.active_connections: Dict[str, Connection] = {}
        
        # Metrics and monitoring
        self.pool_metrics = {
            "total_connections": 0,
            "idle_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "pool_hits": 0,
            "pool_misses": 0
        }
        
        # Async management
        self._lock = asyncio.Lock()
        self.health_check_task = None
        self.cleanup_task = None
        
        # Connection factory
        self.connection_factory = self._default_connection_factory
        
        self.logger = logging.getLogger(f"ConnectionPool.{pool_id}")
    
    async def initialize(self):
        """Initialize the connection pool."""
        async with self._lock:
            if self.state != PoolState.INITIALIZING:
                return
            
            # Create minimum connections
            for i in range(self.config.min_connections):
                connection = await self._create_connection()
                if connection:
                    self.connections[connection.connection_id] = connection
                    self.idle_connections.append(connection.connection_id)
            
            self.state = PoolState.ACTIVE
            
            # Start background tasks
            if self.config.enable_health_checks:
                self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._update_metrics()
    
    async def _create_connection(self) -> Optional[Connection]:
        """Create a new connection."""
        try:
            # Create SSL context for secure connections
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Create session with connection limits
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ssl=ssl_context,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": f"FederatedDP-LLM-Pool/{self.pool_id}"}
            )
            
            connection_id = f"{self.pool_id}_{len(self.connections)}_{int(time.time())}"
            connection = Connection(connection_id, session)
            
            # Validate new connection
            if await connection.validate():
                return connection
            else:
                await connection.close()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            return None
    
    def _default_connection_factory(self) -> aiohttp.ClientSession:
        """Default connection factory."""
        return aiohttp.ClientSession()
    
    async def acquire_connection(self, timeout: Optional[float] = None) -> Optional[Connection]:
        """Acquire a connection from the pool."""
        timeout = timeout or self.config.connection_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self._lock:
                if self.state != PoolState.ACTIVE:
                    return None
                
                # Try to get idle connection
                if self.idle_connections:
                    connection_id = self.idle_connections.popleft()
                    connection = self.connections.get(connection_id)
                    
                    if connection and await connection.acquire():
                        self.active_connections[connection_id] = connection
                        self.pool_metrics["pool_hits"] += 1
                        self._update_metrics()
                        return connection
                
                # Create new connection if under limit
                if len(self.connections) < self.config.max_connections:
                    connection = await self._create_connection()
                    if connection:
                        self.connections[connection.connection_id] = connection
                        if await connection.acquire():
                            self.active_connections[connection.connection_id] = connection
                            self.pool_metrics["pool_misses"] += 1
                            self._update_metrics()
                            return connection
                        else:
                            # Failed to acquire, remove from pool
                            del self.connections[connection.connection_id]
                            await connection.close()
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        return None
    
    async def release_connection(self, connection: Connection):
        """Release a connection back to the pool."""
        async with self._lock:
            connection_id = connection.connection_id
            
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
                
                # Check if connection is still healthy
                if (connection.state == ConnectionState.ACTIVE and 
                    not connection.is_expired and
                    connection.metrics.success_rate > 0.8):
                    
                    await connection.release()
                    self.idle_connections.append(connection_id)
                else:
                    # Remove unhealthy/expired connection
                    await self._remove_connection(connection_id)
            
            self._update_metrics()
    
    async def _remove_connection(self, connection_id: str):
        """Remove a connection from the pool."""
        connection = self.connections.get(connection_id)
        if connection:
            await connection.close()
            del self.connections[connection_id]
            
            # Remove from idle queue if present
            try:
                self.idle_connections.remove(connection_id)
            except ValueError:
                pass
            
            # Remove from active connections if present
            self.active_connections.pop(connection_id, None)
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.state == PoolState.ACTIVE:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on idle connections."""
        async with self._lock:
            connections_to_check = list(self.idle_connections)
        
        for connection_id in connections_to_check:
            connection = self.connections.get(connection_id)
            if connection:
                if not await connection.validate():
                    async with self._lock:
                        await self._remove_connection(connection_id)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.state == PoolState.ACTIVE:
            try:
                await self._cleanup_expired_connections()
                await self._ensure_minimum_connections()
                await asyncio.sleep(30)  # Cleanup every 30 seconds
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_connections(self):
        """Clean up expired connections."""
        async with self._lock:
            expired_connections = []
            
            for connection_id, connection in self.connections.items():
                if connection.is_expired and connection_id not in self.active_connections:
                    expired_connections.append(connection_id)
            
            for connection_id in expired_connections:
                await self._remove_connection(connection_id)
    
    async def _ensure_minimum_connections(self):
        """Ensure minimum number of connections."""
        async with self._lock:
            current_count = len(self.connections)
            needed = self.config.min_connections - current_count
            
            for _ in range(needed):
                connection = await self._create_connection()
                if connection:
                    self.connections[connection.connection_id] = connection
                    self.idle_connections.append(connection.connection_id)
    
    def _update_metrics(self):
        """Update pool metrics."""
        self.pool_metrics.update({
            "total_connections": len(self.connections),
            "idle_connections": len(self.idle_connections),
            "active_connections": len(self.active_connections),
            "failed_connections": len([c for c in self.connections.values() 
                                    if c.state == ConnectionState.FAILED])
        })
    
    async def scale_pool(self, target_size: int):
        """Scale pool to target size."""
        async with self._lock:
            if self.state != PoolState.ACTIVE:
                return False
            
            current_size = len(self.connections)
            target_size = max(self.config.min_connections, 
                            min(target_size, self.config.max_connections))
            
            if target_size > current_size:
                # Scale up
                self.state = PoolState.SCALING
                for _ in range(target_size - current_size):
                    connection = await self._create_connection()
                    if connection:
                        self.connections[connection.connection_id] = connection
                        self.idle_connections.append(connection.connection_id)
                self.state = PoolState.ACTIVE
            
            elif target_size < current_size:
                # Scale down - remove idle connections first
                self.state = PoolState.SCALING
                to_remove = current_size - target_size
                removed = 0
                
                while removed < to_remove and self.idle_connections:
                    connection_id = self.idle_connections.pop()
                    await self._remove_connection(connection_id)
                    removed += 1
                
                self.state = PoolState.ACTIVE
            
            self._update_metrics()
            return True
    
    async def drain_pool(self, timeout: float = 60.0):
        """Drain all connections from pool."""
        self.state = PoolState.DRAINING
        start_time = time.time()
        
        # Wait for active connections to finish
        while self.active_connections and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        # Force close remaining connections
        async with self._lock:
            for connection in list(self.connections.values()):
                await connection.close()
            
            self.connections.clear()
            self.idle_connections.clear()
            self.active_connections.clear()
        
        self.state = PoolState.CLOSED
    
    async def close(self):
        """Close the connection pool."""
        if self.health_check_task:
            self.health_check_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        await self.drain_pool()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get detailed pool statistics."""
        # Calculate aggregate metrics
        total_requests = sum(c.metrics.total_requests for c in self.connections.values())
        successful_requests = sum(c.metrics.total_requests - c.metrics.failed_requests 
                                for c in self.connections.values())
        
        if total_requests > 0:
            avg_response_time = sum(c.metrics.avg_response_time * c.metrics.total_requests 
                                  for c in self.connections.values()) / total_requests
            success_rate = successful_requests / total_requests
        else:
            avg_response_time = 0.0
            success_rate = 1.0
        
        connection_details = []
        for conn in self.connections.values():
            connection_details.append({
                "id": conn.connection_id,
                "state": conn.state.value,
                "age_seconds": conn.metrics.age_seconds,
                "idle_time": conn.metrics.idle_time,
                "total_requests": conn.metrics.total_requests,
                "success_rate": conn.metrics.success_rate,
                "avg_response_time": conn.metrics.avg_response_time
            })
        
        return {
            "pool_id": self.pool_id,
            "state": self.state.value,
            "config": {
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "max_idle_time": self.config.max_idle_time,
                "max_lifetime": self.config.max_lifetime
            },
            "metrics": {
                **self.pool_metrics,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time
            },
            "connections": connection_details
        }


class ConnectionPoolManager:
    """Manages multiple connection pools."""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.default_config = PoolConfig()
        self._lock = asyncio.Lock()
    
    async def create_pool(self, pool_id: str, config: Optional[PoolConfig] = None) -> ConnectionPool:
        """Create a new connection pool."""
        async with self._lock:
            if pool_id in self.pools:
                return self.pools[pool_id]
            
            config = config or self.default_config
            pool = ConnectionPool(pool_id, config)
            await pool.initialize()
            
            self.pools[pool_id] = pool
            return pool
    
    async def get_pool(self, pool_id: str) -> Optional[ConnectionPool]:
        """Get an existing pool."""
        return self.pools.get(pool_id)
    
    async def remove_pool(self, pool_id: str):
        """Remove and close a pool."""
        async with self._lock:
            pool = self.pools.get(pool_id)
            if pool:
                await pool.close()
                del self.pools[pool_id]
    
    async def close_all_pools(self):
        """Close all pools."""
        async with self._lock:
            for pool in self.pools.values():
                await pool.close()
            self.pools.clear()
    
    def get_all_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        return {
            pool_id: pool.get_pool_stats()
            for pool_id, pool in self.pools.items()
        }


# Global pool manager
_global_pool_manager: Optional[ConnectionPoolManager] = None


async def get_pool_manager() -> ConnectionPoolManager:
    """Get global pool manager."""
    global _global_pool_manager
    if _global_pool_manager is None:
        _global_pool_manager = ConnectionPoolManager()
    return _global_pool_manager


class PooledHTTPClient:
    """HTTP client that uses connection pooling."""
    
    def __init__(self, pool_id: str = "default"):
        self.pool_id = pool_id
        self.pool: Optional[ConnectionPool] = None
    
    async def _ensure_pool(self):
        """Ensure pool is initialized."""
        if self.pool is None:
            manager = await get_pool_manager()
            self.pool = await manager.create_pool(self.pool_id)
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request using pooled connection."""
        await self._ensure_pool()
        
        connection = await self.pool.acquire_connection()
        if not connection:
            raise RuntimeError("Unable to acquire connection from pool")
        
        try:
            start_time = time.time()
            async with connection.session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                connection.record_request(
                    response_time=response_time,
                    success=200 <= response.status < 400
                )
                return response
        
        except Exception as e:
            connection.record_request(
                response_time=time.time() - start_time,
                success=False
            )
            raise e
        
        finally:
            await self.pool.release_connection(connection)
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request."""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request."""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request."""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request."""
        return await self.request("DELETE", url, **kwargs)