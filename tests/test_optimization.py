"""
Performance optimization and caching tests.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

from federated_dp_llm.optimization.performance_optimizer import (
    PerformanceOptimizer, LoadPredictor, ResourceOptimizer, AdaptiveScaler,
    ResourceMetrics, PerformanceTarget, OptimizationStrategy
)
from federated_dp_llm.optimization.caching import (
    CacheManager, MemoryCache, CacheEntry, CacheLevel
)
from federated_dp_llm.optimization.connection_pool import (
    ConnectionPoolManager, ConnectionPool, PoolConfig
)


@pytest.mark.unit
class TestLoadPredictor:
    """Test load prediction functionality."""
    
    def test_predictor_initialization(self):
        """Test load predictor initialization."""
        predictor = LoadPredictor(history_size=100)
        
        assert predictor.history_size == 100
        assert len(predictor.metrics_history) == 0
        assert predictor.short_term_window == 10
    
    def test_metrics_addition(self):
        """Test adding metrics to predictor."""
        predictor = LoadPredictor()
        
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=0.6,
            memory_usage=0.7,
            gpu_usage=0.5,
            network_io=0.3,
            storage_io=0.2,
            active_requests=25,
            queue_length=5
        )
        
        predictor.add_metrics(metrics)
        
        assert len(predictor.metrics_history) == 1
        assert predictor.metrics_history[0] == metrics
    
    def test_load_prediction(self):
        """Test load prediction with sample data."""
        predictor = LoadPredictor()
        
        # Add sample metrics data
        base_time = time.time()
        for i in range(20):
            metrics = ResourceMetrics(
                timestamp=base_time + i * 60,
                cpu_usage=0.5 + 0.1 * np.sin(i * 0.1),
                memory_usage=0.6 + 0.05 * np.cos(i * 0.1),
                gpu_usage=0.4,
                network_io=0.3,
                storage_io=0.2,
                active_requests=20 + i,
                queue_length=max(0, i - 15)
            )
            predictor.add_metrics(metrics)
        
        # Predict load
        prediction = predictor.predict_load(horizon_minutes=30)
        
        assert "predicted_requests" in prediction
        assert "predicted_cpu" in prediction
        assert "confidence" in prediction
        assert prediction["confidence"] > 0
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        predictor = LoadPredictor()
        
        # Add normal metrics
        for i in range(25):
            metrics = ResourceMetrics(
                timestamp=time.time() + i * 60,
                cpu_usage=0.5,
                memory_usage=0.6,
                gpu_usage=0.4,
                network_io=0.3,
                storage_io=0.2,
                active_requests=20,
                queue_length=2
            )
            predictor.add_metrics(metrics)
        
        # Add anomalous metrics
        anomalous_metrics = ResourceMetrics(
            timestamp=time.time() + 30 * 60,
            cpu_usage=0.95,  # Very high CPU
            memory_usage=0.6,
            gpu_usage=0.4,
            network_io=0.3,
            storage_io=0.2,
            active_requests=20,
            queue_length=2
        )
        
        anomalies = predictor.detect_anomalies(anomalous_metrics)
        
        assert "cpu_anomaly" in anomalies


@pytest.mark.unit
class TestResourceOptimizer:
    """Test resource optimization functionality."""
    
    def test_optimizer_initialization(self):
        """Test resource optimizer initialization."""
        optimizer = ResourceOptimizer(OptimizationStrategy.BALANCED)
        
        assert optimizer.strategy == OptimizationStrategy.BALANCED
        assert len(optimizer.optimization_history) == 0
    
    def test_cpu_optimization(self):
        """Test CPU optimization actions."""
        optimizer = ResourceOptimizer(OptimizationStrategy.THROUGHPUT_FOCUSED)
        targets = PerformanceTarget(max_cpu_usage=0.7)
        
        # High CPU usage metrics
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=0.9,  # Above threshold
            memory_usage=0.5,
            gpu_usage=0.4,
            network_io=0.3,
            storage_io=0.2,
            active_requests=60,
            queue_length=10
        )
        
        actions = optimizer.analyze_resource_usage(metrics, targets)
        
        assert len(actions) > 0
        assert any(action.action_type == "scale_out" for action in actions)
    
    def test_memory_optimization(self):
        """Test memory optimization actions."""
        optimizer = ResourceOptimizer()
        targets = PerformanceTarget(max_memory_usage=0.8)
        
        # High memory usage metrics
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=0.5,
            memory_usage=0.9,  # Above threshold
            gpu_usage=0.4,
            network_io=0.3,
            storage_io=0.2,
            active_requests=30,
            queue_length=5
        )
        
        actions = optimizer.analyze_resource_usage(metrics, targets)
        
        assert len(actions) > 0
        assert any(action.action_type == "optimize_cache" for action in actions)
    
    def test_queue_optimization(self):
        """Test queue optimization actions."""
        optimizer = ResourceOptimizer(OptimizationStrategy.LATENCY_FOCUSED)
        targets = PerformanceTarget(max_queue_length=20)
        
        # High queue length metrics
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=0.5,
            memory_usage=0.6,
            gpu_usage=0.4,
            network_io=0.3,
            storage_io=0.2,
            active_requests=30,
            queue_length=30  # Above threshold
        )
        
        actions = optimizer.analyze_resource_usage(metrics, targets)
        
        assert len(actions) > 0
        assert any(action.action_type == "increase_workers" for action in actions)
        assert any(action.action_type == "optimize_request_scheduling" for action in actions)


@pytest.mark.unit
class TestAdaptiveScaler:
    """Test adaptive scaling functionality."""
    
    def test_scaler_initialization(self):
        """Test adaptive scaler initialization."""
        scaler = AdaptiveScaler(min_instances=2, max_instances=10)
        
        assert scaler.min_instances == 2
        assert scaler.max_instances == 10
        assert scaler.current_instances == 2
    
    def test_scale_up_decision(self):
        """Test scale up decision logic."""
        scaler = AdaptiveScaler(min_instances=2, max_instances=10)
        
        # High resource pressure
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=0.9,
            memory_usage=0.85,
            gpu_usage=0.8,
            network_io=0.5,
            storage_io=0.3,
            active_requests=100,
            queue_length=20
        )
        
        # High predicted load
        prediction = {
            "predicted_cpu": 0.9,
            "predicted_memory": 0.85,
            "confidence": 0.8
        }
        
        decision = scaler.should_scale(metrics, prediction)
        assert decision == "scale_up"
    
    def test_scale_down_decision(self):
        """Test scale down decision logic."""
        scaler = AdaptiveScaler(min_instances=2, max_instances=10)
        scaler.current_instances = 6  # Start with more instances
        
        # Low resource pressure
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=0.2,
            memory_usage=0.25,
            gpu_usage=0.1,
            network_io=0.1,
            storage_io=0.1,
            active_requests=5,
            queue_length=0
        )
        
        # Low predicted load
        prediction = {
            "predicted_cpu": 0.2,
            "predicted_memory": 0.25,
            "confidence": 0.7
        }
        
        # Set last scale down time to allow scaling
        scaler.last_scale_down = time.time() - 700  # 700 seconds ago
        
        decision = scaler.should_scale(metrics, prediction)
        assert decision == "scale_down"
    
    def test_scaling_execution(self):
        """Test scaling execution."""
        scaler = AdaptiveScaler(min_instances=2, max_instances=10)
        initial_instances = scaler.current_instances
        
        # Execute scale up
        result = scaler.execute_scaling("scale_up", factor=0.5)
        
        assert result["action"] == "scale_up"
        assert scaler.current_instances > initial_instances
        assert result["new_instances"] == scaler.current_instances
        
        # Execute scale down
        result = scaler.execute_scaling("scale_down", factor=0.25)
        
        assert result["action"] == "scale_down"
        assert scaler.current_instances < result["old_instances"]


@pytest.mark.unit
class TestCacheManager:
    """Test caching functionality."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False,
            "memory_cache": {
                "max_size": 1000,
                "default_ttl": 3600
            }
        }
        
        cache_manager = CacheManager(config)
        
        assert CacheLevel.L1_MEMORY in cache_manager.backends
        assert CacheLevel.L2_REDIS not in cache_manager.backends
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False
        }
        
        cache_manager = CacheManager(config)
        
        # Set value
        success = await cache_manager.set("test_key", "test_value", ttl=60)
        assert success is True
        
        # Get value
        value = await cache_manager.get("test_key")
        assert value == "test_value"
        
        # Get non-existent value
        missing_value = await cache_manager.get("missing_key")
        assert missing_value is None
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test cache entry expiry."""
        config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False
        }
        
        cache_manager = CacheManager(config)
        
        # Set value with short TTL
        await cache_manager.set("expire_key", "expire_value", ttl=1)
        
        # Should be available immediately
        value = await cache_manager.get("expire_key")
        assert value == "expire_value"
        
        # Wait for expiry
        await asyncio.sleep(2)
        
        # Should be expired
        expired_value = await cache_manager.get("expire_key")
        assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation by tags."""
        config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False
        }
        
        cache_manager = CacheManager(config)
        
        # Set values with tags
        await cache_manager.set("key1", "value1", tags=["user:123"])
        await cache_manager.set("key2", "value2", tags=["user:123", "department:cardiology"])
        await cache_manager.set("key3", "value3", tags=["department:emergency"])
        
        # Invalidate by tag
        invalidated = await cache_manager.invalidate_by_tags(["user:123"])
        
        assert invalidated == 2  # key1 and key2 should be invalidated
        
        # Check values
        assert await cache_manager.get("key1") is None
        assert await cache_manager.get("key2") is None
        assert await cache_manager.get("key3") == "value3"  # Still available


@pytest.mark.unit
class TestConnectionPool:
    """Test connection pooling functionality."""
    
    def test_pool_config(self):
        """Test pool configuration."""
        config = PoolConfig(
            min_size=3,
            max_size=15,
            max_idle_time=600,
            request_timeout=45
        )
        
        assert config.min_size == 3
        assert config.max_size == 15
        assert config.max_idle_time == 600
        assert config.request_timeout == 45
    
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test connection pool creation."""
        config = PoolConfig(min_size=2, max_size=5)
        endpoint = "http://test.example.com"
        
        pool = ConnectionPool(endpoint, config)
        
        assert pool.endpoint == endpoint
        assert pool.config == config
        assert len(pool.connections) == 0  # Initially empty
        
        # Cleanup
        await pool.close()
    
    @pytest.mark.asyncio 
    async def test_pool_manager(self):
        """Test connection pool manager."""
        config = PoolConfig(min_size=1, max_size=3)
        manager = ConnectionPoolManager(default_config=config)
        
        # Get pool for endpoint
        pool = await manager.get_pool("http://test.example.com")
        
        assert pool is not None
        assert pool.endpoint == "http://test.example.com"
        assert "http://test.example.com" in manager.pools
        
        # Get same pool again (should reuse)
        same_pool = await manager.get_pool("http://test.example.com")
        assert same_pool is pool
        
        # Cleanup
        await manager.close_all()


@pytest.mark.integration
class TestOptimizationIntegration:
    """Integration tests for optimization components."""
    
    @pytest.mark.asyncio
    async def test_performance_optimizer_workflow(self):
        """Test complete performance optimization workflow."""
        config = {
            "strategy": "balanced",
            "targets": {
                "max_cpu_usage": 0.8,
                "max_memory_usage": 0.8,
                "max_latency": 2.0
            },
            "min_instances": 2,
            "max_instances": 10
        }
        
        optimizer = PerformanceOptimizer(config)
        
        # Start optimizer
        await optimizer.start()
        
        # Wait a moment for initialization
        await asyncio.sleep(0.1)
        
        # Check that optimizer is running
        assert optimizer.running is True
        assert optimizer.optimization_task is not None
        assert optimizer.metrics_collection_task is not None
        
        # Stop optimizer
        await optimizer.stop()
        
        assert optimizer.running is False
    
    @pytest.mark.asyncio
    async def test_cache_and_optimization_integration(self):
        """Test integration between caching and optimization."""
        # Setup cache
        cache_config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False,
            "memory_cache": {"max_size": 100}
        }
        
        cache_manager = CacheManager(cache_config)
        
        # Setup optimizer
        opt_config = {
            "strategy": "resource_efficient",
            "min_instances": 1,
            "max_instances": 5
        }
        
        optimizer = PerformanceOptimizer(opt_config)
        
        # Cache some data
        for i in range(10):
            await cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Get cache stats
        cache_stats = cache_manager.get_stats()
        
        assert cache_stats["sets"] == 10
        assert cache_stats["hit_rate"] >= 0  # Should be defined
        
        # Cleanup
        await cache_manager.close()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_auto_scaling_behavior(self):
        """Test auto-scaling behavior under load."""
        scaler = AdaptiveScaler(min_instances=2, max_instances=8)
        
        # Simulate increasing load
        load_levels = [0.3, 0.5, 0.7, 0.9, 0.95]  # Increasing CPU usage
        
        for load in load_levels:
            metrics = ResourceMetrics(
                timestamp=time.time(),
                cpu_usage=load,
                memory_usage=load * 0.8,
                gpu_usage=load * 0.6,
                network_io=0.3,
                storage_io=0.2,
                active_requests=int(load * 100),
                queue_length=max(0, int(load * 50) - 25)
            )
            
            prediction = {
                "predicted_cpu": min(load + 0.1, 1.0),
                "predicted_memory": min(load * 0.8 + 0.1, 1.0),
                "confidence": 0.8
            }
            
            # Check if scaling is needed
            decision = scaler.should_scale(metrics, prediction)
            
            if decision == "scale_up" and scaler.current_instances < scaler.max_instances:
                result = scaler.execute_scaling("scale_up", factor=0.5)
                assert result["action"] == "scale_up"
                
                # Wait for cooldown
                scaler.last_scale_up = time.time() - 400  # Override cooldown for testing
        
        # Should have scaled up due to high load
        assert scaler.current_instances > 2
    
    @pytest.mark.asyncio
    async def test_comprehensive_monitoring_integration(self):
        """Test integration of all monitoring components."""
        # Setup all components
        cache_config = {"enable_memory_cache": True, "enable_redis_cache": False}
        cache_manager = CacheManager(cache_config)
        
        pool_manager = ConnectionPoolManager()
        
        opt_config = {
            "strategy": "balanced",
            "min_instances": 2,
            "max_instances": 6
        }
        optimizer = PerformanceOptimizer(opt_config)
        
        # Start systems
        await optimizer.start()
        
        # Simulate some activity
        await cache_manager.set("test", "data")
        cached_value = await cache_manager.get("test")
        assert cached_value == "data"
        
        # Get performance report
        report = optimizer.get_performance_report()
        
        assert "strategy" in report
        assert "scaling_status" in report
        assert "targets" in report
        
        # Get cache info
        cache_info = await cache_manager.get_cache_info()
        
        assert "stats" in cache_info
        assert "backends" in cache_info
        
        # Cleanup
        await optimizer.stop()
        await cache_manager.close()
        await pool_manager.close_all()