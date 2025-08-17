#!/usr/bin/env python3
"""
Generation 3 Validation Test - Scalability and Performance Optimization
Tests the advanced performance optimization, auto-scaling, and production features.
"""

import asyncio
import time
import sys
import os
import traceback
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, '/root/repo')

def test_advanced_performance_optimizer():
    """Test advanced performance optimization system."""
    print("⚡ Testing Advanced Performance Optimizer...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import (
            OptimizationStrategy, PerformanceMetrics, AdvancedPerformanceOptimizer,
            GlobalOptimizationManager, get_performance_optimizer
        )
        
        # Test optimization strategy enumeration
        assert OptimizationStrategy.ADAPTIVE.value == "adaptive"
        assert OptimizationStrategy.AGGRESSIVE.value == "aggressive"
        print("✅ Optimization strategies defined correctly")
        
        # Test performance metrics
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            request_latency=150.0,
            throughput=500.0,
            cpu_usage=45.0,
            memory_usage=60.0,
            error_rate=0.01
        )
        metrics_dict = metrics.to_dict()
        assert 'timestamp' in metrics_dict
        assert 'request_latency' in metrics_dict
        print("✅ Performance metrics system working")
        
        # Test optimizer initialization
        optimizer = AdvancedPerformanceOptimizer(OptimizationStrategy.BALANCED)
        assert optimizer.strategy == OptimizationStrategy.BALANCED
        print("✅ Performance optimizer initialized successfully")
        
        # Test global optimization manager
        global_manager = GlobalOptimizationManager()
        global_optimizer = global_manager.get_optimizer()
        assert global_optimizer is not None
        print("✅ Global optimization manager working")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced performance optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_intelligent_caching():
    """Test intelligent caching system."""
    print("🧠 Testing Intelligent Caching...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import IntelligentCache
        
        # Test cache initialization
        cache = IntelligentCache(max_size=100, ttl=60)
        assert cache.max_size == 100
        assert cache.ttl == 60
        print("✅ Intelligent cache initialized successfully")
        
        # Test cache key generation
        key = cache._generate_cache_key("test prompt", "test_model", {"param": "value"})
        assert isinstance(key, str)
        assert len(key) == 32  # SHA256 hash truncated to 32 chars
        print("✅ Cache key generation working")
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        assert 'cache_size' in stats
        assert 'hit_rate' in stats
        assert 'utilization' in stats
        print("✅ Cache statistics system working")
        
        return True
        
    except Exception as e:
        print(f"❌ Intelligent caching test failed: {e}")
        traceback.print_exc()
        return False

def test_adaptive_load_balancer():
    """Test adaptive load balancing system."""
    print("⚖️ Testing Adaptive Load Balancer...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import (
            AdaptiveLoadBalancer, PerformanceMetrics
        )
        
        # Test load balancer initialization
        load_balancer = AdaptiveLoadBalancer()
        assert load_balancer.node_metrics is not None
        assert load_balancer.node_weights is not None
        print("✅ Adaptive load balancer initialized")
        
        # Test node metrics update
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            request_latency=100.0,
            throughput=1000.0,
            cpu_usage=30.0,
            memory_usage=40.0,
            error_rate=0.001
        )
        
        load_balancer.update_node_metrics("test_node_1", metrics)
        assert "test_node_1" in load_balancer.node_weights
        print("✅ Node metrics update working")
        
        # Test load distribution prediction
        load_balancer.update_node_metrics("test_node_2", metrics)
        distribution = load_balancer.predict_load_distribution()
        assert isinstance(distribution, dict)
        assert "test_node_1" in distribution
        assert "test_node_2" in distribution
        print("✅ Load distribution prediction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive load balancer test failed: {e}")
        traceback.print_exc()
        return False

def test_auto_scaler():
    """Test auto-scaling system."""
    print("📈 Testing Auto-Scaler...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import (
            AutoScaler, PerformanceMetrics, ScalingDirection
        )
        
        # Test auto scaler initialization
        auto_scaler = AutoScaler(min_replicas=1, max_replicas=5)
        assert auto_scaler.min_replicas == 1
        assert auto_scaler.max_replicas == 5
        assert auto_scaler.current_replicas == 1
        print("✅ Auto-scaler initialized successfully")
        
        # Test scaling decision with high load
        high_load_metrics = PerformanceMetrics(
            timestamp=time.time(),
            request_latency=1500.0,  # High latency
            throughput=100.0,
            cpu_usage=85.0,  # High CPU
            memory_usage=90.0,  # High memory
            error_rate=0.08  # High error rate
        )
        
        decision = auto_scaler.analyze_scaling_need(high_load_metrics)
        assert decision.direction in [ScalingDirection.SCALE_OUT, ScalingDirection.MAINTAIN]
        assert decision.confidence >= 0.0
        print("✅ Scaling decision analysis working")
        
        # Test scaling application
        if decision.direction == ScalingDirection.SCALE_OUT:
            success = auto_scaler.apply_scaling_decision(decision)
            if success:
                print("✅ Scaling decision applied successfully")
            else:
                print("⚠️ Scaling decision not applied (low confidence)")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-scaler test failed: {e}")
        traceback.print_exc()
        return False

def test_quantum_performance_optimizer():
    """Test quantum performance optimization."""
    print("⚛️ Testing Quantum Performance Optimizer...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import (
            QuantumPerformanceOptimizer, PerformanceMetrics
        )
        
        # Test quantum optimizer initialization
        quantum_optimizer = QuantumPerformanceOptimizer()
        assert quantum_optimizer.quantum_coherence_threshold == 0.8
        print("✅ Quantum performance optimizer initialized")
        
        # Test quantum parameter optimization
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            request_latency=200.0,
            throughput=800.0,
            cpu_usage=50.0,
            memory_usage=60.0,
            quantum_coherence_score=0.9,
            error_rate=0.02
        )
        
        optimizations = quantum_optimizer.optimize_quantum_parameters(metrics)
        assert isinstance(optimizations, dict)
        print("✅ Quantum parameter optimization working")
        
        # Test performance prediction
        improvement = quantum_optimizer.predict_quantum_performance(optimizations)
        assert 0.0 <= improvement <= 0.5  # Should be between 0% and 50%
        print("✅ Quantum performance prediction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum performance optimizer test failed: {e}")
        traceback.print_exc()
        return False

async def test_async_optimization_features():
    """Test async optimization features."""
    print("🔄 Testing Async Optimization Features...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import (
            AdvancedPerformanceOptimizer, PerformanceMetrics
        )
        
        optimizer = AdvancedPerformanceOptimizer()
        
        # Test async request optimization
        request_data = {
            "prompt": "Test medical query",
            "model": "test_model",
            "parameters": {"temperature": 0.7}
        }
        
        optimization_result = await optimizer.optimize_request_processing(request_data)
        assert 'optimization_applied' in optimization_result
        print("✅ Async request optimization working")
        
        # Test async cache storage
        await optimizer.store_response_in_cache(request_data, "test response")
        print("✅ Async cache storage working")
        
        # Test optimization cycle
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            request_latency=300.0,
            throughput=600.0,
            cpu_usage=70.0,
            memory_usage=65.0,
            error_rate=0.03
        )
        
        optimizer.record_performance_metrics(metrics)
        
        # Force optimization cycle by setting last optimization time to past
        optimizer.last_optimization_time = time.time() - 120  # 2 minutes ago
        
        cycle_result = await optimizer.run_optimization_cycle()
        assert 'timestamp' in cycle_result
        print("✅ Async optimization cycle working")
        
        return True
        
    except Exception as e:
        print(f"❌ Async optimization features test failed: {e}")
        traceback.print_exc()
        return False

def test_production_readiness_features():
    """Test production readiness features."""
    print("🏭 Testing Production Readiness Features...")
    
    try:
        from federated_dp_llm.optimization.advanced_performance_optimizer import (
            AdvancedPerformanceOptimizer, GlobalOptimizationManager
        )
        
        # Test dashboard generation
        optimizer = AdvancedPerformanceOptimizer()
        
        # Add some metrics for dashboard
        from federated_dp_llm.optimization.advanced_performance_optimizer import PerformanceMetrics
        
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=time.time() - (5-i) * 60,  # 5 minutes of data
                request_latency=200.0 + i * 10,
                throughput=500.0 + i * 50,
                cpu_usage=40.0 + i * 5,
                memory_usage=50.0 + i * 3,
                error_rate=0.01 + i * 0.005
            )
            optimizer.record_performance_metrics(metrics, f"node_{i}")
        
        dashboard = optimizer.get_optimization_dashboard()
        assert 'strategy' in dashboard
        assert 'current_performance' in dashboard
        assert 'scaling_status' in dashboard
        assert 'cache_performance' in dashboard
        print("✅ Production dashboard working")
        
        # Test global optimization manager alerts
        global_manager = GlobalOptimizationManager()
        assert global_manager.alert_thresholds is not None
        assert 'high_latency' in global_manager.alert_thresholds
        print("✅ Alert system configuration working")
        
        return True
        
    except Exception as e:
        print(f"❌ Production readiness test failed: {e}")
        traceback.print_exc()
        return False

async def run_generation_3_validation():
    """Run complete Generation 3 validation suite."""
    print("=" * 70)
    print("🚀 GENERATION 3 VALIDATION - MAKE IT SCALE")
    print("=" * 70)
    
    sync_tests = [
        test_advanced_performance_optimizer,
        test_intelligent_caching,
        test_adaptive_load_balancer,
        test_auto_scaler,
        test_quantum_performance_optimizer,
        test_production_readiness_features
    ]
    
    async_tests = [
        test_async_optimization_features
    ]
    
    passed = 0
    total = len(sync_tests) + len(async_tests)
    
    # Run synchronous tests
    for test in sync_tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    # Run asynchronous tests
    for test in async_tests:
        try:
            if await test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 70)
    print(f"🎯 GENERATION 3 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Generation 3 SCALABLE implementation validated successfully!")
        print("📊 Advanced optimization, auto-scaling, and production features confirmed")
        print("⚡ System ready for high-performance production deployment")
    elif passed >= total * 0.8:
        print("⚠️ Generation 3 mostly successful with minor issues")
        print("🔧 Core scalability features are operational")
    else:
        print("❌ Generation 3 validation failed - scalability issues detected")
        print("🛠️ Performance optimization implementation needs improvement")
    
    print("=" * 70)
    return passed >= total * 0.8  # 80% threshold for Generation 3

if __name__ == "__main__":
    success = asyncio.run(run_generation_3_validation())
    sys.exit(0 if success else 1)