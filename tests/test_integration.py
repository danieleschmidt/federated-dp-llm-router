"""
Integration tests for the complete federated DP-LLM system.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch
import httpx

from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
from federated_dp_llm.routing.load_balancer import FederatedRouter, InferenceRequest
from federated_dp_llm.federation.client import HospitalNode, PrivateInferenceClient
from federated_dp_llm.security.compliance import BudgetManager, ComplianceMonitor
from federated_dp_llm.monitoring.metrics import MetricsCollector, PrivacyDashboard
from federated_dp_llm.optimization.caching import CacheManager
from federated_dp_llm.optimization.performance_optimizer import PerformanceOptimizer


@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_inference_flow(self, hospital_nodes, test_config):
        """Test complete inference flow from client to response."""
        # Setup components
        dp_config = DPConfig(epsilon_per_query=0.1, delta=1e-5)
        privacy_accountant = PrivacyAccountant(dp_config)
        
        router = FederatedRouter(model_name="test-model")
        router.register_nodes(hospital_nodes)
        
        # Create inference request
        request = InferenceRequest(
            request_id="integration_test_001",
            user_id="test_doctor",
            prompt="Test medical query for patient evaluation",
            model_name="test-model",
            max_privacy_budget=0.1,
            require_consensus=False,
            department="cardiology"
        )
        
        # Route request (mocked for integration test)
        with patch.object(router, '_simulate_inference') as mock_inference:
            mock_inference.return_value = "Mocked medical response"
            
            response = await router.route_request(request)
            
            assert response.request_id == request.request_id
            assert response.text == "Mocked medical response"
            assert response.privacy_cost > 0
            assert len(response.processing_nodes) > 0
    
    @pytest.mark.asyncio
    async def test_privacy_budget_enforcement(self, privacy_accountant, budget_manager):
        """Test privacy budget enforcement across system."""
        user_id = "budget_test_user"
        department = "emergency"
        
        # Initial budget check
        initial_remaining = privacy_accountant.get_remaining_budget(user_id)
        assert initial_remaining == privacy_accountant.config.max_budget_per_user
        
        # Spend budget through normal flow
        success = privacy_accountant.spend_budget(user_id, 5.0, "test_query")
        assert success is True
        
        # Check remaining budget
        remaining = privacy_accountant.get_remaining_budget(user_id)
        assert remaining == initial_remaining - 5.0
        
        # Try to exceed budget
        success = privacy_accountant.spend_budget(user_id, 10.0, "exceed_test")
        assert success is False
        
        # Budget should remain unchanged
        final_remaining = privacy_accountant.get_remaining_budget(user_id)
        assert final_remaining == remaining
    
    @pytest.mark.asyncio
    async def test_consensus_mechanism(self, hospital_nodes):
        """Test consensus mechanism with multiple nodes."""
        router = FederatedRouter(model_name="consensus-test-model")
        router.register_nodes(hospital_nodes)
        
        # Create consensus request
        request = InferenceRequest(
            request_id="consensus_test_001",
            user_id="consensus_doctor",
            prompt="Critical diagnosis requiring consensus",
            model_name="consensus-test-model",
            max_privacy_budget=0.2,
            require_consensus=True,
            department="emergency"
        )
        
        # Mock consensus responses
        with patch.object(router, '_simulate_inference') as mock_inference:
            mock_inference.return_value = "Consensus medical response"
            
            response = await router.route_request(request)
            
            assert response.consensus_achieved is True
            assert len(response.processing_nodes) >= 3  # Minimum for consensus
            assert response.confidence_score > 0.9  # Higher confidence for consensus
    
    @pytest.mark.asyncio
    async def test_monitoring_and_metrics(self, metrics_collector):
        """Test monitoring and metrics collection."""
        from federated_dp_llm.monitoring.metrics import PrivacyMetric, PerformanceMetric
        
        # Record some metrics
        privacy_metric = PrivacyMetric(
            user_id="metrics_test_user",
            department="radiology",
            epsilon_spent=0.15,
            delta_spent=1e-5,
            query_type="inference",
            timestamp=time.time(),
            node_id="hospital_a"
        )
        
        performance_metric = PerformanceMetric(
            metric_name="inference_latency",
            value=0.75,
            labels={"model_name": "test-model", "node_id": "hospital_a"},
            timestamp=time.time()
        )
        
        metrics_collector.record_privacy_metric(privacy_metric)
        metrics_collector.record_performance_metric(performance_metric)
        
        # Get summaries
        privacy_summary = metrics_collector.get_privacy_summary(3600)
        performance_summary = metrics_collector.get_performance_summary(3600)
        
        assert privacy_summary["total_queries"] == 1
        assert privacy_summary["total_epsilon_spent"] == 0.15
        assert "inference_latency" in performance_summary
    
    @pytest.mark.asyncio
    async def test_caching_system(self, temp_dir):
        """Test caching system integration."""
        cache_config = {
            "enable_memory_cache": True,
            "enable_redis_cache": False,  # Disable Redis for testing
            "memory_cache": {
                "max_size": 100,
                "default_ttl": 300
            }
        }
        
        cache_manager = CacheManager(cache_config)
        
        # Test cache operations
        await cache_manager.set("test_key", "test_value", ttl=60, privacy_level="public")
        
        cached_value = await cache_manager.get("test_key")
        assert cached_value == "test_value"
        
        # Test privacy-aware caching
        await cache_manager.set("sensitive_key", "sensitive_data", 
                               ttl=30, privacy_level="sensitive")
        
        sensitive_value = await cache_manager.get("sensitive_key")
        assert sensitive_value == "sensitive_data"
        
        # Test cache stats
        stats = cache_manager.get_stats()
        assert stats["hits"] >= 2
        assert stats["sets"] >= 2
    
    @pytest.mark.asyncio
    async def test_compliance_monitoring(self, compliance_test_events):
        """Test compliance monitoring system."""
        monitor = ComplianceMonitor()
        
        # Record test events
        for event in compliance_test_events:
            monitor.record_event(event)
        
        # Generate compliance report
        report = monitor.generate_report(period="daily")
        
        assert report["report_metadata"]["total_events"] == len(compliance_test_events)
        assert report["violation_summary"]["total"] >= 0
        assert "compliance_score" in report
        assert report["compliance_score"] <= 100.0
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self):
        """Test performance optimization system."""
        config = {
            "strategy": "balanced",
            "targets": {
                "max_latency": 2.0,
                "min_throughput": 100.0,
                "max_cpu_usage": 0.8
            },
            "min_instances": 2,
            "max_instances": 10
        }
        
        optimizer = PerformanceOptimizer(config)
        
        # Start optimizer
        await optimizer.start()
        
        # Wait a bit for initialization
        await asyncio.sleep(1)
        
        # Get performance report
        report = optimizer.get_performance_report()
        
        assert "strategy" in report
        assert "current_metrics" in report
        assert "scaling_status" in report
        
        # Stop optimizer
        await optimizer.stop()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_federated_training_simulation(self, hospital_nodes):
        """Test federated training simulation."""
        from federated_dp_llm.federation.server import FederatedTrainer
        
        dp_config = DPConfig(epsilon_per_query=0.1, delta=1e-5)
        
        trainer = FederatedTrainer(
            base_model="test-model",
            dp_config=dp_config,
            rounds=3,  # Short test
            clients_per_round=2
        )
        
        # Run federated training
        history = await trainer.train_federated(hospital_nodes[:2])  # Use only 2 nodes for speed
        
        assert history["total_rounds"] > 0
        assert history["final_loss"] < float('inf')
        assert history["total_participants"] == 2
        assert "metrics_history" in history
    
    @pytest.mark.asyncio
    async def test_security_integration(self, test_certificates):
        """Test security system integration."""
        from federated_dp_llm.security.authentication import AuthenticationManager
        
        auth_manager = AuthenticationManager("test-jwt-secret")
        
        # Create test user
        from federated_dp_llm.security.authentication import Role
        user = auth_manager.create_user(
            username="test_security_user",
            email="test@hospital.test",
            password="SecurePassword123!",
            department="security_test",
            roles=[Role.DOCTOR]
        )
        
        # Test authentication
        authenticated_user, message = auth_manager.authenticate_user(
            "test_security_user", "SecurePassword123!"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.user_id == user.user_id
        assert message == "Authentication successful"
        
        # Test session management
        session = auth_manager.create_session(
            user, "127.0.0.1", "test-user-agent"
        )
        
        session_user = auth_manager.validate_session(session.session_id)
        assert session_user is not None
        assert session_user.user_id == user.user_id
        
        # Test certificate validation
        cert_manager = test_certificates["cert_manager"]
        ca_cert = test_certificates["ca_cert"]
        client_cert = test_certificates["client_cert"]
        
        # Verify certificate chain
        is_valid = cert_manager.verify_certificate_chain(ca_cert.fingerprint)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, hospital_nodes):
        """Test system error handling and recovery."""
        router = FederatedRouter(model_name="error-test-model")
        router.register_nodes(hospital_nodes)
        
        # Test with failing node
        request = InferenceRequest(
            request_id="error_test_001",
            user_id="error_test_user",
            prompt="Test error handling",
            model_name="error-test-model",
            max_privacy_budget=0.1,
            department="test"
        )
        
        # Mock node failure
        with patch.object(router, '_simulate_inference') as mock_inference:
            mock_inference.side_effect = Exception("Simulated node failure")
            
            # Should handle error gracefully
            with pytest.raises(Exception):
                await router.route_request(request)
        
        # Test recovery with working node
        with patch.object(router, '_simulate_inference') as mock_inference:
            mock_inference.return_value = "Recovery response"
            
            response = await router.route_request(request)
            assert response.text == "Recovery response"


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_api_health_endpoint(self, test_server):
        """Test API health endpoint."""
        handler, base_url = test_server
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
    
    @pytest.mark.asyncio 
    async def test_api_inference_endpoint(self, test_server):
        """Test API inference endpoint."""
        handler, base_url = test_server
        
        # Create test token
        token = handler.create_test_token("test_api_user", "cardiology", "doctor", ["read", "query_inference"])
        
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "prompt": "Test API inference request",
            "model_name": "test-model",
            "max_privacy_budget": 0.1,
            "require_consensus": False
        }
        
        with patch.object(handler.router, 'route_request') as mock_route:
            from federated_dp_llm.routing.load_balancer import InferenceResponse
            
            mock_response = InferenceResponse(
                request_id="api_test_001",
                text="API test response",
                privacy_cost=0.1,
                remaining_budget=9.9,
                processing_nodes=["test_node"],
                latency=0.5,
                confidence_score=0.8
            )
            mock_route.return_value = mock_response
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/inference",
                    json=payload,
                    headers=headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["text"] == "API test response"
                assert data["privacy_cost"] == 0.1
                assert data["remaining_budget"] == 9.9


@pytest.mark.integration
@pytest.mark.slow
class TestScalabilityAndPerformance:
    """Test system scalability and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, hospital_nodes):
        """Test handling of concurrent inference requests."""
        router = FederatedRouter(model_name="concurrent-test-model")
        router.register_nodes(hospital_nodes)
        
        # Create multiple concurrent requests
        async def make_request(request_id: str):
            request = InferenceRequest(
                request_id=request_id,
                user_id=f"concurrent_user_{request_id}",
                prompt="Concurrent test request",
                model_name="concurrent-test-model",
                max_privacy_budget=0.05,
                department="concurrent_test"
            )
            
            with patch.object(router, '_simulate_inference') as mock_inference:
                mock_inference.return_value = f"Response for {request_id}"
                return await router.route_request(request)
        
        # Execute concurrent requests
        num_requests = 20
        tasks = [make_request(f"req_{i}") for i in range(num_requests)]
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        failed_responses = [r for r in responses if isinstance(r, Exception)]
        
        assert len(successful_responses) > 0, "At least some requests should succeed"
        
        # Performance metrics
        total_time = end_time - start_time
        throughput = len(successful_responses) / total_time
        
        print(f"Processed {len(successful_responses)}/{num_requests} requests in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Failed requests: {len(failed_responses)}")
        
        # Basic performance assertions
        assert throughput > 1.0, "Should process at least 1 request per second"
        assert len(failed_responses) / num_requests < 0.5, "Failure rate should be < 50%"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, metrics_collector):
        """Test memory usage patterns under load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Generate load
        for i in range(1000):
            from federated_dp_llm.monitoring.metrics import PrivacyMetric
            
            metric = PrivacyMetric(
                user_id=f"load_user_{i}",
                department="load_test",
                epsilon_spent=0.01,
                delta_spent=1e-5,
                query_type="load_test",
                timestamp=time.time(),
                node_id="load_test_node"
            )
            
            metrics_collector.record_privacy_metric(metric)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage increased by {memory_increase / 1024 / 1024:.2f} MB")
        
        # Memory should not increase excessively
        max_allowed_increase = 100 * 1024 * 1024  # 100 MB
        assert memory_increase < max_allowed_increase, "Memory usage increased too much"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])