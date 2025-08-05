"""
Tests for routing and load balancing functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch

from federated_dp_llm.routing.load_balancer import (
    FederatedRouter, InferenceRequest, InferenceResponse, RoutingStrategy
)
from federated_dp_llm.federation.client import HospitalNode


@pytest.mark.unit
class TestFederatedRouter:
    """Test cases for FederatedRouter."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = FederatedRouter(model_name="test-model", num_shards=2)
        
        assert router.model_name == "test-model"
        assert router.num_shards == 2
        assert len(router.registered_nodes) == 0
        assert router.strategy == RoutingStrategy.LOAD_BALANCED
    
    def test_node_registration(self, hospital_nodes):
        """Test node registration."""
        router = FederatedRouter(model_name="test-model")
        
        # Register nodes
        router.register_nodes(hospital_nodes)
        
        assert len(router.registered_nodes) == 3
        for node in hospital_nodes:
            assert node.id in router.registered_nodes
    
    def test_node_removal(self, hospital_nodes):
        """Test node removal."""
        router = FederatedRouter(model_name="test-model")
        router.register_nodes(hospital_nodes)
        
        # Remove a node
        removed = router.remove_node("hospital_a")
        
        assert removed is True
        assert "hospital_a" not in router.registered_nodes
        assert len(router.registered_nodes) == 2
    
    @pytest.mark.asyncio
    async def test_route_request_load_balanced(self, federated_router, sample_inference_request):
        """Test load-balanced routing."""
        # Mock the _route_to_node method
        with patch.object(federated_router, '_route_to_node') as mock_route:
            mock_response = InferenceResponse(
                request_id=sample_inference_request.request_id,
                text="Test response",
                privacy_cost=0.05,
                remaining_budget=9.95,
                processing_nodes=["hospital_a"],
                latency=0.5
            )
            mock_route.return_value = mock_response
            
            # Route request
            response = await federated_router.route_request(sample_inference_request)
            
            assert response.request_id == sample_inference_request.request_id
            assert response.text == "Test response"
            assert response.privacy_cost == 0.05
            mock_route.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_request_consensus(self, federated_router, sample_inference_request):
        """Test consensus-based routing."""
        # Enable consensus
        sample_inference_request.require_consensus = True
        
        with patch.object(federated_router, '_route_with_consensus') as mock_consensus:
            mock_response = InferenceResponse(
                request_id=sample_inference_request.request_id,
                text="Consensus response",
                privacy_cost=0.1,
                remaining_budget=9.9,
                processing_nodes=["hospital_a", "hospital_b"],
                latency=1.2,
                consensus_achieved=True
            )
            mock_consensus.return_value = mock_response
            
            response = await federated_router.route_request(sample_inference_request)
            
            assert response.consensus_achieved is True
            assert len(response.processing_nodes) == 2
            mock_consensus.assert_called_once()
    
    def test_select_best_node(self, federated_router):
        """Test node selection logic."""
        # Update node metrics to test selection
        for node_id in federated_router.registered_nodes:
            federated_router.node_metrics[node_id] = {
                "load": 0.5,
                "response_time": 0.8,
                "success_rate": 0.95,
                "last_updated": time.time()
            }
        
        # Make one node clearly better
        federated_router.node_metrics["hospital_b"]["load"] = 0.2
        federated_router.node_metrics["hospital_b"]["response_time"] = 0.3
        
        best_node = federated_router._select_best_node("cardiology")
        assert best_node == "hospital_b"
    
    @pytest.mark.asyncio
    async def test_health_check(self, federated_router):
        """Test health check functionality."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock successful health check
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "load": 0.6,
                "response_time": 0.4
            }
            mock_get.return_value = mock_response
            
            health_status = await federated_router.health_check()
            
            assert "overall_status" in health_status
            assert "nodes" in health_status
            assert len(health_status["nodes"]) == 3


@pytest.mark.unit  
class TestInferenceRequest:
    """Test cases for InferenceRequest."""
    
    def test_request_creation(self):
        """Test inference request creation."""
        request = InferenceRequest(
            request_id="test_123",
            user_id="user_456",
            prompt="Test prompt",
            model_name="test-model"
        )
        
        assert request.request_id == "test_123"
        assert request.user_id == "user_456"
        assert request.max_privacy_budget == 1.0  # default
        assert request.priority == 1  # default
        assert request.timeout == 30.0  # default
    
    def test_request_validation(self):
        """Test request validation."""
        # Valid request
        request = InferenceRequest(
            request_id="test_123",
            user_id="user_456", 
            prompt="Test prompt",
            model_name="test-model"
        )
        
        assert request.is_valid()
        
        # Invalid request (empty prompt)
        invalid_request = InferenceRequest(
            request_id="test_123",
            user_id="user_456",
            prompt="",
            model_name="test-model"
        )
        
        assert not invalid_request.is_valid()


@pytest.mark.unit
class TestInferenceResponse:
    """Test cases for InferenceResponse."""
    
    def test_response_creation(self):
        """Test inference response creation."""
        response = InferenceResponse(
            request_id="test_123",
            text="Response text",
            privacy_cost=0.1,
            remaining_budget=9.9,
            processing_nodes=["node1"],
            latency=0.5
        )
        
        assert response.request_id == "test_123"
        assert response.text == "Response text"
        assert response.privacy_cost == 0.1
        assert response.latency == 0.5
        assert response.confidence_score == 0.0  # default
    
    def test_response_serialization(self):
        """Test response serialization."""
        response = InferenceResponse(
            request_id="test_123",
            text="Response text",
            privacy_cost=0.1,
            remaining_budget=9.9,
            processing_nodes=["node1"],
            latency=0.5
        )
        
        response_dict = response.to_dict()
        
        assert "request_id" in response_dict
        assert "text" in response_dict
        assert "privacy_cost" in response_dict
        assert response_dict["request_id"] == "test_123"


@pytest.mark.integration
class TestRoutingIntegration:
    """Integration tests for routing system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_routing(self, federated_router, sample_inference_request):
        """Test end-to-end routing workflow."""
        # Mock external dependencies
        with patch('httpx.AsyncClient.post') as mock_post:
            # Mock node response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "text": "Generated response",
                "confidence": 0.85,
                "processing_time": 0.8
            }
            mock_post.return_value = mock_response
            
            # Route request
            response = await federated_router.route_request(sample_inference_request)
            
            assert response is not None
            assert response.request_id == sample_inference_request.request_id
            assert len(response.processing_nodes) > 0
    
    @pytest.mark.asyncio
    async def test_node_failure_handling(self, federated_router, sample_inference_request):
        """Test handling of node failures."""
        # Mock node failure and recovery
        with patch('httpx.AsyncClient.post') as mock_post:
            # First call fails
            mock_post.side_effect = [
                Exception("Connection failed"),
                AsyncMock(status_code=200, json=lambda: {"text": "Fallback response"})
            ]
            
            # Should fallback to another node
            response = await federated_router.route_request(sample_inference_request)
            
            # Should still get a response from fallback node
            assert response is not None
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_balancing_distribution(self, federated_router):
        """Test that load balancing distributes requests across nodes."""
        requests = []
        for i in range(10):
            request = InferenceRequest(
                request_id=f"test_{i}",
                user_id=f"user_{i}",
                prompt=f"Test prompt {i}",
                model_name="test-model",
                department="general"
            )
            requests.append(request)
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "Response"}
            mock_post.return_value = mock_response
            
            # Process all requests
            responses = []
            for request in requests:
                response = await federated_router.route_request(request)
                responses.append(response)
            
            # All requests should be processed
            assert len(responses) == 10
            
            # Should use multiple nodes (basic check)
            used_nodes = set()
            for response in responses:
                used_nodes.update(response.processing_nodes)
            
            assert len(used_nodes) > 1  # Should distribute across nodes