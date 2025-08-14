"""
Request Handler for Federated DP-LLM Router

Handles HTTP requests for the federated differential privacy LLM system,
including request validation, routing, and response formatting.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request model for LLM inference."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    prompt: str = Field(..., description="Input prompt for inference")
    model_name: str = Field(default="medllama-7b", description="Target model name")
    max_privacy_budget: float = Field(default=0.1, description="Maximum privacy budget to spend")
    require_consensus: bool = Field(default=False, description="Require multiple node consensus")
    priority: int = Field(default=5, description="Request priority (0=highest, 10=lowest)")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    department: Optional[str] = Field(default=None, description="User department")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class InferenceResponse(BaseModel):
    """Response model for LLM inference."""
    request_id: str
    user_id: str
    text: str
    confidence_score: float
    privacy_cost: float
    remaining_budget: float
    processing_nodes: List[str]
    latency: float
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    nodes: Dict[str, str]
    metrics: Dict[str, Any]


class RequestHandler:
    """Handles HTTP requests for the federated LLM system."""
    
    def __init__(self, router=None, privacy_accountant=None, auth_manager=None):
        """Initialize request handler with dependencies."""
        from ..routing.load_balancer import FederatedRouter
        from ..core.privacy_accountant import PrivacyAccountant
        from ..security.authentication import AuthenticationManager
        
        self.router = router
        self.privacy_accountant = privacy_accountant
        self.auth_manager = auth_manager
        self.request_metrics = {}
        
    async def handle_inference_request(self, request: InferenceRequest, 
                                     http_request: Optional[Request] = None) -> InferenceResponse:
        """Handle an inference request with privacy and authentication checks."""
        start_time = time.time()
        
        try:
            # Validate request
            await self._validate_request(request)
            
            # Check authentication if auth manager is available
            if self.auth_manager and http_request:
                await self._authenticate_request(http_request, request.user_id)
            
            # Check privacy budget
            if self.privacy_accountant:
                if not self.privacy_accountant.check_budget(request.user_id, request.max_privacy_budget):
                    raise HTTPException(
                        status_code=429,
                        detail=f"Insufficient privacy budget for user {request.user_id}"
                    )
            
            # Route request if router is available
            if self.router:
                # Convert Pydantic model to dataclass for router
                from ..routing.load_balancer import InferenceRequest as RouterRequest
                router_request = RouterRequest(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    prompt=request.prompt,
                    model_name=request.model_name,
                    max_privacy_budget=request.max_privacy_budget,
                    require_consensus=request.require_consensus,
                    priority=request.priority,
                    timeout=request.timeout,
                    department=request.department
                )
                
                router_response = await self.router.route_request(router_request)
                
                # Spend privacy budget
                if self.privacy_accountant:
                    self.privacy_accountant.spend_budget(
                        request.user_id, 
                        router_response.privacy_cost, 
                        request.prompt[:100]  # First 100 chars for audit
                    )
                
                # Convert router response to HTTP response
                response = InferenceResponse(
                    request_id=router_response.request_id,
                    user_id=router_response.user_id,
                    text=router_response.text,
                    confidence_score=router_response.confidence_score,
                    privacy_cost=router_response.privacy_cost,
                    remaining_budget=self.privacy_accountant.get_remaining_budget(request.user_id) if self.privacy_accountant else 0.0,
                    processing_nodes=router_response.processing_nodes,
                    latency=time.time() - start_time,
                    timestamp=time.time(),
                    metadata=router_response.metadata
                )
                
            else:
                # Fallback response when router is not available
                response = InferenceResponse(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    text=f"Simulated response for: {request.prompt[:50]}...",
                    confidence_score=0.85,
                    privacy_cost=request.max_privacy_budget * 0.5,
                    remaining_budget=10.0,
                    processing_nodes=["fallback_node"],
                    latency=time.time() - start_time,
                    timestamp=time.time(),
                    metadata={"mode": "fallback", "model": request.model_name}
                )
            
            # Record metrics
            self._record_request_metrics(request, response, time.time() - start_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing inference request {request.request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def handle_health_check(self) -> HealthCheckResponse:
        """Handle health check request."""
        try:
            # Get router health if available
            nodes_status = {}
            if self.router:
                nodes_status = await self.router.health_check()
            
            # Calculate basic metrics
            total_requests = sum(self.request_metrics.values()) if self.request_metrics else 0
            
            response = HealthCheckResponse(
                status="healthy",
                timestamp=time.time(),
                version="0.1.0",
                nodes=nodes_status,
                metrics={
                    "total_requests": total_requests,
                    "active_nodes": len(nodes_status),
                    "system_status": "operational"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    
    async def handle_metrics_request(self) -> Dict[str, Any]:
        """Handle request for system metrics."""
        try:
            metrics = {
                "request_count": sum(self.request_metrics.values()),
                "request_breakdown": self.request_metrics,
                "timestamp": time.time()
            }
            
            # Add router metrics if available
            if self.router:
                router_stats = self.router.get_routing_stats()
                metrics.update(router_stats)
            
            # Add privacy metrics if available
            if self.privacy_accountant:
                privacy_stats = self.privacy_accountant.get_privacy_stats()
                metrics["privacy"] = privacy_stats
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
    
    async def _validate_request(self, request: InferenceRequest):
        """Validate incoming request."""
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if len(request.prompt) > 10000:  # 10K character limit
            raise HTTPException(status_code=400, detail="Prompt too long (max 10,000 characters)")
        
        if request.max_privacy_budget <= 0:
            raise HTTPException(status_code=400, detail="Privacy budget must be positive")
        
        if request.max_privacy_budget > 5.0:  # Reasonable upper limit
            raise HTTPException(status_code=400, detail="Privacy budget too high (max 5.0)")
        
        if not (0 <= request.priority <= 10):
            raise HTTPException(status_code=400, detail="Priority must be between 0 and 10")
        
        if request.timeout <= 0 or request.timeout > 300:  # 5 minute max
            raise HTTPException(status_code=400, detail="Timeout must be between 0 and 300 seconds")
    
    async def _authenticate_request(self, http_request: Request, user_id: str):
        """Authenticate the HTTP request."""
        # Extract token from Authorization header
        auth_header = http_request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        token = auth_header.split(" ")[1]
        
        # Validate token
        try:
            user = self.auth_manager.verify_token(token)
            if not user or user.user_id != user_id:
                raise HTTPException(status_code=403, detail="Token user mismatch")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    def _record_request_metrics(self, request: InferenceRequest, response: InferenceResponse, latency: float):
        """Record request metrics for monitoring."""
        # Simple metrics tracking
        metric_key = f"{request.model_name}_{request.department or 'unknown'}"
        
        if metric_key not in self.request_metrics:
            self.request_metrics[metric_key] = 0
        
        self.request_metrics[metric_key] += 1
        
        # Log important metrics
        logger.info(f"Request {request.request_id} - Latency: {latency:.3f}s, "
                   f"Privacy Cost: {response.privacy_cost:.3f}, "
                   f"Confidence: {response.confidence_score:.3f}")
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request handling statistics."""
        return {
            "total_requests": sum(self.request_metrics.values()),
            "request_breakdown": dict(self.request_metrics),
            "handler_status": "active"
        }
    
    async def handle_batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Handle multiple inference requests in batch."""
        if len(requests) > 10:  # Reasonable batch limit
            raise HTTPException(status_code=400, detail="Batch size too large (max 10 requests)")
        
        responses = []
        
        # Process requests concurrently
        tasks = [self.handle_inference_request(req) for req in requests]
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error responses
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    error_response = InferenceResponse(
                        request_id=requests[i].request_id,
                        user_id=requests[i].user_id,
                        text=f"Error: {str(response)}",
                        confidence_score=0.0,
                        privacy_cost=0.0,
                        remaining_budget=0.0,
                        processing_nodes=[],
                        latency=0.0,
                        timestamp=time.time(),
                        metadata={"error": True}
                    )
                    processed_responses.append(error_response)
                else:
                    processed_responses.append(response)
            
            return processed_responses
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")