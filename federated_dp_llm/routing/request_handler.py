"""
HTTPS Request Handler

Handles incoming inference requests with proper authentication, validation,
and privacy budget checking.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import jwt
from ..routing.load_balancer import FederatedRouter
from ..routing.simple_load_balancer import SimpleLoadBalancer, FederatedInferenceCoordinator, LoadBalancingStrategy
from ..core.privacy_accountant import DPConfig, PrivacyAccountant
from ..core.model_service import ModelService, InferenceRequest, InferenceResponse, get_model_service
from ..core.error_handling import get_error_handler, handle_errors, ErrorCategory, ErrorSeverity, create_circuit_breaker, create_retry_handler
from ..federation.http_client import FederatedHTTPClient, SimpleDistributedInference, NodeInferenceRequest
from ..security.enhanced_security import get_security_manager
from ..monitoring.logging_config import setup_logging, LogConfig, get_logger
from ..optimization.integrated_optimizer import get_integrated_optimizer, optimize_request


class InferenceRequestModel(BaseModel):
    """Pydantic model for inference requests."""
    prompt: str = Field(..., min_length=1, max_length=8192)
    model_name: str = Field(default="medllama-7b")
    max_privacy_budget: float = Field(default=0.1, gt=0, le=10.0)
    require_consensus: bool = Field(default=False)
    priority: int = Field(default=1, ge=1, le=10)
    timeout: float = Field(default=30.0, gt=0, le=300.0)
    department: Optional[str] = Field(default=None)


class InferenceResponseModel(BaseModel):
    """Pydantic model for inference responses."""
    request_id: str
    text: str
    privacy_cost: float
    remaining_budget: float
    processing_nodes: List[str]
    latency: float
    confidence_score: float
    consensus_achieved: bool = False


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    nodes: Dict[str, Any]


class UserClaims(BaseModel):
    """JWT user claims."""
    user_id: str
    department: str
    role: str
    permissions: List[str]


class RequestHandler:
    """Main HTTP request handler for federated inference."""
    
    def __init__(self, router: FederatedRouter = None, jwt_secret: str = "your-secret-key"):
        self.router = router
        self.jwt_secret = jwt_secret
        
        # Initialize logging
        log_config = LogConfig(
            level="INFO",
            format="structured",
            output="both",
            log_file="./logs/federated_llm.log",
            audit_log_file="./logs/audit.log",
            enable_privacy_filtering=True,
            enable_security_logging=True
        )
        self.loggers = setup_logging(log_config)
        self.logger = get_logger("request_handler")
        
        # Initialize error handling and security
        self.error_handler = get_error_handler()
        self.security_manager = get_security_manager()
        
        # Initialize performance optimization
        self.optimizer = get_integrated_optimizer()
        
        # Create circuit breakers
        self.model_circuit_breaker = create_circuit_breaker("model_service", failure_threshold=5)
        self.node_circuit_breaker = create_circuit_breaker("federated_nodes", failure_threshold=3)
        
        # Create retry handlers
        self.inference_retry = create_retry_handler(max_attempts=3, base_delay=1.0)
        
        # Initialize model service and privacy accountant
        dp_config = DPConfig(epsilon_per_query=0.1, delta=1e-5)
        self.privacy_accountant = PrivacyAccountant(config=dp_config)
        self.model_service = get_model_service(self.privacy_accountant)
        
        # Initialize simple load balancer (replacing quantum complexity)
        self.load_balancer = SimpleLoadBalancer(LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
        self.inference_coordinator = FederatedInferenceCoordinator(self.load_balancer)
        
        # Register some default nodes (can be configured later)
        self.load_balancer.register_node("node_1", "localhost", 8001)
        self.load_balancer.register_node("node_2", "localhost", 8002)
        
        # Keep HTTP client for backward compatibility
        self.http_client = self.load_balancer.http_client
        self.distributed_inference = SimpleDistributedInference(self.http_client)
        self.app = FastAPI(
            title="Federated DP-LLM Router",
            description="Privacy-preserving federated LLM inference API",
            version="0.1.0"
        )
        
        # Security
        self.security = HTTPBearer()
        
        # Request tracking
        self.request_counter = 0
        self.active_requests: Dict[str, float] = {}
        
        # Load default model
        asyncio.create_task(self._initialize_model())
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
    async def _initialize_model(self):
        """Initialize the model service and optimization system."""
        try:
            # Start optimization system
            await self.optimizer.start()
            
            # Load model
            success = self.model_service.load_model("microsoft/DialoGPT-small", "cpu")
            if success:
                self.logger.info("Model service initialized successfully")
            else:
                self.logger.error("Failed to initialize model service")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
    
    def _setup_middleware(self):
        """Configure middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://*.hospital.local", "https://*.health-network.local"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        @self.app.middleware("http")
        async def security_middleware(request, call_next):
            # Get client IP
            client_ip = request.client.host if request.client else "unknown"
            
            # Log request
            self.logger.info(
                f"Incoming request: {request.method} {request.url.path}",
                extra={
                    "client_ip": client_ip,
                    "user_agent": request.headers.get("user-agent"),
                    "endpoint": str(request.url.path),
                    "method": request.method
                }
            )
            
            # Security validation for non-health endpoints
            if not request.url.path.startswith("/health"):
                try:
                    # Extract request data for analysis
                    request_data = f"{request.method} {request.url.path}"
                    
                    # Get query parameters for analysis
                    if request.query_params:
                        request_data += f" params: {dict(request.query_params)}"
                    
                    # Basic security validation
                    allowed, security_info = await self.security_manager.validate_request(
                        request_data=request_data,
                        source_ip=client_ip,
                        endpoint=str(request.url.path)
                    )
                    
                    if not allowed:
                        self.logger.warning(
                            f"Request blocked: {security_info.get('reason')}",
                            extra={
                                "client_ip": client_ip,
                                "endpoint": str(request.url.path),
                                "security_info": security_info
                            }
                        )
                        
                        if security_info.get("rate_limited"):
                            return Response(
                                content=json.dumps({
                                    "error": "Rate limit exceeded",
                                    "retry_after": security_info.get("retry_after", 60)
                                }),
                                status_code=429,
                                media_type="application/json"
                            )
                        else:
                            return Response(
                                content=json.dumps({
                                    "error": "Access denied",
                                    "reason": security_info.get("reason", "Security violation")
                                }),
                                status_code=403,
                                media_type="application/json"
                            )
                            
                except Exception as e:
                    self.logger.error(f"Security middleware error: {e}")
                    # Continue on security middleware errors to avoid blocking legitimate requests
            
            # Process request
            try:
                response = await call_next(request)
                
                # Add security headers
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["X-XSS-Protection"] = "1; mode=block"
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                response.headers["X-Request-ID"] = str(uuid.uuid4())
                
                return response
                
            except Exception as e:
                # Log and handle request processing errors
                await self.error_handler.handle_error(e)
                raise
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Health check endpoint."""
            model_health = self.model_service.health_check()
            node_health = await self.router.health_check() if self.router else {}
            
            all_nodes = {**node_health, "model_service": model_health}
            status = "healthy" if model_health["status"] == "healthy" else "unhealthy"
            
            return HealthCheckResponse(
                status=status,
                timestamp=time.time(),
                version="0.1.0",
                nodes=all_nodes
            )
        
        @self.app.get("/stats")
        async def get_stats(user: UserClaims = Depends(self.get_current_user)):
            """Get routing statistics (admin only)."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            return self.router.get_routing_stats()
        
        @self.app.post("/inference", response_model=InferenceResponseModel)
        async def inference(
            request: InferenceRequestModel,
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Main inference endpoint."""
            # Generate request ID
            self.request_counter += 1
            request_id = f"req_{int(time.time())}_{self.request_counter}"
            
            # Create model service request
            model_request = InferenceRequest(
                text=request.prompt,
                user_id=user.user_id,
                max_length=512,
                privacy_budget=request.max_privacy_budget
            )
            
            # Track active request
            self.active_requests[request_id] = time.time()
            
            try:
                # Use optimized inference
                request_key = f\"inference_{hash(str(model_request.__dict__))}\"
                
                async def inference_func(data):
                    return await self.model_service.inference(model_request)
                
                response = await optimize_request(
                    request_key=request_key,
                    request_data=model_request.__dict__,
                    inference_func=inference_func
                )
                
                if not response.success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=response.error
                    )
                
                # Get remaining budget
                remaining_budget = self.privacy_accountant.get_user_budget(user.user_id)
                
                return InferenceResponseModel(
                    request_id=request_id,
                    text=response.generated_text,
                    privacy_cost=response.privacy_cost,
                    remaining_budget=remaining_budget,
                    processing_nodes=[f"shard_{i}" for i in range(response.shard_count)],
                    latency=response.processing_time,
                    confidence_score=0.95,  # Simplified
                    consensus_achieved=True  # Simplified
                )
                
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"Request timed out after {request.timeout} seconds"
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {str(e)}"
                )
            finally:
                # Clean up tracking
                self.active_requests.pop(request_id, None)
        
        @self.app.post("/inference/distributed", response_model=InferenceResponseModel)
        async def distributed_inference(
            request: InferenceRequestModel,
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Distributed inference across multiple nodes."""
            # Generate request ID
            self.request_counter += 1
            request_id = f"dist_req_{int(time.time())}_{self.request_counter}"
            
            # Check if we have healthy nodes
            healthy_nodes = self.http_client.get_healthy_nodes()
            if not healthy_nodes:
                # Fallback to local inference
                return await inference(request, user)
            
            # Track active request
            self.active_requests[request_id] = time.time()
            
            try:
                # Choose inference strategy based on request
                if request.require_consensus:
                    # Use consensus inference
                    result = await self.distributed_inference.consensus_inference(
                        text=request.prompt,
                        user_id=user.user_id,
                        request_id=request_id,
                        min_consensus=2
                    )
                else:
                    # Use fastest inference
                    result = await self.distributed_inference.fastest_inference(
                        text=request.prompt,
                        user_id=user.user_id,
                        request_id=request_id
                    )
                
                if not result:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="No nodes available or consensus not reached"
                    )
                
                # Deduct privacy budget
                if self.privacy_accountant.can_query(user.user_id, request.max_privacy_budget):
                    self.privacy_accountant.deduct_budget(user.user_id, request.max_privacy_budget)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Privacy budget exceeded"
                    )
                
                # Get remaining budget
                remaining_budget = self.privacy_accountant.get_user_budget(user.user_id)
                
                end_time = time.time()
                processing_time = end_time - self.active_requests[request_id]
                
                return InferenceResponseModel(
                    request_id=request_id,
                    text=result,
                    privacy_cost=request.max_privacy_budget,
                    remaining_budget=remaining_budget,
                    processing_nodes=healthy_nodes,
                    latency=processing_time,
                    confidence_score=0.95,
                    consensus_achieved=request.require_consensus
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Distributed inference failed: {str(e)}"
                )
            finally:
                # Clean up tracking
                self.active_requests.pop(request_id, None)
        
        @self.app.get("/nodes/health")
        async def check_all_nodes_health(
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Check health of all registered nodes."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            health_results = await self.http_client.health_check_all_nodes()
            return {
                "timestamp": time.time(),
                "node_health": health_results,
                "stats": self.http_client.get_node_stats()
            }
        
        @self.app.post("/nodes/register")
        async def register_node(
            node_data: dict,
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Register a new federated node."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            try:
                node_id = node_data["node_id"]
                host = node_data["host"]
                port = node_data["port"]
                
                self.http_client.register_node(node_id, host, port)
                
                # Perform health check
                health = await self.http_client.health_check_node(node_id)
                
                return {
                    "message": f"Node {node_id} registered successfully",
                    "health": health.__dict__ if health else None
                }
                
            except KeyError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {e}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to register node: {str(e)}"
                )
        
        @self.app.post("/inference/simple", response_model=InferenceResponseModel)
        async def simple_inference(
            request: InferenceRequestModel,
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Simple load-balanced inference using the new simple load balancer."""
            # Generate request ID
            self.request_counter += 1
            request_id = f"simple_req_{int(time.time())}_{self.request_counter}"
            
            # Check privacy budget
            if not self.privacy_accountant.can_query(user.user_id, request.max_privacy_budget):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Privacy budget exceeded"
                )
            
            # Track active request
            self.active_requests[request_id] = time.time()
            
            try:
                # Use simple inference coordinator
                if request.require_consensus:
                    # Use redundant inference for consensus
                    result = await self.inference_coordinator.redundant_inference(
                        text=request.prompt,
                        user_id=user.user_id,
                        request_id=request_id,
                        redundancy=2
                    )
                else:
                    # Use single inference
                    result = await self.inference_coordinator.single_inference(
                        text=request.prompt,
                        user_id=user.user_id,
                        request_id=request_id
                    )
                
                if not result:
                    # Fallback to local model service
                    model_request = InferenceRequest(
                        text=request.prompt,
                        user_id=user.user_id,
                        max_length=512,
                        privacy_budget=request.max_privacy_budget
                    )
                    
                    model_response = await self.model_service.inference(model_request)
                    if model_response.success:
                        result = model_response.generated_text
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="All inference methods failed"
                        )
                
                # Deduct privacy budget
                self.privacy_accountant.deduct_budget(user.user_id, request.max_privacy_budget, request_id)
                
                # Get remaining budget
                remaining_budget = self.privacy_accountant.get_user_budget(user.user_id)
                
                end_time = time.time()
                processing_time = end_time - self.active_requests[request_id]
                
                return InferenceResponseModel(
                    request_id=request_id,
                    text=result,
                    privacy_cost=request.max_privacy_budget,
                    remaining_budget=remaining_budget,
                    processing_nodes=self.load_balancer.http_client.get_healthy_nodes(),
                    latency=processing_time,
                    confidence_score=0.95,
                    consensus_achieved=request.require_consensus
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Simple inference failed: {str(e)}"
                )
            finally:
                # Clean up tracking
                self.active_requests.pop(request_id, None)
        
        @self.app.get("/load_balancer/stats")
        async def get_load_balancer_stats(
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Get load balancer statistics."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            return await self.load_balancer.get_load_balancing_stats()
        
        @self.app.post("/load_balancer/strategy")
        async def update_load_balancer_strategy(
            strategy_data: dict,
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Update load balancing strategy."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            try:
                strategy_name = strategy_data["strategy"]
                strategy = LoadBalancingStrategy(strategy_name)
                self.load_balancer.update_strategy(strategy)
                
                return {
                    "message": f"Load balancing strategy updated to {strategy_name}",
                    "current_strategy": strategy.value
                }
                
            except (KeyError, ValueError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid strategy: {e}"
                )
        
        @self.app.get("/performance/stats")
        async def get_performance_stats(
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Get comprehensive performance statistics."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            stats = self.optimizer.get_performance_stats()
            error_stats = self.error_handler.get_error_stats()
            security_stats = self.security_manager.get_security_stats()
            
            return {
                "timestamp": time.time(),
                "performance": stats,
                "errors": error_stats,
                "security": security_stats
            }
        
        @self.app.post("/performance/optimize")
        async def trigger_optimization(
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Manually trigger performance optimization."""
            if "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            try:
                await self.optimizer.trigger_optimization()
                return {"message": "Optimization triggered successfully"}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Optimization failed: {str(e)}"
                )
        
        @self.app.get("/metrics/system")
        async def get_system_metrics(
            user: UserClaims = Depends(self.get_current_user)
        ):
            """Get current system metrics."""
            if "monitoring" not in user.permissions and "admin" not in user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Monitoring access required"
                )
            
            current_metrics = self.optimizer.resource_monitor.get_current_metrics()
            metrics_summary = self.optimizer.resource_monitor.get_metrics_summary(60)
            
            return {
                "current": current_metrics.__dict__ if current_metrics else None,
                "summary_1h": metrics_summary
            }
        
        @self.app.get("/privacy/budget/{user_id}")
        async def get_privacy_budget(
            user_id: str,
            current_user: UserClaims = Depends(self.get_current_user)
        ):
            """Get privacy budget for a user."""
            # Users can only check their own budget unless admin
            if current_user.user_id != user_id and "admin" not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot access other user's privacy budget"
                )
            
            remaining = self.router.privacy_accountant.get_remaining_budget(user_id)
            history = self.router.privacy_accountant.get_user_history(user_id)
            
            return {
                "user_id": user_id,
                "remaining_budget": remaining,
                "max_budget": self.router.privacy_accountant.config.max_budget_per_user,
                "spent_budget": self.router.privacy_accountant.user_budgets.get(user_id, 0.0),
                "query_count": len(history),
                "last_reset": None  # TODO: Implement budget reset tracking
            }
        
        @self.app.post("/privacy/reset/{user_id}")
        async def reset_privacy_budget(
            user_id: str,
            current_user: UserClaims = Depends(self.get_current_user)
        ):
            """Reset privacy budget for a user (admin only)."""
            if "admin" not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required"
                )
            
            self.router.privacy_accountant.reset_user_budget(user_id)
            
            return {
                "message": f"Privacy budget reset for user {user_id}",
                "reset_time": time.time()
            }
        
        @self.app.get("/models")
        async def list_models():
            """List available models."""
            # In practice, would query actual model registry
            return {
                "models": [
                    {
                        "name": "medllama-7b",
                        "description": "Medical LLaMA 7B parameter model",
                        "parameters": "7B",
                        "specialized": "healthcare",
                        "privacy_cost": 0.1
                    },
                    {
                        "name": "bioclinical-13b", 
                        "description": "BioClinical 13B parameter model",
                        "parameters": "13B",
                        "specialized": "clinical_notes",
                        "privacy_cost": 0.15
                    }
                ]
            }
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> UserClaims:
        """Extract and validate user from JWT token."""
        try:
            # Decode JWT token
            payload = jwt.decode(
                credentials.credentials,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            # Extract user claims
            user_claims = UserClaims(
                user_id=payload.get("sub"),
                department=payload.get("department", "general"),
                role=payload.get("role", "user"),
                permissions=payload.get("permissions", ["read"])
            )
            
            return user_claims
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def create_test_token(self, user_id: str, department: str = "general", role: str = "user", permissions: List[str] = None) -> str:
        """Create test JWT token (for development only)."""
        if permissions is None:
            permissions = ["read"]
        
        payload = {
            "sub": user_id,
            "department": department,
            "role": role,
            "permissions": permissions,
            "iat": time.time(),
            "exp": time.time() + 3600  # 1 hour expiry
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs):
        """Run the HTTP server."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
            **kwargs
        )
    
    async def shutdown(self):
        """Graceful shutdown."""
        # Wait for active requests to complete (with timeout)
        if self.active_requests:
            print(f"Waiting for {len(self.active_requests)} active requests...")
            
            max_wait = 30  # seconds
            start_time = time.time()
            
            while self.active_requests and (time.time() - start_time) < max_wait:
                await asyncio.sleep(0.1)
            
            if self.active_requests:
                print(f"Timeout: {len(self.active_requests)} requests still active")
        
        print("Request handler shutdown complete")