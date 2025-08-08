"""
HTTPS Request Handler

Handles incoming inference requests with proper authentication, validation,
and privacy budget checking.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt
from ..routing.load_balancer import FederatedRouter, InferenceRequest, InferenceResponse
from ..core.privacy_accountant import DPConfig


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
    
    def __init__(self, router: FederatedRouter, jwt_secret: str):
        self.router = router
        self.jwt_secret = jwt_secret
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
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
    
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
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            return response
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Health check endpoint."""
            node_health = await self.router.health_check()
            
            return HealthCheckResponse(
                status="healthy",
                timestamp=time.time(),
                version="0.1.0",
                nodes=node_health
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
            
            # Create internal request
            inference_request = InferenceRequest(
                request_id=request_id,
                user_id=user.user_id,
                prompt=request.prompt,
                model_name=request.model_name,
                max_privacy_budget=request.max_privacy_budget,
                require_consensus=request.require_consensus,
                priority=request.priority,
                timeout=request.timeout,
                department=user.department
            )
            
            # Track active request
            self.active_requests[request_id] = time.time()
            
            try:
                # Route request
                response = await asyncio.wait_for(
                    self.router.route_request(inference_request),
                    timeout=request.timeout
                )
                
                return InferenceResponseModel(**asdict(response))
                
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