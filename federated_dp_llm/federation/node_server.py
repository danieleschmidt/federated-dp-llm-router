"""
Federated Node Server

A simple HTTP server that can host model shards and participate in federated inference.
"""

import asyncio
import logging
import uvicorn
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

from ..core.model_service import ModelService, InferenceRequest, InferenceResponse, get_model_service
from ..core.privacy_accountant import PrivacyAccountant, DPConfig


class NodeConfig(BaseModel):
    """Configuration for a federated node."""
    node_id: str
    host: str = "localhost"
    port: int = 8001
    model_name: str = "microsoft/DialoGPT-small"
    max_concurrent_requests: int = 10
    device: str = "cpu"


class NodeInferenceRequest(BaseModel):
    """Request for node-level inference."""
    text: str
    user_id: str
    request_id: str
    max_length: int = 512
    privacy_budget: float = 0.1


class NodeInferenceResponse(BaseModel):
    """Response from node-level inference."""
    request_id: str
    generated_text: str
    processing_time: float
    node_id: str
    success: bool
    error: Optional[str] = None


class NodeHealthResponse(BaseModel):
    """Node health check response."""
    node_id: str
    status: str
    model_loaded: bool
    active_requests: int
    max_concurrent: int
    device: str
    memory_usage_mb: float


class FederatedNodeServer:
    """HTTP server for federated learning node."""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"node_{config.node_id}")
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"Federated Node {config.node_id}",
            description="Federated learning node for distributed inference",
            version="0.1.0"
        )
        
        # Initialize services
        dp_config = DPConfig(epsilon_per_query=0.1, delta=1e-5)
        privacy_accountant = PrivacyAccountant(config=dp_config)
        self.model_service = get_model_service(privacy_accountant)
        
        # Request tracking
        self.active_requests = 0
        self.request_history: Dict[str, float] = {}
        
        self._setup_routes()
        
    async def initialize(self):
        """Initialize the node (load model)."""
        try:
            self.logger.info(f"Initializing node {self.config.node_id}")
            success = self.model_service.load_model(
                self.config.model_name, 
                self.config.device
            )
            
            if success:
                self.logger.info(f"Node {self.config.node_id} initialized successfully")
            else:
                self.logger.error(f"Failed to initialize node {self.config.node_id}")
                
            return success
        except Exception as e:
            self.logger.error(f"Error initializing node {self.config.node_id}: {e}")
            return False
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/inference", response_model=NodeInferenceResponse)
        async def node_inference(request: NodeInferenceRequest):
            """Perform inference on this node."""
            if self.active_requests >= self.config.max_concurrent_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Node at maximum capacity"
                )
            
            self.active_requests += 1
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Create model service request
                model_request = InferenceRequest(
                    text=request.text,
                    user_id=request.user_id,
                    max_length=request.max_length,
                    privacy_budget=request.privacy_budget
                )
                
                # Perform inference
                response = await self.model_service.inference(model_request)
                
                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - start_time
                
                return NodeInferenceResponse(\n                    request_id=request.request_id,
                    generated_text=response.generated_text,
                    processing_time=processing_time,
                    node_id=self.config.node_id,
                    success=response.success,
                    error=response.error
                )
                
            except Exception as e:
                self.logger.error(f"Inference failed: {e}")
                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - start_time
                
                return NodeInferenceResponse(
                    request_id=request.request_id,
                    generated_text="",
                    processing_time=processing_time,
                    node_id=self.config.node_id,
                    success=False,
                    error=str(e)
                )
                
            finally:
                self.active_requests -= 1
        
        @self.app.get("/health", response_model=NodeHealthResponse)
        async def health_check():
            """Health check endpoint."""
            health = self.model_service.health_check()
            
            # Get memory usage
            memory_mb = 0.0
            if torch.cuda.is_available() and self.config.device != "cpu":
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            return NodeHealthResponse(
                node_id=self.config.node_id,
                status=health["status"],
                model_loaded=health["model_loaded"],
                active_requests=self.active_requests,
                max_concurrent=self.config.max_concurrent_requests,
                device=self.config.device,
                memory_usage_mb=memory_mb
            )
        
        @self.app.get("/info")
        async def node_info():
            """Get detailed node information."""
            return {
                "node_id": self.config.node_id,
                "config": self.config.dict(),
                "model_info": self.model_service.get_model_info(),
                "health": self.model_service.health_check()
            }
        
        @self.app.post("/shutdown")
        async def shutdown():
            """Graceful shutdown endpoint."""
            self.logger.info(f"Shutting down node {self.config.node_id}")
            return {"message": f"Node {self.config.node_id} shutting down"}
    
    def run(self, log_level: str = "info"):
        """Run the node server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=log_level
        )
    
    async def start_async(self):
        """Start the server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Example of running a federated node."""
    # Configure node
    config = NodeConfig(
        node_id="node_1",
        host="localhost",
        port=8001,
        model_name="microsoft/DialoGPT-small"
    )
    
    # Create and initialize node
    node = FederatedNodeServer(config)
    await node.initialize()
    
    # Start server
    await node.start_async()


if __name__ == "__main__":
    asyncio.run(main())