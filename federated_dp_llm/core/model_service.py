"""
Model Service for Real LLM Integration

Provides a simplified interface for loading and serving real models
with distributed sharding capabilities.
"""

import logging
import torch
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from transformers import pipeline, Pipeline

from .model_sharding import ModelSharder, ModelMetadata
from .privacy_accountant import PrivacyAccountant


@dataclass
class InferenceRequest:
    """Request for model inference."""
    text: str
    user_id: str
    max_length: int = 512
    temperature: float = 0.7
    privacy_budget: float = 0.1


@dataclass
class InferenceResponse:
    """Response from model inference."""
    generated_text: str
    privacy_cost: float
    processing_time: float
    shard_count: int
    model_name: str
    success: bool = True
    error: Optional[str] = None


class ModelService:
    """Main service for model loading and distributed inference."""
    
    def __init__(self, privacy_accountant: Optional[PrivacyAccountant] = None):
        self.logger = logging.getLogger(__name__)
        self.sharder = ModelSharder()
        self.privacy_accountant = privacy_accountant
        self.pipeline: Optional[Pipeline] = None
        self.is_loaded = False
        self.model_metadata: Optional[ModelMetadata] = None
        
    def load_model(self, model_name: str = "microsoft/DialoGPT-small", device: str = "cpu") -> bool:
        """Load a model for inference. Uses a small model by default for testing."""
        try:
            self.logger.info(f"Loading model service with {model_name}")
            
            # For simplicity, we'll use a pipeline for now
            # This can be extended to use the sharder for distributed inference
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if device != "cpu" and torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if device != "cpu" and torch.cuda.is_available() else torch.float32
            )
            
            # Also load through sharder for distributed capabilities
            model = self.sharder.load_model_from_pretrained(model_name, device)
            
            # Create simple shards (for demonstration)
            node_ids = ["node_1", "node_2"]  # Simple 2-node setup
            self.model_metadata = self.sharder.create_simple_shards(
                model, num_shards=2, node_ids=node_ids
            )
            
            self.is_loaded = True
            self.logger.info(f"Model service loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model service: {e}")
            self.is_loaded = False
            return False
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform privacy-aware inference."""
        start_time = asyncio.get_event_loop().time()
        
        if not self.is_loaded:
            return InferenceResponse(
                generated_text="",
                privacy_cost=0.0,
                processing_time=0.0,
                shard_count=0,
                model_name="none",
                success=False,
                error="Model not loaded"
            )
        
        try:
            # Check privacy budget
            if self.privacy_accountant:
                can_query = self.privacy_accountant.can_query(
                    request.user_id, request.privacy_budget
                )
                if not can_query:
                    return InferenceResponse(
                        generated_text="",
                        privacy_cost=0.0,
                        processing_time=0.0,
                        shard_count=0,
                        model_name=self.sharder.model_name,
                        success=False,
                        error="Privacy budget exceeded"
                    )
            
            # Perform inference
            generated_text = await self._run_inference(request)
            
            # Deduct privacy budget
            if self.privacy_accountant:
                self.privacy_accountant.deduct_budget(
                    request.user_id, request.privacy_budget
                )
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            return InferenceResponse(
                generated_text=generated_text,
                privacy_cost=request.privacy_budget,
                processing_time=processing_time,
                shard_count=len(self.sharder.shards),
                model_name=self.sharder.model_name,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            return InferenceResponse(
                generated_text="",
                privacy_cost=0.0,
                processing_time=processing_time,
                shard_count=0,
                model_name=self.sharder.model_name,
                success=False,
                error=str(e)
            )
    
    async def _run_inference(self, request: InferenceRequest) -> str:
        """Run the actual inference."""
        try:
            # Use the pipeline for now (faster and more reliable)
            if self.pipeline:
                result = self.pipeline(
                    request.text,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                
                if result and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    # Remove the input text from the response
                    if generated.startswith(request.text):
                        generated = generated[len(request.text):].strip()
                    return generated or "[No response generated]"
            
            # Fallback to distributed inference
            return self.sharder.inference_with_shards(request.text, request.max_length)
            
        except Exception as e:
            self.logger.warning(f"Pipeline inference failed, using fallback: {e}")
            # Fallback to simple response
            return f"[Model response to: {request.text[:50]}...]"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.sharder.model_name,
            "num_shards": len(self.sharder.shards),
            "metadata": self.model_metadata.__dict__ if self.model_metadata else None,
            "shard_info": {
                shard_id: {
                    "node_id": shard.metadata.node_id,
                    "parameters": shard.metadata.parameters_count,
                    "memory_mb": shard.metadata.memory_footprint / (1024 * 1024),
                    "layer_range": shard.metadata.layer_range
                }
                for shard_id, shard in self.sharder.shards.items()
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the model service."""
        return {
            "status": "healthy" if self.is_loaded else "unhealthy",
            "model_loaded": self.is_loaded,
            "model_name": self.sharder.model_name if self.is_loaded else None,
            "shard_count": len(self.sharder.shards),
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


# Global model service instance
_model_service: Optional[ModelService] = None


def get_model_service(privacy_accountant: Optional[PrivacyAccountant] = None) -> ModelService:
    """Get or create the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService(privacy_accountant)
    return _model_service