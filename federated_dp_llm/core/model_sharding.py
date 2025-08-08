"""
Model Sharding for Distributed LLM Inference

Implements efficient distribution of large language models across multiple
nodes while maintaining privacy and enabling secure aggregation.
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging


class ShardingStrategy(Enum):
    """Model sharding strategies."""
    LAYER_WISE = "layer_wise"  # Distribute by transformer layers
    ATTENTION_SPLIT = "attention_split"  # Split attention heads
    TENSOR_PARALLEL = "tensor_parallel"  # Split tensors across dimensions
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline stages


@dataclass
class ShardMetadata:
    """Metadata for a model shard."""
    shard_id: str
    node_id: str
    shard_type: ShardingStrategy
    layer_range: Optional[Tuple[int, int]] = None
    attention_heads: Optional[List[int]] = None
    tensor_dims: Optional[Tuple[int, ...]] = None
    parameters_count: int = 0
    memory_footprint: int = 0  # in bytes
    dependencies: List[str] = None  # Other shards this depends on


@dataclass
class ModelMetadata:
    """Metadata for the complete sharded model."""
    model_name: str
    total_parameters: int
    num_shards: int
    sharding_strategy: ShardingStrategy
    shards: List[ShardMetadata]
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int


class ModelShard:
    """Represents a single shard of a distributed model."""
    
    def __init__(self, metadata: ShardMetadata, weights: Optional[Dict[str, torch.Tensor]] = None):
        self.metadata = metadata
        self.weights = weights or {}
        self.is_loaded = weights is not None
    
    def load_weights(self, weights_path: Path):
        """Load weights from disk."""
        if weights_path.suffix == '.pt':
            self.weights = torch.load(weights_path, map_location='cpu')
        elif weights_path.suffix == '.pkl':
            with open(weights_path, 'rb') as f:
                self.weights = pickle.load(f)
        else:
            raise ValueError(f"Unsupported weights format: {weights_path.suffix}")
        
        self.is_loaded = True
    
    def save_weights(self, weights_path: Path):
        """Save weights to disk."""
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        if weights_path.suffix == '.pt':
            torch.save(self.weights, weights_path)
        elif weights_path.suffix == '.pkl':
            with open(weights_path, 'wb') as f:
                pickle.dump(self.weights, f)
        else:
            raise ValueError(f"Unsupported weights format: {weights_path.suffix}")
    
    def get_memory_usage(self) -> int:
        """Calculate memory usage of loaded weights."""
        if not self.is_loaded:
            return 0
        
        total_bytes = 0
        for tensor in self.weights.values():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * tensor.element_size()
        
        return total_bytes
    
    def forward_pass(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Execute forward pass for this shard."""
        if not self.is_loaded:
            raise RuntimeError("Shard weights not loaded")
        
        # This is a simplified implementation - actual forward pass depends on model architecture
        if self.metadata.shard_type == ShardingStrategy.LAYER_WISE:
            return self._layer_wise_forward(input_tensor, **kwargs)
        elif self.metadata.shard_type == ShardingStrategy.ATTENTION_SPLIT:
            return self._attention_split_forward(input_tensor, **kwargs)
        else:
            raise NotImplementedError(f"Forward pass not implemented for {self.metadata.shard_type}")
    
    def _layer_wise_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for layer-wise sharding."""
        # Simplified transformer layer implementation
        for layer_name, layer_weights in self.weights.items():
            if 'attention' in layer_name:
                x = self._attention_forward(x, layer_weights)
            elif 'feed_forward' in layer_name:
                x = self._feed_forward(x, layer_weights)
        return x
    
    def _attention_split_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for attention head splitting."""
        # Simplified multi-head attention with split heads
        if 'attention_weights' in self.weights:
            attention_output = torch.matmul(x, self.weights['attention_weights'])
            return attention_output
        return x
    
    def _attention_forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Simplified attention computation."""
        return torch.matmul(x, weights)
    
    def _feed_forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Simplified feed-forward computation."""
        return torch.relu(torch.matmul(x, weights))


class ModelSharder:
    """Main class for model sharding operations."""
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.shards: Dict[str, ModelShard] = {}
        self.metadata: Optional[ModelMetadata] = None
        self.tokenizer = None
        self.config = None
        self.logger = logging.getLogger(__name__)
        
    def load_model_from_pretrained(self, model_path_or_name: str, device: str = "cpu") -> nn.Module:
        """Load a real model from HuggingFace or local path."""
        try:
            self.logger.info(f"Loading model: {model_path_or_name}")
            
            # Load config and tokenizer
            self.config = AutoConfig.from_pretrained(model_path_or_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
            
            # Load model
            model = AutoModel.from_pretrained(
                model_path_or_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
            self.model_name = model_path_or_name
            self.logger.info(f"Model loaded successfully: {model.config.model_type}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path_or_name}: {e}")
            raise
    
    def create_simple_shards(self, model: nn.Module, num_shards: int, node_ids: List[str]) -> ModelMetadata:
        """Create simple layer-wise shards for real models."""
        if len(node_ids) != num_shards:
            raise ValueError(f"Number of node_ids ({len(node_ids)}) must match num_shards ({num_shards})")
        
        # Get model layers
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layers = model.encoder.layer
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            self.logger.warning("Unknown model structure, using simplified sharding")
            layers = []
        
        total_layers = len(layers)
        layers_per_shard = max(1, total_layers // num_shards)
        
        shard_metadata = []
        self.shards = {}
        
        for i in range(num_shards):
            start_layer = i * layers_per_shard
            end_layer = min((i + 1) * layers_per_shard, total_layers)
            
            shard_id = f"shard_{i}"
            node_id = node_ids[i]
            
            # Extract weights for this shard
            shard_weights = {}
            param_count = 0
            
            if layers:
                for layer_idx in range(start_layer, end_layer):
                    if layer_idx < len(layers):
                        layer = layers[layer_idx]
                        for name, param in layer.named_parameters():
                            shard_weights[f"layer_{layer_idx}_{name}"] = param.detach().clone()
                            param_count += param.numel()
            
            # Create shard metadata
            metadata = ShardMetadata(
                shard_id=shard_id,
                node_id=node_id,
                shard_type=ShardingStrategy.LAYER_WISE,
                layer_range=(start_layer, end_layer),
                parameters_count=param_count,
                memory_footprint=sum(t.numel() * t.element_size() for t in shard_weights.values())
            )
            
            # Create shard
            shard = ModelShard(metadata, shard_weights)
            self.shards[shard_id] = shard
            shard_metadata.append(metadata)
        
        # Create model metadata
        total_params = sum(p.numel() for p in model.parameters())
        
        self.metadata = ModelMetadata(
            model_name=self.model_name,
            total_parameters=total_params,
            num_shards=num_shards,
            sharding_strategy=ShardingStrategy.LAYER_WISE,
            shards=shard_metadata,
            vocab_size=getattr(self.config, 'vocab_size', 50000),
            hidden_size=getattr(self.config, 'hidden_size', 768),
            num_layers=getattr(self.config, 'num_hidden_layers', total_layers),
            num_attention_heads=getattr(self.config, 'num_attention_heads', 12)
        )
        
        self.logger.info(f"Created {num_shards} shards with {total_params:,} total parameters")
        return self.metadata
    
    def inference_with_shards(self, text: str, max_length: int = 512) -> str:
        """Run distributed inference across shards."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model_from_pretrained first.")
        
        if not self.shards:
            raise RuntimeError("No shards available. Call create_simple_shards first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = inputs["input_ids"]
            
            # Simple distributed inference simulation
            # In a real implementation, this would coordinate across network nodes
            current_hidden_states = input_ids.float()  # Simplified
            
            for shard_id in sorted(self.shards.keys()):
                shard = self.shards[shard_id]
                if shard.is_loaded and shard.weights:
                    # Simulate processing through this shard
                    # Real implementation would send hidden states to remote node
                    current_hidden_states = self._simulate_shard_forward(current_hidden_states, shard)
            
            # Convert back to tokens (simplified)
            # Real implementation would use proper language model head
            output_text = self._decode_output(current_hidden_states)
            
            return output_text
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    def _simulate_shard_forward(self, hidden_states: torch.Tensor, shard: ModelShard) -> torch.Tensor:
        """Simulate forward pass through a shard."""
        # This is a simplified simulation
        # Real implementation would use actual transformer layers
        if shard.weights:
            # Apply a simple linear transformation
            weight_keys = list(shard.weights.keys())
            if weight_keys:
                first_weight = shard.weights[weight_keys[0]]
                if first_weight.dim() >= 2:
                    # Simple matrix multiplication as proxy for layer processing
                    try:
                        if hidden_states.size(-1) == first_weight.size(0):
                            return torch.matmul(hidden_states, first_weight.T)
                    except RuntimeError:
                        pass
        
        # Fallback: return input unchanged
        return hidden_states
    
    def _decode_output(self, hidden_states: torch.Tensor) -> str:
        """Convert hidden states back to text."""
        # Simplified decoding - real implementation would use language model head
        try:
            # Take the mean of hidden states and convert to token indices
            mean_states = hidden_states.mean(dim=-1).long()
            
            # Clamp to vocabulary size
            vocab_size = getattr(self.config, 'vocab_size', 50000)
            token_ids = torch.clamp(mean_states, 0, vocab_size - 1)
            
            # Decode tokens
            output_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            return output_text
            
        except Exception as e:
            self.logger.warning(f"Decoding failed: {e}")
            return "[Generated response - decoding simplified]" + str(hidden_states.shape)
    
    def shard_model_layer_wise_legacy(self, model: nn.Module, num_shards: int, node_ids: List[str]) -> ModelMetadata:
        """Shard model by distributing layers across nodes."""
        if len(node_ids) != num_shards:
            raise ValueError("Number of node IDs must match number of shards")
        
        # Get model structure
        model_dict = model.state_dict()
        layer_names = list(model_dict.keys())
        
        # Determine model properties
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate model structure (simplified)
        num_layers = len([name for name in layer_names if 'layer' in name or 'block' in name])
        hidden_size = 768  # Default - should be extracted from model
        num_attention_heads = 12  # Default - should be extracted from model
        vocab_size = 50000  # Default - should be extracted from model
        
        # Distribute layers across shards
        layers_per_shard = max(1, num_layers // num_shards)
        shard_metadata_list = []
        
        for i, node_id in enumerate(node_ids):
            start_layer = i * layers_per_shard
            end_layer = min((i + 1) * layers_per_shard, num_layers)
            
            # Extract weights for this shard
            shard_weights = {}
            shard_param_count = 0
            
            for name, param in model_dict.items():
                # Simple heuristic to assign layers to shards
                layer_idx = self._extract_layer_index(name)
                if layer_idx is not None and start_layer <= layer_idx < end_layer:
                    shard_weights[name] = param.clone()
                    shard_param_count += param.numel()
            
            # Create shard metadata
            shard_metadata = ShardMetadata(
                shard_id=f"shard_{i}",
                node_id=node_id,
                shard_type=ShardingStrategy.LAYER_WISE,
                layer_range=(start_layer, end_layer),
                parameters_count=shard_param_count,
                memory_footprint=sum(t.numel() * t.element_size() for t in shard_weights.values())
            )
            
            # Create shard
            shard = ModelShard(shard_metadata, shard_weights)
            self.shards[shard_metadata.shard_id] = shard
            shard_metadata_list.append(shard_metadata)
        
        # Create model metadata
        self.metadata = ModelMetadata(
            model_name=self.model_name,
            total_parameters=total_params,
            num_shards=num_shards,
            sharding_strategy=ShardingStrategy.LAYER_WISE,
            shards=shard_metadata_list,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads
        )
        
        return self.metadata
    
    def shard_model_attention_split(self, model: nn.Module, num_shards: int, node_ids: List[str]) -> ModelMetadata:
        """Shard model by splitting attention heads across nodes."""
        if len(node_ids) != num_shards:
            raise ValueError("Number of node IDs must match number of shards")
        
        model_dict = model.state_dict()
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate attention heads (simplified)
        num_attention_heads = 12  # Should be extracted from model config
        heads_per_shard = max(1, num_attention_heads // num_shards)
        
        shard_metadata_list = []
        
        for i, node_id in enumerate(node_ids):
            start_head = i * heads_per_shard
            end_head = min((i + 1) * heads_per_shard, num_attention_heads)
            head_indices = list(range(start_head, end_head))
            
            # Extract attention weights for these heads
            shard_weights = {}
            shard_param_count = 0
            
            for name, param in model_dict.items():
                if 'attention' in name.lower():
                    # Split attention weights by heads (simplified)
                    if len(param.shape) >= 2:
                        head_dim = param.shape[0] // num_attention_heads
                        start_idx = start_head * head_dim
                        end_idx = end_head * head_dim
                        shard_weights[name] = param[start_idx:end_idx].clone()
                        shard_param_count += shard_weights[name].numel()
            
            shard_metadata = ShardMetadata(
                shard_id=f"attention_shard_{i}",
                node_id=node_id,
                shard_type=ShardingStrategy.ATTENTION_SPLIT,
                attention_heads=head_indices,
                parameters_count=shard_param_count,
                memory_footprint=sum(t.numel() * t.element_size() for t in shard_weights.values())
            )
            
            shard = ModelShard(shard_metadata, shard_weights)
            self.shards[shard_metadata.shard_id] = shard
            shard_metadata_list.append(shard_metadata)
        
        self.metadata = ModelMetadata(
            model_name=self.model_name,
            total_parameters=total_params,
            num_shards=num_shards,
            sharding_strategy=ShardingStrategy.ATTENTION_SPLIT,
            shards=shard_metadata_list,
            vocab_size=50000,  # Default
            hidden_size=768,   # Default
            num_layers=12,     # Default
            num_attention_heads=num_attention_heads
        )
        
        return self.metadata
    
    def _extract_layer_index(self, parameter_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        import re
        
        # Common patterns for layer indices
        patterns = [
            r'layer\.(\d+)\.',
            r'blocks\.(\d+)\.',
            r'h\.(\d+)\.',
            r'transformer\.(\d+)\.'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, parameter_name)
            if match:
                return int(match.group(1))
        
        return None
    
    def save_sharded_model(self, output_dir: Path):
        """Save sharded model to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)
        
        # Save each shard
        for shard_id, shard in self.shards.items():
            shard_dir = output_dir / shard_id
            shard_dir.mkdir(exist_ok=True)
            
            # Save shard metadata
            shard_metadata_path = shard_dir / "metadata.json"
            with open(shard_metadata_path, 'w') as f:
                json.dump(asdict(shard.metadata), f, indent=2, default=str)
            
            # Save shard weights
            weights_path = shard_dir / "weights.pt"
            shard.save_weights(weights_path)
    
    def load_sharded_model(self, model_dir: Path):
        """Load sharded model from disk."""
        # Load metadata
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Reconstruct metadata (simplified - would need proper deserialization)
        self.metadata = ModelMetadata(**metadata_dict)
        
        # Load shards
        for shard_metadata in self.metadata.shards:
            shard_dir = model_dir / shard_metadata.shard_id
            weights_path = shard_dir / "weights.pt"
            
            shard = ModelShard(shard_metadata)
            shard.load_weights(weights_path)
            self.shards[shard_metadata.shard_id] = shard
    
    def get_shard(self, shard_id: str) -> Optional[ModelShard]:
        """Get a specific shard by ID."""
        return self.shards.get(shard_id)
    
    def get_shards_for_node(self, node_id: str) -> List[ModelShard]:
        """Get all shards assigned to a specific node."""
        return [
            shard for shard in self.shards.values()
            if shard.metadata.node_id == node_id
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.metadata:
            return {}
        
        total_memory = sum(shard.get_memory_usage() for shard in self.shards.values())
        
        return {
            "model_name": self.metadata.model_name,
            "total_parameters": self.metadata.total_parameters,
            "num_shards": self.metadata.num_shards,
            "sharding_strategy": self.metadata.sharding_strategy.value,
            "total_memory_usage": total_memory,
            "loaded_shards": len([s for s in self.shards.values() if s.is_loaded]),
            "shards_info": [
                {
                    "shard_id": shard.metadata.shard_id,
                    "node_id": shard.metadata.node_id,
                    "parameters": shard.metadata.parameters_count,
                    "memory_usage": shard.get_memory_usage(),
                    "is_loaded": shard.is_loaded
                }
                for shard in self.shards.values()
            ]
        }