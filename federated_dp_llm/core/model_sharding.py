"""
Model Sharding for Federated LLM Distribution

Implements model sharding strategies for distributing large language models
across federated nodes while maintaining privacy and efficiency.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Model sharding strategies for distributed deployment."""
    LAYER_WISE = "layer_wise"          # Distribute layers across nodes
    ATTENTION_HEAD = "attention_head"   # Split attention heads
    PARAMETER_WISE = "parameter_wise"   # Distribute parameters
    HYBRID = "hybrid"                   # Combination of strategies


@dataclass
class ModelShard:
    """Represents a shard of a federated model."""
    shard_id: str
    node_id: str
    layer_range: Tuple[int, int]
    parameters: Dict[str, Any]
    shard_size: int
    dependencies: List[str]
    
    def __post_init__(self):
        """Validate shard configuration."""
        if self.layer_range[0] >= self.layer_range[1]:
            raise ValueError(f"Invalid layer range: {self.layer_range}")


@dataclass
class ShardingConfig:
    """Configuration for model sharding."""
    strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE
    num_shards: int = 4
    overlap_layers: int = 0
    load_balancing: bool = True
    privacy_constraints: bool = True
    
    def validate(self):
        """Validate sharding configuration."""
        if self.num_shards < 1:
            raise ValueError("Number of shards must be positive")
        if self.overlap_layers < 0:
            raise ValueError("Overlap layers cannot be negative")


class ModelSharder:
    """Handles sharding of large language models for federated deployment."""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
        self.config.validate()
        self.shards: Dict[str, ModelShard] = {}
        self.model_metadata: Dict[str, Any] = {}
        
    async def shard_model(self, model_name: str, model_config: Dict[str, Any], 
                         available_nodes: List[str]) -> Dict[str, ModelShard]:
        """
        Shard a model across available nodes.
        
        Args:
            model_name: Name of the model to shard
            model_config: Model configuration including layer count, parameters
            available_nodes: List of available federated nodes
            
        Returns:
            Dictionary mapping shard IDs to ModelShard objects
        """
        logger.info(f"Sharding model {model_name} with strategy {self.config.strategy.value}")
        
        if len(available_nodes) < self.config.num_shards:
            raise ValueError(f"Not enough nodes ({len(available_nodes)}) for {self.config.num_shards} shards")
        
        # Extract model metadata
        num_layers = model_config.get('num_layers', 32)
        total_params = model_config.get('total_parameters', 7_000_000_000)
        
        self.model_metadata[model_name] = {
            'num_layers': num_layers,
            'total_parameters': total_params,
            'sharding_strategy': self.config.strategy.value
        }
        
        # Generate shards based on strategy
        shards = await self._generate_shards(model_name, model_config, available_nodes)
        
        # Store shards
        for shard in shards.values():
            self.shards[shard.shard_id] = shard
            
        logger.info(f"Successfully created {len(shards)} shards for {model_name}")
        return shards
    
    async def _generate_shards(self, model_name: str, model_config: Dict[str, Any], 
                             nodes: List[str]) -> Dict[str, ModelShard]:
        """Generate shards based on the configured strategy."""
        
        if self.config.strategy == ShardingStrategy.LAYER_WISE:
            return await self._layer_wise_sharding(model_name, model_config, nodes)
        elif self.config.strategy == ShardingStrategy.ATTENTION_HEAD:
            return await self._attention_head_sharding(model_name, model_config, nodes)
        elif self.config.strategy == ShardingStrategy.PARAMETER_WISE:
            return await self._parameter_wise_sharding(model_name, model_config, nodes)
        elif self.config.strategy == ShardingStrategy.HYBRID:
            return await self._hybrid_sharding(model_name, model_config, nodes)
        else:
            raise ValueError(f"Unknown sharding strategy: {self.config.strategy}")
    
    async def _layer_wise_sharding(self, model_name: str, model_config: Dict[str, Any],
                                 nodes: List[str]) -> Dict[str, ModelShard]:
        """Implement layer-wise sharding strategy."""
        num_layers = model_config.get('num_layers', 32)
        layers_per_shard = max(1, num_layers // self.config.num_shards)
        
        shards = {}
        for i in range(self.config.num_shards):
            start_layer = i * layers_per_shard
            end_layer = min((i + 1) * layers_per_shard + self.config.overlap_layers, num_layers)
            
            if start_layer >= num_layers:
                break
                
            shard_id = f"{model_name}_shard_{i}"
            node_id = nodes[i % len(nodes)]
            
            # Calculate shard size (simplified)
            shard_size = int((end_layer - start_layer) / num_layers * model_config.get('total_parameters', 7_000_000_000))
            
            # Determine dependencies
            dependencies = []
            if i > 0:
                dependencies.append(f"{model_name}_shard_{i-1}")
            
            shard = ModelShard(
                shard_id=shard_id,
                node_id=node_id,
                layer_range=(start_layer, end_layer),
                parameters={
                    'layers': list(range(start_layer, end_layer)),
                    'model_name': model_name,
                    'shard_index': i
                },
                shard_size=shard_size,
                dependencies=dependencies
            )
            
            shards[shard_id] = shard
            
        return shards
    
    async def _attention_head_sharding(self, model_name: str, model_config: Dict[str, Any],
                                     nodes: List[str]) -> Dict[str, ModelShard]:
        """Implement attention head sharding strategy."""
        num_heads = model_config.get('num_attention_heads', 32)
        num_layers = model_config.get('num_layers', 32)
        heads_per_shard = max(1, num_heads // self.config.num_shards)
        
        shards = {}
        for i in range(self.config.num_shards):
            start_head = i * heads_per_shard
            end_head = min((i + 1) * heads_per_shard, num_heads)
            
            if start_head >= num_heads:
                break
                
            shard_id = f"{model_name}_head_shard_{i}"
            node_id = nodes[i % len(nodes)]
            
            # Calculate shard size
            head_params = model_config.get('total_parameters', 7_000_000_000) / num_heads
            shard_size = int((end_head - start_head) * head_params)
            
            shard = ModelShard(
                shard_id=shard_id,
                node_id=node_id,
                layer_range=(0, num_layers),  # All layers, specific heads
                parameters={
                    'attention_heads': list(range(start_head, end_head)),
                    'model_name': model_name,
                    'shard_index': i
                },
                shard_size=shard_size,
                dependencies=[]  # Attention heads can be parallel
            )
            
            shards[shard_id] = shard
            
        return shards
    
    async def _parameter_wise_sharding(self, model_name: str, model_config: Dict[str, Any],
                                     nodes: List[str]) -> Dict[str, ModelShard]:
        """Implement parameter-wise sharding strategy."""
        total_params = model_config.get('total_parameters', 7_000_000_000)
        params_per_shard = total_params // self.config.num_shards
        
        shards = {}
        for i in range(self.config.num_shards):
            start_param = i * params_per_shard
            end_param = min((i + 1) * params_per_shard, total_params)
            
            shard_id = f"{model_name}_param_shard_{i}"
            node_id = nodes[i % len(nodes)]
            
            shard = ModelShard(
                shard_id=shard_id,
                node_id=node_id,
                layer_range=(0, model_config.get('num_layers', 32)),
                parameters={
                    'parameter_range': (start_param, end_param),
                    'model_name': model_name,
                    'shard_index': i
                },
                shard_size=int(end_param - start_param),
                dependencies=[]
            )
            
            shards[shard_id] = shard
            
        return shards
    
    async def _hybrid_sharding(self, model_name: str, model_config: Dict[str, Any],
                             nodes: List[str]) -> Dict[str, ModelShard]:
        """Implement hybrid sharding strategy combining multiple approaches."""
        # Use layer-wise for first half, attention-head for second half
        half_shards = self.config.num_shards // 2
        
        # Layer-wise shards
        layer_config = ShardingConfig(
            strategy=ShardingStrategy.LAYER_WISE,
            num_shards=half_shards,
            overlap_layers=self.config.overlap_layers
        )
        layer_sharder = ModelSharder(layer_config)
        layer_shards = await layer_sharder._layer_wise_sharding(
            f"{model_name}_layers", model_config, nodes[:half_shards]
        )
        
        # Attention head shards
        head_config = ShardingConfig(
            strategy=ShardingStrategy.ATTENTION_HEAD,
            num_shards=self.config.num_shards - half_shards
        )
        head_sharder = ModelSharder(head_config)
        head_shards = await head_sharder._attention_head_sharding(
            f"{model_name}_heads", model_config, nodes[half_shards:]
        )
        
        # Combine shards
        all_shards = {**layer_shards, **head_shards}
        return all_shards
    
    def get_shard_info(self, shard_id: str) -> Optional[ModelShard]:
        """Get information about a specific shard."""
        return self.shards.get(shard_id)
    
    def get_node_shards(self, node_id: str) -> List[ModelShard]:
        """Get all shards assigned to a specific node."""
        return [shard for shard in self.shards.values() if shard.node_id == node_id]
    
    def get_model_shards(self, model_name: str) -> List[ModelShard]:
        """Get all shards for a specific model."""
        return [shard for shard in self.shards.values() 
                if shard.parameters.get('model_name') == model_name]
    
    async def optimize_sharding(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize sharding based on performance metrics."""
        optimization_results = {
            'original_latency': performance_metrics.get('average_latency', 0),
            'memory_usage': performance_metrics.get('memory_usage', 0),
            'throughput': performance_metrics.get('throughput', 0)
        }
        
        # Simple optimization: if latency is high, suggest more shards
        if performance_metrics.get('average_latency', 0) > 500:  # 500ms threshold
            suggested_shards = min(self.config.num_shards * 2, 16)
            optimization_results['suggestion'] = f"Increase shards to {suggested_shards}"
        
        # If memory usage is high, suggest parameter-wise sharding
        if performance_metrics.get('memory_usage', 0) > 0.8:  # 80% memory usage
            optimization_results['suggestion'] = "Switch to parameter-wise sharding"
        
        return optimization_results
    
    def get_sharding_stats(self) -> Dict[str, Any]:
        """Get statistics about current sharding configuration."""
        if not self.shards:
            return {"total_shards": 0}
        
        node_distribution = {}
        total_size = 0
        
        for shard in self.shards.values():
            node_id = shard.node_id
            node_distribution[node_id] = node_distribution.get(node_id, 0) + 1
            total_size += shard.shard_size
        
        return {
            "total_shards": len(self.shards),
            "node_distribution": node_distribution,
            "total_size": total_size,
            "average_shard_size": total_size / len(self.shards) if self.shards else 0,
            "strategy": self.config.strategy.value
        }