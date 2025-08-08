"""Core privacy and differential privacy components."""

from .privacy_accountant import PrivacyAccountant, DPConfig
from .secure_aggregation import SecureAggregator

# Conditional import for model sharding (requires torch)
try:
    from .model_sharding import ModelSharder, ShardingStrategy
    __all__ = ["PrivacyAccountant", "DPConfig", "SecureAggregator", "ModelSharder", "ShardingStrategy"]
except ImportError:
    __all__ = ["PrivacyAccountant", "DPConfig", "SecureAggregator"]