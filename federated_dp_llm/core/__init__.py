```python
"""Core privacy and differential privacy components."""

from .privacy_accountant import PrivacyAccountant, DPConfig
from .secure_aggregation import SecureAggregator

# Conditional imports for optional dependencies
__all__ = ["PrivacyAccountant", "DPConfig", "SecureAggregator"]

# Model sharding (requires torch)
try:
    from .model_sharding import ModelSharder, ShardingStrategy
    __all__.extend(["ModelSharder", "ShardingStrategy"])
except ImportError:
    pass

# Model service components
try:
    from .model_service import ModelService, InferenceRequest, InferenceResponse, get_model_service
    __all__.extend(["ModelService", "InferenceRequest", "InferenceResponse", "get_model_service"])
except ImportError:
    pass
```
