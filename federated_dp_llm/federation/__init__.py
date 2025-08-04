"""Federation components for distributed learning and inference."""

from .client import HospitalNode, PrivateInferenceClient
from .server import FederatedTrainer
from .protocols import FederatedProtocol

__all__ = ["HospitalNode", "PrivateInferenceClient", "FederatedTrainer", "FederatedProtocol"]