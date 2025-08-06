"""
Federated Differential Privacy LLM Router

A production-ready system for serving privacy-budget-aware LLM shards across 
distributed healthcare institutions with differential privacy guarantees.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core.privacy_accountant import PrivacyAccountant, DPConfig
from .routing.load_balancer import FederatedRouter
from .federation.client import HospitalNode, PrivateInferenceClient
from .federation.server import FederatedTrainer
from .security.compliance import BudgetManager
from .quantum_planning import (
    QuantumTaskPlanner,
    QuantumState,
    TaskPriority,
    SuperpositionScheduler,
    TaskSuperposition,
    EntanglementOptimizer,
    ResourceEntanglement,
    InterferenceBalancer,
    TaskInterference
)

__all__ = [
    "PrivacyAccountant",
    "DPConfig", 
    "FederatedRouter",
    "HospitalNode",
    "PrivateInferenceClient", 
    "FederatedTrainer",
    "BudgetManager",
    "QuantumTaskPlanner",
    "QuantumState",
    "TaskPriority", 
    "SuperpositionScheduler",
    "TaskSuperposition",
    "EntanglementOptimizer",
    "ResourceEntanglement",
    "InterferenceBalancer",
    "TaskInterference",
]