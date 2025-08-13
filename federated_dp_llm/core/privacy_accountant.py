"""
Differential Privacy Accountant

Implements privacy budget tracking with (ε, δ)-DP guarantees using various
composition mechanisms including RDP (Rényi Differential Privacy).
"""

import math
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List, Tuple
import numpy as np
from abc import ABC, abstractmethod
from .storage import get_budget_storage, SimpleBudgetStorage
from ..security.enhanced_privacy_validator import get_privacy_validator, ValidationResult


class DPMechanism(Enum):
    """Supported differential privacy mechanisms."""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace" 
    EXPONENTIAL = "exponential"


class CompositionMethod(Enum):
    """Privacy composition methods."""
    BASIC = "basic"
    ADVANCED = "advanced"
    RDP = "rdp"  # Rényi Differential Privacy


@dataclass
class DPConfig:
    """Differential privacy configuration."""
    epsilon_per_query: float = 0.1
    delta: float = 1e-5
    max_budget_per_user: float = 10.0
    noise_multiplier: float = 1.1
    clip_norm: float = 1.0
    mechanism: DPMechanism = DPMechanism.GAUSSIAN
    composition: CompositionMethod = CompositionMethod.RDP


@dataclass
class PrivacySpend:
    """Record of privacy budget expenditure."""
    user_id: str
    epsilon: float
    delta: float
    timestamp: float
    query_type: str
    mechanism: DPMechanism


class PrivacyMechanism(ABC):
    """Abstract base class for privacy mechanisms."""
    
    @abstractmethod
    def add_noise(self, value: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """Add calibrated noise to maintain differential privacy."""
        pass
    
    @abstractmethod
    def get_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Calculate noise scale for given sensitivity and epsilon."""
        pass


class GaussianMechanism(PrivacyMechanism):
    """Gaussian noise mechanism for differential privacy."""
    
    def __init__(self, delta: float = 1e-5):
        self.delta = delta
    
    def add_noise(self, value: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """Add Gaussian noise calibrated for (ε, δ)-DP."""
        sigma = self.get_noise_scale(sensitivity, epsilon)
        noise = np.random.normal(0, sigma, value.shape)
        return value + noise
    
    def get_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Calculate Gaussian noise scale for (ε, δ)-DP."""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Standard Gaussian mechanism noise scale
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        return (sensitivity * c) / epsilon


class LaplaceMechanism(PrivacyMechanism):
    """Laplace noise mechanism for pure differential privacy."""
    
    def add_noise(self, value: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """Add Laplace noise calibrated for ε-DP."""
        scale = self.get_noise_scale(sensitivity, epsilon)
        noise = np.random.laplace(0, scale, value.shape)
        return value + noise
    
    def get_noise_scale(self, sensitivity: float, epsilon: float) -> float:
        """Calculate Laplace noise scale for ε-DP."""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        return sensitivity / epsilon


class RDPAccountant:
    """Rényi Differential Privacy accountant for tight composition."""
    
    def __init__(self, orders: List[float] = None):
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.orders = orders
        self.rdp_eps = {order: 0.0 for order in orders}
    
    def compose(self, sigma: float, steps: int = 1):
        """Compose RDP guarantee for Gaussian mechanism."""
        for order in self.orders:
            if order == 1.0:
                continue
            rdp_order = steps * order / (2 * sigma**2)
            self.rdp_eps[order] += rdp_order
    
    def get_privacy_spent(self, delta: float) -> Tuple[float, float]:
        """Convert RDP to (ε, δ)-DP."""
        min_eps = float('inf')
        
        for order in self.orders:
            if order == 1.0:
                continue
            eps = self.rdp_eps[order] + math.log(1/delta) / (order - 1)
            min_eps = min(min_eps, eps)
        
        return min_eps, delta


class PrivacyAccountant:
    """Main privacy accountant for tracking differential privacy budgets."""
    
    def __init__(self, config: DPConfig, storage: Optional[SimpleBudgetStorage] = None):
        self.config = config
        self.storage = storage or get_budget_storage()
        self.user_budgets: Dict[str, float] = {}
        self.privacy_history: List[PrivacySpend] = []
        self._lock = threading.RLock()
        
        # Initialize mechanism
        if config.mechanism == DPMechanism.GAUSSIAN:
            self.mechanism = GaussianMechanism(config.delta)
        elif config.mechanism == DPMechanism.LAPLACE:
            self.mechanism = LaplaceMechanism()
        else:
            raise ValueError(f"Unsupported mechanism: {config.mechanism}")
        
        # Initialize composition tracking
        if config.composition == CompositionMethod.RDP:
            self.rdp_accountant = RDPAccountant()
        
        # Load existing budgets from storage
        self._load_budgets_from_storage()
    
    def check_budget(self, user_id: str, requested_epsilon: float, 
                    department: str = "general", data_sensitivity: str = "medium",
                    user_role: str = "doctor", query_type: str = "inference") -> Tuple[bool, Optional[ValidationResult]]:
        """Check if user has sufficient privacy budget with enhanced validation."""
        # Enhanced privacy parameter validation
        validator = get_privacy_validator()
        validation_result = validator.validate_privacy_parameters(
            requested_epsilon, self.config.delta, self.config.noise_multiplier,
            department, data_sensitivity, user_role, query_type
        )
        
        if not validation_result.valid:
            return False, validation_result
            
        # Check budget limits
        current_budget = self.user_budgets.get(user_id, 0.0)
        budget_available = current_budget + requested_epsilon <= self.config.max_budget_per_user
        
        return budget_available, validation_result
    
    def spend_budget(self, user_id: str, epsilon: float, query_type: str = "inference",
                    department: str = "general", data_sensitivity: str = "medium",
                    user_role: str = "doctor") -> Tuple[bool, Optional[ValidationResult]]:
        """Spend privacy budget and record the transaction with enhanced validation."""
        with self._lock:
            budget_ok, validation_result = self.check_budget(
                user_id, epsilon, department, data_sensitivity, user_role, query_type
            )
            
            if not budget_ok:
                return False, validation_result
            
            # Record spend atomically
            self.user_budgets[user_id] = self.user_budgets.get(user_id, 0.0) + epsilon
            
            spend_record = PrivacySpend(
                user_id=user_id,
                epsilon=epsilon,
                delta=self.config.delta,
                timestamp=time.time(),
                query_type=query_type,
                mechanism=self.config.mechanism
            )
            self.privacy_history.append(spend_record)
            
            # Update composition tracking
            if self.config.composition == CompositionMethod.RDP:
                sigma = self.config.noise_multiplier
                self.rdp_accountant.compose(sigma, steps=1)
            
            return True, validation_result
    
    def get_remaining_budget(self, user_id: str) -> float:
        """Get remaining privacy budget for user."""
        spent = self.user_budgets.get(user_id, 0.0)
        return max(0.0, self.config.max_budget_per_user - spent)
    
    def add_noise_to_query(self, query_result: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """Add calibrated noise to query result."""
        return self.mechanism.add_noise(query_result, sensitivity, epsilon)
    
    def get_privacy_spent_total(self) -> Tuple[float, float]:
        """Get total privacy spent using composition."""
        if self.config.composition == CompositionMethod.RDP:
            return self.rdp_accountant.get_privacy_spent(self.config.delta)
        
        # Basic composition (worst case)
        total_epsilon = sum(spend.epsilon for spend in self.privacy_history)
        total_delta = len(self.privacy_history) * self.config.delta
        return total_epsilon, total_delta
    
    def reset_user_budget(self, user_id: str):
        """Reset privacy budget for a user (e.g., daily reset)."""
        with self._lock:
            self.user_budgets[user_id] = 0.0
            # Remove user's history for the reset period
            self.privacy_history = [
                spend for spend in self.privacy_history 
                if spend.user_id != user_id
            ]
    
    def get_user_history(self, user_id: str) -> List[PrivacySpend]:
        """Get privacy spending history for a user."""
        return [spend for spend in self.privacy_history if spend.user_id == user_id]
    
    def _load_budgets_from_storage(self):
        """Load existing budget records from persistent storage."""
        try:
            all_budgets = self.storage.get_all_budgets()
            for user_id, budget_record in all_budgets.items():
                self.user_budgets[user_id] = budget_record.spent_budget
        except Exception as e:
            print(f"Warning: Could not load budgets from storage: {e}")
    
    def can_query(self, user_id: str, epsilon: float) -> bool:
        """Check if user can make a query with given epsilon."""
        # Get budget from storage
        budget_record = self.storage.get_user_budget(user_id)
        if not budget_record:
            # Create new user with default budget
            self.storage.create_user_budget(user_id, "unknown", self.config.max_budget_per_user)
            return epsilon <= self.config.max_budget_per_user
        
        return budget_record.remaining_budget >= epsilon
    
    def deduct_budget(self, user_id: str, epsilon: float, request_id: Optional[str] = None) -> bool:
        """Deduct privacy budget from user account."""
        # Update storage
        success = self.storage.update_user_budget(user_id, epsilon, request_id)
        
        if success:
            # Update in-memory cache
            self.user_budgets[user_id] = self.user_budgets.get(user_id, 0.0) + epsilon
            
            # Add to history
            self.privacy_history.append(PrivacySpend(
                user_id=user_id,
                epsilon=epsilon,
                delta=self.config.delta,
                timestamp=time.time(),
                query_type="inference"
            ))
        
        return success
    
    def get_user_budget(self, user_id: str) -> float:
        """Get remaining privacy budget for user from storage."""
        budget_record = self.storage.get_user_budget(user_id)
        if not budget_record:
            # Create new user
            self.storage.create_user_budget(user_id, "unknown", self.config.max_budget_per_user)
            return self.config.max_budget_per_user
        
        return budget_record.remaining_budget
    
    def reset_user_budget_persistent(self, user_id: str) -> bool:
        """Reset user budget in persistent storage."""
        success = self.storage.reset_user_budget(user_id)
        
        if success:
            # Update in-memory cache
            self.user_budgets[user_id] = 0.0
            
            # Remove from history
            self.privacy_history = [
                spend for spend in self.privacy_history 
                if spend.user_id != user_id
            ]
        
        return success