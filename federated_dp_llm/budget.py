"""
PrivacyBudgetTracker — per-user/session ε accounting.

Uses basic sequential composition: ε_total = Σ ε_i per user.
Raises BudgetExhaustedError when a user exceeds their limit.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List


class BudgetExhaustedError(Exception):
    """Raised when a user's privacy budget is exhausted."""


@dataclass
class BudgetRecord:
    user_id: str
    epsilon_used: float = 0.0
    epsilon_limit: float = 10.0
    query_count: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def epsilon_remaining(self) -> float:
        return max(0.0, self.epsilon_limit - self.epsilon_used)

    @property
    def exhausted(self) -> bool:
        return self.epsilon_used >= self.epsilon_limit


class PrivacyBudgetTracker:
    """
    Thread-safe per-user privacy budget tracker.

    Parameters
    ----------
    default_limit : float
        Default ε limit per user (default 10.0).
    """

    def __init__(self, default_limit: float = 10.0):
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.default_limit = default_limit
        self._lock = threading.Lock()
        self._records: Dict[str, BudgetRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_budget(self, user_id: str, epsilon_needed: float) -> bool:
        """Return True if the user has enough budget for *epsilon_needed*."""
        record = self._get_or_create(user_id)
        return record.epsilon_remaining >= epsilon_needed

    def consume(self, user_id: str, epsilon: float, query_type: str = "query") -> BudgetRecord:
        """
        Consume *epsilon* from *user_id*'s budget.

        Raises BudgetExhaustedError if the budget would be exceeded.
        """
        with self._lock:
            record = self._get_or_create(user_id)
            if record.epsilon_used + epsilon > record.epsilon_limit:
                raise BudgetExhaustedError(
                    f"User '{user_id}' budget exhausted: "
                    f"used={record.epsilon_used:.3f}, limit={record.epsilon_limit:.3f}, "
                    f"requested={epsilon:.3f}"
                )
            record.epsilon_used += epsilon
            record.query_count += 1
            record.history.append({
                "timestamp": time.time(),
                "epsilon": epsilon,
                "query_type": query_type,
                "total_after": record.epsilon_used,
            })
            return record

    def get_record(self, user_id: str) -> BudgetRecord:
        return self._get_or_create(user_id)

    def set_limit(self, user_id: str, epsilon_limit: float) -> None:
        """Override the ε limit for a specific user."""
        with self._lock:
            record = self._get_or_create(user_id)
            record.epsilon_limit = epsilon_limit

    def reset(self, user_id: str) -> None:
        """Reset a user's budget (e.g., new session / time window)."""
        with self._lock:
            if user_id in self._records:
                r = self._records[user_id]
                r.epsilon_used = 0.0
                r.query_count = 0
                r.history.clear()

    def summary(self) -> Dict[str, dict]:
        """Return a summary dict of all tracked users."""
        with self._lock:
            return {
                uid: {
                    "used": r.epsilon_used,
                    "limit": r.epsilon_limit,
                    "remaining": r.epsilon_remaining,
                    "queries": r.query_count,
                    "exhausted": r.exhausted,
                }
                for uid, r in self._records.items()
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, user_id: str) -> BudgetRecord:
        """Must be called with lock held (or from consume which holds it)."""
        if user_id not in self._records:
            self._records[user_id] = BudgetRecord(
                user_id=user_id,
                epsilon_limit=self.default_limit,
            )
        return self._records[user_id]
