"""
RouterMetrics — per-route latency, privacy budget consumption, tier distribution.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RouteEvent:
    route: str
    tier: str
    latency_ms: float
    epsilon_spent: float
    user_id: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True


class RouterMetrics:
    """
    Collects and summarises routing telemetry.

    All operations are thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._events: List[RouteEvent] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        route: str,
        tier: str,
        latency_ms: float,
        epsilon_spent: float,
        user_id: str,
        success: bool = True,
    ) -> None:
        with self._lock:
            self._events.append(
                RouteEvent(
                    route=route,
                    tier=tier,
                    latency_ms=latency_ms,
                    epsilon_spent=epsilon_spent,
                    user_id=user_id,
                    success=success,
                )
            )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def latency_stats(self) -> Dict[str, dict]:
        """Per-route latency statistics (mean, min, max, p95)."""
        with self._lock:
            by_route: Dict[str, List[float]] = defaultdict(list)
            for e in self._events:
                by_route[e.route].append(e.latency_ms)

        result = {}
        for route, latencies in by_route.items():
            sorted_l = sorted(latencies)
            n = len(sorted_l)
            p95_idx = max(0, int(0.95 * n) - 1)
            result[route] = {
                "count": n,
                "mean_ms": round(sum(sorted_l) / n, 2),
                "min_ms": round(sorted_l[0], 2),
                "max_ms": round(sorted_l[-1], 2),
                "p95_ms": round(sorted_l[p95_idx], 2),
            }
        return result

    def tier_distribution(self) -> Dict[str, int]:
        """Count of routing events per tier."""
        with self._lock:
            counts: Dict[str, int] = defaultdict(int)
            for e in self._events:
                counts[e.tier] += 1
        return dict(counts)

    def budget_consumption(self) -> Dict[str, float]:
        """Total ε consumed per user across all recorded events."""
        with self._lock:
            by_user: Dict[str, float] = defaultdict(float)
            for e in self._events:
                by_user[e.user_id] += e.epsilon_spent
        return {uid: round(v, 4) for uid, v in by_user.items()}

    def success_rate(self) -> Dict[str, float]:
        """Per-route success rate (0.0–1.0)."""
        with self._lock:
            total: Dict[str, int] = defaultdict(int)
            success: Dict[str, int] = defaultdict(int)
            for e in self._events:
                total[e.route] += 1
                if e.success:
                    success[e.route] += 1
        return {
            r: round(success[r] / total[r], 4) if total[r] else 0.0
            for r in total
        }

    def summary(self) -> dict:
        """Consolidated summary of all metrics."""
        return {
            "total_events": len(self._events),
            "tier_distribution": self.tier_distribution(),
            "latency_stats": self.latency_stats(),
            "budget_consumption": self.budget_consumption(),
            "success_rate": self.success_rate(),
        }
