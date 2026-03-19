"""
FederatedRouter — route sanitised queries to appropriate stub backends.

Architecture
------------
The router maintains a local *routing table* mapping RoutingTier → backend.
Backends are pluggable callables (stubs here).  The router:

  1. Sanitises the query via DPQuerySanitizer.
  2. Classifies via RoutingPolicy.
  3. Checks/consumes PrivacyBudget.
  4. Dispatches to the appropriate backend.
  5. Records metrics.

No real LLM calls are made; backends return stub responses.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .policy import RoutingPolicy, RoutingTier, PolicyDecision
from .sanitizer import DPQuerySanitizer, SanitisedQuery
from .budget import PrivacyBudgetTracker, BudgetExhaustedError
from .metrics import RouterMetrics


# ---------------------------------------------------------------------------
# Backend stubs
# ---------------------------------------------------------------------------

def _local_backend(sanitised: SanitisedQuery) -> str:
    """Stub: on-device / edge model."""
    return (
        f"[LOCAL MODEL] Processed {sanitised.noisy_features.get('word_count', '?'):.0f}~"
        f" words. (query redacted: {bool(sanitised.redacted_types)})"
    )


def _cloud_backend(sanitised: SanitisedQuery) -> str:
    """Stub: cloud LLM API (OpenAI/Anthropic-compatible)."""
    return (
        f"[CLOUD API] Response for ~{sanitised.noisy_features.get('word_count', '?'):.0f}"
        f"-word query. Complexity≈{sanitised.noisy_features.get('complexity_score', 0):.3f}"
    )


def _sensitive_backend(sanitised: SanitisedQuery) -> str:
    """Stub: sensitive routing — use private/on-prem model only."""
    redacted = ", ".join(sanitised.redacted_types) if sanitised.redacted_types else "topic"
    return (
        f"[PRIVATE MODEL] Handling sensitive query ({redacted} detected). "
        "Processing on isolated node."
    )


# ---------------------------------------------------------------------------
# Routing table entry
# ---------------------------------------------------------------------------

@dataclass
class BackendEntry:
    name: str
    handler: Callable[[SanitisedQuery], str]
    base_latency_ms: float   # simulated min latency
    jitter_ms: float = 5.0   # simulated jitter range


_DEFAULT_TABLE: Dict[RoutingTier, BackendEntry] = {
    RoutingTier.LOCAL:     BackendEntry("local-model",   _local_backend,     latency_ms := 8.0,  jitter_ms=3.0) if False else BackendEntry("local-model",     _local_backend,     8.0,  3.0),
    RoutingTier.CLOUD:     BackendEntry("cloud-api",     _cloud_backend,     45.0, 15.0),
    RoutingTier.SENSITIVE: BackendEntry("private-model", _sensitive_backend, 20.0,  5.0),
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    request_id: str
    user_id: str
    tier: RoutingTier
    backend: str
    response: str
    latency_ms: float
    epsilon_spent: float
    reasons: list = field(default_factory=list)
    sanitised_query: Optional[SanitisedQuery] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class FederatedRouter:
    """
    Routes queries through the full DP pipeline.

    Parameters
    ----------
    policy : RoutingPolicy | None
    sanitizer : DPQuerySanitizer | None
    budget_tracker : PrivacyBudgetTracker | None
    metrics : RouterMetrics | None
    routing_table : dict | None
        Override the default tier→backend mapping.
    """

    def __init__(
        self,
        policy: Optional[RoutingPolicy] = None,
        sanitizer: Optional[DPQuerySanitizer] = None,
        budget_tracker: Optional[PrivacyBudgetTracker] = None,
        metrics: Optional[RouterMetrics] = None,
        routing_table: Optional[Dict[RoutingTier, BackendEntry]] = None,
    ):
        self.policy        = policy        or RoutingPolicy()
        self.sanitizer     = sanitizer     or DPQuerySanitizer(epsilon=0.5)
        self.budget_tracker = budget_tracker or PrivacyBudgetTracker(default_limit=10.0)
        self.metrics       = metrics       or RouterMetrics()
        self.routing_table = routing_table or _DEFAULT_TABLE.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str, user_id: str = "anonymous") -> RoutingResult:
        """
        Full pipeline: classify → sanitise → budget check → dispatch.

        Always returns a RoutingResult; sets .error on failure.
        """
        request_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()

        # 1. Classify (on raw query for accurate PII detection)
        decision: PolicyDecision = self.policy.classify(query)
        tier = decision.tier

        # 2. Sanitise
        sanitised: SanitisedQuery = self.sanitizer.sanitise(
            query, complexity_score=decision.complexity_score
        )
        epsilon_needed = sanitised.epsilon_spent

        # 3. Budget check
        try:
            self.budget_tracker.consume(user_id, epsilon_needed, query_type=tier.value)
        except BudgetExhaustedError as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000
            result = RoutingResult(
                request_id=request_id,
                user_id=user_id,
                tier=tier,
                backend="none",
                response="",
                latency_ms=latency_ms,
                epsilon_spent=0.0,
                reasons=decision.reasons,
                error=str(exc),
            )
            self.metrics.record(
                route="budget-exhausted",
                tier=tier.value,
                latency_ms=latency_ms,
                epsilon_spent=0.0,
                user_id=user_id,
                success=False,
            )
            return result

        # 4. Dispatch to backend
        backend_entry = self.routing_table[tier]
        response = self._dispatch(backend_entry, sanitised)

        latency_ms = (time.perf_counter() - t_start) * 1000
        # Add simulated network/model latency on top of real wall-clock
        simulated_latency = backend_entry.base_latency_ms
        total_latency = round(latency_ms + simulated_latency, 2)

        # 5. Record metrics
        self.metrics.record(
            route=backend_entry.name,
            tier=tier.value,
            latency_ms=total_latency,
            epsilon_spent=epsilon_needed,
            user_id=user_id,
            success=True,
        )

        return RoutingResult(
            request_id=request_id,
            user_id=user_id,
            tier=tier,
            backend=backend_entry.name,
            response=response,
            latency_ms=total_latency,
            epsilon_spent=epsilon_needed,
            reasons=decision.reasons,
            sanitised_query=sanitised,
        )

    def update_backend(self, tier: RoutingTier, entry: BackendEntry) -> None:
        """Dynamically update the backend for a tier (e.g., A/B testing)."""
        self.routing_table[tier] = entry

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch(entry: BackendEntry, sanitised: SanitisedQuery) -> str:
        return entry.handler(sanitised)
