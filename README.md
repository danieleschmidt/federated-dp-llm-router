# federated-dp-llm-router

A **federated, differentially-private LLM request router** — routes queries to appropriate model backends while preserving user privacy.

```
Query → RoutingPolicy → DPQuerySanitizer → PrivacyBudgetTracker → FederatedRouter → Backend
                              ↑ Laplace noise + PII redaction          ↑ ε enforcement
```

## Why this exists

When routing queries to LLM backends (local model, cloud API, private server), you face two competing pressures:

1. **Correctness** — complex queries need capable (cloud) models; simple ones can stay local.
2. **Privacy** — queries may contain PII, medical data, credentials, or other sensitive content that must never reach a cloud API.

This router resolves that tension with a principled DP pipeline.

## Architecture

### `RoutingPolicy`
Classifies each query into one of three **routing tiers**:

| Tier | Description | Example |
|------|-------------|---------|
| `LOCAL` | Simple, non-sensitive — safe for on-device model | "What is a variable?" |
| `CLOUD` | Complex, no PII — suitable for cloud API | "Explain transformers in detail" |
| `SENSITIVE` | Contains PII or regulated content — stays private | "My SSN is 123-45-6789" |

Classification uses regex-based PII detection (SSN, email, credit cards, IPs, etc.), sensitive topic matching (medical records, passwords, credentials), and complexity heuristics (word count, action verbs).

### `DPQuerySanitizer`
Applies **Laplace-mechanism differential privacy** before routing:

- **Text redaction**: replaces PII tokens with `<REDACTED_TYPE>` placeholders
- **Feature noise**: adds calibrated Laplace noise to numerical features (word count, complexity score, question depth)
- Each call consumes `ε` from the privacy budget

Noise scale follows the Laplace mechanism: `noise ~ Lap(0, Δf/ε)` where `Δf` is the feature's global sensitivity.

### `PrivacyBudgetTracker`
Per-user **ε accounting** with hard limits:

- Uses sequential composition: `ε_total = Σ εᵢ`
- Raises `BudgetExhaustedError` when a user exceeds their limit
- Thread-safe; supports per-user limit overrides and session reset

### `FederatedRouter`
Orchestrates the full pipeline and maintains the **routing table** (tier → backend):

- Sanitises query, checks/consumes privacy budget, dispatches to appropriate backend
- Backends are pluggable callables (stubs provided; swap in real HTTP clients)
- Returns `RoutingResult` with full audit trail: tier, backend, ε spent, latency, redactions

### `RouterMetrics`
Collects telemetry per route event:
- Per-route latency (mean, min, max, p95)
- Tier distribution
- ε consumed per user
- Success rate per backend

## Quick start

```python
from federated_dp_llm import FederatedRouter

router = FederatedRouter()
result = router.route("Explain overfitting", user_id="alice")

print(result.tier)      # RoutingTier.CLOUD
print(result.backend)   # cloud-api
print(result.epsilon_spent)  # 0.5
```

## Demo

```bash
python demo.py
```

Output shows 10 queries across three users with routing decisions, tier distribution, latency stats, and privacy budget consumption.

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

48 tests covering all components. No external dependencies — stdlib only.

## Privacy guarantees

| Property | Value |
|----------|-------|
| Mechanism | Laplace |
| Composition | Sequential (basic) |
| Default ε per query | 0.5 |
| Default ε budget per user | 10.0 |
| PII redaction | SSN, email, credit card, phone, IP, passport, DOB |

**Note**: This implementation provides ε-DP on the **routing features**, not on the query content itself (which is redacted for PII but not DP-noised at the token level). For full text-level DP, consider mechanisms like SanText or DP-Forward.

## Structure

```
federated_dp_llm/
├── __init__.py       # public API
├── policy.py         # RoutingPolicy + tier classification
├── sanitizer.py      # DPQuerySanitizer (Laplace noise + PII redaction)
├── budget.py         # PrivacyBudgetTracker
├── router.py         # FederatedRouter + backend stubs
└── metrics.py        # RouterMetrics

tests/
├── test_policy.py
├── test_sanitizer.py
├── test_budget.py
├── test_metrics.py
└── test_router.py

demo.py               # end-to-end walkthrough
```

## License

MIT
