#!/usr/bin/env python3
"""
demo.py — 10 queries with varying sensitivity.

Shows routing decisions and privacy budget consumption per user.
"""

from federated_dp_llm import (
    FederatedRouter,
    PrivacyBudgetTracker,
    RouterMetrics,
    RoutingPolicy,
)
from federated_dp_llm.sanitizer import DPQuerySanitizer

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

budget   = PrivacyBudgetTracker(default_limit=10.0)
policy   = RoutingPolicy(complexity_threshold=20)
sanitizer= DPQuerySanitizer(epsilon=0.5, seed=42)
metrics  = RouterMetrics()
router   = FederatedRouter(policy=policy, sanitizer=sanitizer,
                            budget_tracker=budget, metrics=metrics)

# ---------------------------------------------------------------------------
# 10 sample queries (varied sensitivity)
# ---------------------------------------------------------------------------

QUERIES = [
    # (query_text, user_id)
    ("What is the capital of France?",                         "alice"),
    ("Explain the transformer architecture in detail.",        "alice"),
    ("My SSN is 123-45-6789, can you help me verify it?",      "alice"),
    ("Define overfitting in machine learning.",                 "bob"),
    ("Write a comprehensive analysis of climate change policy.","bob"),
    ("My email is john.doe@example.com and I need help.",       "bob"),
    ("Is Python faster than JavaScript?",                       "carol"),
    ("Create a step-by-step guide to building a REST API.",     "carol"),
    ("Patient ID: P-993812, prescription renewal needed.",      "carol"),
    ("Summarize the French Revolution comprehensively.",        "alice"),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

SEP = "─" * 72

print(f"\n{'FEDERATED DP-LLM ROUTER — DEMO':^72}")
print(SEP)
print(f"{'#':<3}  {'User':<8}  {'Tier':<12}  {'Backend':<16}  {'ε spent':<10}  {'OK'}")
print(SEP)

for i, (query, user_id) in enumerate(QUERIES, 1):
    result = router.route(query, user_id=user_id)
    ok_marker = "✓" if result.success else "✗ BUDGET"
    print(
        f"{i:<3}  {user_id:<8}  {result.tier.value:<12}  "
        f"{result.backend:<16}  {result.epsilon_spent:<10.3f}  {ok_marker}"
    )
    if not result.success:
        print(f"      ERROR: {result.error}")
    # Show first reason
    if result.reasons:
        print(f"      ↳ {result.reasons[0]}")

print(SEP)

# ---------------------------------------------------------------------------
# Budget summary
# ---------------------------------------------------------------------------

print("\n📊 PRIVACY BUDGET SUMMARY")
print(SEP)
print(f"{'User':<10}  {'Used ε':<10}  {'Limit ε':<10}  {'Remaining':<10}  {'Queries'}")
print(SEP)
for uid, rec in budget.summary().items():
    print(
        f"{uid:<10}  {rec['used']:<10.3f}  {rec['limit']:<10.1f}  "
        f"{rec['remaining']:<10.3f}  {rec['queries']}"
    )

# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

summary = metrics.summary()
print(f"\n📈 ROUTING METRICS  (total events: {summary['total_events']})")
print(SEP)

print("Tier distribution:")
for tier, count in sorted(summary["tier_distribution"].items()):
    bar = "█" * count
    print(f"  {tier:<12} {bar} ({count})")

print("\nLatency (simulated) per backend:")
for route, stats in sorted(summary["latency_stats"].items()):
    print(f"  {route:<18}  mean={stats['mean_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms")

print("\nε consumed per user (recorded in metrics):")
for uid, eps in sorted(summary["budget_consumption"].items()):
    print(f"  {uid:<10}  ε={eps:.3f}")

print(SEP)
print("Demo complete.\n")
