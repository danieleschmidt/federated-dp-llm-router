"""
RoutingPolicy — classify queries into routing tiers.

Tiers
-----
LOCAL     : safe for on-device / local model (simple, non-sensitive)
CLOUD     : suitable for cloud API (complex, no PII detected)
SENSITIVE : must stay local or be refused (contains PII / regulated content)

Classification uses pattern matching (regex) + heuristics:
  1. PII detection  → SENSITIVE
  2. Topic sensitivity → SENSITIVE
  3. Query complexity (token count, question depth) → LOCAL vs CLOUD
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class RoutingTier(Enum):
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
    SENSITIVE = "SENSITIVE"


# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# PII patterns — presence → SENSITIVE
_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ssn",        re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card",re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("email",      re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone",      re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("dob",        re.compile(r"\b(?:born|dob|date of birth)[^\n]{0,20}\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", re.I)),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("passport",   re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
]

# Sensitive topic keywords → SENSITIVE
_SENSITIVE_TOPICS: List[re.Pattern] = [
    re.compile(r"\b(medical record|diagnosis|prescription|patient id)\b", re.I),
    re.compile(r"\b(password|secret key|api[ _]key|private key|token)\b", re.I),
    re.compile(r"\b(bank account|routing number|swift code|iban)\b", re.I),
    re.compile(r"\b(classified|top secret|confidential)\b", re.I),
    re.compile(r"\b(ssn|social security)\b", re.I),
]

# Complexity markers that push toward CLOUD
_COMPLEX_MARKERS: List[re.Pattern] = [
    re.compile(r"\b(explain|analyze|compare|summarize|evaluate|critique)\b", re.I),
    re.compile(r"\b(write a|generate|create|design|implement)\b", re.I),
    re.compile(r"\b(step.by.step|in detail|comprehensively)\b", re.I),
]

# Simple query markers that stay LOCAL
_SIMPLE_MARKERS: List[re.Pattern] = [
    re.compile(r"^(what|who|when|where|is|are|do|does|can|will)\b", re.I),
    re.compile(r"\b(define|spell|meaning of|synonym|antonym)\b", re.I),
]


@dataclass
class PolicyDecision:
    tier: RoutingTier
    reasons: List[str] = field(default_factory=list)
    pii_types_found: List[str] = field(default_factory=list)
    complexity_score: float = 0.0


class RoutingPolicy:
    """
    Classify a query string into a RoutingTier.

    Parameters
    ----------
    complexity_threshold : int
        Word count above which a clean query is sent to CLOUD (default 30).
    """

    def __init__(self, complexity_threshold: int = 30):
        self.complexity_threshold = complexity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> PolicyDecision:
        """Return a PolicyDecision for *query*."""
        reasons: List[str] = []
        pii_types: List[str] = []

        # 1. PII detection
        for name, pattern in _PII_PATTERNS:
            if pattern.search(query):
                pii_types.append(name)
                reasons.append(f"PII detected: {name}")

        # 2. Sensitive topics
        for pattern in _SENSITIVE_TOPICS:
            m = pattern.search(query)
            if m:
                reasons.append(f"Sensitive topic: '{m.group(0)}'")

        if pii_types or any("Sensitive topic" in r for r in reasons):
            return PolicyDecision(
                tier=RoutingTier.SENSITIVE,
                reasons=reasons,
                pii_types_found=pii_types,
                complexity_score=self._complexity(query),
            )

        # 3. Complexity → CLOUD vs LOCAL
        complexity = self._complexity(query)
        complex_hits = sum(1 for p in _COMPLEX_MARKERS if p.search(query))
        simple_hits  = sum(1 for p in _SIMPLE_MARKERS  if p.search(query))

        word_count = len(query.split())

        if complex_hits > 0 or word_count > self.complexity_threshold:
            reasons.append(
                f"Complex query (words={word_count}, complex_markers={complex_hits})"
            )
            return PolicyDecision(
                tier=RoutingTier.CLOUD,
                reasons=reasons,
                complexity_score=complexity,
            )

        reasons.append(
            f"Simple query (words={word_count}, simple_markers={simple_hits})"
        )
        return PolicyDecision(
            tier=RoutingTier.LOCAL,
            reasons=reasons,
            complexity_score=complexity,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _complexity(query: str) -> float:
        """Heuristic complexity score in [0, 1]."""
        words = query.split()
        n = len(words)
        # Normalised by a 200-word "ceiling"
        length_score = min(n / 200.0, 1.0)
        # Count unique words ratio
        vocab_ratio = len(set(w.lower() for w in words)) / max(n, 1)
        return round((length_score + vocab_ratio) / 2, 4)
