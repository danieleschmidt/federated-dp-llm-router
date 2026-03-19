"""
DPQuerySanitizer — apply differential privacy before routing.

Strategy
--------
* Numerical features extracted from the query (word count, question depth,
  complexity score) receive **Laplace noise** calibrated to sensitivity/ε.
* Tokens matching PII patterns are **redacted** (replaced with <REDACTED_TYPE>).
* The sanitised query text is returned alongside noisy feature values.

The Laplace mechanism: noise ~ Lap(0, Δf/ε)
  Δf = global sensitivity of the feature (we use max observed range).
  ε  = privacy budget consumed by this sanitisation step.
"""

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List

from .policy import _PII_PATTERNS


# ---------------------------------------------------------------------------
# Token-level redaction patterns (applied to query text)
# ---------------------------------------------------------------------------

_REDACT_PATTERNS: List[tuple] = _PII_PATTERNS  # reuse policy patterns


@dataclass
class SanitisedQuery:
    original_length: int            # word count before sanitisation
    sanitised_text: str             # text with PII replaced
    noisy_features: Dict[str, float] = field(default_factory=dict)
    epsilon_spent: float = 0.0
    redacted_types: List[str] = field(default_factory=list)


class DPQuerySanitizer:
    """
    Apply Laplace-mechanism differential privacy to query features.

    Parameters
    ----------
    epsilon : float
        Privacy budget to spend per sanitisation call (default 0.5).
    delta_sensitivity : float
        Global sensitivity for the Laplace mechanism (default 1.0).
    seed : int | None
        Seed for reproducible noise (default None = non-deterministic).
    """

    # Feature sensitivities (Δf) — conservative upper bounds
    _FEATURE_SENSITIVITY: Dict[str, float] = {
        "word_count":        200.0,   # max reasonable query length
        "complexity_score":    1.0,   # always in [0, 1]
        "question_depth":     10.0,   # number of nested clauses / punctuation
    }

    def __init__(
        self,
        epsilon: float = 0.5,
        delta_sensitivity: float = 1.0,
        seed: int | None = None,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.delta_sensitivity = delta_sensitivity
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitise(self, query: str, complexity_score: float = 0.0) -> SanitisedQuery:
        """
        Redact PII from *query* and add Laplace noise to numerical features.

        Returns a SanitisedQuery with noisy feature values.
        """
        words = query.split()
        original_length = len(words)

        # Step 1: text-level redaction
        sanitised_text, redacted_types = self._redact_pii(query)

        # Step 2: extract true features
        true_features = {
            "word_count":     float(original_length),
            "complexity_score": complexity_score,
            "question_depth":  self._question_depth(query),
        }

        # Step 3: add Laplace noise
        noisy_features = {}
        for feature, true_val in true_features.items():
            sensitivity = self._FEATURE_SENSITIVITY[feature]
            scale = sensitivity / self.epsilon          # Lap scale = Δf / ε
            noise = self._laplace_noise(scale)
            # Clip to valid range
            noisy_val = true_val + noise
            if feature == "complexity_score":
                noisy_val = max(0.0, min(1.0, noisy_val))
            elif feature in ("word_count", "question_depth"):
                noisy_val = max(0.0, noisy_val)
            noisy_features[feature] = round(noisy_val, 4)

        return SanitisedQuery(
            original_length=original_length,
            sanitised_text=sanitised_text,
            noisy_features=noisy_features,
            epsilon_spent=self.epsilon,
            redacted_types=redacted_types,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _redact_pii(self, text: str) -> tuple[str, List[str]]:
        """Replace PII tokens with <REDACTED_TYPE> placeholders."""
        redacted_types: List[str] = []
        result = text
        for name, pattern in _REDACT_PATTERNS:
            def _replace(m, _name=name):
                return f"<REDACTED_{_name.upper()}>"
            new_result, n_subs = pattern.subn(_replace, result)
            if n_subs:
                result = new_result
                redacted_types.append(name)
        return result, redacted_types

    def _question_depth(self, text: str) -> float:
        """Proxy for syntactic depth: count punctuation / clause markers."""
        markers = re.findall(r"[,;:\(\)\[\]]|(?:\band\b|\bor\b|\bbut\b|\bif\b|\bwhen\b)", text, re.I)
        return float(len(markers))

    def _laplace_noise(self, scale: float) -> float:
        """Sample from Laplace(0, scale) using inverse CDF."""
        u = self._rng.uniform(-0.5, 0.5)
        # Lap(0, b): F^{-1}(u) = -b * sign(u - 0.5) * ln(1 - 2|u - 0.5|)
        # Equivalent: X = -b * sign(u) * ln(1 - 2|u|)  where u ~ U(-0.5, 0.5)
        if u == 0:
            return 0.0
        sign = 1.0 if u > 0 else -1.0
        return -scale * sign * math.log(1.0 - 2.0 * abs(u))
