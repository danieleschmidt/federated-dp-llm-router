"""Tests for DPQuerySanitizer."""

import math
import unittest
from federated_dp_llm.sanitizer import DPQuerySanitizer


class TestDPQuerySanitizer(unittest.TestCase):

    def setUp(self):
        # Seeded for reproducibility
        self.san = DPQuerySanitizer(epsilon=1.0, seed=0)

    def test_epsilon_spent_equals_config(self):
        result = self.san.sanitise("Hello world")
        self.assertAlmostEqual(result.epsilon_spent, 1.0)

    def test_pii_redacted_in_text(self):
        result = self.san.sanitise("My email is test@example.com")
        self.assertNotIn("test@example.com", result.sanitised_text)
        self.assertIn("<REDACTED_EMAIL>", result.sanitised_text)
        self.assertIn("email", result.redacted_types)

    def test_ssn_redacted(self):
        result = self.san.sanitise("SSN: 111-22-3333")
        self.assertNotIn("111-22-3333", result.sanitised_text)
        self.assertIn("ssn", result.redacted_types)

    def test_original_length_correct(self):
        query = "This is a test query with seven words here"
        result = self.san.sanitise(query)
        self.assertEqual(result.original_length, len(query.split()))

    def test_noisy_features_present(self):
        result = self.san.sanitise("Short query")
        self.assertIn("word_count", result.noisy_features)
        self.assertIn("complexity_score", result.noisy_features)
        self.assertIn("question_depth", result.noisy_features)

    def test_complexity_score_clamped(self):
        # Even with large noise the complexity score should stay in [0,1]
        san = DPQuerySanitizer(epsilon=0.001, seed=1)  # large noise
        for _ in range(20):
            result = san.sanitise("test")
            score = result.noisy_features["complexity_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_word_count_non_negative(self):
        san = DPQuerySanitizer(epsilon=0.001, seed=2)
        for _ in range(20):
            result = san.sanitise("hi")
            self.assertGreaterEqual(result.noisy_features["word_count"], 0.0)

    def test_no_redaction_clean_query(self):
        result = self.san.sanitise("What is the capital of France?")
        self.assertEqual(result.redacted_types, [])
        self.assertNotIn("REDACTED", result.sanitised_text)

    def test_invalid_epsilon_raises(self):
        with self.assertRaises(ValueError):
            DPQuerySanitizer(epsilon=-1.0)

    def test_laplace_noise_statistical(self):
        """With many samples the mean of Laplace(0, b) should be near 0."""
        san = DPQuerySanitizer(epsilon=1.0, seed=None)
        samples = [san._laplace_noise(1.0) for _ in range(2000)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 0.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
