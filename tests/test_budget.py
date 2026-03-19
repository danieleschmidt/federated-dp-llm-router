"""Tests for PrivacyBudgetTracker."""

import threading
import unittest
from federated_dp_llm.budget import PrivacyBudgetTracker, BudgetExhaustedError


class TestPrivacyBudgetTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = PrivacyBudgetTracker(default_limit=5.0)

    def test_consume_reduces_budget(self):
        self.tracker.consume("alice", 1.0)
        rec = self.tracker.get_record("alice")
        self.assertAlmostEqual(rec.epsilon_used, 1.0)
        self.assertAlmostEqual(rec.epsilon_remaining, 4.0)

    def test_multiple_consumes_accumulate(self):
        self.tracker.consume("bob", 1.0)
        self.tracker.consume("bob", 2.0)
        rec = self.tracker.get_record("bob")
        self.assertAlmostEqual(rec.epsilon_used, 3.0)

    def test_budget_exhausted_raises(self):
        self.tracker.consume("carol", 4.9)
        with self.assertRaises(BudgetExhaustedError):
            self.tracker.consume("carol", 0.2)  # 4.9 + 0.2 > 5.0

    def test_check_budget_true_when_sufficient(self):
        self.assertTrue(self.tracker.check_budget("dave", 3.0))

    def test_check_budget_false_when_insufficient(self):
        self.tracker.consume("dave", 4.5)
        self.assertFalse(self.tracker.check_budget("dave", 1.0))

    def test_reset_clears_budget(self):
        self.tracker.consume("eve", 3.0)
        self.tracker.reset("eve")
        rec = self.tracker.get_record("eve")
        self.assertAlmostEqual(rec.epsilon_used, 0.0)
        self.assertEqual(rec.query_count, 0)

    def test_set_custom_limit(self):
        self.tracker.set_limit("frank", 20.0)
        rec = self.tracker.get_record("frank")
        self.assertAlmostEqual(rec.epsilon_limit, 20.0)

    def test_summary_returns_all_users(self):
        self.tracker.consume("u1", 1.0)
        self.tracker.consume("u2", 2.0)
        s = self.tracker.summary()
        self.assertIn("u1", s)
        self.assertIn("u2", s)

    def test_invalid_limit_raises(self):
        with self.assertRaises(ValueError):
            PrivacyBudgetTracker(default_limit=0)

    def test_thread_safety(self):
        """Concurrent consumes should not exceed budget."""
        tracker = PrivacyBudgetTracker(default_limit=5.0)
        errors = []
        successes = []

        def worker():
            try:
                tracker.consume("shared", 0.5)
                successes.append(1)
            except BudgetExhaustedError:
                errors.append(1)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rec = tracker.get_record("shared")
        # epsilon_used must not exceed limit
        self.assertLessEqual(rec.epsilon_used, 5.0 + 1e-9)
        # successes + exhaustions == 20
        self.assertEqual(len(successes) + len(errors), 20)


if __name__ == "__main__":
    unittest.main()
