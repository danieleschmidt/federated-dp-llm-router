"""Integration tests for FederatedRouter."""

import unittest
from federated_dp_llm import FederatedRouter, RoutingTier, PrivacyBudgetTracker
from federated_dp_llm.sanitizer import DPQuerySanitizer
from federated_dp_llm.router import BackendEntry, _local_backend


class TestFederatedRouter(unittest.TestCase):

    def setUp(self):
        self.san = DPQuerySanitizer(epsilon=0.5, seed=99)
        self.budget = PrivacyBudgetTracker(default_limit=10.0)
        self.router = FederatedRouter(sanitizer=self.san, budget_tracker=self.budget)

    def test_simple_query_routes_local(self):
        result = self.router.route("What is Python?", user_id="u1")
        self.assertTrue(result.success)
        self.assertEqual(result.tier, RoutingTier.LOCAL)
        self.assertEqual(result.backend, "local-model")

    def test_complex_query_routes_cloud(self):
        result = self.router.route(
            "Explain the transformer architecture step-by-step in detail.", user_id="u2"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.tier, RoutingTier.CLOUD)

    def test_pii_query_routes_sensitive(self):
        result = self.router.route("My SSN is 999-88-7777", user_id="u3")
        self.assertTrue(result.success)
        self.assertEqual(result.tier, RoutingTier.SENSITIVE)
        self.assertEqual(result.backend, "private-model")

    def test_budget_exhausted_returns_error(self):
        budget = PrivacyBudgetTracker(default_limit=0.4)  # tiny limit
        router = FederatedRouter(sanitizer=self.san, budget_tracker=budget)
        # First call will exhaust budget (epsilon=0.5 > limit=0.4)
        result = router.route("What is Python?", user_id="cheap")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("exhausted", result.error.lower())

    def test_epsilon_consumed_accumulates(self):
        self.router.route("What is Python?", user_id="acc")
        self.router.route("Define overfitting.", user_id="acc")
        rec = self.budget.get_record("acc")
        self.assertAlmostEqual(rec.epsilon_used, 1.0)  # 2 × 0.5

    def test_result_has_request_id(self):
        result = self.router.route("Hello", user_id="u4")
        self.assertIsNotNone(result.request_id)
        self.assertTrue(len(result.request_id) > 0)

    def test_result_has_latency(self):
        result = self.router.route("Hello world", user_id="u5")
        self.assertGreater(result.latency_ms, 0.0)

    def test_update_backend(self):
        custom_entry = BackendEntry("custom", _local_backend, 1.0)
        self.router.update_backend(RoutingTier.CLOUD, custom_entry)
        result = self.router.route(
            "Write a comprehensive guide to machine learning.", user_id="u6"
        )
        self.assertEqual(result.backend, "custom")

    def test_anonymous_user_works(self):
        result = self.router.route("What time is it?")
        self.assertEqual(result.user_id, "anonymous")

    def test_metrics_populated_after_routing(self):
        router = FederatedRouter(sanitizer=self.san)
        router.route("What is Python?", user_id="m1")
        router.route("Analyze this in depth.", user_id="m1")
        summary = router.metrics.summary()
        self.assertEqual(summary["total_events"], 2)


if __name__ == "__main__":
    unittest.main()
