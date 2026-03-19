"""Tests for RouterMetrics."""

import unittest
from federated_dp_llm.metrics import RouterMetrics


class TestRouterMetrics(unittest.TestCase):

    def setUp(self):
        self.m = RouterMetrics()
        self.m.record("local-model", "LOCAL",  10.0, 0.5, "alice")
        self.m.record("local-model", "LOCAL",  12.0, 0.5, "alice")
        self.m.record("cloud-api",   "CLOUD",  50.0, 0.5, "bob")
        self.m.record("cloud-api",   "CLOUD",  60.0, 0.5, "bob",  success=False)
        self.m.record("private-model","SENSITIVE", 25.0, 0.5, "carol")

    def test_total_events(self):
        self.assertEqual(self.m.summary()["total_events"], 5)

    def test_tier_distribution(self):
        dist = self.m.tier_distribution()
        self.assertEqual(dist["LOCAL"], 2)
        self.assertEqual(dist["CLOUD"], 2)
        self.assertEqual(dist["SENSITIVE"], 1)

    def test_latency_stats_local(self):
        stats = self.m.latency_stats()["local-model"]
        self.assertAlmostEqual(stats["mean_ms"], 11.0)
        self.assertAlmostEqual(stats["min_ms"], 10.0)
        self.assertAlmostEqual(stats["max_ms"], 12.0)

    def test_budget_consumption_per_user(self):
        consumption = self.m.budget_consumption()
        self.assertAlmostEqual(consumption["alice"], 1.0)
        self.assertAlmostEqual(consumption["bob"],   1.0)
        self.assertAlmostEqual(consumption["carol"], 0.5)

    def test_success_rate(self):
        sr = self.m.success_rate()
        self.assertAlmostEqual(sr["local-model"],    1.0)
        self.assertAlmostEqual(sr["cloud-api"],      0.5)
        self.assertAlmostEqual(sr["private-model"],  1.0)

    def test_empty_metrics_summary(self):
        m2 = RouterMetrics()
        s = m2.summary()
        self.assertEqual(s["total_events"], 0)
        self.assertEqual(s["tier_distribution"], {})


if __name__ == "__main__":
    unittest.main()
