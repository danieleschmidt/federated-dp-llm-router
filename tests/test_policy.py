"""Tests for RoutingPolicy."""

import unittest
from federated_dp_llm.policy import RoutingPolicy, RoutingTier


class TestRoutingPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = RoutingPolicy(complexity_threshold=20)

    # --- SENSITIVE ---

    def test_ssn_is_sensitive(self):
        d = self.policy.classify("My SSN is 123-45-6789")
        self.assertEqual(d.tier, RoutingTier.SENSITIVE)
        self.assertIn("ssn", d.pii_types_found)

    def test_email_is_sensitive(self):
        d = self.policy.classify("Contact me at foo@bar.com please")
        self.assertEqual(d.tier, RoutingTier.SENSITIVE)
        self.assertIn("email", d.pii_types_found)

    def test_credit_card_is_sensitive(self):
        d = self.policy.classify("My card number is 4111 1111 1111 1111")
        self.assertEqual(d.tier, RoutingTier.SENSITIVE)
        self.assertIn("credit_card", d.pii_types_found)

    def test_password_keyword_is_sensitive(self):
        d = self.policy.classify("Store my password securely")
        self.assertEqual(d.tier, RoutingTier.SENSITIVE)

    def test_medical_record_is_sensitive(self):
        d = self.policy.classify("Access patient ID 12345 medical record")
        self.assertEqual(d.tier, RoutingTier.SENSITIVE)

    # --- CLOUD ---

    def test_complex_verb_goes_to_cloud(self):
        d = self.policy.classify("Explain the transformer model architecture")
        self.assertEqual(d.tier, RoutingTier.CLOUD)

    def test_long_query_goes_to_cloud(self):
        long_q = "word " * 25  # 25 words > threshold 20
        d = self.policy.classify(long_q)
        self.assertEqual(d.tier, RoutingTier.CLOUD)

    def test_write_query_goes_to_cloud(self):
        d = self.policy.classify("Write a function to sort a list")
        self.assertEqual(d.tier, RoutingTier.CLOUD)

    # --- LOCAL ---

    def test_simple_definition_is_local(self):
        d = self.policy.classify("What is a variable?")
        self.assertEqual(d.tier, RoutingTier.LOCAL)

    def test_short_factual_is_local(self):
        d = self.policy.classify("Who invented Python?")
        self.assertEqual(d.tier, RoutingTier.LOCAL)

    def test_complexity_score_range(self):
        d = self.policy.classify("Hello")
        self.assertGreaterEqual(d.complexity_score, 0.0)
        self.assertLessEqual(d.complexity_score, 1.0)

    # --- Reasons populated ---

    def test_reasons_populated(self):
        d = self.policy.classify("Analyze the economy in detail")
        self.assertTrue(len(d.reasons) > 0)


if __name__ == "__main__":
    unittest.main()
