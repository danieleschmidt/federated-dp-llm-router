"""
Tests for privacy accountant and differential privacy mechanisms.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from federated_dp_llm.core.privacy_accountant import (
    PrivacyAccountant, DPConfig, DPMechanism, CompositionMethod,
    GaussianMechanism, LaplaceMechanism, RDPAccountant, PrivacySpend
)


class TestDPConfig:
    """Tests for DP configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DPConfig()
        assert config.epsilon_per_query == 0.1
        assert config.delta == 1e-5
        assert config.max_budget_per_user == 10.0
        assert config.mechanism == DPMechanism.GAUSSIAN
        assert config.composition == CompositionMethod.RDP
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DPConfig(
            epsilon_per_query=0.5,
            delta=1e-4,
            max_budget_per_user=20.0,
            mechanism=DPMechanism.LAPLACE,
            composition=CompositionMethod.BASIC
        )
        assert config.epsilon_per_query == 0.5
        assert config.delta == 1e-4
        assert config.max_budget_per_user == 20.0
        assert config.mechanism == DPMechanism.LAPLACE
        assert config.composition == CompositionMethod.BASIC


class TestGaussianMechanism:
    """Tests for Gaussian mechanism."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        mechanism = GaussianMechanism(delta=1e-5)
        assert mechanism.delta == 1e-5
    
    def test_noise_scale_calculation(self):
        """Test noise scale calculation."""
        mechanism = GaussianMechanism(delta=1e-5)
        
        sensitivity = 1.0
        epsilon = 1.0
        scale = mechanism.get_noise_scale(sensitivity, epsilon)
        
        assert scale > 0
        assert isinstance(scale, float)
    
    def test_noise_scale_epsilon_validation(self):
        """Test epsilon validation in noise scale calculation."""
        mechanism = GaussianMechanism()
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            mechanism.get_noise_scale(1.0, 0.0)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            mechanism.get_noise_scale(1.0, -1.0)
    
    @pytest.mark.privacy
    def test_add_noise(self, privacy_test_data):
        """Test noise addition."""
        mechanism = GaussianMechanism(delta=1e-5)
        
        for value in privacy_test_data["plaintext_values"]:
            for epsilon in privacy_test_data["epsilon_values"]:
                noisy_value = mechanism.add_noise(
                    np.array([value]), 
                    privacy_test_data["sensitivity"], 
                    epsilon
                )
                
                assert isinstance(noisy_value, np.ndarray)
                assert noisy_value.shape == (1,)
                # Value should be different (with high probability)
                assert noisy_value[0] != value or epsilon > 10  # High epsilon = low noise
    
    def test_add_noise_array(self):
        """Test noise addition to arrays."""
        mechanism = GaussianMechanism()
        
        data = np.array([1.0, 2.0, 3.0, 4.0])
        noisy_data = mechanism.add_noise(data, sensitivity=1.0, epsilon=1.0)
        
        assert noisy_data.shape == data.shape
        assert isinstance(noisy_data, np.ndarray)


class TestLaplaceMechanism:
    """Tests for Laplace mechanism."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        mechanism = LaplaceMechanism()
        assert mechanism is not None
    
    def test_noise_scale_calculation(self):
        """Test Laplace noise scale calculation."""
        mechanism = LaplaceMechanism()
        
        sensitivity = 1.0
        epsilon = 1.0
        scale = mechanism.get_noise_scale(sensitivity, epsilon)
        
        expected_scale = sensitivity / epsilon
        assert scale == expected_scale
    
    @pytest.mark.privacy
    def test_add_noise(self, privacy_test_data):
        """Test Laplace noise addition."""
        mechanism = LaplaceMechanism()
        
        for value in privacy_test_data["plaintext_values"]:
            for epsilon in privacy_test_data["epsilon_values"]:
                noisy_value = mechanism.add_noise(
                    np.array([value]),
                    privacy_test_data["sensitivity"],
                    epsilon
                )
                
                assert isinstance(noisy_value, np.ndarray)
                assert noisy_value.shape == (1,)


class TestRDPAccountant:
    """Tests for RDP accountant."""
    
    def test_initialization(self):
        """Test RDP accountant initialization."""
        accountant = RDPAccountant()
        assert len(accountant.orders) > 0
        assert all(order >= 1.0 for order in accountant.orders)
        assert all(accountant.rdp_eps[order] == 0.0 for order in accountant.orders)
    
    def test_custom_orders(self):
        """Test RDP accountant with custom orders."""
        custom_orders = [1.5, 2.0, 3.0, 5.0]
        accountant = RDPAccountant(orders=custom_orders)
        assert accountant.orders == custom_orders
    
    def test_compose(self):
        """Test RDP composition."""
        accountant = RDPAccountant(orders=[2.0, 3.0])
        
        # Initial privacy spent should be zero
        epsilon, delta = accountant.get_privacy_spent(delta=1e-5)
        assert epsilon == 0.0
        
        # After composition, privacy should be non-zero
        accountant.compose(sigma=1.0, steps=1)
        epsilon_after, delta_after = accountant.get_privacy_spent(delta=1e-5)
        
        assert epsilon_after > 0.0
        assert delta_after == 1e-5
    
    def test_multiple_compositions(self):
        """Test multiple RDP compositions."""
        accountant = RDPAccountant()
        
        accountant.compose(sigma=1.0, steps=1)
        epsilon_1, _ = accountant.get_privacy_spent(delta=1e-5)
        
        accountant.compose(sigma=1.0, steps=1) 
        epsilon_2, _ = accountant.get_privacy_spent(delta=1e-5)
        
        # Privacy cost should increase with more compositions
        assert epsilon_2 > epsilon_1


class TestPrivacyAccountant:
    """Tests for main privacy accountant."""
    
    def test_initialization(self, dp_config):
        """Test privacy accountant initialization."""
        accountant = PrivacyAccountant(dp_config)
        
        assert accountant.config == dp_config
        assert len(accountant.user_budgets) == 0
        assert len(accountant.privacy_history) == 0
        assert accountant.mechanism is not None
    
    def test_unsupported_mechanism(self):
        """Test initialization with unsupported mechanism."""
        config = DPConfig(mechanism=DPMechanism.EXPONENTIAL)
        
        with pytest.raises(ValueError, match="Unsupported mechanism"):
            PrivacyAccountant(config)
    
    def test_check_budget_new_user(self, privacy_accountant):
        """Test budget check for new user."""
        user_id = "new_user"
        requested_epsilon = 0.5
        
        can_spend = privacy_accountant.check_budget(user_id, requested_epsilon)
        assert can_spend is True
    
    def test_check_budget_insufficient(self, privacy_accountant):
        """Test budget check with insufficient budget."""
        user_id = "test_user"
        
        # Spend most of the budget
        privacy_accountant.spend_budget(user_id, 9.5)
        
        # Try to spend more than remaining
        can_spend = privacy_accountant.check_budget(user_id, 1.0)
        assert can_spend is False
    
    def test_spend_budget_success(self, privacy_accountant):
        """Test successful budget spending."""
        user_id = "test_user"
        epsilon = 0.5
        
        success = privacy_accountant.spend_budget(user_id, epsilon, "test_query")
        
        assert success is True
        assert privacy_accountant.user_budgets[user_id] == epsilon
        assert len(privacy_accountant.privacy_history) == 1
        
        # Check history record
        history_record = privacy_accountant.privacy_history[0]
        assert history_record.user_id == user_id
        assert history_record.epsilon == epsilon
        assert history_record.query_type == "test_query"
    
    def test_spend_budget_insufficient(self, privacy_accountant):
        """Test budget spending with insufficient budget."""
        user_id = "test_user"
        
        # Spend all budget
        privacy_accountant.spend_budget(user_id, 10.0)
        
        # Try to spend more
        success = privacy_accountant.spend_budget(user_id, 0.1)
        assert success is False
    
    def test_get_remaining_budget(self, privacy_accountant):
        """Test remaining budget calculation."""
        user_id = "test_user"
        
        # Initially full budget
        remaining = privacy_accountant.get_remaining_budget(user_id)
        assert remaining == privacy_accountant.config.max_budget_per_user
        
        # After spending
        privacy_accountant.spend_budget(user_id, 3.0)
        remaining = privacy_accountant.get_remaining_budget(user_id)
        assert remaining == privacy_accountant.config.max_budget_per_user - 3.0
    
    def test_get_remaining_budget_unknown_user(self, privacy_accountant):
        """Test remaining budget for unknown user."""
        remaining = privacy_accountant.get_remaining_budget("unknown_user")
        assert remaining == privacy_accountant.config.max_budget_per_user
    
    def test_add_noise_to_query(self, privacy_accountant):
        """Test adding noise to query results."""
        query_result = np.array([1.0, 2.0, 3.0])
        sensitivity = 1.0
        epsilon = 1.0
        
        noisy_result = privacy_accountant.add_noise_to_query(
            query_result, sensitivity, epsilon
        )
        
        assert isinstance(noisy_result, np.ndarray)
        assert noisy_result.shape == query_result.shape
        # Results should be different (with high probability)
        assert not np.array_equal(noisy_result, query_result) or epsilon > 10
    
    def test_get_privacy_spent_total_basic(self):
        """Test total privacy spent calculation with basic composition."""
        config = DPConfig(composition=CompositionMethod.BASIC)
        accountant = PrivacyAccountant(config)
        
        # Spend some budget
        accountant.spend_budget("user1", 0.5)
        accountant.spend_budget("user2", 0.3)
        
        total_epsilon, total_delta = accountant.get_privacy_spent_total()
        
        assert total_epsilon == 0.8  # 0.5 + 0.3
        assert total_delta == 2 * config.delta  # 2 queries
    
    def test_reset_user_budget(self, privacy_accountant):
        """Test user budget reset."""
        user_id = "test_user"
        
        # Spend some budget
        privacy_accountant.spend_budget(user_id, 5.0)
        assert privacy_accountant.user_budgets[user_id] == 5.0
        
        # Reset budget
        privacy_accountant.reset_user_budget(user_id)
        assert privacy_accountant.user_budgets[user_id] == 0.0
        
        # History should be cleared for this user
        user_history = privacy_accountant.get_user_history(user_id)
        assert len(user_history) == 0
    
    def test_get_user_history(self, privacy_accountant):
        """Test user history retrieval."""
        user_id = "test_user"
        
        # Spend budget multiple times
        privacy_accountant.spend_budget(user_id, 1.0, "query1")
        privacy_accountant.spend_budget(user_id, 2.0, "query2")
        privacy_accountant.spend_budget("other_user", 1.0, "query3")
        
        history = privacy_accountant.get_user_history(user_id)
        
        assert len(history) == 2
        assert all(record.user_id == user_id for record in history)
        assert history[0].query_type == "query1"
        assert history[1].query_type == "query2"
    
    @pytest.mark.privacy
    def test_privacy_guarantees(self, privacy_accountant, test_utils):
        """Test that privacy guarantees are maintained."""
        # This is a simplified test - in practice would use more rigorous DP testing
        sensitivity = 1.0
        epsilon = 1.0
        
        true_values = np.array([1.0, 5.0, -2.0, 0.0])
        
        for true_value in true_values:
            noisy_value = privacy_accountant.add_noise_to_query(
                np.array([true_value]), sensitivity, epsilon
            )[0]
            
            # Check that noise is within reasonable bounds
            test_utils.assert_privacy_bounds(
                noisy_value, true_value, sensitivity, epsilon
            )
    
    @pytest.mark.slow
    def test_composition_accuracy(self):
        """Test accuracy of composition methods."""
        # Compare basic vs RDP composition
        basic_config = DPConfig(composition=CompositionMethod.BASIC)
        rdp_config = DPConfig(composition=CompositionMethod.RDP)
        
        basic_accountant = PrivacyAccountant(basic_config)
        rdp_accountant = PrivacyAccountant(rdp_config)
        
        # Perform same operations
        num_queries = 10
        epsilon_per_query = 0.1
        
        for i in range(num_queries):
            basic_accountant.spend_budget(f"user_{i}", epsilon_per_query)
            rdp_accountant.spend_budget(f"user_{i}", epsilon_per_query)
        
        basic_total, _ = basic_accountant.get_privacy_spent_total()
        rdp_total, _ = rdp_accountant.get_privacy_spent_total()
        
        # RDP should give tighter bounds (lower total epsilon)
        assert rdp_total <= basic_total
        
        # Basic composition should be num_queries * epsilon_per_query
        assert basic_total == num_queries * epsilon_per_query


class TestPrivacySpend:
    """Tests for privacy spend records."""
    
    def test_privacy_spend_creation(self):
        """Test privacy spend record creation."""
        timestamp = time.time()
        spend = PrivacySpend(
            user_id="test_user",
            epsilon=0.1,
            delta=1e-5,
            timestamp=timestamp,
            query_type="inference",
            mechanism=DPMechanism.GAUSSIAN
        )
        
        assert spend.user_id == "test_user"
        assert spend.epsilon == 0.1
        assert spend.delta == 1e-5
        assert spend.timestamp == timestamp
        assert spend.query_type == "inference"
        assert spend.mechanism == DPMechanism.GAUSSIAN


@pytest.mark.integration
class TestPrivacyAccountantIntegration:
    """Integration tests for privacy accountant."""
    
    @pytest.mark.asyncio
    async def test_concurrent_budget_spending(self, privacy_accountant):
        """Test concurrent budget spending."""
        import asyncio
        
        user_id = "concurrent_user"
        
        async def spend_budget():
            return privacy_accountant.spend_budget(user_id, 1.0)
        
        # Try to spend budget concurrently
        tasks = [spend_budget() for _ in range(15)]  # More than max budget
        results = await asyncio.gather(*tasks)
        
        # Only 10 should succeed (max budget is 10.0)
        successful_spends = sum(results)
        assert successful_spends == 10
        
        # Total spent should equal max budget
        assert privacy_accountant.user_budgets[user_id] == 10.0
    
    def test_multiple_users_isolation(self, privacy_accountant):
        """Test that users' budgets are isolated."""
        users = ["user1", "user2", "user3"]
        
        # Each user spends different amounts
        for i, user in enumerate(users, 1):
            privacy_accountant.spend_budget(user, float(i))
        
        # Check isolation
        for i, user in enumerate(users, 1):
            remaining = privacy_accountant.get_remaining_budget(user)
            expected_remaining = privacy_accountant.config.max_budget_per_user - float(i)
            assert remaining == expected_remaining
    
    @pytest.mark.slow
    def test_large_scale_operations(self, privacy_accountant):
        """Test privacy accountant with large number of operations."""
        num_users = 1000
        queries_per_user = 10
        
        for user_id in range(num_users):
            for query_id in range(queries_per_user):
                if privacy_accountant.check_budget(f"user_{user_id}", 0.01):
                    privacy_accountant.spend_budget(f"user_{user_id}", 0.01, f"query_{query_id}")
        
        # Check that history was recorded
        assert len(privacy_accountant.privacy_history) > 0
        
        # Check that budgets were tracked
        assert len(privacy_accountant.user_budgets) == num_users