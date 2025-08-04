"""
Test configuration and fixtures for pytest.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np

from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
from federated_dp_llm.routing.load_balancer import FederatedRouter
from federated_dp_llm.federation.client import HospitalNode
from federated_dp_llm.security.compliance import BudgetManager
from federated_dp_llm.monitoring.metrics import MetricsCollector


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "privacy": {
            "epsilon_per_query": 0.1,
            "delta": 1e-5,
            "max_budget_per_user": 10.0,
            "noise_multiplier": 1.1
        },
        "security": {
            "jwt_secret": "test-secret-key"
        },
        "monitoring": {
            "enable_prometheus": False
        }
    }


@pytest.fixture
def dp_config():
    """Default differential privacy configuration for tests."""
    return DPConfig(
        epsilon_per_query=0.1,
        delta=1e-5,
        max_budget_per_user=10.0,
        noise_multiplier=1.1
    )


@pytest.fixture
def privacy_accountant(dp_config):
    """Privacy accountant instance for tests."""
    return PrivacyAccountant(dp_config)


@pytest.fixture
def hospital_nodes():
    """Sample hospital nodes for testing."""
    return [
        HospitalNode(
            id="hospital_a",
            endpoint="https://hospital-a.test:8443",
            data_size=50000,
            compute_capacity="4xA100",
            department="cardiology"
        ),
        HospitalNode(
            id="hospital_b", 
            endpoint="https://hospital-b.test:8443",
            data_size=75000,
            compute_capacity="8xA100",
            department="emergency"
        ),
        HospitalNode(
            id="hospital_c",
            endpoint="https://hospital-c.test:8443",
            data_size=30000,
            compute_capacity="2xV100",
            department="radiology"
        )
    ]


@pytest.fixture
def federated_router(hospital_nodes):
    """Federated router instance for tests."""
    router = FederatedRouter(
        model_name="test-model",
        num_shards=2
    )
    router.register_nodes(hospital_nodes)
    return router


@pytest.fixture
def budget_manager():
    """Budget manager for tests."""
    department_budgets = {
        "emergency": 20.0,
        "cardiology": 15.0,
        "radiology": 10.0,
        "general": 5.0,
        "research": 2.0
    }
    return BudgetManager(department_budgets)


@pytest.fixture
def metrics_collector():
    """Metrics collector for tests."""
    return MetricsCollector(enable_prometheus=False)


@pytest.fixture
def sample_inference_request():
    """Sample inference request for testing."""
    from federated_dp_llm.routing.load_balancer import InferenceRequest
    
    return InferenceRequest(
        request_id="test_request_123",
        user_id="test_user",
        prompt="Test medical query about patient symptoms",
        model_name="test-model",
        max_privacy_budget=0.1,
        require_consensus=False,
        priority=1,
        timeout=30.0,
        department="cardiology"
    )


@pytest.fixture
def sample_model_parameters():
    """Sample model parameters for testing."""
    return {
        "layer1.weight": np.random.normal(0, 0.1, (128, 64)),
        "layer1.bias": np.random.normal(0, 0.01, (128,)),
        "layer2.weight": np.random.normal(0, 0.1, (64, 32)),
        "layer2.bias": np.random.normal(0, 0.01, (64,))
    }


@pytest.fixture
def mock_user_data():
    """Mock user data for authentication tests."""
    from federated_dp_llm.security.authentication import Role
    
    return {
        "user_id": "test_user_123",
        "username": "doctor_smith",
        "email": "doctor.smith@hospital.test",
        "department": "cardiology",
        "roles": [Role.DOCTOR],
        "password": "SecurePassword123!"
    }


@pytest.fixture
def test_certificates(temp_dir):
    """Generate test certificates for mTLS testing."""
    from federated_dp_llm.security.authentication import CertificateManager
    
    cert_manager = CertificateManager()
    
    # Generate CA certificate
    ca_cert, ca_cert_pem, ca_key_pem = cert_manager.generate_ca_certificate("Test CA")
    
    # Generate client certificate
    client_cert, client_cert_pem, client_key_pem = cert_manager.generate_client_certificate(
        ca_cert, ca_key_pem, "test_client", "cardiology"
    )
    
    # Save certificates to temp directory
    ca_cert_file = temp_dir / "ca.crt"
    ca_key_file = temp_dir / "ca.key"
    client_cert_file = temp_dir / "client.crt"
    client_key_file = temp_dir / "client.key"
    
    ca_cert_file.write_bytes(ca_cert_pem)
    ca_key_file.write_bytes(ca_key_pem)
    client_cert_file.write_bytes(client_cert_pem)
    client_key_file.write_bytes(client_key_pem)
    
    return {
        "ca_cert": ca_cert,
        "ca_cert_file": ca_cert_file,
        "ca_key_file": ca_key_file,
        "client_cert": client_cert,
        "client_cert_file": client_cert_file,
        "client_key_file": client_key_file,
        "cert_manager": cert_manager
    }


@pytest.fixture
async def test_server(federated_router, test_config):
    """Start test server for integration tests."""
    from federated_dp_llm.routing.request_handler import RequestHandler
    import uvicorn
    
    # Create request handler
    handler = RequestHandler(federated_router, test_config["security"]["jwt_secret"])
    
    # Configure test server
    config = uvicorn.Config(
        handler.app,
        host="127.0.0.1",
        port=8888,  # Different port for testing
        log_level="error"  # Reduce noise in tests
    )
    
    server = uvicorn.Server(config)
    
    # Start server in background
    task = asyncio.create_task(server.serve())
    
    # Wait for server to start
    await asyncio.sleep(0.5)
    
    yield handler, "http://127.0.0.1:8888"
    
    # Cleanup
    server.should_exit = True
    await task


@pytest.fixture
def privacy_test_data():
    """Test data for privacy mechanism testing."""
    return {
        "plaintext_values": [1.0, 2.5, -0.5, 10.0, 0.0],
        "sensitivity": 1.0,
        "epsilon_values": [0.1, 0.5, 1.0, 2.0],
        "delta": 1e-5
    }


@pytest.fixture  
def compliance_test_events():
    """Sample audit events for compliance testing."""
    from federated_dp_llm.security.compliance import AuditEvent, AuditEventType
    import time
    
    return [
        AuditEvent(
            event_id="event_1",
            event_type=AuditEventType.QUERY_SUBMITTED,
            user_id="doctor_123",
            department="cardiology",
            timestamp=time.time() - 3600,
            details={"query": "Patient symptom analysis", "epsilon_spent": 0.1}
        ),
        AuditEvent(
            event_id="event_2", 
            event_type=AuditEventType.PRIVACY_BUDGET_SPENT,
            user_id="nurse_456",
            department="emergency", 
            timestamp=time.time() - 1800,
            details={"epsilon_spent": 0.2}
        ),
        AuditEvent(
            event_id="event_3",
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id="unknown",
            department="general",
            timestamp=time.time() - 900,
            details={"violation_type": "unauthorized_access", "ip": "192.168.1.100"},
            risk_level="high"
        )
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "privacy: mark test as a privacy/DP test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_privacy_bounds(noisy_value: float, true_value: float, 
                            sensitivity: float, epsilon: float, confidence: float = 0.95):
        """Assert that noisy value is within expected privacy bounds."""
        # This is a simplified check - in practice would use more sophisticated bounds
        max_noise = sensitivity / epsilon * 3  # Rough 3-sigma bound
        assert abs(noisy_value - true_value) <= max_noise, \
            f"Noisy value {noisy_value} outside expected bounds for true value {true_value}"
    
    @staticmethod
    def generate_mock_training_data(num_samples: int = 1000, num_features: int = 10):
        """Generate mock training data."""
        X = np.random.normal(0, 1, (num_samples, num_features))
        y = np.random.randint(0, 2, num_samples)
        return X, y
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
        """Wait for a condition to become true."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False


@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils