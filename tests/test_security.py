"""
Security and authentication tests.
"""

import pytest
import time
import jwt
from unittest.mock import patch, MagicMock

from federated_dp_llm.security.authentication import (
    AuthenticationManager, UserManager, Role, User, CertificateManager
)
from federated_dp_llm.security.compliance import (
    BudgetManager, ComplianceChecker, AuditEvent, AuditEventType
)


@pytest.mark.security
@pytest.mark.unit
class TestAuthenticationManager:
    """Test authentication functionality."""
    
    def test_auth_manager_creation(self):
        """Test authentication manager creation."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        assert auth_manager.jwt_secret == "test-secret"
        assert auth_manager.token_expiry == 3600  # default
        assert isinstance(auth_manager.user_manager, UserManager)
    
    @pytest.mark.asyncio
    async def test_user_registration(self, mock_user_data):
        """Test user registration."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Register user
        success = await auth_manager.register_user(
            username=mock_user_data["username"],
            email=mock_user_data["email"],
            password=mock_user_data["password"],
            department=mock_user_data["department"],
            roles=mock_user_data["roles"]
        )
        
        assert success is True
        
        # Check user exists
        user = await auth_manager.user_manager.get_user(mock_user_data["username"])
        assert user is not None
        assert user.username == mock_user_data["username"]
        assert user.department == mock_user_data["department"]
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, mock_user_data):
        """Test user authentication."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Register user first
        await auth_manager.register_user(
            username=mock_user_data["username"],
            email=mock_user_data["email"],
            password=mock_user_data["password"],
            department=mock_user_data["department"],
            roles=mock_user_data["roles"]
        )
        
        # Authenticate user
        token = await auth_manager.authenticate(
            mock_user_data["username"],
            mock_user_data["password"]
        )
        
        assert token is not None
        assert isinstance(token, str)
        
        # Verify token
        user_info = auth_manager.verify_token(token)
        assert user_info["username"] == mock_user_data["username"]
        assert user_info["department"] == mock_user_data["department"]
    
    @pytest.mark.asyncio
    async def test_failed_authentication(self, mock_user_data):
        """Test failed authentication."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Try to authenticate non-existent user
        token = await auth_manager.authenticate("nonexistent", "password")
        assert token is None
        
        # Register user and try wrong password
        await auth_manager.register_user(
            username=mock_user_data["username"],
            email=mock_user_data["email"],
            password=mock_user_data["password"],
            department=mock_user_data["department"],
            roles=mock_user_data["roles"]
        )
        
        token = await auth_manager.authenticate(
            mock_user_data["username"],
            "wrong_password"
        )
        assert token is None
    
    def test_token_expiry(self):
        """Test JWT token expiry."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret", token_expiry=1)
        
        # Create token that expires in 1 second
        token = auth_manager._create_token("test_user", "cardiology", [Role.DOCTOR])
        
        # Should be valid initially
        user_info = auth_manager.verify_token(token)
        assert user_info is not None
        
        # Wait for expiry and test again
        time.sleep(2)
        
        with pytest.raises(jwt.ExpiredSignatureError):
            auth_manager.verify_token(token)
    
    def test_invalid_token(self):
        """Test invalid token handling."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Test invalid token
        with pytest.raises(jwt.InvalidTokenError):
            auth_manager.verify_token("invalid.token.here")
        
        # Test token with wrong secret
        wrong_auth = AuthenticationManager(jwt_secret="wrong-secret")
        token = wrong_auth._create_token("test_user", "cardiology", [Role.DOCTOR])
        
        with pytest.raises(jwt.InvalidTokenError):
            auth_manager.verify_token(token)


@pytest.mark.security
@pytest.mark.unit
class TestUserManager:
    """Test user management functionality."""
    
    @pytest.mark.asyncio
    async def test_user_creation(self):
        """Test user creation."""
        user_manager = UserManager()
        
        user = User(
            user_id="test_123",
            username="testuser",
            email="test@example.com",
            department="cardiology",
            roles=[Role.DOCTOR]
        )
        
        success = await user_manager.create_user(user, "password123")
        assert success is True
        
        # Retrieve user
        retrieved = await user_manager.get_user("testuser")
        assert retrieved is not None
        assert retrieved.username == "testuser"
        assert retrieved.department == "cardiology"
    
    @pytest.mark.asyncio
    async def test_password_verification(self):
        """Test password verification."""
        user_manager = UserManager()
        
        user = User(
            user_id="test_123",
            username="testuser",
            email="test@example.com",
            department="cardiology",
            roles=[Role.DOCTOR]
        )
        
        await user_manager.create_user(user, "password123")
        
        # Correct password
        valid = await user_manager.verify_password("testuser", "password123")
        assert valid is True
        
        # Wrong password
        invalid = await user_manager.verify_password("testuser", "wrongpassword")
        assert invalid is False
    
    @pytest.mark.asyncio
    async def test_role_authorization(self):
        """Test role-based authorization."""
        user_manager = UserManager()
        
        # Create doctor user
        doctor = User(
            user_id="doctor_123",
            username="doctor",
            email="doctor@example.com",
            department="cardiology",
            roles=[Role.DOCTOR]
        )
        await user_manager.create_user(doctor, "password")
        
        # Create admin user
        admin = User(
            user_id="admin_123",
            username="admin",
            email="admin@example.com",
            department="it",
            roles=[Role.ADMIN]
        )
        await user_manager.create_user(admin, "password")
        
        # Test authorization
        assert user_manager.has_role("doctor", Role.DOCTOR) is True
        assert user_manager.has_role("doctor", Role.ADMIN) is False
        assert user_manager.has_role("admin", Role.ADMIN) is True
        assert user_manager.has_role("admin", Role.DOCTOR) is False


@pytest.mark.security
@pytest.mark.unit  
class TestCertificateManager:
    """Test certificate management for mTLS."""
    
    def test_ca_certificate_generation(self):
        """Test CA certificate generation."""
        cert_manager = CertificateManager()
        
        ca_cert, ca_cert_pem, ca_key_pem = cert_manager.generate_ca_certificate("Test CA")
        
        assert ca_cert is not None
        assert ca_cert_pem is not None
        assert ca_key_pem is not None
        assert b"BEGIN CERTIFICATE" in ca_cert_pem
        assert b"BEGIN PRIVATE KEY" in ca_key_pem
    
    def test_client_certificate_generation(self):
        """Test client certificate generation."""
        cert_manager = CertificateManager()
        
        # Generate CA first
        ca_cert, ca_cert_pem, ca_key_pem = cert_manager.generate_ca_certificate("Test CA")
        
        # Generate client certificate
        client_cert, client_cert_pem, client_key_pem = cert_manager.generate_client_certificate(
            ca_cert, ca_key_pem, "test_client", "cardiology"
        )
        
        assert client_cert is not None
        assert client_cert_pem is not None
        assert client_key_pem is not None
        assert b"BEGIN CERTIFICATE" in client_cert_pem
        assert b"BEGIN PRIVATE KEY" in client_key_pem
    
    def test_certificate_verification(self):
        """Test certificate verification."""
        cert_manager = CertificateManager()
        
        # Generate certificates
        ca_cert, ca_cert_pem, ca_key_pem = cert_manager.generate_ca_certificate("Test CA")
        client_cert, client_cert_pem, client_key_pem = cert_manager.generate_client_certificate(
            ca_cert, ca_key_pem, "test_client", "cardiology"
        )
        
        # Verify client certificate against CA
        is_valid = cert_manager.verify_certificate(client_cert_pem, ca_cert_pem)
        assert is_valid is True
        
        # Test with wrong CA
        wrong_ca, wrong_ca_pem, _ = cert_manager.generate_ca_certificate("Wrong CA")
        is_invalid = cert_manager.verify_certificate(client_cert_pem, wrong_ca_pem)
        assert is_invalid is False


@pytest.mark.security
@pytest.mark.unit
class TestBudgetManager:
    """Test privacy budget management."""
    
    def test_budget_initialization(self):
        """Test budget manager initialization."""
        department_budgets = {
            "cardiology": 10.0,
            "emergency": 15.0
        }
        
        budget_manager = BudgetManager(department_budgets)
        
        assert budget_manager.get_department_budget("cardiology") == 10.0
        assert budget_manager.get_department_budget("emergency") == 15.0
        assert budget_manager.get_department_budget("unknown") == 5.0  # default
    
    def test_budget_allocation(self):
        """Test budget allocation to users."""
        budget_manager = BudgetManager({"cardiology": 10.0})
        
        # Allocate budget
        success = budget_manager.allocate_budget("user1", "cardiology", 2.0)
        assert success is True
        
        # Check remaining budget
        remaining = budget_manager.get_remaining_budget("user1")
        assert remaining == 2.0
        
        # Try to allocate more than available
        success = budget_manager.allocate_budget("user2", "cardiology", 15.0)
        assert success is False
    
    def test_budget_spending(self):
        """Test budget spending."""
        budget_manager = BudgetManager({"cardiology": 10.0})
        
        # Allocate and spend budget
        budget_manager.allocate_budget("user1", "cardiology", 5.0)
        
        success = budget_manager.spend_budget("user1", 2.0)
        assert success is True
        
        remaining = budget_manager.get_remaining_budget("user1")
        assert remaining == 3.0
        
        # Try to spend more than available
        success = budget_manager.spend_budget("user1", 5.0)
        assert success is False
    
    def test_budget_reset(self):
        """Test budget reset functionality."""
        budget_manager = BudgetManager({"cardiology": 10.0})
        
        # Allocate and spend some budget
        budget_manager.allocate_budget("user1", "cardiology", 5.0)
        budget_manager.spend_budget("user1", 3.0)
        
        # Reset budget
        budget_manager.reset_user_budget("user1")
        
        remaining = budget_manager.get_remaining_budget("user1")
        assert remaining == 0.0


@pytest.mark.security
@pytest.mark.unit
class TestComplianceChecker:
    """Test compliance checking functionality."""
    
    def test_hipaa_compliance_check(self, compliance_test_events):
        """Test HIPAA compliance checking."""
        compliance_checker = ComplianceChecker()
        
        # Mock configuration for HIPAA
        with patch.object(compliance_checker, 'hipaa_config', {
            'max_queries_per_hour': 100,
            'max_privacy_budget_per_day': 10.0,
            'required_audit_fields': ['user_id', 'department', 'timestamp']
        }):
            
            # Check compliance for events
            for event in compliance_test_events:
                is_compliant, violations = compliance_checker.check_hipaa_compliance(event)
                
                if event.event_type == AuditEventType.SECURITY_VIOLATION:
                    assert is_compliant is False
                    assert len(violations) > 0
                else:
                    assert is_compliant is True
                    assert len(violations) == 0
    
    def test_gdpr_compliance_check(self, compliance_test_events):
        """Test GDPR compliance checking.""" 
        compliance_checker = ComplianceChecker()
        
        # Test data retention policies
        current_time = time.time()
        old_event = AuditEvent(
            event_id="old_event",
            event_type=AuditEventType.QUERY_SUBMITTED,
            user_id="user_123",
            department="cardiology",
            timestamp=current_time - (90 * 24 * 3600),  # 90 days old
            details={"query": "old query"}
        )
        
        is_compliant, violations = compliance_checker.check_gdpr_compliance(old_event)
        
        # Should flag for data retention violation
        assert is_compliant is False
        assert any("retention" in v.lower() for v in violations)
    
    def test_audit_trail_generation(self, compliance_test_events):
        """Test audit trail generation."""
        compliance_checker = ComplianceChecker()
        
        # Add events to audit trail
        for event in compliance_test_events:
            compliance_checker.add_audit_event(event)
        
        # Generate audit trail
        audit_trail = compliance_checker.generate_audit_trail(
            start_time=time.time() - 7200,  # 2 hours ago
            end_time=time.time()
        )
        
        assert len(audit_trail) == len(compliance_test_events)
        
        # Test filtering by department
        cardiology_trail = compliance_checker.generate_audit_trail(
            start_time=time.time() - 7200,
            end_time=time.time(),
            department="cardiology"
        )
        
        assert len(cardiology_trail) == 1
        assert cardiology_trail[0].department == "cardiology"


@pytest.mark.security
@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_authentication(self, mock_user_data):
        """Test complete authentication workflow."""
        auth_manager = AuthenticationManager(jwt_secret="test-secret")
        
        # Register user
        success = await auth_manager.register_user(
            username=mock_user_data["username"],
            email=mock_user_data["email"],
            password=mock_user_data["password"],
            department=mock_user_data["department"],
            roles=mock_user_data["roles"]
        )
        assert success is True
        
        # Authenticate and get token
        token = await auth_manager.authenticate(
            mock_user_data["username"],
            mock_user_data["password"]
        )
        assert token is not None
        
        # Use token for authorization
        user_info = auth_manager.verify_token(token)
        assert user_info["username"] == mock_user_data["username"]
        
        # Check role authorization
        has_doctor_role = auth_manager.user_manager.has_role(
            mock_user_data["username"], 
            Role.DOCTOR
        )
        assert has_doctor_role is True
    
    @pytest.mark.asyncio
    async def test_mtls_certificate_workflow(self, test_certificates):
        """Test mTLS certificate workflow."""
        cert_manager = test_certificates["cert_manager"]
        
        # Verify certificate chain
        is_valid = cert_manager.verify_certificate(
            test_certificates["client_cert_file"].read_bytes(),
            test_certificates["ca_cert_file"].read_bytes()
        )
        assert is_valid is True
        
        # Test certificate extraction
        cert_info = cert_manager.extract_certificate_info(
            test_certificates["client_cert"]
        )
        
        assert "subject" in cert_info
        assert "department" in cert_info
        assert cert_info["department"] == "cardiology"
    
    def test_compliance_and_audit_integration(self, compliance_test_events):
        """Test compliance checking with audit trail."""
        compliance_checker = ComplianceChecker()
        budget_manager = BudgetManager({"cardiology": 10.0, "emergency": 15.0})
        
        # Process events through compliance checker
        for event in compliance_test_events:
            compliance_checker.add_audit_event(event)
            
            # Check compliance
            hipaa_compliant, _ = compliance_checker.check_hipaa_compliance(event)
            gdpr_compliant, _ = compliance_checker.check_gdpr_compliance(event)
            
            # Log compliance results
            compliance_result = {
                "event_id": event.event_id,
                "hipaa_compliant": hipaa_compliant,
                "gdpr_compliant": gdpr_compliant
            }
            
            # In a real system, this would be logged
            assert isinstance(compliance_result, dict)
        
        # Generate compliance report
        report = compliance_checker.generate_compliance_report()
        
        assert "total_events" in report
        assert "violations" in report
        assert report["total_events"] == len(compliance_test_events)