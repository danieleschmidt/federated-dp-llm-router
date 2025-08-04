# Contributing to Federated DP-LLM Router

Thank you for your interest in contributing to the Federated DP-LLM Router! This document provides guidelines and information for contributors.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Security Considerations](#security-considerations)
- [Privacy Guidelines](#privacy-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment of any kind
- Discriminatory language or imagery
- Personal attacks or insults
- Public or private harassment
- Publishing others' private information without permission

## Getting Started

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git
- Basic understanding of:
  - Machine Learning and Large Language Models
  - Differential Privacy concepts
  - Federated Learning principles
  - Healthcare data privacy (HIPAA, GDPR)

### Areas for Contribution

We welcome contributions in several areas:

#### 🔒 Privacy & Security
- Differential privacy mechanisms
- Secure aggregation protocols
- Privacy budget optimization
- Security vulnerability fixes
- Compliance features (HIPAA, GDPR)

#### 🚀 Performance & Scalability
- Model serving optimization
- Caching improvements
- Connection pooling enhancements
- Auto-scaling algorithms
- Load balancing strategies

#### 🤖 Federated Learning
- New aggregation algorithms
- Byzantine fault tolerance
- Model compression techniques
- Communication efficiency
- Convergence improvements

#### 🏥 Healthcare Integration
- EHR system connectors
- Medical data parsers
- Clinical workflow integration
- Regulatory compliance tools
- Healthcare-specific models

#### 📊 Monitoring & Observability
- Metrics collection
- Dashboard improvements
- Alerting systems
- Performance profiling
- Privacy monitoring

#### 🧪 Testing & Quality
- Unit tests
- Integration tests
- Security tests
- Privacy tests
- Performance benchmarks

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/federated-dp-llm-router.git
cd federated-dp-llm-router
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 3. Environment Configuration

```bash
# Copy example configuration
cp configs/development.yaml.example configs/development.yaml

# Set environment variables
export FEDERATED_CONFIG_FILE=configs/development.yaml
export PYTHONPATH=$PWD
```

### 4. Start Development Services

```bash
# Start development stack
docker-compose -f docker-compose.dev.yml up -d

# Run tests to verify setup
pytest tests/ -v
```

## Contributing Process

### 1. Create an Issue

Before starting work, please create an issue to discuss:
- Bug reports with reproduction steps
- Feature requests with use cases
- Performance improvements with benchmarks
- Security vulnerabilities (see [SECURITY.md](SECURITY.md))

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Use descriptive branch names:
- `feature/privacy-budget-optimization`
- `fix/memory-leak-in-cache`
- `security/fix-jwt-validation`
- `docs/improve-api-documentation`

### 3. Make Changes

Follow our [coding standards](#coding-standards) and ensure:
- Code is well-documented
- Tests are included
- Security implications are considered
- Privacy guarantees are maintained

### 4. Test Your Changes

```bash
# Run full test suite
pytest tests/ --cov=federated_dp_llm --cov-report=html

# Run specific test types
pytest tests/ -m unit          # Unit tests
pytest tests/ -m integration   # Integration tests
pytest tests/ -m security      # Security tests
pytest tests/ -m privacy       # Privacy tests

# Run performance tests
pytest tests/ -m slow --benchmark-only
```

### 5. Security and Privacy Validation

```bash
# Static security analysis
bandit -r federated_dp_llm/

# Dependency vulnerability check
safety check

# Privacy test validation
pytest tests/test_privacy_accountant.py -v
```

### 6. Create Pull Request

When creating a PR, please:
- Use a descriptive title
- Fill out the PR template completely
- Reference related issues
- Include screenshots/demos for UI changes
- Ensure CI passes

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Maximum line length: 100 characters
# Use type hints for all functions
def process_inference_request(
    request: InferenceRequest,
    privacy_budget: float,
    timeout: Optional[float] = None
) -> InferenceResponse:
    """
    Process an inference request with privacy guarantees.
    
    Args:
        request: The inference request to process
        privacy_budget: Maximum privacy budget to consume
        timeout: Request timeout in seconds
        
    Returns:
        The inference response with privacy metadata
        
    Raises:
        PrivacyBudgetExceeded: If privacy budget is insufficient
        RequestTimeout: If request exceeds timeout
    """
    # Implementation here
    pass
```

### Code Organization

```
federated_dp_llm/
├── core/              # Core privacy and crypto components
├── routing/           # Request routing and load balancing
├── federation/        # Federated learning coordination
├── security/          # Authentication and compliance
├── monitoring/        # Metrics and health checking
├── optimization/      # Performance and caching
└── models/           # Model-specific implementations
```

### Error Handling

```python
class PrivacyBudgetExceeded(Exception):
    """Raised when privacy budget is insufficient."""
    
    def __init__(self, requested: float, available: float, user_id: str):
        self.requested = requested
        self.available = available
        self.user_id = user_id
        super().__init__(
            f"Privacy budget exceeded for user {user_id}: "
            f"requested {requested}, available {available}"
        )

# Always use specific exceptions
try:
    result = process_with_privacy(data, epsilon=0.1)
except PrivacyBudgetExceeded as e:
    logger.warning(f"Privacy budget exceeded: {e}")
    return create_error_response("Insufficient privacy budget")
```

### Logging

```python
import logging

# Use structured logging
logger = logging.getLogger(__name__)

# Log with context
logger.info(
    "Privacy query processed",
    extra={
        "user_id": user_id,
        "epsilon_spent": epsilon_spent,
        "query_type": "inference",
        "processing_time": processing_time
    }
)
```

## Testing Guidelines

### Test Categories

#### Unit Tests
```python
@pytest.mark.unit
def test_privacy_accountant_budget_tracking():
    """Test privacy budget tracking functionality."""
    accountant = PrivacyAccountant(DPConfig(max_budget_per_user=10.0))
    
    # Test budget spending
    success = accountant.spend_budget("user1", 5.0)
    assert success is True
    
    # Test remaining budget
    remaining = accountant.get_remaining_budget("user1")
    assert remaining == 5.0
```

#### Integration Tests
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_inference():
    """Test complete inference pipeline."""
    # Setup components
    router = create_test_router()
    client = create_test_client()
    
    # Make request
    response = await client.query("Test medical query")
    
    # Verify response
    assert response.privacy_cost > 0
    assert len(response.text) > 0
```

#### Privacy Tests
```python
@pytest.mark.privacy
def test_differential_privacy_guarantee():
    """Test that DP mechanism satisfies privacy guarantee."""
    mechanism = GaussianMechanism(delta=1e-5)
    
    # Test with adjacent datasets
    dataset1 = [1, 2, 3, 4, 5]
    dataset2 = [1, 2, 3, 4, 6]  # Differs by one record
    
    # Verify privacy bound holds
    assert verify_dp_guarantee(mechanism, dataset1, dataset2, epsilon=0.1)
```

#### Security Tests
```python
@pytest.mark.security
def test_authentication_security():
    """Test authentication security measures."""
    auth = AuthenticationManager("secret")
    
    # Test rate limiting
    for _ in range(10):
        auth.authenticate_user("baduser", "wrongpass")
    
    # Should be rate limited now
    with pytest.raises(RateLimitExceeded):
        auth.authenticate_user("baduser", "wrongpass")
```

### Test Data

Use realistic but anonymized test data:

```python
@pytest.fixture
def sample_medical_queries():
    """Sample medical queries for testing."""
    return [
        "Patient presents with chest pain and shortness of breath",
        "Reviewing lab results for diabetic patient",
        "Interpreting chest X-ray findings"
    ]
```

## Security Considerations

### Security Review Process

All security-related changes require:
1. Security review by maintainers
2. Automated security testing
3. Manual penetration testing (for significant changes)
4. Documentation of security implications

### Sensitive Code Areas

Exercise extra caution when modifying:
- Authentication and authorization logic
- Privacy budget enforcement
- Cryptographic implementations
- Input validation and sanitization
- Database queries and ORM usage

### Security Testing

```bash
# Run security linters
bandit -r federated_dp_llm/
semgrep --config=auto federated_dp_llm/

# Check dependencies
safety check
pip-audit

# Test for common vulnerabilities
pytest tests/security/ -v
```

## Privacy Guidelines

### Privacy Impact Assessment

For changes affecting privacy:
1. Document privacy implications
2. Verify differential privacy guarantees
3. Test with privacy test suite
4. Update privacy documentation

### Privacy-Preserving Development

```python
# Always use privacy accountant for budget tracking
def process_query(query: str, user_id: str) -> str:
    # Check budget before processing
    if not privacy_accountant.check_budget(user_id, EPSILON_COST):
        raise PrivacyBudgetExceeded()
    
    # Process with differential privacy
    result = model.inference_with_dp(query, epsilon=EPSILON_COST)
    
    # Record privacy expenditure
    privacy_accountant.spend_budget(user_id, EPSILON_COST)
    
    return result
```

### Privacy Testing

```python
def test_privacy_composition():
    """Test privacy composition bounds."""
    accountant = PrivacyAccountant(DPConfig())
    
    # Multiple queries should compose correctly
    for i in range(10):
        accountant.spend_budget(f"user_{i}", 0.1)
    
    total_epsilon, total_delta = accountant.get_privacy_spent_total()
    
    # Verify composition bounds
    assert total_epsilon <= expected_bound(0.1, 10)
    assert total_delta <= 10 * accountant.config.delta
```

## Documentation

### Code Documentation

```python
class PrivacyAccountant:
    """
    Manages differential privacy budgets for federated learning.
    
    This class implements privacy budget tracking and enforcement
    according to differential privacy principles. It supports
    multiple composition methods and noise mechanisms.
    
    Example:
        >>> config = DPConfig(epsilon_per_query=0.1, delta=1e-5)
        >>> accountant = PrivacyAccountant(config)
        >>> success = accountant.spend_budget("user123", 0.1)
    
    Attributes:
        config: Differential privacy configuration
        user_budgets: Per-user budget tracking
        privacy_history: Historical privacy expenditures
    """
```

### API Documentation

Use OpenAPI/Swagger for API documentation:

```python
@app.post("/inference", response_model=InferenceResponse)
async def inference(
    request: InferenceRequestModel,
    user: UserClaims = Depends(get_current_user)
):
    """
    Submit an inference request to the federated model.
    
    This endpoint processes inference requests while maintaining
    differential privacy guarantees. The privacy cost is automatically
    deducted from the user's privacy budget.
    
    Args:
        request: The inference request parameters
        user: Authenticated user information
        
    Returns:
        Inference response with privacy metadata
        
    Raises:
        HTTPException: If privacy budget is insufficient or request fails
    """
```

### Architecture Documentation

Update architecture docs when making structural changes:
- System architecture diagrams
- Data flow descriptions
- Security model documentation
- Privacy guarantee explanations

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: security@terragonlabs.com (security issues only)

### Getting Help

1. Check existing issues and documentation
2. Search previous discussions
3. Create a new issue with detailed information
4. Be patient and respectful in interactions

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Conference presentations (with permission)

## Release Process

### Version Numbering

We use semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

For maintainers preparing releases:

1. **Code Review**
   - All PRs reviewed and approved
   - CI/CD pipeline passes
   - Security scans clean

2. **Testing**
   - Full test suite passes
   - Performance benchmarks acceptable
   - Security tests pass
   - Privacy tests pass

3. **Documentation**
   - CHANGELOG.md updated
   - API documentation current
   - Architecture docs updated

4. **Security**
   - Security audit completed
   - Vulnerability scan clean
   - Dependencies updated

5. **Compliance**
   - Privacy impact assessment
   - Regulatory compliance verified
   - Audit logs reviewed

## Thank You!

We appreciate your contributions to making federated learning more private, secure, and accessible for healthcare applications. Every contribution, whether code, documentation, testing, or feedback, helps advance privacy-preserving AI for healthcare.

---

*This contributing guide is a living document. Please suggest improvements via pull requests.*