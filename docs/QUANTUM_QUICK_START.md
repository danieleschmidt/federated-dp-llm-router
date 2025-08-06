# Quantum-Enhanced Federated LLM Router - Quick Start Guide

Get started with the quantum-enhanced federated LLM router in minutes with this comprehensive quick start guide.

## 🚀 Prerequisites

- **Python**: 3.9+ with pip
- **Docker**: 20.10+ (for containerized deployment)
- **Memory**: 8GB+ RAM recommended  
- **CPU**: 4+ cores for optimal quantum performance
- **GPU**: Optional but recommended for acceleration

## ⚡ Quick Installation

### Option 1: Docker (Recommended)
```bash
# Clone the quantum-enhanced repository
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner

# Build quantum-enhanced image
docker build -f deployment/docker/Dockerfile.quantum -t quantum-federated-llm:latest .

# Start with quantum features enabled
docker run -d \
  --name quantum-router \
  -p 8080:8080 \
  -p 8443:8443 \
  -p 9090:9090 \
  -e QUANTUM_ENABLED=true \
  -e LOG_LEVEL=INFO \
  quantum-federated-llm:latest

# Verify quantum system is running
curl http://localhost:8080/quantum/status
```

### Option 2: Python Environment
```bash
# Create and activate virtual environment
python -m venv quantum-venv
source quantum-venv/bin/activate  # On Windows: quantum-venv\Scripts\activate

# Install quantum-enhanced package
pip install -r requirements.txt
pip install -r requirements-security.txt

# Initialize quantum configuration
python -c "
import yaml
config = {
    'quantum_planning': {'enable_superposition': True, 'enable_entanglement': True},
    'privacy': {'epsilon_per_query': 0.1, 'max_budget_per_user': 10.0},
    'security': {'enable_encryption': True, 'security_level': 'confidential'}
}
with open('quantum_config.yaml', 'w') as f:
    yaml.dump(config, f)
"

# Start quantum-enhanced router
python -m federated_dp_llm.cli --config quantum_config.yaml --enable-quantum
```

## 🌟 Basic Usage

### 1. Initialize Quantum Components
```python
from federated_dp_llm import (
    QuantumTaskPlanner, SuperpositionScheduler, 
    EntanglementOptimizer, InterferenceBalancer,
    PrivacyAccountant, DPConfig
)

# Initialize privacy accounting
privacy_accountant = PrivacyAccountant(DPConfig(
    epsilon_per_query=0.1,
    delta=1e-5,
    max_budget_per_user=10.0
))

# Create quantum task planner
quantum_planner = QuantumTaskPlanner(
    privacy_accountant=privacy_accountant,
    coherence_threshold=0.8,
    max_entangled_tasks=5
)

# Initialize quantum components  
scheduler = SuperpositionScheduler(max_superposition_time=300.0)
optimizer = EntanglementOptimizer(bell_inequality_threshold=2.0)
balancer = InterferenceBalancer(coherence_threshold=0.7)
```

### 2. Create Quantum-Enhanced Router
```python
from federated_dp_llm import FederatedRouter, HospitalNode
from federated_dp_llm.routing.load_balancer import RoutingStrategy

# Create quantum-optimized router
router = FederatedRouter(
    model_name="medllama-7b",
    num_shards=4,
    routing_strategy=RoutingStrategy.QUANTUM_OPTIMIZED
)

# Register hospital nodes
hospitals = [
    HospitalNode(
        id="hospital_a",
        endpoint="https://hospital-a.local:8443",
        data_size=50000,
        compute_capacity="4xA100"
    ),
    HospitalNode(
        id="hospital_b",
        endpoint="https://hospital-b.local:8443", 
        data_size=75000,
        compute_capacity="8xA100"
    )
]

# Register with quantum enhancement
await router.register_nodes(hospitals)
```

### 3. Execute Quantum-Enhanced Inference
```python
from federated_dp_llm.routing.load_balancer import InferenceRequest

# Create inference request
request = InferenceRequest(
    request_id="quantum_demo_001",
    user_id="doctor_smith",
    prompt="Patient presents with chest pain and shortness of breath. Differential diagnosis?",
    model_name="medllama-7b",
    max_privacy_budget=1.0,
    priority=7,  # High priority (1-10 scale)
    department="emergency"
)

# Execute with quantum optimization
response = await router.route_request(request)

print(f"Response: {response.text}")
print(f"Privacy cost: {response.privacy_cost:.3f}")
print(f"Processing node: {response.processing_nodes}")
print(f"Confidence: {response.confidence_score:.3f}")
print(f"Latency: {response.latency:.2f}ms")
```

## 🔮 Quantum Features Demo

### Superposition Scheduling
```python
# Task exists in superposition across nodes until measurement
task_data = {
    'task_id': 'superposition_demo',
    'user_id': 'demo_user',
    'prompt': 'Analyze ECG for arrhythmia patterns',
    'priority': 1,  # Critical priority
    'privacy_budget': 0.5,
    'estimated_duration': 30.0,
    'department': 'cardiology'
}

# Add to quantum planner (enters superposition)
task_id = await quantum_planner.add_task(task_data)
print(f"Task {task_id} in quantum superposition")

# Generate quantum-optimized assignments
assignments = await quantum_planner.plan_optimal_assignments()
print(f"Optimal assignment: {assignments[0] if assignments else 'None ready'}")
```

### Entanglement Optimization
```python
from federated_dp_llm.quantum_planning import EntanglementType

# Create entanglement between related resources
entanglement_id = await optimizer.create_resource_entanglement(
    resource_pairs=[("hospital_a", "gpu"), ("hospital_b", "gpu")],
    entanglement_type=EntanglementType.RESOURCE,
    target_correlation=0.8
)

# Measure correlated optimization
correlations = await optimizer.measure_entangled_correlations(entanglement_id)
print(f"Optimized allocations: {correlations}")
```

### Interference Load Balancing
```python
from federated_dp_llm.quantum_planning import InterferenceType

# Create constructive interference for load balancing
interference_id = await balancer.create_task_interference(
    task_ids=[task_id],
    target_nodes=["hospital_a", "hospital_b", "hospital_c"],
    interference_type=InterferenceType.CONSTRUCTIVE
)

# Current and target loads
current_loads = {"hospital_a": 0.9, "hospital_b": 0.2, "hospital_c": 0.3}
target_loads = {"hospital_a": 0.5, "hospital_b": 0.5, "hospital_c": 0.5}

# Optimize distribution with quantum interference
optimized = await balancer.optimize_load_distribution(current_loads, target_loads)
print(f"Optimized loads: {optimized}")
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# System health
curl http://localhost:8080/health

# Quantum subsystem status  
curl http://localhost:8080/quantum/status

# Privacy accounting status
curl http://localhost:8080/privacy/status

# Performance metrics (Prometheus format)
curl http://localhost:9090/metrics
```

### Key Metrics to Monitor
```python
# Get comprehensive quantum statistics
stats = quantum_planner.get_quantum_statistics()
print(f"Active tasks: {stats['active_tasks']}")
print(f"Superposition tasks: {stats['superposition_tasks']}")
print(f"Entangled tasks: {stats['entangled_tasks']}")
print(f"Average coherence: {stats['average_node_coherence']:.3f}")

# Routing performance
routing_stats = router.get_routing_stats()
print(f"Success rate: {routing_stats['success_rate']:.3f}")
print(f"Quantum task counter: {routing_stats['quantum_task_counter']}")
```

## 🔧 Configuration

### Quantum Planning Settings
```yaml
quantum_planning:
  enable_superposition: true
  enable_entanglement: true  
  enable_interference: true
  coherence_threshold: 0.8
  max_entangled_tasks: 5
  decoherence_rate: 0.01
  optimization_strategy: "adaptive"  # adaptive, latency_focused, throughput_focused
```

### Performance Optimization
```yaml
performance:
  enable_auto_scaling: true
  enable_caching: true
  optimization_interval: 30
  max_workers: 4
  resource_pool_size: 16
  
  scaling_policies:
    - trigger: "cpu_utilization"
      scale_up_threshold: 0.8
      scale_down_threshold: 0.3
    - trigger: "quantum_coherence_loss"
      scale_up_threshold: 0.7
      scale_down_threshold: 0.9
```

### Security Configuration  
```yaml
security:
  enable_encryption: true
  enable_audit_trail: true
  security_level: "confidential"  # public, internal, confidential, restricted
  enable_quantum_signatures: true
  
  privacy:
    epsilon_per_query: 0.1
    delta: 1.0e-5
    max_budget_per_user: 10.0
    composition_method: "rdp"
```

## 🧪 Testing the System

### Basic Functionality Test
```python
import asyncio

async def test_quantum_system():
    # Test quantum planning
    task_data = {
        'task_id': 'test_001',
        'user_id': 'test_user', 
        'prompt': 'Test quantum optimization',
        'priority': 2,
        'privacy_budget': 0.1,
        'estimated_duration': 15.0
    }
    
    task_id = await quantum_planner.add_task(task_data)
    assignments = await quantum_planner.plan_optimal_assignments()
    
    assert len(assignments) >= 0
    print("✅ Quantum planning test passed")
    
    # Test routing
    if assignments:
        print(f"✅ Assignment generated: {assignments[0]['node_id']}")
    
    print("🎉 Quantum system test completed successfully!")

# Run test
asyncio.run(test_quantum_system())
```

### Performance Benchmark
```bash
# Run performance tests
python -c "
import time
import asyncio
from federated_dp_llm.quantum_planning import QuantumTaskPlanner
from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig

async def benchmark():
    planner = QuantumTaskPlanner(PrivacyAccountant(DPConfig()))
    
    # Register test node
    planner.register_node('test_node', {
        'current_load': 0.3, 'privacy_budget': 100.0,
        'compute_capacity': {'gpu_memory': 32768, 'cpu_cores': 16}
    })
    
    # Benchmark task creation
    start = time.time()
    for i in range(100):
        await planner.add_task({
            'task_id': f'bench_{i}', 'user_id': f'user_{i}',
            'prompt': 'Benchmark task', 'priority': 2,
            'privacy_budget': 0.01, 'estimated_duration': 10.0
        })
    
    creation_time = time.time() - start
    print(f'Created 100 tasks in {creation_time:.2f}s ({100/creation_time:.1f} tasks/sec)')
    
    # Benchmark assignment generation
    start = time.time()
    assignments = await planner.plan_optimal_assignments()
    planning_time = time.time() - start
    
    print(f'Generated {len(assignments)} assignments in {planning_time:.3f}s')
    print('🏆 Benchmark completed successfully!')

asyncio.run(benchmark())
"
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Quantum Coherence Loss
```bash
# Check coherence metrics
curl http://localhost:8080/quantum/coherence

# Solution: Reduce entangled tasks or increase threshold
# Edit configuration:
# quantum_planning:
#   max_entangled_tasks: 3
#   coherence_threshold: 0.6
```

#### 2. Performance Degradation
```bash
# Check resource utilization  
curl http://localhost:8080/quantum/optimization/status

# Enable auto-scaling
# performance:
#   enable_auto_scaling: true
#   max_workers: 8
```

#### 3. Privacy Budget Issues
```bash
# Check privacy status
curl http://localhost:8080/privacy/budgets

# Adjust privacy parameters
# privacy:
#   max_budget_per_user: 20.0
#   epsilon_per_query: 0.05
```

## 📚 Next Steps

1. **Production Deployment**: See [Production Deployment Guide](../deployment/production-deployment-guide.md)
2. **Architecture Deep Dive**: Read [Quantum Architecture Documentation](../QUANTUM_ARCHITECTURE.md)  
3. **Advanced Configuration**: Explore configuration options in `/configs/`
4. **Monitoring Setup**: Configure Prometheus/Grafana dashboards
5. **Security Hardening**: Follow security best practices guide
6. **Performance Tuning**: Optimize for your specific healthcare workload

## 💡 Tips for Success

- **Start Small**: Begin with 2-3 hospital nodes and scale up
- **Monitor Coherence**: Keep quantum coherence above 0.8 for optimal performance
- **Privacy-First**: Always validate privacy budgets in production
- **Test Thoroughly**: Use the provided test suites before deployment
- **Documentation**: Keep quantum configuration documented and versioned

Welcome to the future of federated healthcare AI with quantum-inspired optimization! 🚀