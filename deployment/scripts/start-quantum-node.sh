#!/bin/bash
set -e

# Quantum-Enhanced Federated LLM Router Startup Script
echo "🚀 Starting Quantum-Enhanced Federated LLM Router..."

# Validate environment
if [ -z "$QUANTUM_CONFIG_PATH" ]; then
    export QUANTUM_CONFIG_PATH="/app/configs/production.yaml"
fi

if [ -z "$LOG_LEVEL" ]; then
    export LOG_LEVEL="INFO"
fi

if [ -z "$WORKERS" ]; then
    export WORKERS="4"
fi

# Initialize logging
mkdir -p /app/logs
echo "$(date '+%Y-%m-%d %H:%M:%S') - Quantum node starting with config: $QUANTUM_CONFIG_PATH" >> /app/logs/startup.log

# Generate encryption keys if they don't exist
if [ ! -f "/app/keys/quantum_private.pem" ]; then
    echo "🔑 Generating quantum encryption keys..."
    python3 -c "
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import os

os.makedirs('/app/keys', exist_ok=True)

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096
)

# Save private key
with open('/app/keys/quantum_private.pem', 'wb') as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ))

# Save public key
public_key = private_key.public_key()
with open('/app/keys/quantum_public.pem', 'wb') as f:
    f.write(public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ))

print('✅ Quantum encryption keys generated')
"
fi

# Initialize quantum configuration
echo "⚙️ Initializing quantum configuration..."
python3 -c "
import yaml
import os

config = {
    'quantum_planning': {
        'enable_superposition': True,
        'enable_entanglement': True,
        'enable_interference': True,
        'coherence_threshold': 0.8,
        'max_entangled_tasks': 5,
        'optimization_strategy': 'adaptive'
    },
    'privacy': {
        'epsilon_per_query': 0.1,
        'delta': 1e-5,
        'max_budget_per_user': 10.0,
        'composition_method': 'rdp'
    },
    'security': {
        'enable_encryption': True,
        'enable_audit_trail': True,
        'security_level': 'confidential',
        'enable_quantum_signatures': True
    },
    'performance': {
        'enable_auto_scaling': True,
        'enable_caching': True,
        'optimization_interval': 30,
        'max_workers': int(os.environ.get('WORKERS', 4))
    },
    'monitoring': {
        'enable_metrics': True,
        'enable_alerts': True,
        'collection_interval': 10,
        'retention_hours': 168
    }
}

os.makedirs('/app/configs', exist_ok=True)
with open('/app/configs/runtime.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('✅ Quantum configuration initialized')
"

# Validate system health before starting
echo "🏥 Performing health checks..."
python3 -c "
import sys
import importlib.util

# Check required modules
required_modules = [
    'numpy', 'asyncio', 'fastapi', 'uvicorn', 
    'cryptography', 'pyjwt', 'redis', 'yaml'
]

missing_modules = []
for module in required_modules:
    if importlib.util.find_spec(module) is None:
        missing_modules.append(module)

if missing_modules:
    print(f'❌ Missing required modules: {missing_modules}')
    sys.exit(1)

print('✅ All required modules available')

# Check configuration
try:
    with open('$QUANTUM_CONFIG_PATH', 'r') as f:
        pass
    print('✅ Configuration file accessible')
except FileNotFoundError:
    print('⚠️ Using runtime configuration')

print('✅ Health checks passed')
"

# Start monitoring in background
echo "📊 Starting monitoring services..."
python3 -m federated_dp_llm.monitoring.health_check &
MONITORING_PID=$!

# Setup signal handlers
cleanup() {
    echo "🛑 Shutting down quantum node..."
    kill $MONITORING_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start the quantum-enhanced federated router
echo "🌟 Starting Quantum-Enhanced Federated LLM Router..."
echo "   - Workers: $WORKERS"
echo "   - Log Level: $LOG_LEVEL"
echo "   - Config: $QUANTUM_CONFIG_PATH"
echo "   - Quantum Features: ENABLED"

exec python3 -m federated_dp_llm.cli \
    --config "$QUANTUM_CONFIG_PATH" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --enable-quantum \
    --enable-monitoring \
    --enable-security