#!/usr/bin/env python3
"""
Debug router initialization to find the validation error.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_router_components():
    """Test each router component individually."""
    print("🔍 Testing individual router components...")
    
    try:
        from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig
        config = DPConfig()
        privacy_accountant = PrivacyAccountant(config)
        print("✅ PrivacyAccountant works")
    except Exception as e:
        print(f"❌ PrivacyAccountant failed: {e}")
        return False
        
    try:
        from federated_dp_llm.quantum_planning import QuantumTaskPlanner
        planner = QuantumTaskPlanner(privacy_accountant)
        print("✅ QuantumTaskPlanner works")
    except Exception as e:
        print(f"❌ QuantumTaskPlanner failed: {e}")
        return False
        
    try:
        from federated_dp_llm.quantum_planning import SuperpositionScheduler
        scheduler = SuperpositionScheduler()
        print("✅ SuperpositionScheduler works")
    except Exception as e:
        print(f"❌ SuperpositionScheduler failed: {e}")
        return False
        
    try:
        from federated_dp_llm.quantum_planning import EntanglementOptimizer
        optimizer = EntanglementOptimizer()
        print("✅ EntanglementOptimizer works")
    except Exception as e:
        print(f"❌ EntanglementOptimizer failed: {e}")
        return False
        
    try:
        from federated_dp_llm.quantum_planning import InterferenceBalancer
        balancer = InterferenceBalancer()
        print("✅ InterferenceBalancer works")
    except Exception as e:
        print(f"❌ InterferenceBalancer failed: {e}")
        return False
    
    return True

def test_router_minimal():
    """Test minimal router initialization."""
    print("🔍 Testing minimal router initialization...")
    
    try:
        from federated_dp_llm.routing.load_balancer import FederatedRouter, RoutingStrategy
        
        print("✅ Import successful, creating router...")
        router = FederatedRouter(
            model_name="test-model",
            num_shards=2,
            routing_strategy=RoutingStrategy.ROUND_ROBIN  # Try simpler strategy
        )
        print("✅ FederatedRouter created successfully!")
        return True
    except Exception as e:
        print(f"❌ FederatedRouter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Debug Router Initialization")
    print("=" * 50)
    
    if test_router_components():
        print("\n🔄 All components work, testing full router...")
        test_router_minimal()
    else:
        print("\n❌ Component test failed")