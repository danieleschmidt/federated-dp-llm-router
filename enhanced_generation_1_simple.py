#!/usr/bin/env python3
"""
Enhanced Generation 1: Simple Core Functionality
Demonstrates core federated DP-LLM router capabilities with minimal viable features.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimplePrivacyConfig:
    """Simplified privacy configuration for Generation 1."""
    epsilon_per_query: float = 0.1
    max_budget_per_user: float = 5.0
    delta: float = 1e-5

@dataclass
class SimpleNode:
    """Simplified node representation."""
    node_id: str
    endpoint: str
    status: str = "active"
    load: float = 0.0

@dataclass
class SimpleRequest:
    """Simplified inference request."""
    user_id: str
    prompt: str
    privacy_budget: float = 0.1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SimpleFederatedRouter:
    """
    Generation 1: Basic federated router with core functionality.
    Demonstrates value with minimal viable features.
    """
    
    def __init__(self, privacy_config: SimplePrivacyConfig = None):
        self.privacy_config = privacy_config or SimplePrivacyConfig()
        self.nodes: Dict[str, SimpleNode] = {}
        self.user_budgets: Dict[str, float] = {}
        self.request_history: List[Dict] = []
        
        logger.info("🚀 Generation 1: Simple Federated Router initialized")
    
    def register_node(self, node: SimpleNode) -> bool:
        """Register a new federated node."""
        try:
            self.nodes[node.node_id] = node
            logger.info(f"✅ Node registered: {node.node_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Node registration failed: {e}")
            return False
    
    def check_privacy_budget(self, user_id: str, requested_budget: float) -> bool:
        """Check if user has sufficient privacy budget."""
        current_budget = self.user_budgets.get(user_id, 0.0)
        remaining = self.privacy_config.max_budget_per_user - current_budget
        
        if requested_budget <= remaining:
            return True
        
        logger.warning(f"⚠️ Privacy budget exceeded for user {user_id}")
        return False
    
    def consume_privacy_budget(self, user_id: str, amount: float) -> None:
        """Consume privacy budget for a user."""
        current = self.user_budgets.get(user_id, 0.0)
        self.user_budgets[user_id] = current + amount
        
        remaining = self.privacy_config.max_budget_per_user - self.user_budgets[user_id]
        logger.info(f"💰 Privacy budget consumed: {amount:.3f}, remaining: {remaining:.3f}")
    
    def select_optimal_node(self) -> Optional[SimpleNode]:
        """Select node with lowest load."""
        if not self.nodes:
            return None
        
        active_nodes = [node for node in self.nodes.values() if node.status == "active"]
        if not active_nodes:
            return None
        
        # Simple load balancing - select node with lowest load
        optimal_node = min(active_nodes, key=lambda n: n.load)
        return optimal_node
    
    async def process_request(self, request: SimpleRequest) -> Dict:
        """Process federated inference request with privacy protection."""
        logger.info(f"🔄 Processing request for user: {request.user_id}")
        
        # Check privacy budget
        if not self.check_privacy_budget(request.user_id, request.privacy_budget):
            return {
                "success": False,
                "error": "Privacy budget exceeded",
                "remaining_budget": (
                    self.privacy_config.max_budget_per_user - 
                    self.user_budgets.get(request.user_id, 0.0)
                )
            }
        
        # Select optimal node
        node = self.select_optimal_node()
        if not node:
            return {
                "success": False,
                "error": "No active nodes available"
            }
        
        try:
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Simulate LLM response with privacy noise
            mock_response = f"Processed: {request.prompt[:50]}... (via {node.node_id})"
            
            # Consume privacy budget
            self.consume_privacy_budget(request.user_id, request.privacy_budget)
            
            # Update node load
            node.load += 0.1
            
            # Record request
            self.request_history.append({
                "timestamp": request.timestamp.isoformat(),
                "user_id": request.user_id,
                "node_id": node.node_id,
                "privacy_cost": request.privacy_budget,
                "success": True
            })
            
            return {
                "success": True,
                "response": mock_response,
                "node_id": node.node_id,
                "privacy_cost": request.privacy_budget,
                "remaining_budget": (
                    self.privacy_config.max_budget_per_user - 
                    self.user_budgets[request.user_id]
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Request processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_status(self) -> Dict:
        """Get router status information."""
        return {
            "active_nodes": len([n for n in self.nodes.values() if n.status == "active"]),
            "total_nodes": len(self.nodes),
            "total_requests": len(self.request_history),
            "user_count": len(self.user_budgets),
            "privacy_config": asdict(self.privacy_config)
        }

class SimpleHealthChecker:
    """Simple health monitoring for Generation 1."""
    
    def __init__(self, router: SimpleFederatedRouter):
        self.router = router
        self.last_check = datetime.now()
    
    def health_check(self) -> Dict:
        """Perform basic health check."""
        status = self.router.get_status()
        
        health_status = "healthy"
        if status["active_nodes"] == 0:
            health_status = "unhealthy"
        elif status["active_nodes"] < status["total_nodes"] * 0.5:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "nodes": status,
            "uptime_check": "passed"
        }

async def demo_generation_1():
    """Demonstrate Generation 1 core functionality."""
    print("🌟 GENERATION 1 DEMO: Simple Core Functionality")
    print("=" * 60)
    
    # Initialize router
    config = SimplePrivacyConfig(epsilon_per_query=0.1, max_budget_per_user=2.0)
    router = SimpleFederatedRouter(config)
    
    # Register nodes
    nodes = [
        SimpleNode("hospital_a", "https://hospital-a.local:8443"),
        SimpleNode("hospital_b", "https://hospital-b.local:8443"),
        SimpleNode("hospital_c", "https://hospital-c.local:8443")
    ]
    
    for node in nodes:
        router.register_node(node)
    
    # Initialize health checker
    health_checker = SimpleHealthChecker(router)
    
    # Process sample requests
    requests = [
        SimpleRequest("doctor_001", "Patient presents with chest pain", 0.2),
        SimpleRequest("doctor_002", "Review lab results for diabetes patient", 0.1),
        SimpleRequest("doctor_001", "Follow-up consultation needed", 0.3),
        SimpleRequest("doctor_003", "Emergency case assessment", 0.5)
    ]
    
    print("\n📋 Processing Requests:")
    for i, request in enumerate(requests, 1):
        print(f"\n{i}. Request from {request.user_id}")
        result = await router.process_request(request)
        
        if result["success"]:
            print(f"   ✅ Success: {result['response'][:50]}...")
            print(f"   💰 Privacy cost: {result['privacy_cost']}")
            print(f"   🏦 Node: {result['node_id']}")
            print(f"   💳 Remaining budget: {result['remaining_budget']:.3f}")
        else:
            print(f"   ❌ Error: {result['error']}")
    
    # Show final status
    print(f"\n📊 Final Status:")
    status = router.get_status()
    print(f"   Nodes: {status['active_nodes']}/{status['total_nodes']} active")
    print(f"   Requests processed: {status['total_requests']}")
    print(f"   Users served: {status['user_count']}")
    
    # Health check
    health = health_checker.health_check()
    print(f"\n🏥 Health Check: {health['status']}")
    
    print(f"\n✅ Generation 1 Complete: Basic functionality demonstrated!")
    return router, health_checker

if __name__ == "__main__":
    asyncio.run(demo_generation_1())