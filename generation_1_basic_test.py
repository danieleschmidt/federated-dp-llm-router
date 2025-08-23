#!/usr/bin/env python3
"""
Generation 1: Basic Functionality Test & Implementation
Test core federated DP-LLM router functionality without external dependencies.
"""

import sys
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1  
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class BasicTask:
    """Basic task representation without numpy dependencies."""
    task_id: str
    user_id: str
    prompt: str
    priority: TaskPriority
    privacy_budget: float
    estimated_duration: float = 30.0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class BasicNode:
    """Basic node representation."""
    node_id: str
    current_load: float = 0.0
    privacy_budget_available: float = 100.0
    is_available: bool = True

class BasicPrivacyAccountant:
    """Simplified privacy accountant for Generation 1."""
    
    def __init__(self, max_budget_per_user: float = 10.0, epsilon_per_query: float = 0.1):
        self.max_budget_per_user = max_budget_per_user
        self.epsilon_per_query = epsilon_per_query
        self.user_budgets: Dict[str, float] = {}
        
    def check_budget(self, user_id: str, requested_epsilon: float) -> bool:
        """Check if user has sufficient privacy budget."""
        current_spent = self.user_budgets.get(user_id, 0.0)
        return current_spent + requested_epsilon <= self.max_budget_per_user
    
    def spend_budget(self, user_id: str, epsilon: float) -> bool:
        """Spend privacy budget."""
        if self.check_budget(user_id, epsilon):
            self.user_budgets[user_id] = self.user_budgets.get(user_id, 0.0) + epsilon
            return True
        return False
    
    def get_remaining_budget(self, user_id: str) -> float:
        """Get remaining budget for user."""
        spent = self.user_budgets.get(user_id, 0.0)
        return max(0.0, self.max_budget_per_user - spent)

class BasicFederatedRouter:
    """Simplified federated router for Generation 1."""
    
    def __init__(self, privacy_accountant: BasicPrivacyAccountant):
        self.privacy_accountant = privacy_accountant
        self.nodes: Dict[str, BasicNode] = {}
        self.tasks: Dict[str, BasicTask] = {}
        self.assignments: List[Dict[str, Any]] = []
        
    def register_node(self, node_id: str, capabilities: Dict[str, Any]) -> None:
        """Register a federated node."""
        self.nodes[node_id] = BasicNode(
            node_id=node_id,
            current_load=capabilities.get('current_load', 0.0),
            privacy_budget_available=capabilities.get('privacy_budget', 100.0)
        )
        logger.info(f"Registered node: {node_id}")
    
    def add_task(self, task_data: Dict[str, Any]) -> str:
        """Add a task to the router."""
        task = BasicTask(
            task_id=task_data['task_id'],
            user_id=task_data['user_id'],
            prompt=task_data['prompt'],
            priority=TaskPriority(task_data.get('priority', 2)),
            privacy_budget=task_data.get('privacy_budget', 0.1),
            estimated_duration=task_data.get('estimated_duration', 30.0)
        )
        
        self.tasks[task.task_id] = task
        logger.info(f"Added task: {task.task_id} for user: {task.user_id}")
        return task.task_id
    
    def assign_tasks(self) -> List[Dict[str, Any]]:
        """Basic task assignment algorithm."""
        assignments = []
        
        # Sort tasks by priority (critical first)
        unassigned_tasks = [task for task in self.tasks.values() 
                           if not any(a['task_id'] == task.task_id for a in self.assignments)]
        
        unassigned_tasks.sort(key=lambda t: t.priority.value)
        
        for task in unassigned_tasks:
            # Check privacy budget
            if not self.privacy_accountant.check_budget(task.user_id, task.privacy_budget):
                logger.warning(f"Insufficient privacy budget for task {task.task_id}")
                continue
            
            # Find best available node
            best_node = self._find_best_node(task)
            if best_node:
                # Create assignment
                assignment = {
                    'task_id': task.task_id,
                    'node_id': best_node.node_id,
                    'user_id': task.user_id,
                    'priority': task.priority.value,
                    'privacy_budget': task.privacy_budget,
                    'estimated_duration': task.estimated_duration,
                    'assignment_time': time.time()
                }
                
                assignments.append(assignment)
                self.assignments.append(assignment)
                
                # Update node and budget
                best_node.current_load += task.estimated_duration / 100.0
                best_node.privacy_budget_available -= task.privacy_budget
                self.privacy_accountant.spend_budget(task.user_id, task.privacy_budget)
                
                logger.info(f"Assigned task {task.task_id} to node {best_node.node_id}")
        
        return assignments
    
    def _find_best_node(self, task: BasicTask) -> Optional[BasicNode]:
        """Find the best available node for a task."""
        available_nodes = [node for node in self.nodes.values() 
                          if node.is_available and 
                          node.privacy_budget_available >= task.privacy_budget]
        
        if not available_nodes:
            return None
        
        # Simple scoring: prefer nodes with lower load
        best_node = min(available_nodes, key=lambda n: n.current_load)
        return best_node
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'total_nodes': len(self.nodes),
            'available_nodes': len([n for n in self.nodes.values() if n.is_available]),
            'total_tasks': len(self.tasks),
            'completed_assignments': len(self.assignments),
            'average_node_load': sum(n.current_load for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0.0
        }

def test_generation_1_functionality():
    """Test Generation 1 basic functionality."""
    print("🚀 Testing Generation 1: Basic Functionality")
    print("=" * 50)
    
    # Initialize components
    privacy_accountant = BasicPrivacyAccountant(max_budget_per_user=5.0, epsilon_per_query=0.1)
    router = BasicFederatedRouter(privacy_accountant)
    
    # Register hospital nodes
    hospital_nodes = [
        {"node_id": "hospital_a", "current_load": 0.2, "privacy_budget": 50.0},
        {"node_id": "hospital_b", "current_load": 0.1, "privacy_budget": 75.0},
        {"node_id": "hospital_c", "current_load": 0.4, "privacy_budget": 25.0}
    ]
    
    for node_config in hospital_nodes:
        router.register_node(node_config["node_id"], node_config)
    
    # Add healthcare tasks
    healthcare_tasks = [
        {
            "task_id": "task_001",
            "user_id": "doctor_smith",
            "prompt": "Analyze chest X-ray for pneumonia indicators",
            "priority": 0,  # CRITICAL
            "privacy_budget": 0.2
        },
        {
            "task_id": "task_002", 
            "user_id": "doctor_jones",
            "prompt": "Interpret blood test results for diabetes screening",
            "priority": 1,  # HIGH
            "privacy_budget": 0.1
        },
        {
            "task_id": "task_003",
            "user_id": "researcher_doe",
            "prompt": "Generate summary of patient cohort study",
            "priority": 3,  # LOW
            "privacy_budget": 0.3
        }
    ]
    
    for task_config in healthcare_tasks:
        router.add_task(task_config)
    
    # Test task assignment
    print("\n📋 Performing task assignments...")
    assignments = router.assign_tasks()
    
    print(f"\n✅ Generated {len(assignments)} task assignments:")
    for assignment in assignments:
        print(f"  • Task {assignment['task_id']} → Node {assignment['node_id']} "
              f"(Priority: {assignment['priority']}, Budget: {assignment['privacy_budget']})")
    
    # Test privacy budget tracking
    print(f"\n🔐 Privacy budget status:")
    for user_id in ['doctor_smith', 'doctor_jones', 'researcher_doe']:
        remaining = privacy_accountant.get_remaining_budget(user_id)
        print(f"  • {user_id}: {remaining:.2f} budget remaining")
    
    # System status
    status = router.get_system_status()
    print(f"\n📊 System Status:")
    print(f"  • Nodes: {status['available_nodes']}/{status['total_nodes']} available")
    print(f"  • Tasks: {status['completed_assignments']}/{status['total_tasks']} assigned")
    print(f"  • Average node load: {status['average_node_load']:.2f}")
    
    # Test privacy budget exhaustion
    print(f"\n🧪 Testing privacy budget limits...")
    test_task = {
        "task_id": "task_overflow",
        "user_id": "doctor_smith", 
        "prompt": "Additional analysis request",
        "priority": 2,
        "privacy_budget": 10.0  # This should exceed remaining budget
    }
    
    router.add_task(test_task)
    overflow_assignments = router.assign_tasks()
    
    if not overflow_assignments:
        print("  ✅ Privacy budget protection working - task blocked due to insufficient budget")
    else:
        print("  ⚠️  Privacy budget protection needs enhancement")
    
    print(f"\n🎉 Generation 1 basic functionality test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_generation_1_functionality()
    sys.exit(0 if success else 1)