"""
Adaptive Optimization Engine

Continuously learns from system performance and adapts routing, scheduling,
and resource allocation strategies based on real-world usage patterns.
"""

import asyncio
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from collections import deque, defaultdict
import logging
from pathlib import Path

from ..quantum_planning.numpy_fallback import get_numpy_backend
from ..monitoring.metrics import MetricsCollector
from ..core.privacy_accountant import DPConfig
from ..security.input_validator import validate_input

HAS_NUMPY, np = get_numpy_backend()
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies available."""
    GREEDY = "greedy"
    GRADIENT_DESCENT = "gradient_descent" 
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    QUANTUM_ANNEALING = "quantum_annealing"


class LearningMode(Enum):
    """Learning modes for adaptation."""
    ONLINE = "online"           # Learn continuously from live data
    OFFLINE = "offline"         # Learn from historical data
    HYBRID = "hybrid"          # Combine online and offline learning
    TRANSFER = "transfer"      # Transfer learning from similar tasks


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: float
    accuracy: float
    latency_p95: float
    throughput: float
    resource_utilization: float
    privacy_budget_remaining: float
    error_rate: float
    user_satisfaction: float
    energy_efficiency: float
    configuration: Dict[str, Any]
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML algorithms."""
        return [
            self.accuracy,
            self.latency_p95,
            self.throughput,
            self.resource_utilization,
            self.privacy_budget_remaining,
            1.0 - self.error_rate,  # Convert to success rate
            self.user_satisfaction,
            self.energy_efficiency
        ]


@dataclass
class OptimizationResult:
    """Result of optimization iteration."""
    strategy_used: OptimizationStrategy
    old_configuration: Dict[str, Any]
    new_configuration: Dict[str, Any]
    predicted_improvement: float
    actual_improvement: Optional[float]
    convergence_achieved: bool
    iteration_count: int
    optimization_time_seconds: float


class AdaptiveOptimizer:
    """Adaptive optimization engine with multiple learning strategies."""
    
    def __init__(self, 
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.HYBRID,
                 learning_mode: LearningMode = LearningMode.ONLINE,
                 learning_rate: float = 0.01,
                 history_window: int = 1000,
                 adaptation_frequency: int = 100,
                 convergence_threshold: float = 0.001):
        
        self.optimization_strategy = optimization_strategy
        self.learning_mode = learning_mode
        self.learning_rate = learning_rate
        self.history_window = history_window
        self.adaptation_frequency = adaptation_frequency
        self.convergence_threshold = convergence_threshold
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=history_window)
        self.configuration_history: List[Dict[str, Any]] = []
        self.optimization_results: List[OptimizationResult] = []
        
        # Learning state
        self.current_configuration: Dict[str, Any] = self._initialize_configuration()
        self.best_configuration: Dict[str, Any] = self.current_configuration.copy()
        self.best_performance_score: float = 0.0
        self.iterations_since_improvement: int = 0
        self.total_iterations: int = 0
        
        # Strategy-specific state
        self.gradient_momentum: Dict[str, float] = {}
        self.population: List[Dict[str, Any]] = []  # For evolutionary strategy
        self.q_table: Dict[str, Dict[str, float]] = {}  # For RL strategy
        
        # Metrics collection
        self.metrics_collector = MetricsCollector()
        
        logger.info(f"Initialized adaptive optimizer: {optimization_strategy.value}, {learning_mode.value}")
        
    def _initialize_configuration(self) -> Dict[str, Any]:
        """Initialize baseline configuration."""
        return {
            "load_balancer_strategy": "quantum_weighted",
            "privacy_noise_multiplier": 1.1,
            "batch_size": 32,
            "connection_pool_size": 10,
            "cache_size": 1000,
            "timeout_ms": 5000,
            "retry_count": 3,
            "quantum_coherence_threshold": 0.8,
            "entanglement_strength": 0.5,
            "interference_damping": 0.1
        }
    
    async def record_performance(self, 
                               accuracy: float,
                               latency_p95: float,
                               throughput: float,
                               resource_utilization: float,
                               privacy_budget_remaining: float,
                               error_rate: float,
                               user_satisfaction: float = 0.8,
                               energy_efficiency: float = 0.7) -> bool:
        """Record performance observation and trigger adaptation if needed."""
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            accuracy=accuracy,
            latency_p95=latency_p95,
            throughput=throughput,
            resource_utilization=resource_utilization,
            privacy_budget_remaining=privacy_budget_remaining,
            error_rate=error_rate,
            user_satisfaction=user_satisfaction,
            energy_efficiency=energy_efficiency,
            configuration=self.current_configuration.copy()
        )
        
        self.performance_history.append(snapshot)
        
        # Record metrics
        self.metrics_collector.record_metric("performance_score", self._calculate_performance_score(snapshot))
        self.metrics_collector.record_metric("adaptation_trigger_check", 1)
        
        # Check if adaptation should be triggered
        if len(self.performance_history) % self.adaptation_frequency == 0:
            logger.info("Triggering adaptive optimization")
            optimization_result = await self._optimize_configuration()
            
            if optimization_result:
                self.optimization_results.append(optimization_result)
                return True
        
        return False
    
    def _calculate_performance_score(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate composite performance score."""
        weights = {
            'accuracy': 0.25,
            'latency': -0.15,  # Lower is better
            'throughput': 0.20,
            'resource_utilization': -0.10,  # Efficient utilization is good, but not overutilization
            'privacy_budget_remaining': 0.10,
            'error_rate': -0.10,  # Lower is better
            'user_satisfaction': 0.15,
            'energy_efficiency': 0.15
        }
        
        # Normalize and calculate weighted score
        score = 0.0
        score += weights['accuracy'] * snapshot.accuracy
        score += weights['latency'] * (1.0 - min(1.0, snapshot.latency_p95 / 1000.0))  # Normalize latency
        score += weights['throughput'] * min(1.0, snapshot.throughput / 1000.0)  # Normalize throughput
        score += weights['resource_utilization'] * (1.0 - abs(snapshot.resource_utilization - 0.7))  # Optimal around 70%
        score += weights['privacy_budget_remaining'] * snapshot.privacy_budget_remaining
        score += weights['error_rate'] * (1.0 - snapshot.error_rate)
        score += weights['user_satisfaction'] * snapshot.user_satisfaction
        score += weights['energy_efficiency'] * snapshot.energy_efficiency
        
        return max(0.0, min(1.0, score))
    
    async def _optimize_configuration(self) -> Optional[OptimizationResult]:
        """Optimize configuration based on strategy."""
        if len(self.performance_history) < 10:
            return None
        
        start_time = time.time()
        old_config = self.current_configuration.copy()
        
        try:
            if self.optimization_strategy == OptimizationStrategy.GRADIENT_DESCENT:
                new_config = await self._gradient_descent_optimization()
            elif self.optimization_strategy == OptimizationStrategy.EVOLUTIONARY:
                new_config = await self._evolutionary_optimization()
            elif self.optimization_strategy == OptimizationStrategy.BAYESIAN:
                new_config = await self._bayesian_optimization()
            elif self.optimization_strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
                new_config = await self._reinforcement_learning_optimization()
            elif self.optimization_strategy == OptimizationStrategy.QUANTUM_ANNEALING:
                new_config = await self._quantum_annealing_optimization()
            else:
                new_config = await self._greedy_optimization()
            
            # Validate new configuration
            if self._validate_configuration(new_config):
                predicted_improvement = self._predict_improvement(old_config, new_config)
                
                # Apply configuration if improvement is predicted
                if predicted_improvement > self.convergence_threshold:
                    self.current_configuration = new_config
                    self.total_iterations += 1
                    
                    optimization_time = time.time() - start_time
                    
                    return OptimizationResult(
                        strategy_used=self.optimization_strategy,
                        old_configuration=old_config,
                        new_configuration=new_config,
                        predicted_improvement=predicted_improvement,
                        actual_improvement=None,  # Will be measured later
                        convergence_achieved=False,
                        iteration_count=self.total_iterations,
                        optimization_time_seconds=optimization_time
                    )
                else:
                    self.iterations_since_improvement += 1
                    logger.info("No significant improvement predicted, keeping current configuration")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
        
        return None
    
    async def _gradient_descent_optimization(self) -> Dict[str, Any]:
        """Gradient descent-based optimization."""
        if len(self.performance_history) < 20:
            return self.current_configuration.copy()
        
        # Calculate gradients based on recent performance
        recent_snapshots = list(self.performance_history)[-20:]
        new_config = self.current_configuration.copy()
        
        # For each configuration parameter, estimate gradient
        for param_name, current_value in self.current_configuration.items():
            if isinstance(current_value, (int, float)):
                # Find snapshots with different values for this parameter
                param_gradients = []
                
                for i, snapshot in enumerate(recent_snapshots[:-1]):
                    next_snapshot = recent_snapshots[i + 1]
                    
                    old_param = snapshot.configuration.get(param_name, current_value)
                    new_param = next_snapshot.configuration.get(param_name, current_value)
                    
                    if old_param != new_param:
                        old_score = self._calculate_performance_score(snapshot)
                        new_score = self._calculate_performance_score(next_snapshot)
                        
                        if new_param != old_param:
                            gradient = (new_score - old_score) / (new_param - old_param)
                            param_gradients.append(gradient)
                
                if param_gradients:
                    avg_gradient = sum(param_gradients) / len(param_gradients)
                    
                    # Apply momentum
                    momentum_key = f"{param_name}_momentum"
                    if momentum_key not in self.gradient_momentum:
                        self.gradient_momentum[momentum_key] = 0.0
                    
                    self.gradient_momentum[momentum_key] = (0.9 * self.gradient_momentum[momentum_key] + 
                                                          0.1 * avg_gradient)
                    
                    # Update parameter
                    update = self.learning_rate * self.gradient_momentum[momentum_key]
                    
                    if isinstance(current_value, int):
                        new_config[param_name] = max(1, int(current_value + update))
                    else:
                        new_config[param_name] = max(0.01, current_value + update)
        
        return new_config
    
    async def _evolutionary_optimization(self) -> Dict[str, Any]:
        """Evolutionary algorithm optimization."""
        population_size = 10
        mutation_rate = 0.1
        
        # Initialize population if empty
        if not self.population:
            self.population = [self._mutate_configuration(self.current_configuration, 0.2) 
                              for _ in range(population_size)]
        
        # Evaluate population fitness
        fitness_scores = []
        for config in self.population:
            # Estimate fitness based on similar configurations in history
            estimated_score = self._estimate_configuration_score(config)
            fitness_scores.append(estimated_score)
        
        # Select best individuals
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        elite_count = population_size // 3
        elite_indices = sorted_indices[:elite_count]
        
        # Create new generation
        new_population = []
        
        # Keep elite
        for i in elite_indices:
            new_population.append(self.population[i].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Select parents
            parent1_idx = np.random.choice(elite_indices) if HAS_NUMPY else elite_indices[0]
            parent2_idx = np.random.choice(elite_indices) if HAS_NUMPY else elite_indices[-1]
            
            # Crossover
            child = self._crossover_configurations(self.population[parent1_idx], self.population[parent2_idx])
            
            # Mutate
            if HAS_NUMPY and np.random.random() < mutation_rate:
                child = self._mutate_configuration(child, 0.1)
            
            new_population.append(child)
        
        self.population = new_population
        
        # Return best configuration from new generation
        best_idx = np.argmax([self._estimate_configuration_score(config) for config in new_population]) if HAS_NUMPY else 0
        return new_population[best_idx]
    
    async def _bayesian_optimization(self) -> Dict[str, Any]:
        """Bayesian optimization (simplified implementation)."""
        # This is a simplified version - in practice would use gaussian processes
        if len(self.performance_history) < 5:
            return self._mutate_configuration(self.current_configuration, 0.1)
        
        # Find the configuration that led to best performance
        best_snapshot = max(self.performance_history, key=self._calculate_performance_score)
        best_config = best_snapshot.configuration
        
        # Generate candidate configurations around the best one
        candidates = []
        for _ in range(10):
            candidate = self._mutate_configuration(best_config, 0.05)  # Small mutations
            candidates.append(candidate)
        
        # Select candidate with highest expected improvement
        best_candidate = candidates[0]
        best_expected_improvement = self._estimate_configuration_score(best_candidate)
        
        for candidate in candidates[1:]:
            expected_improvement = self._estimate_configuration_score(candidate)
            if expected_improvement > best_expected_improvement:
                best_candidate = candidate
                best_expected_improvement = expected_improvement
        
        return best_candidate
    
    async def _reinforcement_learning_optimization(self) -> Dict[str, Any]:
        """Q-learning based optimization."""
        # Simplified Q-learning implementation
        epsilon = 0.1  # Exploration rate
        alpha = self.learning_rate
        gamma = 0.95  # Discount factor
        
        # Define state and action spaces (simplified)
        current_state = self._configuration_to_state(self.current_configuration)
        
        # Choose action (configuration change) using epsilon-greedy
        if current_state not in self.q_table:
            self.q_table[current_state] = {}
        
        possible_actions = self._get_possible_actions()
        
        if HAS_NUMPY and np.random.random() < epsilon:
            # Explore: choose random action
            action = np.random.choice(possible_actions)
        else:
            # Exploit: choose best known action
            if possible_actions and current_state in self.q_table:
                action = max(possible_actions, 
                           key=lambda a: self.q_table[current_state].get(a, 0.0))
            else:
                action = possible_actions[0] if possible_actions else "no_change"
        
        # Apply action to get new configuration
        new_config = self._apply_action(self.current_configuration, action)
        
        # If we have performance history, update Q-table
        if len(self.performance_history) >= 2:
            prev_snapshot = self.performance_history[-2]
            current_snapshot = self.performance_history[-1]
            
            prev_state = self._configuration_to_state(prev_snapshot.configuration)
            reward = self._calculate_performance_score(current_snapshot) - self._calculate_performance_score(prev_snapshot)
            
            if prev_state not in self.q_table:
                self.q_table[prev_state] = {}
            
            # Q-learning update
            old_q = self.q_table[prev_state].get(action, 0.0)
            max_next_q = max(self.q_table[current_state].values()) if current_state in self.q_table and self.q_table[current_state] else 0.0
            
            new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
            self.q_table[prev_state][action] = new_q
        
        return new_config
    
    async def _quantum_annealing_optimization(self) -> Dict[str, Any]:
        """Quantum annealing inspired optimization."""
        # Simulated annealing with quantum-inspired elements
        current_config = self.current_configuration.copy()
        current_score = self._estimate_configuration_score(current_config)
        
        # Annealing schedule
        initial_temperature = 1.0
        cooling_rate = 0.95
        min_temperature = 0.01
        
        temperature = initial_temperature
        best_config = current_config.copy()
        best_score = current_score
        
        while temperature > min_temperature:
            # Generate neighbor configuration with quantum tunneling probability
            neighbor_config = self._mutate_configuration(current_config, 0.1)
            neighbor_score = self._estimate_configuration_score(neighbor_config)
            
            # Accept or reject based on quantum annealing probability
            score_diff = neighbor_score - current_score
            
            if score_diff > 0:
                # Always accept improvement
                current_config = neighbor_config
                current_score = neighbor_score
                
                if neighbor_score > best_score:
                    best_config = neighbor_config.copy()
                    best_score = neighbor_score
                    
            else:
                # Accept with probability based on temperature (quantum tunneling)
                if HAS_NUMPY:
                    acceptance_probability = np.exp(score_diff / temperature)
                    if np.random.random() < acceptance_probability:
                        current_config = neighbor_config
                        current_score = neighbor_score
            
            temperature *= cooling_rate
        
        return best_config
    
    async def _greedy_optimization(self) -> Dict[str, Any]:
        """Simple greedy optimization."""
        best_config = self.current_configuration.copy()
        best_score = self._estimate_configuration_score(best_config)
        
        # Try small changes to each parameter
        for param_name, current_value in self.current_configuration.items():
            if isinstance(current_value, (int, float)):
                # Try increasing and decreasing the parameter
                for multiplier in [0.9, 1.1]:
                    test_config = self.current_configuration.copy()
                    
                    if isinstance(current_value, int):
                        test_config[param_name] = max(1, int(current_value * multiplier))
                    else:
                        test_config[param_name] = max(0.01, current_value * multiplier)
                    
                    test_score = self._estimate_configuration_score(test_config)
                    
                    if test_score > best_score:
                        best_config = test_config
                        best_score = test_score
        
        return best_config
    
    def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        constraints = {
            "privacy_noise_multiplier": (0.1, 5.0),
            "batch_size": (1, 1000),
            "connection_pool_size": (1, 100),
            "cache_size": (10, 10000),
            "timeout_ms": (100, 60000),
            "retry_count": (0, 10),
            "quantum_coherence_threshold": (0.1, 1.0),
            "entanglement_strength": (0.0, 1.0),
            "interference_damping": (0.0, 1.0)
        }
        
        for param_name, value in config.items():
            if param_name in constraints:
                min_val, max_val = constraints[param_name]
                if not (min_val <= value <= max_val):
                    return False
        
        return True
    
    def _predict_improvement(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> float:
        """Predict improvement from configuration change."""
        old_score = self._estimate_configuration_score(old_config)
        new_score = self._estimate_configuration_score(new_config)
        return new_score - old_score
    
    def _estimate_configuration_score(self, config: Dict[str, Any]) -> float:
        """Estimate performance score for configuration."""
        if not self.performance_history:
            return 0.5  # Default score
        
        # Find most similar configuration in history
        best_similarity = 0.0
        best_score = 0.5
        
        for snapshot in self.performance_history:
            similarity = self._configuration_similarity(config, snapshot.configuration)
            if similarity > best_similarity:
                best_similarity = similarity
                best_score = self._calculate_performance_score(snapshot)
        
        # Adjust score based on configuration changes
        adjustment = 0.0
        
        # Heuristic adjustments based on parameter changes
        if config.get("privacy_noise_multiplier", 1.1) > 1.5:
            adjustment -= 0.1  # Higher noise reduces utility
        if config.get("batch_size", 32) > 64:
            adjustment += 0.05  # Larger batches can improve throughput
        if config.get("cache_size", 1000) > 5000:
            adjustment += 0.03  # Better caching improves performance
        
        return min(1.0, max(0.0, best_score + adjustment))
    
    def _configuration_similarity(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calculate similarity between configurations."""
        if not config1 or not config2:
            return 0.0
        
        common_keys = set(config1.keys()) & set(config2.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        
        for key in common_keys:
            val1, val2 = config1[key], config2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize numeric similarity
                if val1 == val2:
                    similarity_sum += 1.0
                else:
                    max_val = max(abs(val1), abs(val2), 1.0)
                    similarity_sum += 1.0 - (abs(val1 - val2) / max_val)
            elif val1 == val2:
                similarity_sum += 1.0
        
        return similarity_sum / len(common_keys)
    
    def _mutate_configuration(self, config: Dict[str, Any], mutation_strength: float) -> Dict[str, Any]:
        """Create mutated version of configuration."""
        new_config = config.copy()
        
        for param_name, value in config.items():
            if isinstance(value, (int, float)):
                if HAS_NUMPY:
                    noise = np.random.normal(0, mutation_strength)
                else:
                    noise = (hash(str(time.time()) + param_name) % 1000 - 500) / 1000 * mutation_strength
                
                if isinstance(value, int):
                    new_config[param_name] = max(1, int(value * (1 + noise)))
                else:
                    new_config[param_name] = max(0.01, value * (1 + noise))
        
        return new_config
    
    def _crossover_configurations(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring configuration from two parents."""
        child = {}
        
        for key in config1.keys():
            if key in config2:
                # Random selection from parents
                if HAS_NUMPY:
                    use_parent1 = np.random.random() < 0.5
                else:
                    use_parent1 = (hash(key) % 2) == 0
                
                child[key] = config1[key] if use_parent1 else config2[key]
            else:
                child[key] = config1[key]
        
        return child
    
    def _configuration_to_state(self, config: Dict[str, Any]) -> str:
        """Convert configuration to state string for Q-learning."""
        # Discretize continuous parameters
        state_parts = []
        
        for key, value in sorted(config.items()):
            if isinstance(value, float):
                # Discretize to nearest 0.1
                discretized = round(value * 10) / 10
                state_parts.append(f"{key}:{discretized}")
            elif isinstance(value, int):
                state_parts.append(f"{key}:{value}")
            else:
                state_parts.append(f"{key}:{value}")
        
        return "|".join(state_parts)
    
    def _get_possible_actions(self) -> List[str]:
        """Get list of possible actions for RL."""
        return [
            "increase_batch_size",
            "decrease_batch_size", 
            "increase_cache_size",
            "decrease_cache_size",
            "increase_timeout",
            "decrease_timeout",
            "increase_coherence",
            "decrease_coherence",
            "no_change"
        ]
    
    def _apply_action(self, config: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Apply RL action to configuration."""
        new_config = config.copy()
        
        if action == "increase_batch_size":
            new_config["batch_size"] = min(1000, int(config.get("batch_size", 32) * 1.2))
        elif action == "decrease_batch_size":
            new_config["batch_size"] = max(1, int(config.get("batch_size", 32) * 0.8))
        elif action == "increase_cache_size":
            new_config["cache_size"] = min(10000, int(config.get("cache_size", 1000) * 1.5))
        elif action == "decrease_cache_size":
            new_config["cache_size"] = max(10, int(config.get("cache_size", 1000) * 0.7))
        elif action == "increase_timeout":
            new_config["timeout_ms"] = min(60000, int(config.get("timeout_ms", 5000) * 1.3))
        elif action == "decrease_timeout":
            new_config["timeout_ms"] = max(100, int(config.get("timeout_ms", 5000) * 0.8))
        elif action == "increase_coherence":
            new_config["quantum_coherence_threshold"] = min(1.0, config.get("quantum_coherence_threshold", 0.8) + 0.05)
        elif action == "decrease_coherence":
            new_config["quantum_coherence_threshold"] = max(0.1, config.get("quantum_coherence_threshold", 0.8) - 0.05)
        
        return new_config
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress."""
        if not self.optimization_results:
            return {"status": "no_optimizations_completed"}
        
        # Calculate statistics
        improvements = [r.predicted_improvement for r in self.optimization_results]
        avg_improvement = sum(improvements) / len(improvements)
        best_improvement = max(improvements)
        
        convergence_count = sum(1 for r in self.optimization_results if r.convergence_achieved)
        
        return {
            "total_iterations": self.total_iterations,
            "optimization_count": len(self.optimization_results),
            "average_improvement": avg_improvement,
            "best_improvement": best_improvement,
            "convergence_rate": convergence_count / len(self.optimization_results),
            "iterations_since_improvement": self.iterations_since_improvement,
            "current_strategy": self.optimization_strategy.value,
            "learning_mode": self.learning_mode.value,
            "best_performance_score": self.best_performance_score
        }
    
    async def save_state(self, filepath: str):
        """Save optimizer state for persistence."""
        state = {
            "optimization_strategy": self.optimization_strategy.value,
            "learning_mode": self.learning_mode.value,
            "current_configuration": self.current_configuration,
            "best_configuration": self.best_configuration,
            "best_performance_score": self.best_performance_score,
            "total_iterations": self.total_iterations,
            "gradient_momentum": self.gradient_momentum,
            "q_table": self.q_table,
            "optimization_results": [asdict(r) for r in self.optimization_results]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved optimizer state to {filepath}")
    
    async def load_state(self, filepath: str):
        """Load optimizer state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.optimization_strategy = OptimizationStrategy(state.get("optimization_strategy", "greedy"))
            self.learning_mode = LearningMode(state.get("learning_mode", "online"))
            self.current_configuration = state.get("current_configuration", self._initialize_configuration())
            self.best_configuration = state.get("best_configuration", self.current_configuration.copy())
            self.best_performance_score = state.get("best_performance_score", 0.0)
            self.total_iterations = state.get("total_iterations", 0)
            self.gradient_momentum = state.get("gradient_momentum", {})
            self.q_table = state.get("q_table", {})
            
            logger.info(f"Loaded optimizer state from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")
            logger.info("Starting with fresh state")