"""
Adaptive Learning and Self-Improving Systems

Advanced adaptive algorithms that learn from usage patterns and
continuously optimize performance based on real-world feedback.
"""

from .adaptive_optimizer import AdaptiveOptimizer, OptimizationStrategy, LearningMode
from .self_healing_system import SelfHealingSystem, HealthMetrics, RecoveryAction
from .performance_predictor import PerformancePredictor, PredictionModel, MetricType
from .auto_scaling_engine import AutoScalingEngine, ScalingPolicy, ResourceMetrics

__all__ = [
    "AdaptiveOptimizer",
    "OptimizationStrategy",
    "LearningMode",
    "SelfHealingSystem", 
    "HealthMetrics",
    "RecoveryAction",
    "PerformancePredictor",
    "PredictionModel",
    "MetricType",
    "AutoScalingEngine",
    "ScalingPolicy",
    "ResourceMetrics",
]