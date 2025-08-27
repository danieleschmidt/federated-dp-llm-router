"""
Research Module for Federated DP-LLM Router

Advanced research capabilities including comparative studies, 
novel algorithm development, and academic publication support.
"""

from .comparative_analyzer import ComparativeAnalyzer, BaselineAlgorithm, NovelAlgorithm
from .experiment_framework import ExperimentFramework, ExperimentConfig, StatisticalAnalyzer
from .benchmarking_suite import BenchmarkingSuite, PerformanceMetrics, DatasetManager
from .publication_tools import PublicationGenerator, AcademicFormatter, VisualizationDashboard

__all__ = [
    "ComparativeAnalyzer",
    "BaselineAlgorithm", 
    "NovelAlgorithm",
    "ExperimentFramework",
    "ExperimentConfig",
    "StatisticalAnalyzer",
    "BenchmarkingSuite",
    "PerformanceMetrics",
    "DatasetManager",
    "PublicationGenerator", 
    "AcademicFormatter",
    "VisualizationDashboard",
]