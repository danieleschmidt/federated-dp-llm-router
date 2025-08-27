#!/usr/bin/env python3
"""
Comprehensive Test Suite for Research Framework

Tests all research capabilities including comparative analysis,
experimental framework, benchmarking, and publication tools.
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json
from pathlib import Path

# Import research modules
from federated_dp_llm.research.comparative_analyzer import (
    ComparativeAnalyzer, BaselineAlgorithm, NovelAlgorithm, 
    AlgorithmType, PerformanceResult
)
from federated_dp_llm.research.experiment_framework import (
    ExperimentFramework, ExperimentConfig, ExperimentType,
    StatisticalAnalyzer
)
from federated_dp_llm.research.benchmarking_suite import (
    BenchmarkingSuite, DatasetManager, BenchmarkType, PerformanceMetrics
)
from federated_dp_llm.research.publication_tools import (
    PublicationGenerator, PublicationConfig, PublicationType,
    VisualizationDashboard, AcademicFormatter
)

# Import adaptive modules
from federated_dp_llm.adaptive.adaptive_optimizer import (
    AdaptiveOptimizer, OptimizationStrategy, LearningMode
)
from federated_dp_llm.adaptive.self_healing_system import (
    SelfHealingSystem, HealthStatus, RecoveryActionType
)

class TestComparativeAnalyzer(unittest.TestCase):
    """Test comparative analysis functionality."""
    
    def setUp(self):
        self.analyzer = ComparativeAnalyzer()
        self.baseline = BaselineAlgorithm("round_robin")
        self.novel = NovelAlgorithm("quantum_weighted", "quantum_inspired")
        
    def test_algorithm_registration(self):
        """Test algorithm registration."""
        self.analyzer.register_algorithm(self.baseline, AlgorithmType.BASELINE)
        self.analyzer.register_algorithm(self.novel, AlgorithmType.NOVEL)
        
        self.assertEqual(len(self.analyzer.algorithms), 2)
        self.assertIn(self.baseline.get_name(), self.analyzer.algorithms)
        self.assertIn(self.novel.get_name(), self.analyzer.algorithms)
    
    async def test_algorithm_execution(self):
        """Test individual algorithm execution."""
        # Mock task data
        task_data = Mock()
        task_data.task_id = "test_task_123"
        task_data.priority = Mock()
        task_data.priority.value = 1
        
        # Test baseline execution
        baseline_result = await self.baseline.execute(task_data, node_count=4)
        self.assertIsInstance(baseline_result, dict)
        self.assertIn("result", baseline_result)
        self.assertIn("node", baseline_result)
        
        # Test novel algorithm execution  
        novel_result = await self.novel.execute(task_data, node_count=4)
        self.assertIsInstance(novel_result, dict)
        self.assertIn("optimization_applied", novel_result)
        self.assertTrue(novel_result["optimization_applied"])
    
    def test_performance_metrics(self):
        """Test performance metric collection."""
        baseline_metrics = self.baseline.get_metrics()
        novel_metrics = self.novel.get_metrics()
        
        # Verify required metrics are present
        required_metrics = ['accuracy', 'latency_ms', 'throughput_rps', 'memory_mb']
        for metric in required_metrics:
            self.assertIn(metric, baseline_metrics)
            self.assertIn(metric, novel_metrics)
        
        # Novel algorithm should show improvements
        self.assertGreater(novel_metrics['accuracy'], baseline_metrics['accuracy'])
        self.assertLess(novel_metrics['latency_ms'], baseline_metrics['latency_ms'])
    
    async def test_comparative_study(self):
        """Test full comparative study execution."""
        # Register algorithms
        self.analyzer.register_algorithm(self.baseline, AlgorithmType.BASELINE)
        self.analyzer.register_algorithm(self.novel, AlgorithmType.NOVEL)
        
        # Create test datasets
        test_datasets = [
            Mock(task_id=f"task_{i}", priority=Mock(value=i%3)) 
            for i in range(5)
        ]
        
        # Run comparative study
        results = await self.analyzer.run_comparative_study(
            test_datasets, runs_per_algorithm=2, statistical_confidence=0.95
        )
        
        self.assertEqual(len(results), 2)  # Two algorithms
        
        # Verify results structure
        for algo_name, algo_results in results.items():
            self.assertGreater(len(algo_results), 0)
            for result in algo_results:
                self.assertIsInstance(result, PerformanceResult)
    
    def test_comparison_report(self):
        """Test comparison report generation."""
        # Create mock results
        mock_results = {
            "baseline_round_robin": [
                PerformanceResult(
                    algorithm_name="baseline_round_robin",
                    algorithm_type=AlgorithmType.BASELINE,
                    accuracy=0.75, latency_ms=150, throughput_rps=100, 
                    memory_mb=512, privacy_epsilon=1.0, energy_consumption_j=2.5,
                    error_rate=0.05, statistical_significance=0.1, execution_time=1.0,
                    metadata={}
                )
            ],
            "novel_quantum_weighted": [
                PerformanceResult(
                    algorithm_name="novel_quantum_weighted",
                    algorithm_type=AlgorithmType.NOVEL,
                    accuracy=0.85, latency_ms=90, throughput_rps=180,
                    memory_mb=384, privacy_epsilon=1.0, energy_consumption_j=1.8,
                    error_rate=0.025, statistical_significance=0.01, execution_time=0.8,
                    metadata={}
                )
            ]
        }
        
        report = self.analyzer.generate_comparison_report(mock_results)
        
        self.assertIn("Comparative Algorithm Analysis Report", report)
        self.assertIn("Performance Summary", report)
        self.assertIn("Statistical Significance", report)
        self.assertIn("baseline_round_robin", report)
        self.assertIn("novel_quantum_weighted", report)


class TestExperimentFramework(unittest.TestCase):
    """Test experimental framework functionality."""
    
    def setUp(self):
        self.config = ExperimentConfig(
            experiment_id="test_exp_001",
            experiment_type=ExperimentType.COMPARATIVE_STUDY,
            name="Test Comparative Study",
            description="Testing experimental framework",
            hypothesis="Novel algorithms outperform baselines",
            success_criteria={"min_accuracy": 0.8},
            num_runs=3,
            random_seed=42
        )
        self.framework = ExperimentFramework(self.config)
    
    def test_experiment_initialization(self):
        """Test experiment framework initialization."""
        self.assertEqual(self.framework.config.experiment_id, "test_exp_001")
        self.assertEqual(self.framework.config.num_runs, 3)
        self.assertIsInstance(self.framework.results, list)
        self.assertIsInstance(self.framework.experiment_log, list)
    
    def test_statistical_analyzer(self):
        """Test statistical analysis tools."""
        # Test effect size calculation
        control_group = [0.75, 0.76, 0.74, 0.77, 0.75]
        treatment_group = [0.85, 0.84, 0.86, 0.85, 0.87]
        
        effect_size = StatisticalAnalyzer.calculate_effect_size(control_group, treatment_group)
        
        self.assertIn("cohens_d", effect_size)
        self.assertIn("interpretation", effect_size)
        self.assertGreater(effect_size["cohens_d"], 0)  # Should show positive effect
        
        # Test power analysis
        sample_size = StatisticalAnalyzer.power_analysis(0.5, 0.05, 0.8)
        self.assertIsInstance(sample_size, int)
        self.assertGreater(sample_size, 0)
        
        # Test confidence interval
        ci = StatisticalAnalyzer.confidence_interval([0.75, 0.76, 0.74, 0.77, 0.75])
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])  # Lower bound <= upper bound
    
    async def test_experiment_execution(self):
        """Test experiment execution."""
        # Create mock algorithms
        mock_algorithms = {
            "baseline": Mock(),
            "novel": Mock()
        }
        
        # Setup mock execute methods
        async def mock_execute(data, **kwargs):
            return {"accuracy": 0.8, "processing_time": 0.1}
        
        for algo in mock_algorithms.values():
            algo.execute = mock_execute
            algo.get_metrics = lambda: {"accuracy": 0.8, "latency_ms": 100}
        
        # Create mock datasets
        mock_datasets = [{"data": f"dataset_{i}"} for i in range(2)]
        
        # Run experiment
        results = await self.framework.run_experiment(mock_algorithms, mock_datasets)
        
        self.assertIn("experiment_config", results)
        self.assertIn("statistical_analysis", results)
        self.assertIn("hypothesis_validation", results)
        self.assertIn("reproducibility_hash", results)
    
    def test_hypothesis_validation(self):
        """Test hypothesis validation logic."""
        # Create mock statistical results
        statistical_results = {
            "descriptive_statistics": {
                "baseline": {"accuracy_mean": 0.75},
                "novel": {"accuracy_mean": 0.85}
            }
        }
        
        hypothesis_result = self.framework._validate_hypothesis(statistical_results)
        
        self.assertIn("hypothesis", hypothesis_result)
        self.assertIn("validation_results", hypothesis_result)
        self.assertIn("overall_validation", hypothesis_result)


class TestBenchmarkingSuite(unittest.TestCase):
    """Test benchmarking suite functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.suite = BenchmarkingSuite(output_dir=self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_manager_initialization(self):
        """Test dataset manager initialization."""
        self.assertIsInstance(self.suite.dataset_manager, DatasetManager)
        
        # Check if standard datasets are registered
        self.assertIn("synthetic_medical_1k", self.suite.dataset_manager.datasets)
        self.assertIn("clinical_notes_500", self.suite.dataset_manager.datasets)
    
    async def test_dataset_loading(self):
        """Test dataset loading functionality."""
        # Load a standard dataset
        dataset, info = await self.suite.dataset_manager.load_dataset("synthetic_medical_1k")
        
        self.assertIsInstance(dataset, dict)
        self.assertIn("samples", dataset)
        self.assertIn("type", dataset)
        self.assertGreater(len(dataset["samples"]), 0)
        
        # Verify dataset structure
        sample = dataset["samples"][0]
        self.assertIn("id", sample)
        self.assertIn("query", sample)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        metrics = PerformanceMetrics(
            accuracy=0.85, precision=0.83, recall=0.87, f1_score=0.85,
            latency_p50_ms=100, latency_p95_ms=200, latency_p99_ms=300,
            throughput_rps=150, memory_usage_mb=512, cpu_utilization_percent=70,
            gpu_utilization_percent=45, network_io_mbps=100, disk_io_mbps=50,
            energy_consumption_j=2.0, privacy_epsilon_consumed=0.1,
            error_rate=0.02, availability_percent=99.5
        )
        
        # Test composite score calculation
        composite_score = metrics.calculate_composite_score()
        self.assertIsInstance(composite_score, float)
        self.assertGreaterEqual(composite_score, 0.0)
        self.assertLessEqual(composite_score, 1.0)
        
        # Test dictionary conversion
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn("accuracy", metrics_dict)
    
    def test_benchmark_registration(self):
        """Test benchmark registration."""
        def dummy_evaluation(algorithm, data):
            return {"accuracy": 0.8}
        
        self.suite.register_benchmark(
            name="test_benchmark",
            benchmark_type=BenchmarkType.ACCURACY,
            description="Test benchmark",
            datasets=["synthetic_medical_1k"],
            evaluation_function=dummy_evaluation,
            expected_metrics={"accuracy": 0.75}
        )
        
        self.assertIn("test_benchmark", self.suite.benchmarks)
        benchmark_info = self.suite.benchmarks["test_benchmark"]
        self.assertEqual(benchmark_info["type"], BenchmarkType.ACCURACY)
    
    async def test_benchmark_execution(self):
        """Test benchmark execution."""
        # Create mock algorithm
        mock_algorithm = Mock()
        mock_algorithm.name = "test_algorithm"
        mock_algorithm.get_metrics = lambda: {
            "accuracy": 0.85, "latency_ms": 120, "throughput_rps": 140,
            "memory_mb": 400, "privacy_epsilon": 0.1, "energy_consumption_j": 1.8,
            "error_rate": 0.03
        }
        
        async def mock_execute(data, **kwargs):
            return {"accuracy": 0.85}
        
        mock_algorithm.execute = mock_execute
        
        # Register a test benchmark
        self.suite.register_benchmark(
            name="accuracy_test",
            benchmark_type=BenchmarkType.ACCURACY, 
            description="Accuracy test",
            datasets=["synthetic_medical_1k"],
            evaluation_function=lambda algo, data: {"accuracy": 0.85},
            expected_metrics={"accuracy": 0.8}
        )
        
        # Run benchmark
        result = await self.suite.run_benchmark("accuracy_test", mock_algorithm)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm_name, "test_algorithm")
        self.assertIsInstance(result.metrics, PerformanceMetrics)
        self.assertTrue(result.validation_passed)


class TestPublicationTools(unittest.TestCase):
    """Test academic publication tools."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PublicationConfig(
            title="Test Publication",
            authors=["Dr. Test Author"],
            institution="Test University",
            abstract="This is a test abstract for the publication.",
            keywords=["federated learning", "privacy", "healthcare"],
            publication_type=PublicationType.CONFERENCE_PAPER,
            conference_venue="Test Conference 2025"
        )
        self.generator = PublicationGenerator(self.config, self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_visualization_dashboard(self):
        """Test visualization generation."""
        # Create mock results
        mock_results = {
            "baseline": [Mock(accuracy=0.75, latency_ms=150, throughput_rps=100)],
            "novel": [Mock(accuracy=0.85, latency_ms=90, throughput_rps=180)]
        }
        
        viz_file = self.generator.visualizer.generate_performance_comparison(mock_results)
        
        self.assertTrue(Path(viz_file).exists())
        
        # Check file content
        with open(viz_file, 'r') as f:
            content = f.read()
            self.assertIn("Algorithm Performance Comparison", content)
            self.assertIn("baseline", content)
            self.assertIn("novel", content)
    
    def test_academic_formatter(self):
        """Test academic formatting."""
        # Test statistical results formatting
        statistical_results = {
            "descriptive_statistics": {
                "baseline": {
                    "accuracy_mean": 0.75,
                    "accuracy_std": 0.02,
                    "latency_mean": 150,
                    "success_rate": 0.95,
                    "sample_size": 10
                }
            },
            "effect_sizes": {
                "baseline_vs_novel": {
                    "cohens_d": 0.8,
                    "interpretation": "large",
                    "improvement": 13.3
                }
            }
        }
        
        formatted_results = AcademicFormatter.format_statistical_results(statistical_results)
        
        self.assertIn("## Results", formatted_results)
        self.assertIn("Table 1", formatted_results)
        self.assertIn("Effect Size Analysis", formatted_results)
        self.assertIn("Cohen's d", formatted_results)
    
    def test_methodology_section(self):
        """Test methodology section generation."""
        experiment_config = {
            "experiment_type": "comparative_study",
            "num_runs": 10,
            "significance_level": 0.05,
            "statistical_power": 0.8,
            "random_seed": 42
        }
        
        methodology = AcademicFormatter.format_methodology_section(experiment_config)
        
        self.assertIn("## Methodology", methodology)
        self.assertIn("comparative_study", methodology)
        self.assertIn("10 independent runs", methodology)
        self.assertIn("α = 0.05", methodology)
        self.assertIn("Reproducibility", methodology)
    
    async def test_publication_generation(self):
        """Test complete publication generation."""
        research_results = {
            "experiment_config": {"num_runs": 5},
            "statistical_analysis": {
                "descriptive_statistics": {
                    "baseline": {"accuracy_mean": 0.75}
                }
            },
            "key_findings": [
                "Novel algorithms outperform baselines",
                "Privacy guarantees maintained"
            ]
        }
        
        publication_file = await self.generator.generate_publication(research_results)
        
        self.assertTrue(Path(publication_file).exists())
        
        # Check publication content
        with open(publication_file, 'r') as f:
            content = f.read()
            self.assertIn(self.config.title, content)
            self.assertIn(self.config.abstract, content)
            self.assertIn("## Results", content)
            self.assertIn("## Conclusion", content)


class TestAdaptiveOptimizer(unittest.TestCase):
    """Test adaptive optimization functionality."""
    
    def setUp(self):
        self.optimizer = AdaptiveOptimizer(
            optimization_strategy=OptimizationStrategy.GREEDY,
            learning_mode=LearningMode.ONLINE,
            learning_rate=0.01
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.optimization_strategy, OptimizationStrategy.GREEDY)
        self.assertEqual(self.optimizer.learning_mode, LearningMode.ONLINE)
        self.assertIsInstance(self.optimizer.current_configuration, dict)
        self.assertGreater(len(self.optimizer.current_configuration), 0)
    
    async def test_performance_recording(self):
        """Test performance recording and adaptation triggering."""
        # Record multiple performance observations
        for i in range(5):
            triggered = await self.optimizer.record_performance(
                accuracy=0.8 - i*0.01,  # Declining performance
                latency_p95=150 + i*10,
                throughput=100 - i*5,
                resource_utilization=0.7 + i*0.02,
                privacy_budget_remaining=0.5 - i*0.05,
                error_rate=0.02 + i*0.005
            )
        
        # Should have recorded performance history
        self.assertGreater(len(self.optimizer.performance_history), 0)
        
        # Check that performance score calculation works
        latest_snapshot = self.optimizer.performance_history[-1]
        score = self.optimizer._calculate_performance_score(latest_snapshot)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            "privacy_noise_multiplier": 1.1,
            "batch_size": 32,
            "connection_pool_size": 10,
            "cache_size": 1000
        }
        self.assertTrue(self.optimizer._validate_configuration(valid_config))
        
        # Invalid configuration (out of bounds)
        invalid_config = {
            "privacy_noise_multiplier": 10.0,  # Too high
            "batch_size": -5,  # Invalid
            "timeout_ms": 100000  # Too high
        }
        self.assertFalse(self.optimizer._validate_configuration(invalid_config))
    
    def test_configuration_mutation(self):
        """Test configuration mutation for evolutionary algorithms."""
        original_config = self.optimizer.current_configuration.copy()
        mutated_config = self.optimizer._mutate_configuration(original_config, 0.1)
        
        # Should be different from original
        self.assertNotEqual(original_config, mutated_config)
        
        # Should maintain same keys
        self.assertEqual(set(original_config.keys()), set(mutated_config.keys()))
        
        # Should be valid
        self.assertTrue(self.optimizer._validate_configuration(mutated_config))
    
    def test_configuration_similarity(self):
        """Test configuration similarity calculation."""
        config1 = {"param1": 1.0, "param2": 2.0, "param3": "value"}
        config2 = {"param1": 1.1, "param2": 2.0, "param3": "value"}  # Similar
        config3 = {"param1": 5.0, "param2": 10.0, "param3": "different"}  # Different
        
        sim12 = self.optimizer._configuration_similarity(config1, config2)
        sim13 = self.optimizer._configuration_similarity(config1, config3)
        
        self.assertGreater(sim12, sim13)  # config1 more similar to config2
        self.assertGreaterEqual(sim12, 0.0)
        self.assertLessEqual(sim12, 1.0)
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        summary = self.optimizer.get_optimization_summary()
        
        self.assertIn("total_iterations", summary)
        self.assertIn("current_strategy", summary)
        self.assertIn("learning_mode", summary)
        self.assertEqual(summary["current_strategy"], "greedy")
        self.assertEqual(summary["learning_mode"], "online")


class TestSelfHealingSystem(unittest.TestCase):
    """Test self-healing system functionality."""
    
    def setUp(self):
        self.healing_system = SelfHealingSystem(
            health_check_interval=1.0,  # Fast for testing
            critical_threshold=0.3,
            degraded_threshold=0.6
        )
    
    def test_healing_system_initialization(self):
        """Test healing system initialization."""
        self.assertEqual(self.healing_system.system_status, HealthStatus.HEALTHY)
        self.assertFalse(self.healing_system.monitoring_active)
        self.assertGreater(len(self.healing_system.available_actions), 0)
        
        # Check recovery actions are properly initialized
        action_types = {action.action_type for action in self.healing_system.available_actions}
        expected_types = {
            RecoveryActionType.RESTART_COMPONENT,
            RecoveryActionType.SCALE_UP_RESOURCES,
            RecoveryActionType.REDUCE_LOAD,
            RecoveryActionType.CLEAR_CACHE
        }
        self.assertTrue(expected_types.issubset(action_types))
    
    async def test_health_metrics_collection(self):
        """Test health metrics collection."""
        metrics = await self.healing_system._collect_health_metrics()
        
        self.assertIsInstance(metrics.timestamp, float)
        self.assertGreaterEqual(metrics.cpu_utilization, 0.0)
        self.assertLessEqual(metrics.cpu_utilization, 1.0)
        self.assertGreaterEqual(metrics.error_rate, 0.0)
        self.assertLessEqual(metrics.error_rate, 1.0)
        
        # Test health score calculation
        health_score = metrics.overall_health_score()
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)
    
    def test_issue_diagnosis(self):
        """Test issue diagnosis from health metrics."""
        from federated_dp_llm.adaptive.self_healing_system import HealthMetrics
        
        # Create metrics indicating problems
        problematic_metrics = HealthMetrics(
            timestamp=time.time(),
            cpu_utilization=0.95,  # High CPU
            memory_utilization=0.92,  # High memory
            disk_utilization=0.5,
            network_latency=100,
            error_rate=0.08,  # High error rate
            response_time_p95=2500,  # Slow responses
            throughput=50,
            active_connections=200,
            queue_size=6000,  # Large queue
            privacy_budget_remaining=0.05,  # Low privacy budget
            node_availability={"node_0": False, "node_1": True},  # Node down
            component_health={"service_1": HealthStatus.CRITICAL}
        )
        
        issues = self.healing_system._diagnose_issues(problematic_metrics)
        
        expected_issues = [
            "high_cpu_utilization",
            "high_memory_utilization", 
            "high_error_rate",
            "slow_response_times",
            "queue_backlog",
            "privacy_budget_exhausted"
        ]
        
        for expected_issue in expected_issues:
            self.assertIn(expected_issue, issues)
    
    def test_root_cause_analysis(self):
        """Test root cause determination."""
        # Test different issue patterns
        cpu_memory_issues = ["high_cpu_utilization", "slow_response_times"]
        root_cause1 = self.healing_system._determine_root_cause(cpu_memory_issues, None)
        self.assertEqual(root_cause1, "resource_exhaustion_cpu_bottleneck")
        
        memory_queue_issues = ["high_memory_utilization", "queue_backlog"]
        root_cause2 = self.healing_system._determine_root_cause(memory_queue_issues, None)
        self.assertEqual(root_cause2, "resource_exhaustion_memory_bottleneck")
        
        component_failure_issues = ["high_error_rate", "components_unhealthy:service_1"]
        root_cause3 = self.healing_system._determine_root_cause(component_failure_issues, None)
        self.assertEqual(root_cause3, "component_failure_cascade")
    
    def test_recovery_action_selection(self):
        """Test recovery action selection."""
        issues = ["high_cpu_utilization", "slow_response_times"]
        root_cause = "resource_exhaustion_cpu_bottleneck"
        
        selected_actions = self.healing_system._select_recovery_actions(
            issues, root_cause, HealthStatus.CRITICAL
        )
        
        self.assertGreater(len(selected_actions), 0)
        
        # Should prioritize resource-related actions
        action_types = {action.action_type for action in selected_actions}
        self.assertIn(RecoveryActionType.SCALE_UP_RESOURCES, action_types)
    
    async def test_incident_simulation(self):
        """Test incident simulation and recovery."""
        initial_status = self.healing_system.system_status
        
        # Simulate a high load incident
        await self.healing_system.simulate_incident("high_load")
        
        # Should have generated recovery history
        self.assertGreater(len(self.healing_system.recovery_history), 0)
        
        # Check incident report structure
        incident_report = self.healing_system.recovery_history[-1]
        self.assertIsInstance(incident_report.incident_id, str)
        self.assertGreater(len(incident_report.detected_issues), 0)
        self.assertIsInstance(incident_report.root_cause_analysis, str)
        self.assertIsInstance(incident_report.recovery_actions_taken, list)
    
    def test_health_summary(self):
        """Test health summary generation."""
        summary = self.healing_system.get_health_summary()
        
        required_fields = [
            "current_status", "monitoring_active", "recovery_in_progress",
            "total_incidents", "successful_recoveries", "available_actions"
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        self.assertEqual(summary["current_status"], "healthy")
        self.assertFalse(summary["monitoring_active"])
        self.assertEqual(summary["total_incidents"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete research framework."""
    
    async def test_full_research_pipeline(self):
        """Test complete research pipeline from experiment to publication."""
        # Setup comparative analyzer
        analyzer = ComparativeAnalyzer()
        baseline = BaselineAlgorithm("baseline_lb")
        novel = NovelAlgorithm("quantum_lb", "quantum_inspired")
        
        analyzer.register_algorithm(baseline, AlgorithmType.BASELINE)
        analyzer.register_algorithm(novel, AlgorithmType.NOVEL)
        
        # Setup experiment framework
        config = ExperimentConfig(
            experiment_id="integration_test",
            experiment_type=ExperimentType.COMPARATIVE_STUDY,
            name="Integration Test Study",
            description="Full pipeline test",
            hypothesis="Novel algorithms outperform baselines",
            success_criteria={"min_accuracy": 0.8},
            num_runs=2
        )
        
        framework = ExperimentFramework(config)
        
        # Create mock datasets
        test_datasets = [Mock(task_id=f"task_{i}") for i in range(3)]
        
        # Run experiment
        algorithms = {"baseline": baseline, "novel": novel}
        results = await framework.run_experiment(algorithms, test_datasets)
        
        # Verify experiment results
        self.assertIn("statistical_analysis", results)
        self.assertIn("hypothesis_validation", results)
        
        # Test publication generation
        pub_config = PublicationConfig(
            title="Integration Test Publication",
            authors=["Test Author"],
            institution="Test Institution", 
            abstract="Test abstract",
            keywords=["test", "integration"],
            publication_type=PublicationType.TECHNICAL_REPORT
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pub_generator = PublicationGenerator(pub_config, temp_dir)
            pub_file = await pub_generator.generate_publication(results)
            
            self.assertTrue(Path(pub_file).exists())
    
    async def test_adaptive_system_integration(self):
        """Test integration of adaptive optimization and self-healing."""
        # Setup adaptive optimizer
        optimizer = AdaptiveOptimizer(
            optimization_strategy=OptimizationStrategy.GREEDY,
            adaptation_frequency=5  # Adapt every 5 observations
        )
        
        # Setup self-healing system
        healer = SelfHealingSystem(health_check_interval=0.1)
        
        # Simulate system operation with declining performance
        for i in range(10):
            # Record declining performance
            await optimizer.record_performance(
                accuracy=0.9 - i*0.02,
                latency_p95=100 + i*20,
                throughput=200 - i*10,
                resource_utilization=0.6 + i*0.03,
                privacy_budget_remaining=0.8 - i*0.05,
                error_rate=0.01 + i*0.01
            )
        
        # Check that optimization was triggered
        optimization_summary = optimizer.get_optimization_summary()
        self.assertGreaterEqual(optimization_summary["total_iterations"], 0)
        
        # Simulate health check
        health_metrics = await healer.force_health_check()
        self.assertIsNotNone(health_metrics)
        
        # Test system summary
        health_summary = healer.get_health_summary()
        self.assertIn("current_status", health_summary)


def run_tests():
    """Run all tests."""
    print("🧪 Running Research Framework Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestComparativeAnalyzer,
        TestExperimentFramework, 
        TestBenchmarkingSuite,
        TestPublicationTools,
        TestAdaptiveOptimizer,
        TestSelfHealingSystem,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if not result.failures and not result.errors:
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    import time
    
    # Handle async tests
    async def main():
        return run_tests()
    
    success = asyncio.run(main())
    exit(0 if success else 1)