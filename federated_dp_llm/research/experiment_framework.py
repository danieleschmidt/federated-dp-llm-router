"""
Experimental Framework for Reproducible Research

Implements controlled experiments with proper baselines, statistical analysis,
and reproducible research methodologies.
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import logging
from pathlib import Path
import pickle

from ..quantum_planning.numpy_fallback import get_numpy_backend
from .comparative_analyzer import PerformanceResult, AlgorithmInterface

HAS_NUMPY, np = get_numpy_backend()
logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of experiments supported."""
    COMPARATIVE_STUDY = "comparative_study"
    ABLATION_STUDY = "ablation_study" 
    SCALABILITY_TEST = "scalability_test"
    PRIVACY_ANALYSIS = "privacy_analysis"
    LONGITUDINAL_STUDY = "longitudinal_study"


@dataclass
class ExperimentConfig:
    """Configuration for controlled experiments."""
    experiment_id: str
    experiment_type: ExperimentType
    name: str
    description: str
    hypothesis: str
    success_criteria: Dict[str, float]
    num_runs: int = 10
    random_seed: int = 42
    statistical_power: float = 0.8
    significance_level: float = 0.05
    controlled_variables: List[str] = None
    output_directory: str = "./experiment_results"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.controlled_variables is None:
            self.controlled_variables = []
        if self.metadata is None:
            self.metadata = {}


class StatisticalAnalyzer:
    """Statistical analysis tools for experiments."""
    
    @staticmethod
    def calculate_effect_size(control_group: List[float], 
                            treatment_group: List[float]) -> Dict[str, float]:
        """Calculate Cohen's d effect size."""
        if not HAS_NUMPY:
            return {"cohens_d": 0.0, "interpretation": "unable_to_calculate"}
            
        control_mean = np.mean(control_group)
        treatment_mean = np.mean(treatment_group)
        
        control_std = np.std(control_group, ddof=1)
        treatment_std = np.std(treatment_group, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(control_group), len(treatment_group)
        pooled_std = np.sqrt(((n1-1)*control_std**2 + (n2-1)*treatment_std**2) / (n1+n2-2))
        
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small" 
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
            
        return {
            "cohens_d": cohens_d,
            "interpretation": interpretation,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "improvement": (treatment_mean - control_mean) / control_mean * 100
        }
    
    @staticmethod
    def power_analysis(effect_size: float, 
                      alpha: float = 0.05, 
                      power: float = 0.8) -> int:
        """Estimate required sample size for given statistical power."""
        # Simplified power analysis
        if effect_size <= 0:
            return 1000  # Conservative estimate
            
        # Approximation for two-sample t-test
        z_alpha = 1.96  # For alpha = 0.05
        z_beta = 0.84   # For power = 0.8
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return max(int(n), 5)  # Minimum 5 samples
    
    @staticmethod
    def confidence_interval(data: List[float], 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval."""
        if not HAS_NUMPY or len(data) < 2:
            mean_val = sum(data) / len(data) if data else 0
            return (mean_val, mean_val)
            
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(len(data))
        
        # Use t-distribution approximation
        t_value = 2.0  # Approximation for t_{n-1, alpha/2}
        margin = t_value * std_err
        
        return (mean - margin, mean + margin)


class ExperimentFramework:
    """Comprehensive experimental framework for reproducible research."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.experiment_log: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Set random seed for reproducibility
        if HAS_NUMPY:
            np.random.seed(config.random_seed)
        
        # Create output directory
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized experiment: {config.name}")
        
    def log(self, message: str):
        """Log experiment progress."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.experiment_log.append(log_entry)
        logger.info(message)
    
    async def run_experiment(self, 
                           algorithms: Dict[str, AlgorithmInterface],
                           datasets: List[Any],
                           custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Run controlled experiment with statistical rigor."""
        self.start_time = time.time()
        self.log(f"Starting {self.config.experiment_type.value}: {self.config.name}")
        
        # Validate experimental design
        self._validate_experimental_design(algorithms, datasets)
        
        # Execute experimental protocol
        if self.config.experiment_type == ExperimentType.COMPARATIVE_STUDY:
            results = await self._run_comparative_study(algorithms, datasets)
        elif self.config.experiment_type == ExperimentType.ABLATION_STUDY:
            results = await self._run_ablation_study(algorithms, datasets)
        elif self.config.experiment_type == ExperimentType.SCALABILITY_TEST:
            results = await self._run_scalability_test(algorithms, datasets)
        else:
            results = await self._run_generic_experiment(algorithms, datasets)
        
        # Statistical analysis
        statistical_results = self._perform_statistical_analysis(results)
        
        # Validate hypothesis
        hypothesis_validation = self._validate_hypothesis(statistical_results)
        
        self.end_time = time.time()
        experiment_duration = self.end_time - self.start_time
        
        # Compile final results
        final_results = {
            "experiment_config": asdict(self.config),
            "experiment_duration_seconds": experiment_duration,
            "raw_results": results,
            "statistical_analysis": statistical_results,
            "hypothesis_validation": hypothesis_validation,
            "reproducibility_hash": self._generate_reproducibility_hash(),
            "experiment_log": self.experiment_log
        }
        
        # Save results
        await self._save_results(final_results)
        
        self.log(f"Experiment completed in {experiment_duration:.2f} seconds")
        return final_results
    
    def _validate_experimental_design(self, 
                                    algorithms: Dict[str, AlgorithmInterface],
                                    datasets: List[Any]):
        """Validate experimental design for statistical validity."""
        if len(algorithms) < 2:
            self.log("WARNING: Less than 2 algorithms - limited comparative value")
        
        if len(datasets) < 3:
            self.log("WARNING: Less than 3 datasets - limited generalizability")
        
        # Power analysis
        expected_effect_size = self.config.metadata.get("expected_effect_size", 0.5)
        required_sample_size = StatisticalAnalyzer.power_analysis(
            expected_effect_size, 
            self.config.significance_level,
            self.config.statistical_power
        )
        
        total_samples = self.config.num_runs * len(datasets)
        if total_samples < required_sample_size:
            self.log(f"WARNING: Sample size ({total_samples}) may be insufficient. "
                    f"Recommended: {required_sample_size}")
    
    async def _run_comparative_study(self, 
                                   algorithms: Dict[str, AlgorithmInterface],
                                   datasets: List[Any]) -> List[Dict[str, Any]]:
        """Run comparative study between algorithms."""
        all_results = []
        
        for run_idx in range(self.config.num_runs):
            self.log(f"Run {run_idx + 1}/{self.config.num_runs}")
            
            for dataset_idx, dataset in enumerate(datasets):
                for algo_name, algorithm in algorithms.items():
                    result = await self._execute_single_trial(
                        algorithm, dataset, run_idx, dataset_idx, algo_name
                    )
                    all_results.append(result)
                    
        return all_results
    
    async def _run_ablation_study(self,
                                algorithms: Dict[str, AlgorithmInterface], 
                                datasets: List[Any]) -> List[Dict[str, Any]]:
        """Run ablation study to understand component contributions."""
        # Identify base algorithm and variants
        base_algo = None
        variants = []
        
        for name, algo in algorithms.items():
            if "base" in name.lower() or "full" in name.lower():
                base_algo = (name, algo)
            else:
                variants.append((name, algo))
        
        if not base_algo:
            base_algo = list(algorithms.items())[0]
            variants = list(algorithms.items())[1:]
        
        all_results = []
        
        # Test base algorithm
        self.log(f"Testing base algorithm: {base_algo[0]}")
        for run_idx in range(self.config.num_runs):
            for dataset_idx, dataset in enumerate(datasets):
                result = await self._execute_single_trial(
                    base_algo[1], dataset, run_idx, dataset_idx, base_algo[0]
                )
                result["ablation_type"] = "base"
                all_results.append(result)
        
        # Test each variant
        for variant_name, variant_algo in variants:
            self.log(f"Testing variant: {variant_name}")
            for run_idx in range(self.config.num_runs):
                for dataset_idx, dataset in enumerate(datasets):
                    result = await self._execute_single_trial(
                        variant_algo, dataset, run_idx, dataset_idx, variant_name
                    )
                    result["ablation_type"] = "variant"
                    all_results.append(result)
        
        return all_results
    
    async def _run_scalability_test(self,
                                  algorithms: Dict[str, AlgorithmInterface],
                                  datasets: List[Any]) -> List[Dict[str, Any]]:
        """Test algorithm scalability across different loads."""
        all_results = []
        
        # Test different scales
        scales = [1, 2, 4, 8, 16, 32]
        
        for scale in scales:
            self.log(f"Testing scale factor: {scale}x")
            
            for algo_name, algorithm in algorithms.items():
                for run_idx in range(max(1, self.config.num_runs // len(scales))):
                    for dataset_idx, dataset in enumerate(datasets):
                        # Simulate scaled load
                        scaled_dataset = self._scale_dataset(dataset, scale)
                        
                        result = await self._execute_single_trial(
                            algorithm, scaled_dataset, run_idx, dataset_idx, algo_name
                        )
                        result["scale_factor"] = scale
                        all_results.append(result)
        
        return all_results
    
    async def _run_generic_experiment(self,
                                    algorithms: Dict[str, AlgorithmInterface],
                                    datasets: List[Any]) -> List[Dict[str, Any]]:
        """Run generic experimental protocol."""
        return await self._run_comparative_study(algorithms, datasets)
    
    async def _execute_single_trial(self,
                                   algorithm: AlgorithmInterface,
                                   dataset: Any,
                                   run_idx: int,
                                   dataset_idx: int,
                                   algo_name: str) -> Dict[str, Any]:
        """Execute single experimental trial."""
        trial_start = time.time()
        
        try:
            # Execute algorithm
            result = await algorithm.execute(dataset, 
                                          run_idx=run_idx,
                                          experiment_mode=True)
            
            # Get metrics
            metrics = algorithm.get_metrics()
            
            execution_successful = True
            error_message = None
            
        except Exception as e:
            logger.error(f"Trial failed for {algo_name}: {e}")
            result = None
            metrics = {}
            execution_successful = False
            error_message = str(e)
        
        trial_duration = time.time() - trial_start
        
        return {
            "algorithm_name": algo_name,
            "run_idx": run_idx,
            "dataset_idx": dataset_idx,
            "execution_successful": execution_successful,
            "error_message": error_message,
            "trial_duration": trial_duration,
            "metrics": metrics,
            "result": result,
            "timestamp": time.time()
        }
    
    def _scale_dataset(self, dataset: Any, scale_factor: int) -> Any:
        """Create scaled version of dataset for scalability testing."""
        # Simple scaling - in practice, this would be more sophisticated
        if hasattr(dataset, 'scale'):
            return dataset.scale(scale_factor)
        else:
            # Return original dataset with scale metadata
            return {
                "original_dataset": dataset,
                "scale_factor": scale_factor
            }
    
    def _perform_statistical_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Group results by algorithm
        algo_groups = {}
        for result in results:
            algo_name = result["algorithm_name"]
            if algo_name not in algo_groups:
                algo_groups[algo_name] = []
            algo_groups[algo_name].append(result)
        
        statistical_analysis = {
            "descriptive_statistics": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "power_analysis": {}
        }
        
        # Descriptive statistics for each algorithm
        for algo_name, algo_results in algo_groups.items():
            successful_results = [r for r in algo_results if r["execution_successful"]]
            
            if not successful_results:
                continue
            
            # Extract accuracy values (example metric)
            accuracies = [r["metrics"].get("accuracy", 0.0) for r in successful_results]
            latencies = [r["metrics"].get("latency_ms", 1000.0) for r in successful_results]
            
            if accuracies:
                statistical_analysis["descriptive_statistics"][algo_name] = {
                    "accuracy_mean": sum(accuracies) / len(accuracies),
                    "accuracy_std": (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5,
                    "latency_mean": sum(latencies) / len(latencies),
                    "success_rate": len(successful_results) / len(algo_results),
                    "sample_size": len(accuracies)
                }
                
                # Confidence intervals
                ci = StatisticalAnalyzer.confidence_interval(accuracies)
                statistical_analysis["confidence_intervals"][algo_name] = {
                    "accuracy_ci_lower": ci[0],
                    "accuracy_ci_upper": ci[1]
                }
        
        # Pairwise effect sizes
        algo_names = list(algo_groups.keys())
        for i, algo1 in enumerate(algo_names):
            for algo2 in algo_names[i+1:]:
                group1_results = [r for r in algo_groups[algo1] if r["execution_successful"]]
                group2_results = [r for r in algo_groups[algo2] if r["execution_successful"]]
                
                group1_acc = [r["metrics"].get("accuracy", 0.0) for r in group1_results]
                group2_acc = [r["metrics"].get("accuracy", 0.0) for r in group2_results]
                
                if group1_acc and group2_acc:
                    effect_analysis = StatisticalAnalyzer.calculate_effect_size(group1_acc, group2_acc)
                    statistical_analysis["effect_sizes"][f"{algo1}_vs_{algo2}"] = effect_analysis
        
        return statistical_analysis
    
    def _validate_hypothesis(self, statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experimental hypothesis against results."""
        hypothesis_validation = {
            "hypothesis": self.config.hypothesis,
            "success_criteria": self.config.success_criteria,
            "validation_results": {},
            "overall_validation": "unknown"
        }
        
        # Check each success criterion
        criteria_met = 0
        total_criteria = len(self.config.success_criteria)
        
        for criterion, threshold in self.config.success_criteria.items():
            if criterion in statistical_results.get("descriptive_statistics", {}):
                # Example: check if accuracy improvement criterion is met
                best_accuracy = 0
                for algo_stats in statistical_results["descriptive_statistics"].values():
                    if algo_stats.get("accuracy_mean", 0) > best_accuracy:
                        best_accuracy = algo_stats["accuracy_mean"]
                
                if criterion == "min_accuracy" and best_accuracy >= threshold:
                    hypothesis_validation["validation_results"][criterion] = "met"
                    criteria_met += 1
                else:
                    hypothesis_validation["validation_results"][criterion] = "not_met"
            else:
                hypothesis_validation["validation_results"][criterion] = "unable_to_evaluate"
        
        # Overall validation
        if criteria_met == total_criteria:
            hypothesis_validation["overall_validation"] = "supported"
        elif criteria_met > total_criteria / 2:
            hypothesis_validation["overall_validation"] = "partially_supported"
        else:
            hypothesis_validation["overall_validation"] = "not_supported"
        
        return hypothesis_validation
    
    def _generate_reproducibility_hash(self) -> str:
        """Generate hash for reproducibility verification."""
        reproducible_data = {
            "experiment_config": asdict(self.config),
            "python_version": "3.9+",  # Simplified
            "framework_version": "0.1.0"
        }
        
        # Create hash
        data_string = json.dumps(reproducible_data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()[:16]
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save experimental results for reproducibility."""
        output_path = Path(self.config.output_directory)
        
        # Save main results
        results_file = output_path / f"{self.config.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save experiment log
        log_file = output_path / f"{self.config.experiment_id}_log.txt"
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.experiment_log))
        
        # Save raw data (for potential reanalysis)
        data_file = output_path / f"{self.config.experiment_id}_raw_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(results, f)
        
        self.log(f"Results saved to {output_path}")
        
        # Generate summary report
        report = self._generate_experiment_report(results)
        report_file = output_path / f"{self.config.experiment_id}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
    
    def _generate_experiment_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive experiment report."""
        report_lines = [
            f"# Experiment Report: {self.config.name}",
            f"**Experiment ID:** {self.config.experiment_id}",
            f"**Type:** {self.config.experiment_type.value}",
            f"**Duration:** {results['experiment_duration_seconds']:.2f} seconds",
            f"**Reproducibility Hash:** {results['reproducibility_hash']}",
            "",
            "## Hypothesis",
            f"{self.config.hypothesis}",
            "",
            "## Success Criteria",
        ]
        
        for criterion, threshold in self.config.success_criteria.items():
            report_lines.append(f"- **{criterion}**: {threshold}")
        
        report_lines.extend([
            "",
            "## Results Summary",
            ""
        ])
        
        # Statistical results
        if "descriptive_statistics" in results["statistical_analysis"]:
            report_lines.extend([
                "### Performance Statistics",
                "",
                "| Algorithm | Accuracy | Std Dev | Success Rate | Sample Size |",
                "|-----------|----------|---------|--------------|-------------|"
            ])
            
            for algo_name, stats in results["statistical_analysis"]["descriptive_statistics"].items():
                report_lines.append(
                    f"| {algo_name} | {stats.get('accuracy_mean', 0):.3f} | "
                    f"{stats.get('accuracy_std', 0):.3f} | {stats.get('success_rate', 0):.3f} | "
                    f"{stats.get('sample_size', 0)} |"
                )
        
        # Effect sizes
        if "effect_sizes" in results["statistical_analysis"]:
            report_lines.extend([
                "",
                "### Effect Sizes",
                ""
            ])
            
            for comparison, effect_data in results["statistical_analysis"]["effect_sizes"].items():
                cohens_d = effect_data.get("cohens_d", 0)
                interpretation = effect_data.get("interpretation", "unknown")
                improvement = effect_data.get("improvement", 0)
                
                report_lines.append(f"- **{comparison}**: Cohen's d = {cohens_d:.3f} ({interpretation}), "
                                   f"Improvement = {improvement:.1f}%")
        
        # Hypothesis validation
        if "hypothesis_validation" in results:
            validation = results["hypothesis_validation"]["overall_validation"]
            report_lines.extend([
                "",
                "## Hypothesis Validation",
                f"**Result:** {validation}",
                ""
            ])
            
            for criterion, result in results["hypothesis_validation"]["validation_results"].items():
                status = "✅" if result == "met" else "❌" if result == "not_met" else "⚠️"
                report_lines.append(f"{status} {criterion}: {result}")
        
        report_lines.extend([
            "",
            "## Reproducibility",
            f"This experiment can be reproduced using the configuration and data provided.",
            f"Random seed: {self.config.random_seed}",
            f"Reproducibility hash: {results['reproducibility_hash']}",
            "",
            "## Experimental Protocol",
            f"- Number of runs: {self.config.num_runs}",
            f"- Significance level: {self.config.significance_level}",
            f"- Statistical power: {self.config.statistical_power}",
            "",
            "---",
            "*Generated by Federated DP-LLM Research Framework*"
        ])
        
        return "\n".join(report_lines)