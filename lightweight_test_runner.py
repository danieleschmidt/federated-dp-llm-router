#!/usr/bin/env python3
"""
Lightweight Test Runner for Research Framework

Runs basic validation tests without external dependencies to verify
the core research framework functionality.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all research modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        # Test research module imports (core components only)
        from federated_dp_llm.research.comparative_analyzer import ComparativeAnalyzer
        print("✅ ComparativeAnalyzer imported successfully")
        
        from federated_dp_llm.research.experiment_framework import ExperimentFramework
        print("✅ ExperimentFramework imported successfully")
        
        from federated_dp_llm.research.benchmarking_suite import BenchmarkingSuite
        print("✅ BenchmarkingSuite imported successfully")
        
        from federated_dp_llm.research.publication_tools import PublicationGenerator
        print("✅ PublicationGenerator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error during imports: {e}")
        traceback.print_exc()
        return False

def test_adaptive_imports():
    """Test adaptive module imports."""
    print("\n🔍 Testing adaptive module imports...")
    
    try:
        from federated_dp_llm.adaptive.adaptive_optimizer import AdaptiveOptimizer
        print("✅ AdaptiveOptimizer imported successfully")
        
        from federated_dp_llm.adaptive.self_healing_system import SelfHealingSystem
        print("✅ SelfHealingSystem imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test ComparativeAnalyzer basic functionality
        from federated_dp_llm.research.comparative_analyzer import (
            ComparativeAnalyzer, BaselineAlgorithm, NovelAlgorithm, AlgorithmType
        )
        
        analyzer = ComparativeAnalyzer()
        baseline = BaselineAlgorithm("test_baseline")
        novel = NovelAlgorithm("test_novel")
        
        # Test algorithm registration
        analyzer.register_algorithm(baseline, AlgorithmType.BASELINE)
        analyzer.register_algorithm(novel, AlgorithmType.NOVEL)
        
        assert len(analyzer.algorithms) == 2
        print("✅ Algorithm registration works")
        
        # Test metrics collection
        baseline_metrics = baseline.get_metrics()
        novel_metrics = novel.get_metrics()
        
        assert "accuracy" in baseline_metrics
        assert "latency_ms" in baseline_metrics
        assert novel_metrics["accuracy"] > baseline_metrics["accuracy"]
        print("✅ Performance metrics collection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("\n⚡ Testing async functionality...")
    
    try:
        from federated_dp_llm.research.comparative_analyzer import BaselineAlgorithm
        from federated_dp_llm.adaptive.adaptive_optimizer import AdaptiveOptimizer
        
        # Test async algorithm execution
        baseline = BaselineAlgorithm("async_test")
        
        class MockTask:
            def __init__(self, task_id):
                self.task_id = task_id
                
        mock_task = MockTask("test_task_123")
        result = await baseline.execute(mock_task, node_count=4)
        
        assert isinstance(result, dict)
        assert "result" in result
        print("✅ Async algorithm execution works")
        
        # Test adaptive optimizer
        optimizer = AdaptiveOptimizer()
        
        # Test performance recording
        triggered = await optimizer.record_performance(
            accuracy=0.85,
            latency_p95=120.0,
            throughput=150.0,
            resource_utilization=0.7,
            privacy_budget_remaining=0.5,
            error_rate=0.02
        )
        
        assert len(optimizer.performance_history) > 0
        print("✅ Adaptive optimizer performance recording works")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_numpy_fallback():
    """Test numpy fallback functionality."""
    print("\n🔢 Testing numpy fallback...")
    
    try:
        from federated_dp_llm.quantum_planning.numpy_fallback import get_numpy_backend
        
        has_numpy, np = get_numpy_backend()
        
        if has_numpy:
            print("✅ NumPy is available")
            # Test basic numpy operations
            arr = np.array([1, 2, 3, 4, 5])
            assert np.mean(arr) == 3.0
            print("✅ NumPy operations work correctly")
        else:
            print("⚠️  NumPy not available, using fallback")
            # Test that fallback doesn't crash
            assert np is not None
            print("✅ Fallback system works")
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy fallback test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all expected files exist."""
    print("\n📁 Testing file structure...")
    
    expected_files = [
        "federated_dp_llm/__init__.py",
        "federated_dp_llm/research/__init__.py",
        "federated_dp_llm/research/comparative_analyzer.py",
        "federated_dp_llm/research/experiment_framework.py",
        "federated_dp_llm/research/benchmarking_suite.py",
        "federated_dp_llm/research/publication_tools.py",
        "federated_dp_llm/adaptive/__init__.py",
        "federated_dp_llm/adaptive/adaptive_optimizer.py",
        "federated_dp_llm/adaptive/self_healing_system.py",
        "federated_dp_llm/quantum_planning/numpy_fallback.py",
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected files exist")
        return True

def test_configuration_validation():
    """Test configuration validation."""
    print("\n⚙️  Testing configuration validation...")
    
    try:
        from federated_dp_llm.research.experiment_framework import ExperimentConfig, ExperimentType
        from federated_dp_llm.research.publication_tools import PublicationConfig, PublicationType
        
        # Test ExperimentConfig
        exp_config = ExperimentConfig(
            experiment_id="test_001",
            experiment_type=ExperimentType.COMPARATIVE_STUDY,
            name="Test Experiment",
            description="Testing configuration",
            hypothesis="Test hypothesis",
            success_criteria={"accuracy": 0.8}
        )
        
        assert exp_config.experiment_id == "test_001"
        assert exp_config.num_runs == 10  # Default value
        print("✅ ExperimentConfig validation works")
        
        # Test PublicationConfig
        pub_config = PublicationConfig(
            title="Test Publication",
            authors=["Test Author"],
            institution="Test Institution",
            abstract="Test abstract",
            keywords=["test", "validation"],
            publication_type=PublicationType.TECHNICAL_REPORT
        )
        
        assert pub_config.title == "Test Publication"
        assert len(pub_config.authors) == 1
        print("✅ PublicationConfig validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all validation tests."""
    print("🚀 Starting Research Framework Validation Tests")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Adaptive Imports", test_adaptive_imports),
        ("NumPy Fallback", test_numpy_fallback),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Validation", test_configuration_validation),
    ]
    
    # Run sync tests
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"💥 {test_name} - ERROR: {e}")
    
    # Run async test
    print(f"\n📋 Running Async Functionality test...")
    try:
        async_result = asyncio.run(test_async_functionality())
        if async_result:
            passed += 1
            total += 1
            print(f"✅ Async Functionality - PASSED")
        else:
            total += 1
            print(f"❌ Async Functionality - FAILED")
    except Exception as e:
        total += 1
        print(f"💥 Async Functionality - ERROR: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed / total * 100):.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Research framework is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Some functionality may not work as expected.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)