#!/usr/bin/env python3
"""
Lightweight Quality Gates - No External Dependencies
Autonomous SDLC validation with built-in Python libraries only.
"""

import asyncio
import time
import sys
import json
import logging
import traceback
import subprocess
import statistics
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class QualityGateResult(Enum):
    """Quality gate validation results."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class QualityGateReport:
    """Lightweight quality gate report."""
    gate_name: str
    result: QualityGateResult
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class LightweightTestResults:
    """Lightweight test execution results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    warnings: int
    execution_time: float
    overall_quality_score: float
    quality_gates: List[QualityGateReport] = field(default_factory=list)


class LightweightQualityGates:
    """Lightweight quality gates without external dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityGates")
        self.test_results: List[QualityGateReport] = []
        self.start_time = time.time()
    
    async def execute_lightweight_validation(self) -> LightweightTestResults:
        """Execute lightweight quality gate validation."""
        
        print("\n🔍 LIGHTWEIGHT AUTONOMOUS QUALITY GATES")
        print("=" * 70)
        print("Executing validation with built-in Python libraries only...")
        
        # Execute lightweight quality gates
        quality_gates = [
            ("Code Structure", self._validate_code_structure),
            ("File Organization", self._validate_file_organization), 
            ("Basic Functionality", self._validate_basic_functionality),
            ("Documentation", self._validate_documentation),
            ("Configuration", self._validate_configuration),
            ("Deployment Artifacts", self._validate_deployment_artifacts),
            ("Security Basics", self._validate_security_basics),
            ("Performance Structure", self._validate_performance_structure)
        ]
        
        for gate_name, gate_func in quality_gates:
            print(f"\n🔬 Executing: {gate_name}")
            try:
                report = await gate_func()
                self.test_results.append(report)
                
                # Print immediate feedback
                status_icon = "✅" if report.result == QualityGateResult.PASSED else "❌" if report.result == QualityGateResult.FAILED else "⚠️"
                print(f"{status_icon} {gate_name}: {report.result.value} ({report.execution_time:.2f}s)")
                
                if report.errors:
                    for error in report.errors[:2]:
                        print(f"   ❌ {error}")
                
                if report.warnings:
                    for warning in report.warnings[:2]:
                        print(f"   ⚠️ {warning}")
                        
            except Exception as e:
                error_report = QualityGateReport(
                    gate_name=gate_name,
                    result=QualityGateResult.FAILED,
                    execution_time=0.0,
                    errors=[f"Gate execution failed: {str(e)}"]
                )
                self.test_results.append(error_report)
                print(f"❌ {gate_name}: FAILED - Execution error")
        
        # Calculate results
        results = self._calculate_results()
        
        # Generate final report
        await self._generate_report(results)
        
        return results
    
    async def _validate_code_structure(self) -> QualityGateReport:
        """Validate code structure and organization."""
        start_time = time.time()
        errors = []
        warnings = []
        structure_score = 0.0
        
        try:
            # Check for main implementation files
            required_files = [
                "enhanced_core_functionality.py",
                "robust_enhanced_system.py", 
                "scalable_optimized_system.py"
            ]
            
            files_found = 0
            for file_name in required_files:
                file_path = Path(file_name)
                if file_path.exists():
                    files_found += 1
                    file_size = file_path.stat().st_size
                    
                    if file_size > 5000:  # At least 5KB indicates substantial implementation
                        structure_score += 15.0
                    elif file_size > 1000:
                        structure_score += 10.0
                        warnings.append(f"{file_name} is relatively small ({file_size} bytes)")
                    else:
                        warnings.append(f"{file_name} is very small ({file_size} bytes)")
                        structure_score += 5.0
                else:
                    errors.append(f"Required file {file_name} not found")
            
            # Check for federated_dp_llm package structure
            package_path = Path("federated_dp_llm")
            if package_path.exists() and package_path.is_dir():
                structure_score += 10.0
                
                # Check for core modules
                core_modules = ["core", "routing", "federation", "security", "quantum_planning"]
                modules_found = 0
                for module in core_modules:
                    module_path = package_path / module
                    if module_path.exists():
                        modules_found += 1
                
                if modules_found >= 4:
                    structure_score += 15.0
                elif modules_found >= 2:
                    structure_score += 10.0
                    warnings.append(f"Only {modules_found}/{len(core_modules)} core modules found")
                else:
                    warnings.append(f"Insufficient core modules: {modules_found}/{len(core_modules)}")
            else:
                errors.append("federated_dp_llm package not found")
            
            # Check for Python syntax validity
            syntax_valid = 0
            for file_name in required_files:
                if Path(file_name).exists():
                    try:
                        with open(file_name, 'r', encoding='utf-8') as f:
                            content = f.read()
                            compile(content, file_name, 'exec')
                        syntax_valid += 1
                    except SyntaxError as e:
                        errors.append(f"Syntax error in {file_name}: {str(e)}")
                    except Exception as e:
                        warnings.append(f"Could not validate syntax of {file_name}: {str(e)}")
            
            if syntax_valid == files_found:
                structure_score += 10.0
            elif syntax_valid >= files_found // 2:
                structure_score += 5.0
                warnings.append("Some files have syntax issues")
                
        except Exception as e:
            errors.append(f"Code structure validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if structure_score >= 70 and not errors:
            result = QualityGateResult.PASSED
        elif structure_score >= 50:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Code Structure",
            result=result,
            execution_time=execution_time,
            details={"files_checked": len(required_files)},
            metrics={"structure_score": structure_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_file_organization(self) -> QualityGateReport:
        """Validate file organization and project structure."""
        start_time = time.time()
        errors = []
        warnings = []
        organization_score = 0.0
        
        try:
            # Check for proper project structure
            expected_dirs = ["federated_dp_llm", "deployment", "configs", "tests"]
            dirs_found = 0
            
            for dir_name in expected_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists() and dir_path.is_dir():
                    dirs_found += 1
                    organization_score += 15.0
                else:
                    warnings.append(f"Expected directory {dir_name} not found")
            
            # Check for configuration files
            config_files = ["config/production.yaml", "configs/production.yaml", "requirements.txt"]
            config_found = 0
            for config_file in config_files:
                if Path(config_file).exists():
                    config_found += 1
                    organization_score += 5.0
            
            if config_found == 0:
                warnings.append("No configuration files found")
            
            # Check for Docker/deployment files
            deployment_files = ["Dockerfile", "docker-compose.yml", "deployment/"]
            deployment_found = 0
            for dep_file in deployment_files:
                if Path(dep_file).exists():
                    deployment_found += 1
                    organization_score += 5.0
            
            if deployment_found >= 2:
                organization_score += 10.0
            elif deployment_found == 0:
                warnings.append("No deployment files found")
            
            # Check for test files
            test_patterns = ["test_*.py", "*_test.py", "tests/"]
            test_files_found = 0
            
            for pattern in test_patterns:
                if "*" in pattern:
                    # Simple glob-like search
                    if "test_" in pattern:
                        test_files = [f for f in Path(".").iterdir() if f.name.startswith("test_") and f.name.endswith(".py")]
                    elif "_test" in pattern:
                        test_files = [f for f in Path(".").iterdir() if "_test" in f.name and f.name.endswith(".py")]
                    else:
                        continue
                    test_files_found += len(test_files)
                else:
                    if Path(pattern).exists():
                        test_files_found += 1
            
            if test_files_found >= 3:
                organization_score += 15.0
            elif test_files_found >= 1:
                organization_score += 10.0
                warnings.append(f"Limited test files found: {test_files_found}")
            else:
                warnings.append("No test files found")
                
        except Exception as e:
            errors.append(f"File organization validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if organization_score >= 75 and not errors:
            result = QualityGateResult.PASSED
        elif organization_score >= 50:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="File Organization",
            result=result,
            execution_time=execution_time,
            details={"directories_checked": len(expected_dirs)},
            metrics={"organization_score": organization_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_basic_functionality(self) -> QualityGateReport:
        """Validate basic functionality without imports."""
        start_time = time.time()
        errors = []
        warnings = []
        functionality_score = 0.0
        
        try:
            # Check for key classes and functions in files
            files_to_check = [
                ("enhanced_core_functionality.py", ["ProductionFederatedSystem", "EnhancedSystemConfig", "demo_"]),
                ("robust_enhanced_system.py", ["RobustFederatedSystem", "RobustSystemConfig", "ErrorSeverity"]),
                ("scalable_optimized_system.py", ["ScalableOptimizedSystem", "OptimizationConfig", "ScalingConfig"])
            ]
            
            for file_name, expected_elements in files_to_check:
                file_path = Path(file_name)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        elements_found = 0
                        for element in expected_elements:
                            if element in content:
                                elements_found += 1
                        
                        if elements_found == len(expected_elements):
                            functionality_score += 20.0
                        elif elements_found >= len(expected_elements) // 2:
                            functionality_score += 15.0
                            warnings.append(f"{file_name}: {elements_found}/{len(expected_elements)} expected elements found")
                        else:
                            warnings.append(f"{file_name}: missing key elements ({elements_found}/{len(expected_elements)})")
                            functionality_score += 5.0
                            
                    except Exception as e:
                        errors.append(f"Could not analyze {file_name}: {str(e)}")
                else:
                    errors.append(f"File {file_name} not found for functionality check")
            
            # Check for async patterns (modern Python)
            async_patterns = ["async def", "await ", "asyncio"]
            files_with_async = 0
            
            for file_name, _ in files_to_check:
                file_path = Path(file_name)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        has_async = any(pattern in content for pattern in async_patterns)
                        if has_async:
                            files_with_async += 1
                    except Exception:
                        pass
            
            if files_with_async >= 2:
                functionality_score += 15.0
            elif files_with_async >= 1:
                functionality_score += 10.0
                warnings.append("Limited async/await usage detected")
            else:
                warnings.append("No async/await patterns found")
            
            # Check for error handling patterns
            error_patterns = ["try:", "except", "raise", "Error", "Exception"]
            files_with_error_handling = 0
            
            for file_name, _ in files_to_check:
                file_path = Path(file_name)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        error_count = sum(1 for pattern in error_patterns if pattern in content)
                        if error_count >= 3:
                            files_with_error_handling += 1
                    except Exception:
                        pass
            
            if files_with_error_handling >= 2:
                functionality_score += 15.0
            elif files_with_error_handling >= 1:
                functionality_score += 10.0
                warnings.append("Limited error handling patterns")
            else:
                warnings.append("Insufficient error handling patterns")
                
        except Exception as e:
            errors.append(f"Basic functionality validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if functionality_score >= 75 and not errors:
            result = QualityGateResult.PASSED
        elif functionality_score >= 50:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Basic Functionality",
            result=result,
            execution_time=execution_time,
            details={"files_analyzed": len(files_to_check)},
            metrics={"functionality_score": functionality_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_documentation(self) -> QualityGateReport:
        """Validate documentation completeness."""
        start_time = time.time()
        errors = []
        warnings = []
        doc_score = 0.0
        
        try:
            # Check for README
            readme_files = ["README.md", "readme.md", "README.txt", "README"]
            readme_found = False
            readme_size = 0
            
            for readme_file in readme_files:
                readme_path = Path(readme_file)
                if readme_path.exists():
                    readme_found = True
                    readme_size = readme_path.stat().st_size
                    break
            
            if readme_found:
                if readme_size > 5000:  # Substantial README
                    doc_score += 30.0
                elif readme_size > 1000:
                    doc_score += 20.0
                    warnings.append(f"README is relatively short ({readme_size} bytes)")
                else:
                    doc_score += 10.0
                    warnings.append(f"README is very short ({readme_size} bytes)")
            else:
                errors.append("No README file found")
            
            # Check for architecture documentation
            arch_files = ["ARCHITECTURE.md", "architecture.md", "DESIGN.md", "design.md"]
            arch_found = any(Path(f).exists() for f in arch_files)
            
            if arch_found:
                doc_score += 25.0
            else:
                warnings.append("No architecture documentation found")
            
            # Check for security documentation
            security_files = ["SECURITY.md", "security.md", "SECURITY.txt"]
            security_found = any(Path(f).exists() for f in security_files)
            
            if security_found:
                doc_score += 20.0
            else:
                warnings.append("No security documentation found")
            
            # Check for deployment documentation
            deploy_files = ["DEPLOYMENT.md", "deployment.md", "INSTALL.md", "install.md"]
            deploy_found = any(Path(f).exists() for f in deploy_files)
            
            if deploy_found:
                doc_score += 25.0
            else:
                warnings.append("No deployment documentation found")
            
            # Check for inline documentation in code files
            code_files = ["enhanced_core_functionality.py", "robust_enhanced_system.py", "scalable_optimized_system.py"]
            documented_files = 0
            
            for code_file in code_files:
                file_path = Path(code_file)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check for docstrings and comments
                        docstring_count = content.count('"""') + content.count("'''")
                        comment_count = content.count('#')
                        
                        if docstring_count >= 4 or comment_count >= 20:
                            documented_files += 1
                    except Exception:
                        pass
            
            if documented_files >= 2:
                doc_score += 20.0
            elif documented_files >= 1:
                doc_score += 10.0
                warnings.append("Limited inline documentation")
            else:
                warnings.append("Insufficient inline documentation")
                
        except Exception as e:
            errors.append(f"Documentation validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if doc_score >= 90 and not errors:
            result = QualityGateResult.PASSED
        elif doc_score >= 60:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Documentation",
            result=result,
            execution_time=execution_time,
            details={"documentation_types_checked": 4},
            metrics={"documentation_score": doc_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_configuration(self) -> QualityGateReport:
        """Validate configuration management."""
        start_time = time.time()
        errors = []
        warnings = []
        config_score = 0.0
        
        try:
            # Check for configuration files
            config_files = [
                "config/production.yaml",
                "configs/production.yaml", 
                "configs/development.yaml",
                "requirements.txt",
                "setup.py"
            ]
            
            config_found = 0
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    config_found += 1
                    config_score += 15.0
                    
                    # Check file size to ensure it's not empty
                    if config_path.stat().st_size < 50:
                        warnings.append(f"{config_file} is very small or empty")
            
            if config_found == 0:
                errors.append("No configuration files found")
            elif config_found < 2:
                warnings.append(f"Limited configuration files: {config_found}")
            
            # Check for environment-specific configurations
            env_configs = ["development.yaml", "production.yaml", "test.yaml"]
            env_found = 0
            
            for env_config in env_configs:
                if Path(f"config/{env_config}").exists() or Path(f"configs/{env_config}").exists():
                    env_found += 1
            
            if env_found >= 2:
                config_score += 20.0
            elif env_found >= 1:
                config_score += 10.0
                warnings.append("Limited environment-specific configurations")
            else:
                warnings.append("No environment-specific configurations found")
            
            # Check for Docker configuration
            docker_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
            docker_found = any(Path(f).exists() for f in docker_files)
            
            if docker_found:
                config_score += 15.0
            else:
                warnings.append("No Docker configuration found")
                
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if config_score >= 70 and not errors:
            result = QualityGateResult.PASSED
        elif config_score >= 40:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Configuration",
            result=result,
            execution_time=execution_time,
            details={"config_files_checked": len(config_files)},
            metrics={"configuration_score": config_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_deployment_artifacts(self) -> QualityGateReport:
        """Validate deployment readiness artifacts."""
        start_time = time.time()
        errors = []
        warnings = []
        deployment_score = 0.0
        
        try:
            # Check for deployment directory
            deployment_dir = Path("deployment")
            if deployment_dir.exists() and deployment_dir.is_dir():
                deployment_score += 20.0
                
                # Check for subdirectories
                subdirs = ["kubernetes", "docker", "scripts", "monitoring"]
                subdirs_found = 0
                for subdir in subdirs:
                    if (deployment_dir / subdir).exists():
                        subdirs_found += 1
                
                if subdirs_found >= 3:
                    deployment_score += 20.0
                elif subdirs_found >= 2:
                    deployment_score += 15.0
                    warnings.append(f"Some deployment subdirectories missing: {subdirs_found}/4")
                else:
                    warnings.append(f"Limited deployment structure: {subdirs_found}/4 subdirectories")
            else:
                warnings.append("No deployment directory found")
            
            # Check for container files
            container_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
            container_found = 0
            for container_file in container_files:
                if Path(container_file).exists():
                    container_found += 1
                    deployment_score += 10.0
            
            if container_found == 0:
                warnings.append("No containerization files found")
            
            # Check for monitoring configuration
            monitoring_files = ["prometheus.yml", "monitoring/prometheus.yml", "deployment/monitoring/"]
            monitoring_found = any(Path(f).exists() for f in monitoring_files)
            
            if monitoring_found:
                deployment_score += 15.0
            else:
                warnings.append("No monitoring configuration found")
            
            # Check for scripts
            script_files = ["scripts/", "deployment/scripts/", "start.sh", "deploy.sh"]
            script_found = any(Path(f).exists() for f in script_files)
            
            if script_found:
                deployment_score += 15.0
            else:
                warnings.append("No deployment scripts found")
                
        except Exception as e:
            errors.append(f"Deployment artifacts validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if deployment_score >= 70 and not errors:
            result = QualityGateResult.PASSED
        elif deployment_score >= 40:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Deployment Artifacts",
            result=result,
            execution_time=execution_time,
            details={"artifact_categories_checked": 4},
            metrics={"deployment_score": deployment_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_security_basics(self) -> QualityGateReport:
        """Validate basic security patterns."""
        start_time = time.time()
        errors = []
        warnings = []
        security_score = 0.0
        
        try:
            # Check for security-related code patterns
            security_patterns = [
                "encryption", "authentication", "authorization", "validate", "sanitize",
                "privacy", "security", "audit", "logging", "token", "certificate"
            ]
            
            code_files = [
                "enhanced_core_functionality.py",
                "robust_enhanced_system.py",
                "scalable_optimized_system.py"
            ]
            
            files_with_security = 0
            for code_file in code_files:
                file_path = Path(code_file)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                        
                        security_mentions = sum(1 for pattern in security_patterns if pattern in content)
                        if security_mentions >= 5:
                            files_with_security += 1
                    except Exception:
                        pass
            
            if files_with_security >= 2:
                security_score += 25.0
            elif files_with_security >= 1:
                security_score += 15.0
                warnings.append("Limited security patterns in code")
            else:
                warnings.append("Insufficient security patterns found")
            
            # Check for federated_dp_llm security module
            security_module = Path("federated_dp_llm/security")
            if security_module.exists() and security_module.is_dir():
                security_score += 25.0
                
                # Check for security files
                security_files = list(security_module.glob("*.py"))
                if len(security_files) >= 3:
                    security_score += 15.0
                elif len(security_files) >= 1:
                    security_score += 10.0
                    warnings.append(f"Limited security modules: {len(security_files)}")
            else:
                warnings.append("No security module directory found")
            
            # Check for privacy-related patterns
            privacy_patterns = ["privacy", "differential", "budget", "epsilon", "delta", "dp_"]
            files_with_privacy = 0
            
            for code_file in code_files:
                file_path = Path(code_file)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                        
                        privacy_mentions = sum(1 for pattern in privacy_patterns if pattern in content)
                        if privacy_mentions >= 3:
                            files_with_privacy += 1
                    except Exception:
                        pass
            
            if files_with_privacy >= 2:
                security_score += 20.0
            elif files_with_privacy >= 1:
                security_score += 10.0
                warnings.append("Limited privacy patterns")
            else:
                warnings.append("Insufficient privacy patterns found")
            
            # Check for input validation patterns
            validation_patterns = ["validate", "check", "verify", "sanitize", "clean"]
            validation_found = False
            
            for code_file in code_files:
                file_path = Path(code_file)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                        
                        validation_count = sum(1 for pattern in validation_patterns if pattern in content)
                        if validation_count >= 5:
                            validation_found = True
                            break
                    except Exception:
                        pass
            
            if validation_found:
                security_score += 15.0
            else:
                warnings.append("Limited input validation patterns")
                
        except Exception as e:
            errors.append(f"Security basics validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if security_score >= 80 and not errors:
            result = QualityGateResult.PASSED
        elif security_score >= 50:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Security Basics",
            result=result,
            execution_time=execution_time,
            details={"security_categories_checked": 4},
            metrics={"security_score": security_score},
            errors=errors,
            warnings=warnings
        )
    
    async def _validate_performance_structure(self) -> QualityGateReport:
        """Validate performance optimization structure."""
        start_time = time.time()
        errors = []
        warnings = []
        performance_score = 0.0
        
        try:
            # Check for performance optimization patterns
            perf_patterns = [
                "async", "await", "cache", "pool", "batch", "optimize", "scale",
                "concurrent", "parallel", "performance", "speed", "latency"
            ]
            
            code_files = [
                "enhanced_core_functionality.py",
                "robust_enhanced_system.py",
                "scalable_optimized_system.py"
            ]
            
            files_with_perf = 0
            for code_file in code_files:
                file_path = Path(code_file)
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                        
                        perf_mentions = sum(1 for pattern in perf_patterns if pattern in content)
                        if perf_mentions >= 8:
                            files_with_perf += 1
                    except Exception:
                        pass
            
            if files_with_perf >= 2:
                performance_score += 30.0
            elif files_with_perf >= 1:
                performance_score += 20.0
                warnings.append("Limited performance optimization patterns")
            else:
                warnings.append("Insufficient performance patterns found")
            
            # Check for optimization modules
            opt_module = Path("federated_dp_llm/optimization")
            if opt_module.exists() and opt_module.is_dir():
                performance_score += 25.0
                
                opt_files = list(opt_module.glob("*.py"))
                if len(opt_files) >= 3:
                    performance_score += 15.0
                elif len(opt_files) >= 1:
                    performance_score += 10.0
                    warnings.append(f"Limited optimization modules: {len(opt_files)}")
            else:
                warnings.append("No optimization module directory found")
            
            # Check for quantum planning (advanced optimization)
            quantum_module = Path("federated_dp_llm/quantum_planning")
            if quantum_module.exists() and quantum_module.is_dir():
                performance_score += 20.0
                
                quantum_files = list(quantum_module.glob("*.py"))
                if len(quantum_files) >= 5:
                    performance_score += 10.0
                elif len(quantum_files) >= 3:
                    performance_score += 5.0
            else:
                warnings.append("No quantum planning module found")
                
        except Exception as e:
            errors.append(f"Performance structure validation failed: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Determine result
        if performance_score >= 80 and not errors:
            result = QualityGateResult.PASSED
        elif performance_score >= 50:
            result = QualityGateResult.WARNING
        else:
            result = QualityGateResult.FAILED
        
        return QualityGateReport(
            gate_name="Performance Structure",
            result=result,
            execution_time=execution_time,
            details={"performance_categories_checked": 3},
            metrics={"performance_score": performance_score},
            errors=errors,
            warnings=warnings
        )
    
    def _calculate_results(self) -> LightweightTestResults:
        """Calculate comprehensive results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == QualityGateResult.PASSED)
        failed_tests = sum(1 for r in self.test_results if r.result == QualityGateResult.FAILED)
        warnings = sum(1 for r in self.test_results if r.result == QualityGateResult.WARNING)
        
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall quality score
        if total_tests > 0:
            pass_rate = passed_tests / total_tests
            warning_rate = warnings / total_tests
            overall_quality_score = (pass_rate * 100 + warning_rate * 50)
        else:
            overall_quality_score = 0.0
        
        return LightweightTestResults(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warnings=warnings,
            execution_time=total_execution_time,
            overall_quality_score=overall_quality_score,
            quality_gates=self.test_results
        )
    
    async def _generate_report(self, results: LightweightTestResults):
        """Generate final quality report."""
        
        print(f"\n" + "=" * 70)
        print("🏆 LIGHTWEIGHT QUALITY GATES REPORT")
        print("=" * 70)
        
        # Overall results
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Quality Gates: {results.total_tests}")
        print(f"   ✅ Passed: {results.passed_tests}")
        print(f"   ❌ Failed: {results.failed_tests}")
        print(f"   ⚠️  Warnings: {results.warnings}")
        print(f"   ⏱️  Total Execution Time: {results.execution_time:.2f}s")
        print(f"   🎯 Overall Quality Score: {results.overall_quality_score:.1f}/100")
        
        # Individual gate results
        print(f"\n🔍 INDIVIDUAL QUALITY GATES:")
        for gate in results.quality_gates:
            status_icon = "✅" if gate.result == QualityGateResult.PASSED else "❌" if gate.result == QualityGateResult.FAILED else "⚠️"
            print(f"   {status_icon} {gate.gate_name}: {gate.result.value} ({gate.execution_time:.2f}s)")
            
            if gate.metrics:
                for metric, value in gate.metrics.items():
                    print(f"      📊 {metric}: {value:.1f}")
        
        # Production readiness assessment
        print(f"\n🚀 PRODUCTION READINESS ASSESSMENT:")
        
        failed_gates = [g for g in results.quality_gates if g.result == QualityGateResult.FAILED]
        critical_passed = all(g.result != QualityGateResult.FAILED for g in results.quality_gates 
                             if g.gate_name in ["Code Structure", "Security Basics", "Basic Functionality"])
        
        if results.overall_quality_score >= 80 and not failed_gates:
            print(f"   ✅ EXCELLENT - System ready for production deployment")
            print(f"   🌟 All quality gates passed with high score")
        elif results.overall_quality_score >= 70 and len(failed_gates) <= 1:
            print(f"   👍 GOOD - System mostly ready, address remaining issues")
            print(f"   ⚡ Minor improvements recommended before production")
        elif results.overall_quality_score >= 60 and critical_passed:
            print(f"   ⚠️  ACCEPTABLE - Core functionality solid, optimization needed")
            print(f"   🔧 Address warnings and failed gates for production readiness")
        else:
            print(f"   ❌ NEEDS WORK - Significant improvements required")
            print(f"   🚨 Critical issues must be resolved before production")
        
        # Specific recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if failed_gates:
            print(f"   🚨 CRITICAL - Address failed gates:")
            for gate in failed_gates[:3]:  # Show first 3
                print(f"      - {gate.gate_name}")
                for error in gate.errors[:1]:  # Show first error
                    print(f"        • {error}")
        
        warning_gates = [g for g in results.quality_gates if g.result == QualityGateResult.WARNING]
        if warning_gates:
            print(f"   ⚠️  IMPROVE - Address warning gates:")
            for gate in warning_gates[:3]:  # Show first 3
                print(f"      - {gate.gate_name}")
        
        # Next steps
        if results.overall_quality_score >= 75:
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Run full integration tests with dependencies")
            print(f"   2. Perform security penetration testing")
            print(f"   3. Load test with production-like data")
            print(f"   4. Deploy to staging environment")
        else:
            print(f"\n🎯 NEXT STEPS:")
            print(f"   1. Fix failed quality gates")
            print(f"   2. Address warning conditions")  
            print(f"   3. Re-run quality validation")
            print(f"   4. Consider code review and refactoring")
        
        print(f"\n" + "=" * 70)
        print("🎉 LIGHTWEIGHT QUALITY VALIDATION COMPLETE")
        print("=" * 70)


async def main():
    """Main execution for lightweight quality gates."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and execute lightweight quality gates
    quality_gates = LightweightQualityGates()
    
    try:
        results = await quality_gates.execute_lightweight_validation()
        
        # Exit with appropriate code
        if results.overall_quality_score >= 70 and results.failed_tests <= 1:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Needs improvement
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Quality gates execution failed")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(2)  # Critical failure


if __name__ == "__main__":
    asyncio.run(main())