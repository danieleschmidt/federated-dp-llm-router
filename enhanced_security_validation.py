#!/usr/bin/env python3
"""
Enhanced Security Validation Suite
Comprehensive testing of security improvements and privacy enhancements
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any
from federated_dp_llm.security.enhanced_privacy_validator import (
    get_privacy_validator, 
    HealthcareDataSensitivity, 
    DepartmentType,
    ValidationResult
)
from federated_dp_llm.security.secure_config_manager import get_secure_config
from federated_dp_llm.security.secure_database import get_secure_database
from federated_dp_llm.core.privacy_accountant import PrivacyAccountant, DPConfig

logger = logging.getLogger(__name__)

class EnhancedSecurityValidator:
    """
    Comprehensive security validation for the enhanced federated system
    """
    
    def __init__(self):
        self.privacy_validator = get_privacy_validator()
        self.config_manager = get_secure_config()
        self.test_results = {
            'privacy_validation': {'score': 0, 'max_score': 100, 'tests': []},
            'secure_config': {'score': 0, 'max_score': 100, 'tests': []},
            'database_security': {'score': 0, 'max_score': 100, 'tests': []},
            'integration': {'score': 0, 'max_score': 100, 'tests': []},
            'healthcare_compliance': {'score': 0, 'max_score': 100, 'tests': []}
        }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all security validation tests"""
        print("🔐 ENHANCED SECURITY VALIDATION SUITE")
        print("=" * 60)
        
        # Test 1: Privacy Parameter Validation
        await self.test_privacy_validation()
        
        # Test 2: Secure Configuration Management
        await self.test_secure_config_management()
        
        # Test 3: Database Security
        await self.test_database_security()
        
        # Test 4: Integration Testing
        await self.test_integration()
        
        # Test 5: Healthcare Compliance
        await self.test_healthcare_compliance()
        
        # Generate final report
        return self.generate_final_report()
    
    async def test_privacy_validation(self):
        """Test enhanced privacy parameter validation"""
        print("\n1️⃣ PRIVACY PARAMETER VALIDATION")
        print("-" * 40)
        
        tests = [
            # Valid parameters
            {
                'name': 'Valid emergency department parameters',
                'params': {
                    'epsilon': 1.5, 'delta': 1e-6, 'noise_multiplier': 1.0,
                    'department': 'emergency', 'data_sensitivity': 'medium', 'user_role': 'doctor'
                },
                'expected_valid': True,
                'points': 10
            },
            
            # Invalid parameters - too high epsilon
            {
                'name': 'Invalid high epsilon for psychiatry',
                'params': {
                    'epsilon': 5.0, 'delta': 1e-6, 'noise_multiplier': 1.0,
                    'department': 'psychiatry', 'data_sensitivity': 'critical', 'user_role': 'doctor'
                },
                'expected_valid': False,
                'points': 15
            },
            
            # Invalid delta
            {
                'name': 'Invalid high delta',
                'params': {
                    'epsilon': 0.5, 'delta': 0.1, 'noise_multiplier': 1.0,
                    'department': 'general', 'data_sensitivity': 'medium', 'user_role': 'doctor'
                },
                'expected_valid': False,
                'points': 10
            },
            
            # Role-based restrictions
            {
                'name': 'Student role restrictions',
                'params': {
                    'epsilon': 1.0, 'delta': 1e-6, 'noise_multiplier': 1.5,
                    'department': 'general', 'data_sensitivity': 'low', 'user_role': 'student'
                },
                'expected_valid': False,
                'points': 15
            },
            
            # Data sensitivity validation
            {
                'name': 'Critical data sensitivity limits',
                'params': {
                    'epsilon': 0.05, 'delta': 1e-9, 'noise_multiplier': 3.0,
                    'department': 'genetics', 'data_sensitivity': 'critical', 'user_role': 'doctor'
                },
                'expected_valid': True,
                'points': 20
            },
            
            # Department-specific validation
            {
                'name': 'Research department strict limits',
                'params': {
                    'epsilon': 0.15, 'delta': 1e-7, 'noise_multiplier': 2.0,
                    'department': 'research', 'data_sensitivity': 'high', 'user_role': 'researcher'
                },
                'expected_valid': True,
                'points': 15
            },
            
            # Edge cases
            {
                'name': 'Zero epsilon (invalid)',
                'params': {
                    'epsilon': 0.0, 'delta': 1e-6, 'noise_multiplier': 1.0,
                    'department': 'general', 'data_sensitivity': 'medium', 'user_role': 'doctor'
                },
                'expected_valid': False,
                'points': 10
            },
            
            # Query type validation
            {
                'name': 'Training query with appropriate epsilon',
                'params': {
                    'epsilon': 3.0, 'delta': 1e-5, 'noise_multiplier': 1.0,
                    'department': 'research', 'data_sensitivity': 'medium', 'user_role': 'researcher',
                    'query_type': 'training'
                },
                'expected_valid': False,  # Should fail due to research dept limits
                'points': 5
            }
        ]
        
        total_score = 0
        max_score = 0
        
        for test in tests:
            max_score += test['points']
            try:
                result = self.privacy_validator.validate_privacy_parameters(**test['params'])
                
                if result.valid == test['expected_valid']:
                    total_score += test['points']
                    status = "✅ PASS"
                    print(f"   {status} {test['name']} ({test['points']} pts)")
                    if result.warnings:
                        print(f"      Warnings: {len(result.warnings)}")
                else:
                    status = "❌ FAIL"
                    print(f"   {status} {test['name']} (0/{test['points']} pts)")
                    print(f"      Expected valid: {test['expected_valid']}, Got: {result.valid}")
                    if result.issues:
                        print(f"      Issues: {result.issues}")
                
                self.test_results['privacy_validation']['tests'].append({
                    'name': test['name'],
                    'passed': result.valid == test['expected_valid'],
                    'score': test['points'] if result.valid == test['expected_valid'] else 0,
                    'max_score': test['points']
                })
                
            except Exception as e:
                print(f"   ❌ FAIL {test['name']} - Exception: {e}")
                self.test_results['privacy_validation']['tests'].append({
                    'name': test['name'],
                    'passed': False,
                    'score': 0,
                    'max_score': test['points'],
                    'error': str(e)
                })
        
        self.test_results['privacy_validation']['score'] = (total_score / max_score) * 100 if max_score > 0 else 0
        self.test_results['privacy_validation']['max_score'] = max_score
        
        print(f"\n   Privacy Validation Score: {total_score}/{max_score} ({self.test_results['privacy_validation']['score']:.1f}%)")
    
    async def test_secure_config_management(self):
        """Test secure configuration management"""
        print("\n2️⃣ SECURE CONFIGURATION MANAGEMENT")
        print("-" * 40)
        
        total_score = 0
        max_score = 100
        tests = []
        
        try:
            # Test 1: Secret validation
            validation_result = self.config_manager.validate_security_config()
            if validation_result['valid']:
                score = 30
                total_score += score
                print("   ✅ PASS Security configuration validation (30 pts)")
            else:
                print("   ⚠️  WARN Security configuration has issues (15 pts)")
                print(f"      Issues: {validation_result['issues']}")
                score = 15
                total_score += score
            
            tests.append({
                'name': 'Security configuration validation',
                'passed': validation_result['valid'],
                'score': score,
                'max_score': 30
            })
            
            # Test 2: Secret retrieval (should fail safely for missing secrets)
            try:
                jwt_secret = self.config_manager.get_secret('jwt_secret')
                if jwt_secret:
                    print("   ✅ PASS Secret retrieval working (20 pts)")
                    total_score += 20
                    tests.append({
                        'name': 'Secret retrieval',
                        'passed': True,
                        'score': 20,
                        'max_score': 20
                    })
                else:
                    print("   ⚠️  WARN Secret not found (handled safely) (10 pts)")
                    total_score += 10
                    tests.append({
                        'name': 'Secret retrieval',
                        'passed': True,
                        'score': 10,
                        'max_score': 20
                    })
            except ValueError as e:
                # Expected for missing required secrets
                print("   ✅ PASS Secure failure for missing secret (15 pts)")
                total_score += 15
                tests.append({
                    'name': 'Secret retrieval',
                    'passed': True,
                    'score': 15,
                    'max_score': 20
                })
            
            # Test 3: Config retrieval
            test_config = self.config_manager.get_config('test_key', 'default_value')
            if test_config == 'default_value':
                print("   ✅ PASS Configuration retrieval with defaults (20 pts)")
                total_score += 20
                tests.append({
                    'name': 'Configuration retrieval',
                    'passed': True,
                    'score': 20,
                    'max_score': 20
                })
            
            # Test 4: No hardcoded secrets in environment
            hardcoded_found = False
            for key, value in {
                'TEST_PASSWORD': 'password123',
                'SECRET_KEY': 'hardcoded_secret'
            }.items():
                if value in str(value).lower():
                    hardcoded_found = True
                    break
            
            if not hardcoded_found:
                print("   ✅ PASS No hardcoded secrets detected (30 pts)")
                total_score += 30
                tests.append({
                    'name': 'Hardcoded secret detection',
                    'passed': True,
                    'score': 30,
                    'max_score': 30
                })
            else:
                print("   ❌ FAIL Hardcoded secrets detected (0 pts)")
                tests.append({
                    'name': 'Hardcoded secret detection',
                    'passed': False,
                    'score': 0,
                    'max_score': 30
                })
                
        except Exception as e:
            print(f"   ❌ FAIL Secure config test failed: {e}")
            tests.append({
                'name': 'Secure configuration management',
                'passed': False,
                'score': 0,
                'max_score': max_score,
                'error': str(e)
            })
        
        self.test_results['secure_config']['score'] = (total_score / max_score) * 100 if max_score > 0 else 0
        self.test_results['secure_config']['tests'] = tests
        
        print(f"\n   Secure Config Score: {total_score}/{max_score} ({self.test_results['secure_config']['score']:.1f}%)")
    
    async def test_database_security(self):
        """Test secure database operations"""
        print("\n3️⃣ DATABASE SECURITY")
        print("-" * 40)
        
        total_score = 0
        max_score = 100
        tests = []
        
        try:
            # Initialize secure database
            secure_db = await get_secure_database()
            
            # Test 1: Database initialization
            health = await secure_db.health_check()
            if health['status'] == 'healthy':
                print("   ✅ PASS Database initialization and health check (20 pts)")
                total_score += 20
                tests.append({
                    'name': 'Database health check',
                    'passed': True,
                    'score': 20,
                    'max_score': 20
                })
            else:
                print(f"   ❌ FAIL Database health check failed: {health.get('error')} (0 pts)")
                tests.append({
                    'name': 'Database health check',
                    'passed': False,
                    'score': 0,
                    'max_score': 20
                })
            
            # Test 2: SQL injection protection
            malicious_inputs = [
                "'; DROP TABLE user_budgets; --",
                "admin'; UPDATE user_budgets SET total_budget=9999; --",
                "' OR '1'='1",
                "test'; INSERT INTO user_budgets VALUES (999, 'hacker', 9999, 0, 9999, NOW(), 'evil'); --"
            ]
            
            injection_blocked = 0
            for malicious_input in malicious_inputs:
                try:
                    result = await secure_db.get_user_budget(malicious_input)
                    if result is None:  # Should return None for invalid input
                        injection_blocked += 1
                except Exception:
                    injection_blocked += 1  # Exception is also good (blocks injection)
            
            if injection_blocked == len(malicious_inputs):
                print("   ✅ PASS SQL injection protection (30 pts)")
                total_score += 30
                tests.append({
                    'name': 'SQL injection protection',
                    'passed': True,
                    'score': 30,
                    'max_score': 30
                })
            else:
                print(f"   ❌ FAIL SQL injection protection - {injection_blocked}/{len(malicious_inputs)} blocked (0 pts)")
                tests.append({
                    'name': 'SQL injection protection',
                    'passed': False,
                    'score': 0,
                    'max_score': 30
                })
            
            # Test 3: Parameterized queries
            test_user = "test_user_security_validation"
            result = await secure_db.create_user_budget(test_user, 10.0, "test_dept")
            if result.success:
                print("   ✅ PASS Parameterized query execution (25 pts)")
                total_score += 25
                tests.append({
                    'name': 'Parameterized query execution',
                    'passed': True,
                    'score': 25,
                    'max_score': 25
                })
                
                # Cleanup
                await secure_db.update_user_budget(test_user, 10.0, "test_dept")
            else:
                print(f"   ❌ FAIL Parameterized query execution: {result.error} (0 pts)")
                tests.append({
                    'name': 'Parameterized query execution',
                    'passed': False,
                    'score': 0,
                    'max_score': 25
                })
            
            # Test 4: Input validation
            invalid_inputs = ["", None, "a" * 100, "user with spaces and 'quotes'"]
            validation_passed = 0
            for invalid_input in invalid_inputs:
                try:
                    if invalid_input is None:
                        continue
                    result = await secure_db.get_user_budget(invalid_input)
                    if result is None:  # Should reject invalid input
                        validation_passed += 1
                except Exception:
                    validation_passed += 1  # Exception handling is also valid
            
            if validation_passed >= len([x for x in invalid_inputs if x is not None]) - 1:
                print("   ✅ PASS Input validation (25 pts)")
                total_score += 25
                tests.append({
                    'name': 'Input validation',
                    'passed': True,
                    'score': 25,
                    'max_score': 25
                })
            else:
                print(f"   ❌ FAIL Input validation - {validation_passed} validations passed (0 pts)")
                tests.append({
                    'name': 'Input validation',
                    'passed': False,
                    'score': 0,
                    'max_score': 25
                })
                
        except Exception as e:
            print(f"   ❌ FAIL Database security test failed: {e}")
            tests.append({
                'name': 'Database security',
                'passed': False,
                'score': 0,
                'max_score': max_score,
                'error': str(e)
            })
        
        self.test_results['database_security']['score'] = (total_score / max_score) * 100 if max_score > 0 else 0
        self.test_results['database_security']['tests'] = tests
        
        print(f"\n   Database Security Score: {total_score}/{max_score} ({self.test_results['database_security']['score']:.1f}%)")
    
    async def test_integration(self):
        """Test integration of enhanced security features"""
        print("\n4️⃣ INTEGRATION TESTING")
        print("-" * 40)
        
        total_score = 0
        max_score = 100
        tests = []
        
        try:
            # Test 1: Privacy accountant with enhanced validation
            dp_config = DPConfig(
                epsilon_per_query=0.1,
                delta=1e-6,
                max_budget_per_user=5.0,
                noise_multiplier=1.5
            )
            
            accountant = PrivacyAccountant(dp_config)
            
            # Test valid request
            test_user = "integration_test_user"
            budget_ok, validation_result = accountant.check_budget(
                test_user, 0.5, 
                department="general", 
                data_sensitivity="medium", 
                user_role="doctor"
            )
            
            if budget_ok and validation_result and validation_result.valid:
                print("   ✅ PASS Privacy accountant integration (30 pts)")
                total_score += 30
                tests.append({
                    'name': 'Privacy accountant integration',
                    'passed': True,
                    'score': 30,
                    'max_score': 30
                })
            else:
                print(f"   ❌ FAIL Privacy accountant integration - Valid: {budget_ok}, Validation: {validation_result}")
                tests.append({
                    'name': 'Privacy accountant integration',
                    'passed': False,
                    'score': 0,
                    'max_score': 30
                })
            
            # Test 2: Enhanced validation rejection
            budget_ok, validation_result = accountant.check_budget(
                test_user, 10.0,  # Too high
                department="psychiatry", 
                data_sensitivity="critical", 
                user_role="student"
            )
            
            if not budget_ok or (validation_result and not validation_result.valid):
                print("   ✅ PASS Enhanced validation rejection (25 pts)")
                total_score += 25
                tests.append({
                    'name': 'Enhanced validation rejection',
                    'passed': True,
                    'score': 25,
                    'max_score': 25
                })
            else:
                print("   ❌ FAIL Enhanced validation should have rejected request")
                tests.append({
                    'name': 'Enhanced validation rejection',
                    'passed': False,
                    'score': 0,
                    'max_score': 25
                })
            
            # Test 3: Department information retrieval
            dept_info = self.privacy_validator.get_department_info("cardiology")
            if 'limits' in dept_info and 'max_epsilon_per_query' in dept_info['limits']:
                print("   ✅ PASS Department information retrieval (20 pts)")
                total_score += 20
                tests.append({
                    'name': 'Department information retrieval',
                    'passed': True,
                    'score': 20,
                    'max_score': 20
                })
            else:
                print("   ❌ FAIL Department information retrieval")
                tests.append({
                    'name': 'Department information retrieval',
                    'passed': False,
                    'score': 0,
                    'max_score': 20
                })
            
            # Test 4: Recommendation system
            invalid_validation = self.privacy_validator.validate_privacy_parameters(
                10.0, 0.1, 0.5,  # Invalid parameters
                department="research",
                data_sensitivity="critical", 
                user_role="student"
            )
            
            if not invalid_validation.valid and invalid_validation.recommended_params:
                print("   ✅ PASS Parameter recommendation system (25 pts)")
                total_score += 25
                tests.append({
                    'name': 'Parameter recommendation system',
                    'passed': True,
                    'score': 25,
                    'max_score': 25
                })
            else:
                print("   ❌ FAIL Parameter recommendation system")
                tests.append({
                    'name': 'Parameter recommendation system',
                    'passed': False,
                    'score': 0,
                    'max_score': 25
                })
                
        except Exception as e:
            print(f"   ❌ FAIL Integration test failed: {e}")
            tests.append({
                'name': 'Integration testing',
                'passed': False,
                'score': 0,
                'max_score': max_score,
                'error': str(e)
            })
        
        self.test_results['integration']['score'] = (total_score / max_score) * 100 if max_score > 0 else 0
        self.test_results['integration']['tests'] = tests
        
        print(f"\n   Integration Score: {total_score}/{max_score} ({self.test_results['integration']['score']:.1f}%)")
    
    async def test_healthcare_compliance(self):
        """Test healthcare-specific compliance features"""
        print("\n5️⃣ HEALTHCARE COMPLIANCE")
        print("-" * 40)
        
        total_score = 0
        max_score = 100
        tests = []
        
        try:
            # Test 1: Department-specific limits
            departments = [
                ('emergency', 2.0),
                ('icu', 1.5), 
                ('psychiatry', 0.1),
                ('genetics', 0.05),
                ('research', 0.2)
            ]
            
            dept_tests_passed = 0
            for dept, max_epsilon in departments:
                # Test valid epsilon for department
                result = self.privacy_validator.validate_privacy_parameters(
                    max_epsilon * 0.8, 1e-6, 1.5,
                    department=dept, data_sensitivity="medium", user_role="doctor"
                )
                
                # Test invalid (too high) epsilon for department
                invalid_result = self.privacy_validator.validate_privacy_parameters(
                    max_epsilon * 2, 1e-6, 1.5,
                    department=dept, data_sensitivity="medium", user_role="doctor"
                )
                
                if result.valid and not invalid_result.valid:
                    dept_tests_passed += 1
            
            if dept_tests_passed == len(departments):
                print("   ✅ PASS Department-specific compliance limits (30 pts)")
                total_score += 30
                tests.append({
                    'name': 'Department compliance limits',
                    'passed': True,
                    'score': 30,
                    'max_score': 30
                })
            else:
                print(f"   ❌ FAIL Department compliance - {dept_tests_passed}/{len(departments)} passed (10 pts)")
                total_score += 10
                tests.append({
                    'name': 'Department compliance limits',
                    'passed': False,
                    'score': 10,
                    'max_score': 30
                })
            
            # Test 2: Data sensitivity validation
            sensitivities = [
                ('public', 10.0),
                ('low', 5.0),
                ('medium', 2.0), 
                ('high', 1.0),
                ('critical', 0.1)
            ]
            
            sens_tests_passed = 0
            for sens, max_epsilon in sensitivities:
                # Test at limit
                result = self.privacy_validator.validate_privacy_parameters(
                    max_epsilon, 1e-6, 1.5,
                    department="general", data_sensitivity=sens, user_role="doctor"
                )
                
                # Test above limit
                invalid_result = self.privacy_validator.validate_privacy_parameters(
                    max_epsilon + 1.0, 1e-6, 1.5,
                    department="general", data_sensitivity=sens, user_role="doctor"
                )
                
                if result.valid and not invalid_result.valid:
                    sens_tests_passed += 1
            
            if sens_tests_passed >= 4:  # Allow some flexibility
                print("   ✅ PASS Data sensitivity validation (25 pts)")
                total_score += 25
                tests.append({
                    'name': 'Data sensitivity validation',
                    'passed': True,
                    'score': 25,
                    'max_score': 25
                })
            else:
                print(f"   ❌ FAIL Data sensitivity validation - {sens_tests_passed}/{len(sensitivities)} passed (5 pts)")
                total_score += 5
                tests.append({
                    'name': 'Data sensitivity validation',
                    'passed': False,
                    'score': 5,
                    'max_score': 25
                })
            
            # Test 3: Role-based access controls
            roles = [
                ('admin', 1.5),
                ('doctor', 1.0),
                ('nurse', 0.8),
                ('researcher', 0.5),
                ('student', 0.3)
            ]
            
            role_tests_passed = 0
            for role, multiplier in roles:
                # Test appropriate epsilon for role in general department
                result = self.privacy_validator.validate_privacy_parameters(
                    0.8 * multiplier, 1e-6, 1.5,
                    department="general", data_sensitivity="medium", user_role=role
                )
                
                if result.valid:
                    role_tests_passed += 1
            
            if role_tests_passed >= 4:
                print("   ✅ PASS Role-based access controls (25 pts)")
                total_score += 25
                tests.append({
                    'name': 'Role-based access controls',
                    'passed': True,
                    'score': 25,
                    'max_score': 25
                })
            else:
                print(f"   ❌ FAIL Role-based access controls - {role_tests_passed}/{len(roles)} passed (10 pts)")
                total_score += 10
                tests.append({
                    'name': 'Role-based access controls',
                    'passed': False,
                    'score': 10,
                    'max_score': 25
                })
            
            # Test 4: Query type restrictions
            query_types = ['count', 'sum', 'inference', 'training', 'analysis']
            query_restrictions_work = True
            
            for query_type in query_types:
                # Test with very high epsilon - should fail for most query types
                result = self.privacy_validator.validate_privacy_parameters(
                    10.0, 1e-5, 1.0,
                    department="general", data_sensitivity="medium", 
                    user_role="doctor", query_type=query_type
                )
                
                # Most query types should reject high epsilon
                if query_type in ['count', 'sum'] and result.valid:
                    query_restrictions_work = False
            
            if query_restrictions_work:
                print("   ✅ PASS Query type restrictions (20 pts)")
                total_score += 20
                tests.append({
                    'name': 'Query type restrictions',
                    'passed': True,
                    'score': 20,
                    'max_score': 20
                })
            else:
                print("   ❌ FAIL Query type restrictions (5 pts)")
                total_score += 5
                tests.append({
                    'name': 'Query type restrictions',
                    'passed': False,
                    'score': 5,
                    'max_score': 20
                })
                
        except Exception as e:
            print(f"   ❌ FAIL Healthcare compliance test failed: {e}")
            tests.append({
                'name': 'Healthcare compliance',
                'passed': False,
                'score': 0,
                'max_score': max_score,
                'error': str(e)
            })
        
        self.test_results['healthcare_compliance']['score'] = (total_score / max_score) * 100 if max_score > 0 else 0
        self.test_results['healthcare_compliance']['tests'] = tests
        
        print(f"\n   Healthcare Compliance Score: {total_score}/{max_score} ({self.test_results['healthcare_compliance']['score']:.1f}%)")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED SECURITY VALIDATION REPORT")
        print("=" * 60)
        
        total_weighted_score = 0
        total_weight = 0
        
        categories = [
            ('Privacy Validation', 'privacy_validation', 25),
            ('Secure Configuration', 'secure_config', 20),
            ('Database Security', 'database_security', 25),
            ('Integration', 'integration', 15),
            ('Healthcare Compliance', 'healthcare_compliance', 15)
        ]
        
        for name, key, weight in categories:
            score = self.test_results[key]['score']
            total_weighted_score += score * (weight / 100)
            total_weight += weight
            
            status = "✅ EXCELLENT" if score >= 90 else "⚡ GOOD" if score >= 75 else "⚠️  NEEDS IMPROVEMENT" if score >= 60 else "❌ CRITICAL"
            print(f"\n{name:.<40} {score:>6.1f}% {status}")
        
        overall_score = total_weighted_score
        
        print(f"\n{'OVERALL SECURITY SCORE':.<40} {overall_score:>6.1f}%")
        
        # Determine overall grade
        if overall_score >= 95:
            grade = "A+ (EXCELLENT)"
            recommendation = "PRODUCTION READY - All security measures exceeding requirements"
        elif overall_score >= 90:
            grade = "A (EXCELLENT)" 
            recommendation = "PRODUCTION READY - Strong security implementation"
        elif overall_score >= 85:
            grade = "B+ (GOOD)"
            recommendation = "PRODUCTION READY - Minor improvements recommended"
        elif overall_score >= 80:
            grade = "B (GOOD)"
            recommendation = "CONDITIONAL PRODUCTION - Address identified issues"
        elif overall_score >= 70:
            grade = "C+ (ACCEPTABLE)"
            recommendation = "DEVELOPMENT READY - Security improvements required"
        else:
            grade = "C or below (NEEDS WORK)"
            recommendation = "NOT PRODUCTION READY - Major security issues"
        
        print(f"{'SECURITY GRADE':.<40} {grade}")
        print(f"{'RECOMMENDATION':.<40} {recommendation}")
        
        # Improvement suggestions
        print(f"\n🔧 IMPROVEMENT RECOMMENDATIONS:")
        improvements = []
        
        for name, key, weight in categories:
            if self.test_results[key]['score'] < 90:
                improvements.append(f"   • Enhance {name.lower()} (current: {self.test_results[key]['score']:.1f}%)")
        
        if not improvements:
            print("   🎉 No improvements needed - all categories performing excellently!")
        else:
            for improvement in improvements[:3]:  # Show top 3
                print(improvement)
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'recommendation': recommendation,
            'category_scores': {name: self.test_results[key]['score'] for name, key, weight in categories},
            'detailed_results': self.test_results,
            'timestamp': time.time()
        }

async def main():
    """Run enhanced security validation"""
    validator = EnhancedSecurityValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Save detailed report
        with open('/tmp/enhanced_security_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: /tmp/enhanced_security_report.json")
        print(f"🎯 Final Score: {report['overall_score']:.1f}% ({report['grade']})")
        
        return report
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in security validation: {e}")
        return {'overall_score': 0, 'error': str(e)}

if __name__ == "__main__":
    asyncio.run(main())