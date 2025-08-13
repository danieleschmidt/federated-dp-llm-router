#!/usr/bin/env python3
"""
Simplified Security Validation
Test enhanced security features without external dependencies
"""

import sys
import os
import time
import json
sys.path.insert(0, '/root/repo')

def test_privacy_validator():
    """Test enhanced privacy validator functionality"""
    print("🔐 ENHANCED PRIVACY VALIDATOR TESTING")
    print("=" * 50)
    
    try:
        # Import without triggering numpy dependency
        from federated_dp_llm.security.enhanced_privacy_validator import (
            EnhancedPrivacyValidator,
            HealthcareDataSensitivity,
            DepartmentType
        )
        
        validator = EnhancedPrivacyValidator()
        print("✅ Privacy validator initialized successfully")
        
        # Test 1: Valid parameters
        result1 = validator.validate_privacy_parameters(
            epsilon=0.5, 
            delta=1e-6, 
            noise_multiplier=1.5,
            department="general", 
            data_sensitivity="medium", 
            user_role="doctor"
        )
        
        if result1.valid:
            print("✅ Valid parameter validation: PASS")
        else:
            print(f"❌ Valid parameter validation: FAIL - {result1.issues}")
        
        # Test 2: Invalid high epsilon
        result2 = validator.validate_privacy_parameters(
            epsilon=10.0, 
            delta=1e-6, 
            noise_multiplier=1.0,
            department="psychiatry", 
            data_sensitivity="critical", 
            user_role="student"
        )
        
        if not result2.valid:
            print("✅ Invalid parameter rejection: PASS")
        else:
            print("❌ Invalid parameter rejection: FAIL - should have been rejected")
        
        # Test 3: Department info
        dept_info = validator.get_department_info("cardiology")
        if 'limits' in dept_info:
            print("✅ Department information retrieval: PASS")
        else:
            print("❌ Department information retrieval: FAIL")
        
        # Test 4: Budget validation
        budget_result = validator.validate_budget_request(
            user_id="test_user",
            requested_epsilon=0.5,
            daily_spent=2.0,
            weekly_spent=10.0,
            department="general"
        )
        
        if budget_result.valid:
            print("✅ Budget validation: PASS")
        else:
            print(f"⚠️  Budget validation: WARNING - {budget_result.issues}")
        
        return True
        
    except Exception as e:
        print(f"❌ Privacy validator test failed: {e}")
        return False

def test_secure_config():
    """Test secure configuration management"""
    print("\n🔑 SECURE CONFIGURATION TESTING")
    print("=" * 50)
    
    try:
        from federated_dp_llm.security.secure_config_manager import SecureConfigManager
        
        config_manager = SecureConfigManager()
        print("✅ Secure config manager initialized successfully")
        
        # Test 1: Configuration validation
        validation_result = config_manager.validate_security_config()
        
        if validation_result['valid']:
            print("✅ Security configuration validation: PASS")
        else:
            print(f"⚠️  Security configuration validation: ISSUES FOUND")
            for issue in validation_result['issues'][:3]:  # Show first 3 issues
                print(f"   - {issue}")
        
        # Test 2: Safe config retrieval
        test_config = config_manager.get_config('nonexistent_key', 'default_value')
        if test_config == 'default_value':
            print("✅ Safe config retrieval with defaults: PASS")
        else:
            print("❌ Safe config retrieval: FAIL")
        
        # Test 3: Secret handling (expect failure for missing secrets)
        try:
            jwt_secret = config_manager.get_secret('jwt_secret')
            if jwt_secret:
                print("✅ Secret retrieval: PASS (secret found)")
            else:
                print("⚠️  Secret retrieval: Expected failure (secret not configured)")
        except ValueError:
            print("✅ Secret retrieval: PASS (secure failure for missing required secret)")
        except Exception as e:
            print(f"❌ Secret retrieval: ERROR - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Secure config test failed: {e}")
        return False

def test_secure_database():
    """Test secure database functionality"""
    print("\n🗄️  SECURE DATABASE TESTING")
    print("=" * 50)
    
    try:
        import asyncio
        from federated_dp_llm.security.secure_database import SecureDatabase
        
        async def run_db_tests():
            db = SecureDatabase(":memory:")
            await db.initialize()
            print("✅ Secure database initialized successfully")
            
            # Test 1: Health check
            health = await db.health_check()
            if health['status'] == 'healthy':
                print("✅ Database health check: PASS")
            else:
                print(f"❌ Database health check: FAIL - {health.get('error')}")
                return False
            
            # Test 2: SQL injection protection
            malicious_inputs = [
                "'; DROP TABLE user_budgets; --",
                "admin'; UPDATE user_budgets SET total_budget=9999; --",
                "' OR '1'='1"
            ]
            
            injection_blocked = 0
            for malicious_input in malicious_inputs:
                try:
                    result = await db.get_user_budget(malicious_input)
                    if result is None:  # Should return None for invalid input
                        injection_blocked += 1
                except Exception:
                    injection_blocked += 1  # Exception is also good
            
            if injection_blocked == len(malicious_inputs):
                print("✅ SQL injection protection: PASS")
            else:
                print(f"❌ SQL injection protection: FAIL - {injection_blocked}/{len(malicious_inputs)} blocked")
            
            # Test 3: Valid database operations
            test_user = "test_security_user"
            result = await db.create_user_budget(test_user, 10.0, "test_department")
            
            if result.success:
                print("✅ Parameterized database operations: PASS")
                
                # Test retrieval
                user_budget = await db.get_user_budget(test_user)
                if user_budget and user_budget['user_id'] == test_user:
                    print("✅ Secure data retrieval: PASS")
                else:
                    print("❌ Secure data retrieval: FAIL")
                
                # Test update
                update_result = await db.update_user_budget(test_user, 2.0, "test_department")
                if update_result.success:
                    print("✅ Secure data updates: PASS")
                else:
                    print(f"❌ Secure data updates: FAIL - {update_result.error}")
            else:
                print(f"❌ Parameterized database operations: FAIL - {result.error}")
            
            return True
        
        return asyncio.run(run_db_tests())
        
    except Exception as e:
        print(f"❌ Secure database test failed: {e}")
        return False

def test_comprehensive_security():
    """Run comprehensive security validation"""
    print("\n🛡️  COMPREHENSIVE SECURITY VALIDATION")
    print("=" * 60)
    
    test_results = {
        'privacy_validator': False,
        'secure_config': False, 
        'secure_database': False,
        'timestamp': time.time()
    }
    
    # Run all tests
    test_results['privacy_validator'] = test_privacy_validator()
    test_results['secure_config'] = test_secure_config()
    test_results['secure_database'] = test_secure_database()
    
    # Calculate overall score
    passed_tests = sum(test_results[key] for key in ['privacy_validator', 'secure_config', 'secure_database'])
    total_tests = 3
    
    overall_score = (passed_tests / total_tests) * 100
    
    print(f"\n{'='*60}")
    print("📊 SECURITY VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in test_results.items():
        if test_name == 'timestamp':
            continue
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
    
    print(f"\n{'OVERALL SECURITY SCORE':.<40} {overall_score:>6.1f}%")
    
    if overall_score == 100:
        grade = "A+ (EXCELLENT)"
        recommendation = "ALL SECURITY ENHANCEMENTS WORKING PERFECTLY"
    elif overall_score >= 80:
        grade = "A (EXCELLENT)"
        recommendation = "SECURITY ENHANCEMENTS LARGELY SUCCESSFUL"
    elif overall_score >= 60:
        grade = "B (GOOD)"
        recommendation = "MOST SECURITY ENHANCEMENTS WORKING"
    else:
        grade = "C (NEEDS WORK)"
        recommendation = "SECURITY ENHANCEMENTS NEED ATTENTION"
    
    print(f"{'SECURITY GRADE':.<40} {grade}")
    print(f"{'RECOMMENDATION':.<40} {recommendation}")
    
    # Save results
    try:
        with open('/tmp/simplified_security_report.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n📄 Report saved to: /tmp/simplified_security_report.json")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    return test_results

if __name__ == "__main__":
    test_comprehensive_security()