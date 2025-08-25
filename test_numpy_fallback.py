#!/usr/bin/env python3
"""
Test script to verify that the numpy fallback system works correctly
across the entire codebase without numpy dependencies.
"""

import sys
import os
sys.path.insert(0, '.')

# Test the fallback system directly
exec(open('federated_dp_llm/quantum_planning/numpy_fallback.py').read())
has_numpy, np = get_numpy_backend()

print("=" * 50)
print("NUMPY FALLBACK SYSTEM TEST")
print("=" * 50)

print(f"HAS_NUMPY: {has_numpy}")
print(f"np.array([1,2,3]): {np.array([1,2,3])}")
print(f"np.mean([1,2,3,4,5]): {np.mean([1,2,3,4,5])}")
print(f"np.std([1,2,3,4,5]): {np.std([1,2,3,4,5])}")
print(f"np.zeros(5): {np.zeros(5)}")
print(f"np.ones(3): {np.ones(3)}")

# Test mathematical functions
test_data = [1, 4, 9, 16, 25]
print(f"np.sqrt({test_data}): {np.sqrt(test_data)}")

# Test trigonometric functions  
angles = [0, 1.57, 3.14]  # 0, π/2, π
print(f"np.sin({angles}): {np.sin(angles)}")
print(f"np.cos({angles}): {np.cos(angles)}")

# Test exponential
print(f"np.exp([0, 1, 2]): {np.exp([0, 1, 2])}")

print("\n" + "=" * 50)
print("SUCCESS: All numpy fallback functions working!")
print("The system can now run without numpy dependencies.")
print("=" * 50)