"""
Numpy Fallback Implementation for Quantum Planning
Provides mathematical operations without numpy dependency for resilient operation.
"""

import math
import random
from typing import List, Tuple, Union, Optional


class NumpyFallback:
    """Fallback implementation for essential numpy operations."""
    
    @staticmethod
    def array(data: List) -> List:
        """Convert to list-based array."""
        return data if isinstance(data, list) else list(data)
    
    @staticmethod
    def zeros(shape: Union[int, Tuple[int, ...]]) -> List:
        """Create zero-filled array."""
        if isinstance(shape, int):
            return [0.0] * shape
        elif isinstance(shape, tuple) and len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        else:
            raise NotImplementedError("Only 1D and 2D arrays supported")
    
    @staticmethod
    def ones(shape: Union[int, Tuple[int, ...]]) -> List:
        """Create ones-filled array."""
        if isinstance(shape, int):
            return [1.0] * shape
        elif isinstance(shape, tuple) and len(shape) == 2:
            return [[1.0] * shape[1] for _ in range(shape[0])]
        else:
            raise NotImplementedError("Only 1D and 2D arrays supported")
    
    @staticmethod
    def random_normal(mean: float = 0.0, std: float = 1.0, size: Optional[int] = None) -> Union[float, List[float]]:
        """Generate random numbers from normal distribution."""
        if size is None:
            return random.gauss(mean, std)
        return [random.gauss(mean, std) for _ in range(size)]
    
    @staticmethod
    def random_uniform(low: float = 0.0, high: float = 1.0, size: Optional[int] = None) -> Union[float, List[float]]:
        """Generate random numbers from uniform distribution."""
        if size is None:
            return random.uniform(low, high)
        return [random.uniform(low, high) for _ in range(size)]
    
    @staticmethod
    def exp(x: Union[float, List[float]]) -> Union[float, List[float]]:
        """Exponential function."""
        if isinstance(x, (int, float)):
            return math.exp(x)
        return [math.exp(val) for val in x]
    
    @staticmethod
    def sin(x: Union[float, List[float]]) -> Union[float, List[float]]:
        """Sine function."""
        if isinstance(x, (int, float)):
            return math.sin(x)
        return [math.sin(val) for val in x]
    
    @staticmethod
    def cos(x: Union[float, List[float]]) -> Union[float, List[float]]:
        """Cosine function."""
        if isinstance(x, (int, float)):
            return math.cos(x)
        return [math.cos(val) for val in x]
    
    @staticmethod
    def sqrt(x: Union[float, List[float]]) -> Union[float, List[float]]:
        """Square root function."""
        if isinstance(x, (int, float)):
            return math.sqrt(x)
        return [math.sqrt(val) for val in x]
    
    @staticmethod
    def sum(arr: List) -> float:
        """Sum of array elements."""
        return sum(arr) if isinstance(arr[0], (int, float)) else sum(sum(row) for row in arr)
    
    @staticmethod
    def mean(arr: List) -> float:
        """Mean of array elements."""
        if isinstance(arr[0], (int, float)):
            return sum(arr) / len(arr)
        else:
            total_elements = sum(len(row) for row in arr)
            total_sum = sum(sum(row) for row in arr)
            return total_sum / total_elements
    
    @staticmethod
    def std(arr: List) -> float:
        """Standard deviation of array elements."""
        if isinstance(arr[0], (int, float)):
            mean_val = sum(arr) / len(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
            return math.sqrt(variance)
        else:
            flat = [val for row in arr for val in row]
            mean_val = sum(flat) / len(flat)
            variance = sum((x - mean_val) ** 2 for x in flat) / len(flat)
            return math.sqrt(variance)
    
    @staticmethod
    def dot(a: List, b: List) -> float:
        """Dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def argmax(arr: List) -> int:
        """Index of maximum element."""
        return max(range(len(arr)), key=arr.__getitem__)
    
    @staticmethod
    def argmin(arr: List) -> int:
        """Index of minimum element."""
        return min(range(len(arr)), key=arr.__getitem__)
    
    @staticmethod
    def clip(arr: List, min_val: float, max_val: float) -> List:
        """Clip array values to range."""
        return [max(min_val, min(max_val, val)) for val in arr]
    
    @staticmethod
    def abs(arr: Union[float, List]) -> Union[float, List]:
        """Absolute value."""
        if isinstance(arr, (int, float)):
            return abs(arr)
        return [abs(val) for val in arr]
    
    @staticmethod
    def pi() -> float:
        """Pi constant."""
        return math.pi
    
    @staticmethod
    def e() -> float:
        """Euler's number."""
        return math.e


# Conditional import with fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = NumpyFallback()
    HAS_NUMPY = False


def get_numpy_backend() -> Tuple[bool, object]:
    """Get numpy backend with fallback information."""
    return HAS_NUMPY, np


def quantum_wavefunction(amplitudes: List[float], phases: List[float]) -> List[complex]:
    """Create quantum wavefunction from amplitudes and phases."""
    if HAS_NUMPY:
        return [amp * (math.cos(phase) + 1j * math.sin(phase)) 
                for amp, phase in zip(amplitudes, phases)]
    else:
        return [complex(amp * math.cos(phase), amp * math.sin(phase)) 
                for amp, phase in zip(amplitudes, phases)]


def quantum_probability(wavefunction: List[complex]) -> List[float]:
    """Calculate probabilities from quantum wavefunction."""
    return [abs(amplitude) ** 2 for amplitude in wavefunction]


def quantum_interference(wave1: List[complex], wave2: List[complex]) -> List[complex]:
    """Calculate quantum interference between two wavefunctions."""
    return [w1 + w2 for w1, w2 in zip(wave1, wave2)]


def quantum_coherence(wavefunction: List[complex]) -> float:
    """Calculate quantum coherence measure."""
    probabilities = quantum_probability(wavefunction)
    max_prob = max(probabilities)
    return max_prob / sum(probabilities) if sum(probabilities) > 0 else 0.0