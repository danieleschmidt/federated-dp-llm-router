"""
Homomorphic Encryption and Secure Computation

Implements homomorphic encryption schemes for secure computation on encrypted
data in federated learning scenarios.
"""

import secrets
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import struct


@dataclass
class EncryptedValue:
    """Represents an encrypted value with metadata."""
    ciphertext: bytes
    encryption_scheme: str
    key_id: str
    metadata: Dict[str, Any]


@dataclass
class KeyPair:
    """Represents a cryptographic key pair."""
    public_key: bytes
    private_key: bytes
    key_id: str
    algorithm: str
    key_size: int


class HomomorphicScheme:
    """Base class for homomorphic encryption schemes."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self.key_id = None
    
    def generate_keys(self) -> KeyPair:
        """Generate a new key pair."""
        raise NotImplementedError
    
    def encrypt(self, plaintext: Union[int, float, np.ndarray]) -> EncryptedValue:
        """Encrypt plaintext value."""
        raise NotImplementedError
    
    def decrypt(self, ciphertext: EncryptedValue) -> Union[int, float, np.ndarray]:
        """Decrypt ciphertext value."""
        raise NotImplementedError
    
    def add(self, a: EncryptedValue, b: EncryptedValue) -> EncryptedValue:
        """Homomorphic addition of two encrypted values."""
        raise NotImplementedError
    
    def multiply(self, a: EncryptedValue, b: Union[EncryptedValue, int, float]) -> EncryptedValue:
        """Homomorphic multiplication (may be limited to plaintext multiplier)."""
        raise NotImplementedError


class PaillierHomomorphic(HomomorphicScheme):
    """Simplified Paillier homomorphic encryption scheme."""
    
    def __init__(self, key_size: int = 1024):
        super().__init__(key_size)
        self.n = None  # Public key modulus
        self.g = None  # Public key generator
        self.lambda_n = None  # Private key lambda
        self.mu = None  # Private key mu
    
    def generate_keys(self) -> KeyPair:
        """Generate Paillier key pair."""
        # Generate two large primes (simplified for demo)
        p = self._generate_prime(self.key_size // 2)
        q = self._generate_prime(self.key_size // 2)
        
        # Calculate public key parameters
        self.n = p * q
        self.g = self.n + 1  # Simplified generator
        
        # Calculate private key parameters
        self.lambda_n = (p - 1) * (q - 1) // self._gcd(p - 1, q - 1)
        self.mu = self._mod_inverse(self._L(pow(self.g, self.lambda_n, self.n * self.n)), self.n)
        
        # Generate key ID
        self.key_id = secrets.token_hex(16)
        
        # Serialize keys
        public_key_data = struct.pack('!QQ', self.n, self.g)
        private_key_data = struct.pack('!QQ', self.lambda_n, self.mu)
        
        return KeyPair(
            public_key=public_key_data,
            private_key=private_key_data,
            key_id=self.key_id,
            algorithm="paillier",
            key_size=self.key_size
        )
    
    def encrypt(self, plaintext: Union[int, float]) -> EncryptedValue:
        """Encrypt plaintext using Paillier encryption."""
        if self.n is None:
            raise ValueError("Keys not generated")
        
        # Convert to integer if needed
        if isinstance(plaintext, float):
            # Scale float to integer (lose some precision)
            m = int(plaintext * 1000000)  # 6 decimal places
        else:
            m = int(plaintext)
        
        # Ensure plaintext is in valid range
        m = m % self.n
        
        # Generate random r
        r = secrets.randbelow(self.n)
        while self._gcd(r, self.n) != 1:
            r = secrets.randbelow(self.n)
        
        # Compute ciphertext: c = g^m * r^n mod n^2
        n_squared = self.n * self.n
        c = (pow(self.g, m, n_squared) * pow(r, self.n, n_squared)) % n_squared
        
        # Serialize ciphertext
        ciphertext_bytes = struct.pack('!Q', c)
        
        return EncryptedValue(
            ciphertext=ciphertext_bytes,
            encryption_scheme="paillier",
            key_id=self.key_id,
            metadata={"n": self.n, "is_float": isinstance(plaintext, float)}
        )
    
    def decrypt(self, encrypted_value: EncryptedValue) -> Union[int, float]:
        """Decrypt Paillier ciphertext."""
        if self.lambda_n is None or self.mu is None:
            raise ValueError("Private key not available")
        
        # Deserialize ciphertext
        c = struct.unpack('!Q', encrypted_value.ciphertext)[0]
        
        # Decrypt: m = L(c^lambda mod n^2) * mu mod n
        n_squared = self.n * self.n
        c_lambda = pow(c, self.lambda_n, n_squared)
        l_value = self._L(c_lambda)
        m = (l_value * self.mu) % self.n
        
        # Convert back to original format
        if encrypted_value.metadata.get("is_float", False):
            return float(m) / 1000000.0
        else:
            return int(m)
    
    def add(self, a: EncryptedValue, b: EncryptedValue) -> EncryptedValue:
        """Homomorphic addition of two Paillier ciphertexts."""
        if a.key_id != b.key_id:
            raise ValueError("Cannot add ciphertexts encrypted with different keys")
        
        # Deserialize ciphertexts
        c1 = struct.unpack('!Q', a.ciphertext)[0]
        c2 = struct.unpack('!Q', b.ciphertext)[0]
        
        # Homomorphic addition: c1 * c2 mod n^2
        n_squared = self.n * self.n
        c_result = (c1 * c2) % n_squared
        
        # Serialize result
        result_bytes = struct.pack('!Q', c_result)
        
        return EncryptedValue(
            ciphertext=result_bytes,
            encryption_scheme="paillier",
            key_id=self.key_id,
            metadata={"n": self.n, "is_float": a.metadata.get("is_float", False)}
        )
    
    def multiply(self, a: EncryptedValue, b: Union[int, float]) -> EncryptedValue:
        """Homomorphic multiplication by plaintext scalar."""
        if not isinstance(b, (int, float)):
            raise ValueError("Can only multiply by plaintext scalars")
        
        # Deserialize ciphertext
        c = struct.unpack('!Q', a.ciphertext)[0]
        
        # Convert scalar to integer if needed
        if isinstance(b, float):
            scalar = int(b * 1000000)
        else:
            scalar = int(b)
        
        # Homomorphic scalar multiplication: c^scalar mod n^2
        n_squared = self.n * self.n
        c_result = pow(c, scalar, n_squared)
        
        # Serialize result
        result_bytes = struct.pack('!Q', c_result)
        
        return EncryptedValue(
            ciphertext=result_bytes,
            encryption_scheme="paillier",
            key_id=self.key_id,
            metadata={"n": self.n, "is_float": isinstance(b, float) or a.metadata.get("is_float", False)}
        )
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a prime number (simplified for demo)."""
        # In a real implementation, use a proper prime generation algorithm
        # This is a simplified version for demonstration
        import random
        
        min_val = 2 ** (bits - 1)
        max_val = 2 ** bits - 1
        
        while True:
            candidate = random.randint(min_val, max_val)
            if self._is_prime(candidate):
                return candidate
    
    def _is_prime(self, n: int) -> bool:
        """Simple primality test (not cryptographically secure)."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check odd divisors up to sqrt(n)
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        
        return True
    
    def _gcd(self, a: int, b: int) -> int:
        """Greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Modular inverse using extended Euclidean algorithm."""
        if self._gcd(a, m) != 1:
            raise ValueError("Modular inverse does not exist")
        
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        _, x, _ = extended_gcd(a, m)
        return (x % m + m) % m
    
    def _L(self, x: int) -> int:
        """L function for Paillier: L(x) = (x - 1) / n."""
        return (x - 1) // self.n


class HomomorphicEncryption:
    """Main homomorphic encryption manager."""
    
    def __init__(self, scheme: str = "paillier", key_size: int = 1024):
        if scheme == "paillier":
            self.scheme = PaillierHomomorphic(key_size)
        else:
            raise ValueError(f"Unsupported homomorphic scheme: {scheme}")
        
        self.scheme_name = scheme
        self.keys: Optional[KeyPair] = None
    
    def generate_keys(self) -> KeyPair:
        """Generate new encryption keys."""
        self.keys = self.scheme.generate_keys()
        return self.keys
    
    def load_keys(self, key_pair: KeyPair):
        """Load existing key pair."""
        if key_pair.algorithm != self.scheme_name:
            raise ValueError(f"Key algorithm {key_pair.algorithm} doesn't match scheme {self.scheme_name}")
        
        self.keys = key_pair
        
        # Load keys into scheme
        if self.scheme_name == "paillier":
            public_data = struct.unpack('!QQ', key_pair.public_key)
            private_data = struct.unpack('!QQ', key_pair.private_key)
            
            self.scheme.n = public_data[0]
            self.scheme.g = public_data[1]
            self.scheme.lambda_n = private_data[0]
            self.scheme.mu = private_data[1]
            self.scheme.key_id = key_pair.key_id
    
    def encrypt_array(self, array: np.ndarray) -> List[EncryptedValue]:
        """Encrypt a numpy array element-wise."""
        if self.keys is None:
            raise ValueError("No keys loaded")
        
        encrypted_values = []
        flat_array = array.flatten()
        
        for value in flat_array:
            encrypted_value = self.scheme.encrypt(float(value))
            encrypted_values.append(encrypted_value)
        
        return encrypted_values
    
    def decrypt_array(self, encrypted_values: List[EncryptedValue], shape: Tuple[int, ...]) -> np.ndarray:
        """Decrypt a list of encrypted values back to numpy array."""
        if self.keys is None:
            raise ValueError("No keys loaded")
        
        decrypted_values = []
        for encrypted_value in encrypted_values:
            decrypted_value = self.scheme.decrypt(encrypted_value)
            decrypted_values.append(decrypted_value)
        
        return np.array(decrypted_values).reshape(shape)
    
    def add_encrypted_arrays(self, a: List[EncryptedValue], b: List[EncryptedValue]) -> List[EncryptedValue]:
        """Add two encrypted arrays element-wise."""
        if len(a) != len(b):
            raise ValueError("Arrays must have same length")
        
        result = []
        for enc_a, enc_b in zip(a, b):
            sum_encrypted = self.scheme.add(enc_a, enc_b)
            result.append(sum_encrypted)
        
        return result
    
    def multiply_encrypted_array(self, encrypted_array: List[EncryptedValue], scalar: Union[int, float]) -> List[EncryptedValue]:
        """Multiply encrypted array by plaintext scalar."""
        result = []
        for encrypted_value in encrypted_array:
            multiplied_value = self.scheme.multiply(encrypted_value, scalar)
            result.append(multiplied_value)
        
        return result
    
    def secure_average(self, encrypted_arrays: List[List[EncryptedValue]]) -> List[EncryptedValue]:
        """Compute secure average of multiple encrypted arrays."""
        if not encrypted_arrays:
            raise ValueError("No arrays provided")
        
        # Start with first array
        result = encrypted_arrays[0].copy()
        
        # Add remaining arrays
        for encrypted_array in encrypted_arrays[1:]:
            result = self.add_encrypted_arrays(result, encrypted_array)
        
        # Divide by number of arrays
        num_arrays = len(encrypted_arrays)
        result = self.multiply_encrypted_array(result, 1.0 / num_arrays)
        
        return result
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get information about current encryption setup."""
        if self.keys is None:
            return {"status": "no_keys_loaded"}
        
        return {
            "scheme": self.scheme_name,
            "key_id": self.keys.key_id,
            "key_size": self.keys.key_size,
            "algorithm": self.keys.algorithm,
            "supports_addition": True,
            "supports_scalar_multiplication": True,
            "supports_multiplication": False,  # Only scalar multiplication
            "security_level": "additively_homomorphic"
        }


class SecureMultiPartyComputation:
    """Secure multi-party computation utilities."""
    
    def __init__(self, num_parties: int):
        self.num_parties = num_parties
        self.homomorphic_schemes: Dict[str, HomomorphicEncryption] = {}
    
    def register_party(self, party_id: str, homomorphic_encryption: HomomorphicEncryption):
        """Register a party with their homomorphic encryption setup."""
        self.homomorphic_schemes[party_id] = homomorphic_encryption
    
    def secure_sum(self, party_contributions: Dict[str, List[EncryptedValue]]) -> List[EncryptedValue]:
        """Compute secure sum across all parties."""
        if not party_contributions:
            raise ValueError("No contributions provided")
        
        # Use first party's encryption scheme for computation
        first_party = list(party_contributions.keys())[0]
        he_scheme = self.homomorphic_schemes[first_party]
        
        # Start with first contribution
        result = party_contributions[first_party].copy()
        
        # Add remaining contributions
        for party_id, contribution in party_contributions.items():
            if party_id == first_party:
                continue
            
            result = he_scheme.add_encrypted_arrays(result, contribution)
        
        return result
    
    def secure_average(self, party_contributions: Dict[str, List[EncryptedValue]]) -> List[EncryptedValue]:
        """Compute secure average across all parties."""
        secure_sum_result = self.secure_sum(party_contributions)
        
        # Get encryption scheme from first party
        first_party = list(party_contributions.keys())[0]
        he_scheme = self.homomorphic_schemes[first_party]
        
        # Divide by number of parties
        num_parties = len(party_contributions)
        result = he_scheme.multiply_encrypted_array(secure_sum_result, 1.0 / num_parties)
        
        return result
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get statistics about secure computation setup."""
        return {
            "num_parties": self.num_parties,
            "registered_parties": len(self.homomorphic_schemes),
            "encryption_schemes": [
                he.get_encryption_info()
                for he in self.homomorphic_schemes.values()
            ]
        }