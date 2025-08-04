"""
Secure Aggregation Protocols

Implements cryptographic protocols for secure multi-party computation
in federated learning scenarios with privacy preservation.
"""

import hashlib
import secrets
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import numpy as np


@dataclass
class SecureShare:
    """Represents a secure share in the aggregation protocol."""
    node_id: str
    encrypted_data: bytes
    signature: bytes
    commitment: str


@dataclass
class AggregationRound:
    """Tracks a single round of secure aggregation."""
    round_id: str
    participants: List[str]
    shares: Dict[str, SecureShare]
    threshold: int
    status: str = "collecting"


class HomomorphicCipher:
    """Simplified homomorphic encryption for aggregation."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt(self, plaintext: int) -> int:
        """Encrypt integer using RSA (simplified for demo)."""
        plaintext_bytes = plaintext.to_bytes(256, 'big')
        ciphertext = self.public_key.encrypt(
            plaintext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return int.from_bytes(ciphertext, 'big')
    
    def decrypt(self, ciphertext: int) -> int:
        """Decrypt integer using RSA."""
        ciphertext_bytes = ciphertext.to_bytes(512, 'big')
        plaintext_bytes = self.private_key.decrypt(
            ciphertext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return int.from_bytes(plaintext_bytes, 'big')
    
    def add_encrypted(self, c1: int, c2: int) -> int:
        """Homomorphic addition (simplified - not actually homomorphic)."""
        # In a real implementation, this would use actual homomorphic properties
        p1 = self.decrypt(c1)
        p2 = self.decrypt(c2)
        return self.encrypt(p1 + p2)


class SecretSharing:
    """Shamir's Secret Sharing implementation."""
    
    def __init__(self, threshold: int, num_shares: int, prime: int = 2**31 - 1):
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Calculate modular inverse."""
        return pow(a, m - 2, m)
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x."""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, self.prime)) % self.prime
        return result
    
    def create_shares(self, secret: int) -> List[Tuple[int, int]]:
        """Create secret shares using polynomial interpolation."""
        # Generate random coefficients for polynomial
        coefficients = [secret] + [
            secrets.randbelow(self.prime) for _ in range(self.threshold - 1)
        ]
        
        # Create shares
        shares = []
        for i in range(1, self.num_shares + 1):
            share_value = self._evaluate_polynomial(coefficients, i)
            shares.append((i, share_value))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret from threshold number of shares."""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        # Use first threshold shares
        shares = shares[:self.threshold]
        
        # Lagrange interpolation
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (0 - xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            lagrange_coeff = (numerator * self._mod_inverse(denominator, self.prime)) % self.prime
            secret = (secret + yi * lagrange_coeff) % self.prime
        
        return secret


class SecureAggregator:
    """Main secure aggregation coordinator."""
    
    def __init__(self, threshold: int = 3, use_homomorphic: bool = True):
        self.threshold = threshold
        self.use_homomorphic = use_homomorphic
        self.active_rounds: Dict[str, AggregationRound] = {}
        
        if use_homomorphic:
            self.homomorphic_cipher = HomomorphicCipher()
        
        self.secret_sharing = SecretSharing(threshold, threshold * 2)
    
    def start_aggregation_round(self, round_id: str, participants: List[str]) -> AggregationRound:
        """Start a new secure aggregation round."""
        if len(participants) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} participants")
        
        round_obj = AggregationRound(
            round_id=round_id,
            participants=participants,
            shares={},
            threshold=self.threshold
        )
        
        self.active_rounds[round_id] = round_obj
        return round_obj
    
    def create_commitment(self, data: np.ndarray, nonce: bytes = None) -> str:
        """Create cryptographic commitment to data."""
        if nonce is None:
            nonce = secrets.token_bytes(32)
        
        # Serialize data
        data_bytes = data.tobytes()
        
        # Create commitment hash
        hasher = hashlib.sha256()
        hasher.update(data_bytes)
        hasher.update(nonce)
        
        return hasher.hexdigest()
    
    def encrypt_share(self, data: np.ndarray, node_id: str) -> bytes:
        """Encrypt data share for secure transmission."""
        # Convert numpy array to bytes
        data_bytes = data.tobytes()
        
        # Generate symmetric key
        key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        # Encrypt data with AES
        cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
        
        # Combine IV and ciphertext
        return iv + ciphertext
    
    def submit_share(self, round_id: str, node_id: str, data: np.ndarray) -> bool:
        """Submit encrypted share to aggregation round."""
        if round_id not in self.active_rounds:
            raise ValueError(f"Round {round_id} not found")
        
        round_obj = self.active_rounds[round_id]
        
        if node_id not in round_obj.participants:
            raise ValueError(f"Node {node_id} not registered for this round")
        
        if round_obj.status != "collecting":
            raise ValueError(f"Round {round_id} not accepting shares")
        
        # Create secure share
        encrypted_data = self.encrypt_share(data, node_id)
        commitment = self.create_commitment(data)
        signature = self._sign_data(encrypted_data, node_id)
        
        share = SecureShare(
            node_id=node_id,
            encrypted_data=encrypted_data,
            signature=signature,
            commitment=commitment
        )
        
        round_obj.shares[node_id] = share
        
        # Check if we have enough shares
        if len(round_obj.shares) >= round_obj.threshold:
            round_obj.status = "ready"
        
        return True
    
    def aggregate_shares(self, round_id: str) -> Optional[np.ndarray]:
        """Perform secure aggregation of submitted shares."""
        if round_id not in self.active_rounds:
            raise ValueError(f"Round {round_id} not found")
        
        round_obj = self.active_rounds[round_id]
        
        if round_obj.status != "ready":
            return None
        
        if len(round_obj.shares) < round_obj.threshold:
            raise ValueError("Insufficient shares for aggregation")
        
        # Verify all shares first
        for share in round_obj.shares.values():
            if not self._verify_share(share):
                raise ValueError(f"Invalid share from {share.node_id}")
        
        # Decrypt and aggregate
        decrypted_shares = []
        for share in round_obj.shares.values():
            decrypted_data = self._decrypt_share(share.encrypted_data)
            decrypted_shares.append(decrypted_data)
        
        # Simple averaging (in practice, would use more sophisticated aggregation)
        if decrypted_shares:
            aggregated = np.mean(decrypted_shares, axis=0)
            round_obj.status = "completed"
            return aggregated
        
        return None
    
    def _sign_data(self, data: bytes, node_id: str) -> bytes:
        """Create signature for data (simplified)."""
        hasher = hashlib.sha256()
        hasher.update(data)
        hasher.update(node_id.encode())
        return hasher.digest()
    
    def _verify_share(self, share: SecureShare) -> bool:
        """Verify the integrity of a share."""
        expected_signature = self._sign_data(share.encrypted_data, share.node_id)
        return secrets.compare_digest(expected_signature, share.signature)
    
    def _decrypt_share(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt share data (simplified - needs proper key management)."""
        # Extract IV and ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # For demo purposes, use a fixed key (in practice, use proper key exchange)
        key = b'\x00' * 32
        
        cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Convert back to numpy array (assumes float64)
        return np.frombuffer(plaintext, dtype=np.float64)
    
    def get_round_status(self, round_id: str) -> Optional[str]:
        """Get status of aggregation round."""
        round_obj = self.active_rounds.get(round_id)
        return round_obj.status if round_obj else None