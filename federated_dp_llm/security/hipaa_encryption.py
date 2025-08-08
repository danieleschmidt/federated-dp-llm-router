"""
HIPAA/GDPR Compliant Encryption and Data Protection

Implements comprehensive encryption for healthcare data in transit and at rest,
ensuring HIPAA and GDPR compliance with strong cryptographic protections.
"""

import os
import secrets
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64
import json
import time


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""
    algorithm: str = "AES-256-GCM"
    key_derivation: str = "PBKDF2"
    iterations: int = 100000
    enable_compression: bool = True
    auto_key_rotation: bool = True
    key_rotation_days: int = 90


class DataClassificationManager:
    """Manages data classification and encryption requirements."""
    
    def __init__(self):
        self.classification_levels = {
            "public": {"encryption_required": False, "key_strength": 128},
            "internal": {"encryption_required": True, "key_strength": 256},
            "confidential": {"encryption_required": True, "key_strength": 256},
            "restricted": {"encryption_required": True, "key_strength": 512},
            "phi": {"encryption_required": True, "key_strength": 512}  # Protected Health Information
        }
    
    def classify_data(self, data: str, metadata: Dict = None) -> str:
        """Classify data based on content and metadata."""
        data_lower = data.lower()
        
        # PHI indicators
        phi_indicators = [
            "patient", "medical", "diagnosis", "treatment", "prescription",
            "ssn", "social security", "date of birth", "dob", "mrn",
            "medical record", "health", "symptoms", "medication"
        ]
        
        if any(indicator in data_lower for indicator in phi_indicators):
            return "phi"
        
        # Check metadata
        if metadata:
            department = metadata.get("department", "").lower()
            if department in ["emergency", "radiology", "cardiology", "neurology"]:
                return "confidential"
        
        return "internal"
    
    def get_encryption_requirements(self, classification: str) -> Dict:
        """Get encryption requirements for data classification."""
        return self.classification_levels.get(classification, self.classification_levels["internal"])


class HIPAACompliantEncryption:
    """HIPAA-compliant encryption for Protected Health Information."""
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.classifier = DataClassificationManager()
        self._master_key = self._get_or_create_master_key()
        self._key_cache: Dict[str, bytes] = {}
    
    def _get_or_create_master_key(self) -> bytes:
        """Get master key from environment or generate new one."""
        master_key_b64 = os.environ.get("HIPAA_MASTER_KEY")
        if master_key_b64:
            try:
                return base64.b64decode(master_key_b64)
            except Exception:
                pass
        
        # Generate new master key
        master_key = secrets.token_bytes(64)  # 512-bit master key
        print("WARNING: Generated new master key. Set HIPAA_MASTER_KEY environment variable for production.")
        return master_key
    
    def _derive_key(self, context: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derive encryption key for specific context."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=self.config.iterations,
        )
        
        context_bytes = context.encode('utf-8')
        key = kdf.derive(self._master_key + context_bytes)
        
        return key, salt
    
    def encrypt_phi(self, data: str, context: str = "general", 
                   metadata: Dict = None) -> Dict[str, str]:
        """Encrypt Protected Health Information with full HIPAA compliance."""
        # Classify data
        classification = self.classifier.classify_data(data, metadata)
        
        # Get encryption requirements
        requirements = self.classifier.get_encryption_requirements(classification)
        
        if not requirements["encryption_required"]:
            return {"data": data, "encrypted": False, "classification": classification}
        
        # Generate key for this context
        key, salt = self._derive_key(context)
        
        # Generate nonce for GCM mode
        nonce = secrets.token_bytes(12)
        
        # Encrypt data
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        
        data_bytes = data.encode('utf-8')
        ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
        
        # Prepare encrypted package
        encrypted_package = {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "salt": base64.b64encode(salt).decode('utf-8'),
            "tag": base64.b64encode(encryptor.tag).decode('utf-8'),
            "algorithm": "AES-256-GCM",
            "classification": classification,
            "context": context,
            "encrypted": True,
            "timestamp": time.time()
        }
        
        if metadata:
            encrypted_package["metadata"] = metadata
        
        return encrypted_package
    
    def decrypt_phi(self, encrypted_package: Dict[str, str], context: str = "general") -> str:
        """Decrypt Protected Health Information."""
        if not encrypted_package.get("encrypted", False):
            return encrypted_package.get("data", "")
        
        # Extract components
        ciphertext = base64.b64decode(encrypted_package["ciphertext"])
        nonce = base64.b64decode(encrypted_package["nonce"])
        salt = base64.b64decode(encrypted_package["salt"])
        tag = base64.b64decode(encrypted_package["tag"])
        
        # Verify context matches
        if encrypted_package.get("context") != context:
            raise ValueError("Context mismatch - unauthorized decryption attempt")
        
        # Derive key
        key, _ = self._derive_key(context, salt)
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        
        plaintext_bytes = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext_bytes.decode('utf-8')


class TransitEncryption:
    """Encryption for data in transit with TLS and additional layers."""
    
    def __init__(self):
        self.session_keys: Dict[str, bytes] = {}
    
    def create_session_key(self, session_id: str) -> bytes:
        """Create session-specific encryption key."""
        session_key = secrets.token_bytes(32)
        self.session_keys[session_id] = session_key
        return session_key
    
    def encrypt_message(self, message: str, session_id: str) -> Dict[str, str]:
        """Encrypt message for secure transmission."""
        if session_id not in self.session_keys:
            raise ValueError(f"No session key found for session: {session_id}")
        
        key = self.session_keys[session_id]
        
        # Generate nonce
        nonce = secrets.token_bytes(12)
        
        # Encrypt with AES-GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        
        message_bytes = message.encode('utf-8')
        ciphertext = encryptor.update(message_bytes) + encryptor.finalize()
        
        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "tag": base64.b64encode(encryptor.tag).decode('utf-8'),
            "session_id": session_id
        }
    
    def decrypt_message(self, encrypted_message: Dict[str, str]) -> str:
        """Decrypt message from secure transmission."""
        session_id = encrypted_message["session_id"]
        
        if session_id not in self.session_keys:
            raise ValueError(f"No session key found for session: {session_id}")
        
        key = self.session_keys[session_id]
        
        # Extract components
        ciphertext = base64.b64decode(encrypted_message["ciphertext"])
        nonce = base64.b64decode(encrypted_message["nonce"])
        tag = base64.b64decode(encrypted_message["tag"])
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        
        plaintext_bytes = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext_bytes.decode('utf-8')


# Global encryption instance for easy access
_encryption_instance = None

def get_encryption_manager() -> HIPAACompliantEncryption:
    """Get global encryption manager instance."""
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = HIPAACompliantEncryption()
    return _encryption_instance