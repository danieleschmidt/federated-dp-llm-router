#!/usr/bin/env python3
"""
Secure Configuration Manager
Addresses hardcoded secrets and insecure configuration management
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

@dataclass
class SecretConfig:
    """Secure secret configuration"""
    name: str
    env_var: str
    default_file_path: Optional[str] = None
    required: bool = True
    encrypted: bool = True

class SecureConfigManager:
    """
    Secure configuration manager that eliminates hardcoded secrets
    and provides proper secret management
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.environ.get("FEDERATED_CONFIG_FILE", "/etc/federated-dp-llm/config.json")
        self.secrets_config = {
            "jwt_secret": SecretConfig(
                name="jwt_secret",
                env_var="FEDERATED_JWT_SECRET",
                default_file_path="/etc/federated-dp-llm/secrets/jwt.key",
                required=True
            ),
            "master_encryption_key": SecretConfig(
                name="master_encryption_key", 
                env_var="FEDERATED_MASTER_KEY",
                default_file_path="/etc/federated-dp-llm/secrets/master.key",
                required=True
            ),
            "database_password": SecretConfig(
                name="database_password",
                env_var="FEDERATED_DB_PASSWORD", 
                default_file_path="/etc/federated-dp-llm/secrets/db.key",
                required=True
            ),
            "api_key": SecretConfig(
                name="api_key",
                env_var="FEDERATED_API_KEY",
                default_file_path="/etc/federated-dp-llm/secrets/api.key",
                required=False
            )
        }
        self.config_cache = {}
        self.encryption_key = self._get_or_create_encryption_key()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for config encryption"""
        key_path = os.environ.get("FEDERATED_CONFIG_ENCRYPTION_KEY", "/etc/federated-dp-llm/secrets/config.key")
        
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            password = os.environ.get("FEDERATED_CONFIG_PASSWORD", "default-password-change-me").encode()
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Save key securely (in production, use proper key management)
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            logger.warning(f"Generated new config encryption key at {key_path}")
            
            return key
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get secret from secure sources (environment variables or encrypted files)
        Never returns hardcoded secrets
        """
        if secret_name not in self.secrets_config:
            logger.error(f"Unknown secret requested: {secret_name}")
            return None
            
        config = self.secrets_config[secret_name]
        
        # Check environment variable first
        secret_value = os.environ.get(config.env_var)
        if secret_value:
            logger.debug(f"Retrieved secret {secret_name} from environment variable")
            return secret_value
            
        # Check encrypted file
        if config.default_file_path and os.path.exists(config.default_file_path):
            try:
                with open(config.default_file_path, 'rb') as f:
                    encrypted_data = f.read()
                    
                if config.encrypted:
                    fernet = Fernet(self.encryption_key)
                    decrypted_data = fernet.decrypt(encrypted_data)
                    secret_value = decrypted_data.decode('utf-8')
                else:
                    secret_value = encrypted_data.decode('utf-8')
                    
                logger.debug(f"Retrieved secret {secret_name} from encrypted file")
                return secret_value
                
            except Exception as e:
                logger.error(f"Failed to read secret {secret_name} from file: {e}")
        
        if config.required:
            logger.error(f"Required secret {secret_name} not found in environment or secure file")
            raise ValueError(f"Required secret {secret_name} not configured")
        
        logger.warning(f"Optional secret {secret_name} not configured")
        return None
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """
        Set secret in encrypted file storage
        """
        if secret_name not in self.secrets_config:
            logger.error(f"Unknown secret: {secret_name}")
            return False
            
        config = self.secrets_config[secret_name]
        
        if not config.default_file_path:
            logger.error(f"No file path configured for secret {secret_name}")
            return False
            
        try:
            # Encrypt the secret
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(secret_value.encode('utf-8'))
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(config.default_file_path), exist_ok=True)
            
            # Write encrypted secret
            with open(config.default_file_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(config.default_file_path, 0o600)
            
            logger.info(f"Successfully stored secret {secret_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret {secret_name}: {e}")
            return False
    
    def get_config(self, config_key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if config_key in self.config_cache:
            return self.config_cache[config_key]
            
        # Check environment variable first
        env_key = f"FEDERATED_{config_key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            # Try to parse as JSON for complex types
            try:
                value = json.loads(env_value)
            except json.JSONDecodeError:
                value = env_value
            
            self.config_cache[config_key] = value
            return value
        
        # Check config file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                if config_key in config_data:
                    value = config_data[config_key]
                    self.config_cache[config_key] = value
                    return value
                    
            except Exception as e:
                logger.error(f"Failed to read config file {self.config_file}: {e}")
        
        # Return default
        self.config_cache[config_key] = default
        return default
    
    def validate_security_config(self) -> Dict[str, Any]:
        """Validate security configuration"""
        validation_results = {
            "valid": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check that no hardcoded secrets are present in environment
        dangerous_patterns = [
            "password=",
            "secret=", 
            "api_key=",
            "token="
        ]
        
        for key, value in os.environ.items():
            if key.startswith("FEDERATED_"):
                continue  # Skip our managed environment variables
                
            value_lower = str(value).lower()
            for pattern in dangerous_patterns:
                if pattern in value_lower:
                    validation_results["valid"] = False
                    validation_results["issues"].append(
                        f"Potentially hardcoded secret in environment variable {key}"
                    )
        
        # Check that required secrets are available
        for secret_name, config in self.secrets_config.items():
            if config.required:
                try:
                    secret_value = self.get_secret(secret_name)
                    if not secret_value:
                        validation_results["valid"] = False
                        validation_results["issues"].append(f"Required secret {secret_name} not configured")
                except Exception as e:
                    validation_results["valid"] = False
                    validation_results["issues"].append(f"Failed to validate secret {secret_name}: {e}")
        
        # Security recommendations
        if not os.environ.get("FEDERATED_CONFIG_PASSWORD"):
            validation_results["recommendations"].append(
                "Set FEDERATED_CONFIG_PASSWORD environment variable for enhanced security"
            )
        
        return validation_results

# Global secure config manager instance
_secure_config = None

def get_secure_config() -> SecureConfigManager:
    """Get global secure configuration manager"""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfigManager()
    return _secure_config

def get_secret(secret_name: str) -> Optional[str]:
    """Convenience function to get secret"""
    return get_secure_config().get_secret(secret_name)

def get_config(config_key: str, default: Any = None) -> Any:
    """Convenience function to get config"""
    return get_secure_config().get_config(config_key, default)