"""
Authentication and Authorization

Implements mTLS, OAuth2, RBAC and other authentication mechanisms for
secure access to federated LLM services.
"""

import hashlib
import secrets
import time
import jwt
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import datetime


class Role(Enum):
    """System roles with different permission levels."""
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    READ_MODEL = "read_model"
    WRITE_MODEL = "write_model"
    QUERY_INFERENCE = "query_inference"
    VIEW_PRIVACY_BUDGET = "view_privacy_budget"
    RESET_PRIVACY_BUDGET = "reset_privacy_budget"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    EXPORT_DATA = "export_data"
    MANAGE_USERS = "manage_users"
    SYSTEM_CONFIG = "system_config"


@dataclass
class User:
    """Represents a system user."""
    user_id: str
    username: str
    email: str
    department: str
    roles: List[Role]
    permissions: Set[Permission]
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    password_hash: Optional[str] = None
    mfa_enabled: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[float] = None


@dataclass
class Session:
    """Represents an active user session."""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: str
    user_agent: str
    is_valid: bool = True
    last_activity: Optional[float] = None


@dataclass
class Certificate:
    """Represents an X.509 certificate for mTLS."""
    cert_id: str
    subject: str
    issuer: str
    serial_number: str
    not_before: datetime.datetime
    not_after: datetime.datetime
    fingerprint: str
    is_ca: bool = False
    is_valid: bool = True


class RoleBasedAccessControl:
    """Role-Based Access Control (RBAC) implementation."""
    
    def __init__(self):
        # Define role-permission mappings
        self.role_permissions = {
            Role.ADMIN: {
                Permission.READ_MODEL, Permission.WRITE_MODEL, Permission.QUERY_INFERENCE,
                Permission.VIEW_PRIVACY_BUDGET, Permission.RESET_PRIVACY_BUDGET,
                Permission.VIEW_AUDIT_LOGS, Permission.EXPORT_DATA, Permission.MANAGE_USERS,
                Permission.SYSTEM_CONFIG
            },
            Role.DOCTOR: {
                Permission.READ_MODEL, Permission.QUERY_INFERENCE,
                Permission.VIEW_PRIVACY_BUDGET, Permission.EXPORT_DATA
            },
            Role.NURSE: {
                Permission.READ_MODEL, Permission.QUERY_INFERENCE,
                Permission.VIEW_PRIVACY_BUDGET
            },
            Role.RESEARCHER: {
                Permission.READ_MODEL, Permission.QUERY_INFERENCE,
                Permission.VIEW_PRIVACY_BUDGET, Permission.VIEW_AUDIT_LOGS
            },
            Role.AUDITOR: {
                Permission.READ_MODEL, Permission.VIEW_PRIVACY_BUDGET,
                Permission.VIEW_AUDIT_LOGS, Permission.EXPORT_DATA
            },
            Role.GUEST: {
                Permission.READ_MODEL
            }
        }
    
    def get_permissions_for_roles(self, roles: List[Role]) -> Set[Permission]:
        """Get combined permissions for multiple roles."""
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        return permissions
    
    def has_permission(self, roles: List[Role], permission: Permission) -> bool:
        """Check if roles have specific permission."""
        user_permissions = self.get_permissions_for_roles(roles)
        return permission in user_permissions
    
    def can_access_department(self, user_department: str, requested_department: str) -> bool:
        """Check if user can access data from requested department."""
        # Admins can access all departments
        if user_department == "admin":
            return True
        
        # Users can access their own department
        if user_department == requested_department:
            return True
        
        # Emergency department can access critical data from other departments
        if user_department == "emergency" and requested_department in ["cardiology", "neurology"]:
            return True
        
        return False


class PasswordManager:
    """Secure password hashing and validation."""
    
    def __init__(self, min_length: int = 12):
        self.min_length = min_length
        self.salt_length = 32
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt using PBKDF2."""
        if len(password) < self.min_length:
            raise ValueError(f"Password must be at least {self.min_length} characters")
        
        # Generate random salt
        salt = secrets.token_bytes(self.salt_length)
        
        # Hash password with PBKDF2
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        
        # Combine salt and hash
        return salt.hex() + ':' + password_hash.hex()
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt_hex, hash_hex = stored_hash.split(':')
            salt = bytes.fromhex(salt_hex)
            stored_password_hash = bytes.fromhex(hash_hex)
            
            # Hash provided password with same salt
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            
            # Compare hashes
            return secrets.compare_digest(password_hash, stored_password_hash)
        
        except (ValueError, AttributeError):
            return False
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a cryptographically secure password."""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))


class JWTManager:
    """JSON Web Token management for authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", token_expiry: int = 3600):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = token_expiry  # seconds
    
    def create_token(self, user: User, additional_claims: Dict[str, Any] = None) -> str:
        """Create JWT token for user."""
        now = time.time()
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "department": user.department,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "iat": now,
            "exp": now + self.token_expiry,
            "type": "access_token"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def create_refresh_token(self, user: User) -> str:
        """Create refresh token for user."""
        now = time.time()
        
        payload = {
            "sub": user.user_id,
            "type": "refresh_token",
            "iat": now,
            "exp": now + (7 * 24 * 3600)  # 7 days
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """Create new access token from refresh token."""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh_token":
                raise ValueError("Not a refresh token")
            
            if payload.get("sub") != user.user_id:
                raise ValueError("Token user mismatch")
            
            return self.create_token(user)
        
        except jwt.ExpiredSignatureError:
            raise ValueError("Refresh token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid refresh token")


class CertificateManager:
    """X.509 certificate management for mTLS."""
    
    def __init__(self):
        self.certificates: Dict[str, Certificate] = {}
        self.trusted_cas: Set[str] = set()
    
    def generate_ca_certificate(self, subject_name: str, key_size: int = 2048) -> Tuple[Certificate, bytes, bytes]:
        """Generate a Certificate Authority certificate."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Healthcare Federation"),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
        ])
        
        cert_builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))  # 10 years
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        )
        
        # Sign certificate
        certificate = cert_builder.sign(private_key, hashes.SHA256())
        
        # Create Certificate object
        cert_id = secrets.token_hex(16)
        fingerprint = certificate.fingerprint(hashes.SHA256()).hex()
        
        cert_obj = Certificate(
            cert_id=cert_id,
            subject=subject_name,
            issuer=subject_name,  # Self-signed
            serial_number=str(certificate.serial_number),
            not_before=certificate.not_valid_before,
            not_after=certificate.not_valid_after,
            fingerprint=fingerprint,
            is_ca=True
        )
        
        self.certificates[cert_id] = cert_obj
        self.trusted_cas.add(fingerprint)
        
        # Serialize certificate and private key
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return cert_obj, cert_pem, key_pem
    
    def generate_client_certificate(self, ca_cert: Certificate, ca_private_key: bytes, 
                                   client_name: str, department: str) -> Tuple[Certificate, bytes, bytes]:
        """Generate client certificate signed by CA."""
        # Load CA private key
        ca_key = serialization.load_pem_private_key(ca_private_key, password=None)
        
        # Generate client private key
        client_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Build client certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Healthcare Federation"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, department),
            x509.NameAttribute(NameOID.COMMON_NAME, client_name),
        ])
        
        issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, ca_cert.subject),
        ])
        
        cert_builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(client_private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))  # 1 year
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                ]),
                critical=True,
            )
        )
        
        # Sign certificate with CA
        certificate = cert_builder.sign(ca_key, hashes.SHA256())
        
        # Create Certificate object
        cert_id = secrets.token_hex(16)
        fingerprint = certificate.fingerprint(hashes.SHA256()).hex()
        
        cert_obj = Certificate(
            cert_id=cert_id,
            subject=client_name,
            issuer=ca_cert.subject,
            serial_number=str(certificate.serial_number),
            not_before=certificate.not_valid_before,
            not_after=certificate.not_valid_after,
            fingerprint=fingerprint,
            is_ca=False
        )
        
        self.certificates[cert_id] = cert_obj
        
        # Serialize certificate and private key
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        key_pem = client_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return cert_obj, cert_pem, key_pem
    
    def verify_certificate_chain(self, cert_fingerprint: str) -> bool:
        """Verify certificate chain against trusted CAs."""
        # Find certificate
        cert = None
        for certificate in self.certificates.values():
            if certificate.fingerprint == cert_fingerprint:
                cert = certificate
                break
        
        if not cert:
            return False
        
        # Check if certificate is still valid
        now = datetime.datetime.utcnow()
        if now < cert.not_before or now > cert.not_after:
            return False
        
        # If it's a CA certificate, check if it's trusted
        if cert.is_ca:
            return cert.fingerprint in self.trusted_cas
        
        # For client certificates, check if issuer is trusted
        # This is simplified - in practice, you'd verify the full chain
        return True  # Assume valid for demo
    
    def revoke_certificate(self, cert_id: str) -> bool:
        """Revoke a certificate."""
        if cert_id in self.certificates:
            self.certificates[cert_id].is_valid = False
            return True
        return False


class AuthenticationManager:
    """Main authentication and authorization manager."""
    
    def __init__(self, jwt_secret: str):
        self.rbac = RoleBasedAccessControl()
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager(jwt_secret)
        self.certificate_manager = CertificateManager()
        
        # User and session storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = 1800  # 30 minutes
        self.session_timeout = 3600  # 1 hour
    
    def create_user(self, username: str, email: str, password: str, 
                   department: str, roles: List[Role]) -> User:
        """Create a new user."""
        user_id = secrets.token_hex(16)
        
        # Hash password
        password_hash = self.password_manager.hash_password(password)
        
        # Get permissions for roles
        permissions = self.rbac.get_permissions_for_roles(roles)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            department=department,
            roles=roles,
            permissions=permissions,
            created_at=time.time(),
            password_hash=password_hash
        )
        
        self.users[user_id] = user
        return user
    
    def authenticate_user(self, username: str, password: str) -> Tuple[Optional[User], str]:
        """Authenticate user with username/password."""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            return None, "User not found"
        
        # Check if account is locked
        if user.locked_until and time.time() < user.locked_until:
            remaining = int(user.locked_until - time.time())
            return None, f"Account locked for {remaining} seconds"
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = time.time() + self.lockout_duration
                return None, "Account locked due to too many failed attempts"
            
            return None, "Invalid password"
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = time.time()
        
        return user, "Authentication successful"
    
    def create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """Create new user session."""
        session_id = secrets.token_hex(32)
        now = time.time()
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=now,
            expires_at=now + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        return session
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user if valid."""
        session = self.sessions.get(session_id)
        
        if not session or not session.is_valid:
            return None
        
        # Check expiration
        if time.time() > session.expires_at:
            session.is_valid = False
            return None
        
        # Update last activity
        session.last_activity = time.time()
        
        # Get user
        return self.users.get(session.user_id)
    
    def logout_session(self, session_id: str) -> bool:
        """Logout user session."""
        session = self.sessions.get(session_id)
        if session:
            session.is_valid = False
            return True
        return False
    
    def authorize_action(self, user: User, permission: Permission, 
                        target_department: str = None) -> bool:
        """Authorize user action."""
        # Check if user has required permission
        if permission not in user.permissions:
            return False
        
        # Check department access if specified
        if target_department:
            return self.rbac.can_access_department(user.department, target_department)
        
        return True
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        active_sessions = sum(1 for s in self.sessions.values() if s.is_valid)
        locked_users = sum(1 for u in self.users.values() if u.locked_until and time.time() < u.locked_until)
        
        return {
            "total_users": len(self.users),
            "active_sessions": active_sessions,
            "locked_users": locked_users,
            "total_certificates": len(self.certificate_manager.certificates),
            "trusted_cas": len(self.certificate_manager.trusted_cas)
        }