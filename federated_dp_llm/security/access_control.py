"""
Comprehensive Access Control and Authorization System

Implements role-based access control (RBAC), attribute-based access control (ABAC),
and multi-factor authentication for healthcare environments.
"""

import time
import secrets
import hashlib
import json
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pyotp


class AccessLevel(Enum):
    """Access levels for healthcare data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PHI = "phi"  # Protected Health Information


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    QUERY_INFERENCE = "query_inference"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    RESET_PRIVACY_BUDGET = "reset_privacy_budget"
    MANAGE_MODELS = "manage_models"
    CONFIGURE_SYSTEM = "configure_system"


class Role(Enum):
    """Healthcare system roles."""
    PATIENT = "patient"
    NURSE = "nurse"
    DOCTOR = "doctor"
    SPECIALIST = "specialist"
    ADMINISTRATOR = "administrator"
    RESEARCHER = "researcher"
    IT_SUPPORT = "it_support"
    SYSTEM_ADMIN = "system_admin"


@dataclass
class User:
    """User account with role and permissions."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    roles: List[Role]
    permissions: List[Permission]
    department: str
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        self.updated_at = time.time()


@dataclass
class AccessRequest:
    """Access request for authorization."""
    user_id: str
    resource: str
    action: str
    context: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class AccessDecision:
    """Authorization decision."""
    granted: bool
    reason: str
    conditions: List[str]
    expires_at: Optional[float] = None


class RolePermissionManager:
    """Manages role-based permissions."""
    
    def __init__(self):
        self.role_permissions = {
            Role.PATIENT: [
                Permission.READ,
            ],
            Role.NURSE: [
                Permission.READ,
                Permission.WRITE,
                Permission.QUERY_INFERENCE,
            ],
            Role.DOCTOR: [
                Permission.READ,
                Permission.WRITE,
                Permission.QUERY_INFERENCE,
            ],
            Role.SPECIALIST: [
                Permission.READ,
                Permission.WRITE,
                Permission.QUERY_INFERENCE,
            ],
            Role.ADMINISTRATOR: [
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.ADMIN,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.RESET_PRIVACY_BUDGET,
            ],
            Role.RESEARCHER: [
                Permission.READ,
                Permission.QUERY_INFERENCE,
            ],
            Role.IT_SUPPORT: [
                Permission.READ,
                Permission.VIEW_AUDIT_LOGS,
            ],
            Role.SYSTEM_ADMIN: [
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.ADMIN,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.RESET_PRIVACY_BUDGET,
                Permission.MANAGE_MODELS,
                Permission.CONFIGURE_SYSTEM,
            ],
        }
    
    def get_permissions_for_role(self, role: Role) -> List[Permission]:
        """Get permissions for a specific role."""
        return self.role_permissions.get(role, [])
    
    def get_all_permissions_for_roles(self, roles: List[Role]) -> Set[Permission]:
        """Get all permissions for a list of roles."""
        all_permissions = set()
        for role in roles:
            permissions = self.get_permissions_for_role(role)
            all_permissions.update(permissions)
        return all_permissions


class MultiFactorAuth:
    """Multi-factor authentication manager."""
    
    def __init__(self):
        self.backup_codes: Dict[str, List[str]] = {}
    
    def generate_mfa_secret(self, user_id: str) -> str:
        """Generate MFA secret for user."""
        secret = pyotp.random_base32()
        return secret
    
    def get_qr_code_url(self, user_id: str, secret: str, issuer: str = "Federated DP-LLM") -> str:
        """Get QR code URL for MFA setup."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user_id,
            issuer_name=issuer
        )
    
    def verify_mfa_token(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify MFA token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for user."""
        codes = [secrets.token_hex(8) for _ in range(count)]
        self.backup_codes[user_id] = codes
        return codes
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code."""
        user_codes = self.backup_codes.get(user_id, [])
        if code in user_codes:
            user_codes.remove(code)
            return True
        return False


class AccessControlManager:
    """Main access control and authorization manager."""
    
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
        self.users: Dict[str, User] = {}
        self.role_manager = RolePermissionManager()
        self.mfa_manager = MultiFactorAuth()
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        
        # Security policies
        self.max_failed_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.session_timeout = 3600  # 1 hour
        self.password_min_length = 12
        self.require_mfa_for_admin = True
    
    def _hash_password(self, password: str, salt: bytes = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        password_hash = kdf.derive(password.encode('utf-8'))
        
        return password_hash.hex(), salt.hex()
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        computed_hash, _ = self._hash_password(password, bytes.fromhex(salt))
        return secrets.compare_digest(computed_hash, stored_hash)
    
    def _validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password meets security requirements."""
        issues = []
        
        if len(password) < self.password_min_length:
            issues.append(f"Password must be at least {self.password_min_length} characters")
        
        if not any(c.isupper() for c in password):
            issues.append("Password must contain uppercase letter")
        
        if not any(c.islower() for c in password):
            issues.append("Password must contain lowercase letter")
        
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain number")
        
        if not any(c in "!@#$%^&*()_+-=" for c in password):
            issues.append("Password must contain special character")
        
        return len(issues) == 0, issues
    
    def create_user(self, username: str, email: str, password: str, 
                   department: str, roles: List[Role]) -> Tuple[Optional[User], str]:
        """Create new user account."""
        # Validate password
        valid, issues = self._validate_password_strength(password)
        if not valid:
            return None, "; ".join(issues)
        
        # Check if user exists
        if any(user.username == username for user in self.users.values()):
            return None, "Username already exists"
        
        if any(user.email == email for user in self.users.values()):
            return None, "Email already exists"
        
        # Hash password
        password_hash, salt = self._hash_password(password)
        
        # Get permissions for roles
        permissions = list(self.role_manager.get_all_permissions_for_roles(roles))
        
        # Create user
        user_id = secrets.token_hex(16)
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles,
            permissions=permissions,
            department=department
        )
        
        # Enable MFA for admin roles
        if self.require_mfa_for_admin and any(role in [Role.ADMINISTRATOR, Role.SYSTEM_ADMIN] for role in roles):
            user.mfa_enabled = True
            user.mfa_secret = self.mfa_manager.generate_mfa_secret(user_id)
        
        self.users[user_id] = user
        
        return user, "User created successfully"
    
    def authenticate_user(self, username: str, password: str, mfa_token: str = None) -> Tuple[Optional[User], str]:
        """Authenticate user with username/password and optional MFA."""
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            return None, "Invalid credentials"
        
        # Check if account is locked
        if user.locked_until and time.time() < user.locked_until:
            return None, f"Account locked until {time.ctime(user.locked_until)}"
        
        # Verify password
        if not self._verify_password(password, user.password_hash, user.salt):
            self._record_failed_attempt(user.user_id)
            return None, "Invalid credentials"
        
        # Verify MFA if enabled
        if user.mfa_enabled:
            if not mfa_token:
                return None, "MFA token required"
            
            if not self.mfa_manager.verify_mfa_token(user.mfa_secret, mfa_token):
                # Try backup code
                if not self.mfa_manager.verify_backup_code(user.user_id, mfa_token):
                    return None, "Invalid MFA token"
        
        # Clear failed attempts
        if user.user_id in self.failed_attempts:
            del self.failed_attempts[user.user_id]
        
        # Update last login
        user.last_login = time.time()
        user.failed_attempts = 0
        user.locked_until = None
        
        return user, "Authentication successful"
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Add current attempt
        self.failed_attempts[user_id].append(current_time)
        
        # Remove attempts older than 1 hour
        hour_ago = current_time - 3600
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > hour_ago
        ]
        
        # Lock account if too many attempts
        if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            user = self.users.get(user_id)
            if user:
                user.locked_until = current_time + self.lockout_duration
    
    def create_session(self, user: User) -> str:
        """Create authenticated session token."""
        session_id = secrets.token_hex(32)
        
        payload = {
            "session_id": session_id,
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "department": user.department,
            "iat": time.time(),
            "exp": time.time() + self.session_timeout
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Store session info
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "ip_address": None,  # Would be set by request handler
            "user_agent": None   # Would be set by request handler
        }
        
        return token
    
    def validate_session(self, token: str) -> Tuple[Optional[Dict], str]:
        """Validate session token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            session_id = payload.get("session_id")
            
            # Check if session exists
            if session_id not in self.active_sessions:
                return None, "Session not found"
            
            # Update last activity
            self.active_sessions[session_id]["last_activity"] = time.time()
            
            return payload, "Valid session"
            
        except jwt.ExpiredSignatureError:
            return None, "Session expired"
        except jwt.InvalidTokenError:
            return None, "Invalid token"
    
    def authorize_access(self, user_id: str, resource: str, action: str, 
                        context: Dict = None) -> AccessDecision:
        """Authorize user access to resource."""
        user = self.users.get(user_id)
        if not user:
            return AccessDecision(
                granted=False,
                reason="User not found",
                conditions=[]
            )
        
        # Check basic permission
        required_permission = self._map_action_to_permission(action)
        if required_permission not in user.permissions:
            return AccessDecision(
                granted=False,
                reason=f"Missing required permission: {required_permission.value}",
                conditions=[]
            )
        
        # Apply attribute-based access control
        conditions = []
        
        # Department-based restrictions
        if context and "department" in context:
            target_department = context["department"]
            if target_department != user.department and not self._has_cross_department_access(user):
                return AccessDecision(
                    granted=False,
                    reason="Cross-department access denied",
                    conditions=[]
                )
        
        # Time-based restrictions
        if self._is_outside_business_hours() and not self._has_after_hours_access(user):
            conditions.append("After-hours access - additional monitoring applies")
        
        # PHI access restrictions
        if "phi" in resource.lower():
            if not self._has_phi_access(user):
                return AccessDecision(
                    granted=False,
                    reason="PHI access denied - insufficient clearance",
                    conditions=[]
                )
            conditions.append("PHI access - enhanced audit logging enabled")
        
        return AccessDecision(
            granted=True,
            reason="Access granted",
            conditions=conditions
        )
    
    def _map_action_to_permission(self, action: str) -> Permission:
        """Map action to required permission."""
        action_permission_map = {
            "read": Permission.READ,
            "write": Permission.WRITE,
            "delete": Permission.DELETE,
            "query": Permission.QUERY_INFERENCE,
            "admin": Permission.ADMIN,
        }
        
        return action_permission_map.get(action.lower(), Permission.READ)
    
    def _has_cross_department_access(self, user: User) -> bool:
        """Check if user has cross-department access."""
        admin_roles = [Role.ADMINISTRATOR, Role.SYSTEM_ADMIN]
        return any(role in admin_roles for role in user.roles)
    
    def _is_outside_business_hours(self) -> bool:
        """Check if current time is outside business hours."""
        import datetime
        now = datetime.datetime.now()
        # Business hours: 6 AM to 10 PM
        return now.hour < 6 or now.hour >= 22
    
    def _has_after_hours_access(self, user: User) -> bool:
        """Check if user has after-hours access."""
        after_hours_roles = [Role.DOCTOR, Role.NURSE, Role.ADMINISTRATOR, Role.SYSTEM_ADMIN]
        emergency_departments = ["emergency", "icu", "trauma"]
        
        return (any(role in after_hours_roles for role in user.roles) or
                user.department.lower() in emergency_departments)
    
    def _has_phi_access(self, user: User) -> bool:
        """Check if user has PHI access."""
        phi_roles = [Role.DOCTOR, Role.NURSE, Role.SPECIALIST, Role.ADMINISTRATOR]
        return any(role in phi_roles for role in user.roles)
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke active session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get active sessions for user."""
        user_sessions = []
        for session_id, session_info in self.active_sessions.items():
            if session_info["user_id"] == user_id:
                session_info["session_id"] = session_id
                user_sessions.append(session_info)
        return user_sessions
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_info in self.active_sessions.items():
            if current_time - session_info["last_activity"] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        return len(expired_sessions)
    
    def get_access_control_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        total_users = len(self.users)
        active_sessions = len(self.active_sessions)
        locked_users = sum(1 for user in self.users.values() 
                          if user.locked_until and time.time() < user.locked_until)
        mfa_enabled_users = sum(1 for user in self.users.values() if user.mfa_enabled)
        
        # Role distribution
        role_counts = {}
        for user in self.users.values():
            for role in user.roles:
                role_counts[role.value] = role_counts.get(role.value, 0) + 1
        
        return {
            "total_users": total_users,
            "active_sessions": active_sessions,
            "locked_users": locked_users,
            "mfa_enabled_users": mfa_enabled_users,
            "mfa_adoption_rate": mfa_enabled_users / total_users if total_users > 0 else 0,
            "role_distribution": role_counts,
            "security_policies": {
                "max_failed_attempts": self.max_failed_attempts,
                "lockout_duration": self.lockout_duration,
                "session_timeout": self.session_timeout,
                "password_min_length": self.password_min_length,
                "require_mfa_for_admin": self.require_mfa_for_admin
            }
        }


# Global access control instance
_access_control_instance = None

def get_access_control_manager() -> AccessControlManager:
    """Get global access control manager instance."""
    global _access_control_instance
    if _access_control_instance is None:
        jwt_secret = os.environ.get("JWT_SECRET_KEY")
        if not jwt_secret:
            raise ValueError("JWT_SECRET_KEY environment variable must be set")
        _access_control_instance = AccessControlManager(jwt_secret)
    return _access_control_instance