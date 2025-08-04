"""
Logging Configuration

Configures structured logging with privacy-aware log handling, audit trails,
and secure log management for healthcare compliance.
"""

import logging
import logging.handlers
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import re
from dataclasses import dataclass, asdict


@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "structured"  # structured, simple
    output: str = "console"  # console, file, both
    log_file: Optional[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    enable_audit_logging: bool = True
    audit_log_file: Optional[str] = None
    enable_privacy_filtering: bool = True
    enable_security_logging: bool = True


class PrivacyFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    def __init__(self):
        super().__init__()
        
        # Patterns for sensitive data
        self.sensitive_patterns = [
            # Social Security Numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
            (r'\b\d{9}\b', '[SSN_REDACTED]'),
            
            # Credit Card Numbers
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC_REDACTED]'),
            
            # Email addresses (partial redaction)
            (r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', r'\1***@***.\2'),
            
            # Phone numbers
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]'),
            
            # Medical Record Numbers (various formats)
            (r'\bMRN[\s:]?\d+\b', '[MRN_REDACTED]'),
            (r'\b[A-Z]{2,3}\d{6,10}\b', '[MEDICAL_ID_REDACTED]'),
            
            # JWT tokens
            (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', '[JWT_REDACTED]'),
            
            # API keys and secrets
            (r'["\']?(?:api_key|secret|token|password)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', 
             r'[API_KEY_REDACTED]'),
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.sensitive_patterns
        ]
    
    def filter(self, record):
        """Filter log record to remove sensitive information."""
        # Apply privacy filtering to message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._redact_sensitive_data(record.msg)
        
        # Apply filtering to args if present
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    filtered_args.append(self._redact_sensitive_data(arg))
                else:
                    filtered_args.append(arg)
            record.args = tuple(filtered_args)
        
        return True
    
    def _redact_sensitive_data(self, text: str) -> str:
        """Redact sensitive data from text."""
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)
        return text


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""
    
    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': time.time(),
            'datetime': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        if self.include_extra_fields:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }:
                    extra_fields[key] = value
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Prevent propagation to avoid duplicate logs
        self.audit_logger.propagate = False
        
        if config.audit_log_file:
            # Rotating file handler for audit logs
            audit_handler = logging.handlers.RotatingFileHandler(
                config.audit_log_file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count
            )
            
            # Always use structured format for audit logs
            audit_formatter = StructuredFormatter()
            audit_handler.setFormatter(audit_formatter)
            
            # Add privacy filter
            if config.enable_privacy_filtering:
                audit_handler.addFilter(PrivacyFilter())
            
            self.audit_logger.addHandler(audit_handler)
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str = None, 
                          user_agent: str = None, failure_reason: str = None):
        """Log authentication event."""
        self.audit_logger.info(
            "Authentication event",
            extra={
                'event_type': 'authentication',
                'user_id': user_id,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'failure_reason': failure_reason,
                'timestamp': time.time()
            }
        )
    
    def log_privacy_query(self, user_id: str, department: str, epsilon_spent: float,
                         query_type: str, success: bool, error: str = None):
        """Log privacy-preserving query."""
        self.audit_logger.info(
            "Privacy query event",
            extra={
                'event_type': 'privacy_query',
                'user_id': user_id,
                'department': department,
                'epsilon_spent': epsilon_spent,
                'query_type': query_type,
                'success': success,
                'error': error,
                'timestamp': time.time()
            }
        )
    
    def log_model_access(self, user_id: str, model_name: str, action: str,
                        success: bool, details: Dict[str, Any] = None):
        """Log model access event."""
        self.audit_logger.info(
            "Model access event",
            extra={
                'event_type': 'model_access',
                'user_id': user_id,
                'model_name': model_name,
                'action': action,
                'success': success,
                'details': details or {},
                'timestamp': time.time()
            }
        )
    
    def log_configuration_change(self, user_id: str, component: str, 
                               old_value: Any, new_value: Any):
        """Log configuration change."""
        self.audit_logger.info(
            "Configuration change",
            extra={
                'event_type': 'configuration_change',
                'user_id': user_id,
                'component': component,
                'old_value': str(old_value),
                'new_value': str(new_value),
                'timestamp': time.time()
            }
        )
    
    def log_security_event(self, event_type: str, severity: str, description: str,
                          user_id: str = None, ip_address: str = None,
                          details: Dict[str, Any] = None):
        """Log security event."""
        log_level = logging.CRITICAL if severity == 'critical' else logging.WARNING
        
        self.audit_logger.log(
            log_level,
            f"Security event: {description}",
            extra={
                'event_type': 'security',
                'security_event_type': event_type,
                'severity': severity,
                'description': description,
                'user_id': user_id,
                'ip_address': ip_address,
                'details': details or {},
                'timestamp': time.time()
            }
        )


class SecurityLogger:
    """Logger for security-related events."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.security_logger = logging.getLogger('security')
        self.security_logger.setLevel(logging.WARNING)
        self.security_logger.propagate = False
        
        # Always log security events to file
        if config.log_file:
            security_file = config.log_file.replace('.log', '_security.log')
            security_handler = logging.handlers.RotatingFileHandler(
                security_file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count
            )
            
            security_formatter = StructuredFormatter()
            security_handler.setFormatter(security_formatter)
            
            self.security_logger.addHandler(security_handler)
    
    def log_failed_authentication(self, user_id: str, ip_address: str, attempts: int):
        """Log failed authentication attempts."""
        self.security_logger.warning(
            f"Failed authentication attempt for user {user_id}",
            extra={
                'event': 'failed_authentication',
                'user_id': user_id,
                'ip_address': ip_address,
                'attempt_count': attempts,
                'severity': 'medium' if attempts < 5 else 'high'
            }
        )
    
    def log_suspicious_activity(self, description: str, user_id: str = None,
                               ip_address: str = None, details: Dict[str, Any] = None):
        """Log suspicious activity."""
        self.security_logger.warning(
            f"Suspicious activity detected: {description}",
            extra={
                'event': 'suspicious_activity',
                'description': description,
                'user_id': user_id,
                'ip_address': ip_address,
                'details': details or {},
                'severity': 'high'
            }
        )
    
    def log_rate_limit_exceeded(self, user_id: str, endpoint: str, ip_address: str):
        """Log rate limit violations."""
        self.security_logger.warning(
            f"Rate limit exceeded for user {user_id}",
            extra={
                'event': 'rate_limit_exceeded',
                'user_id': user_id,
                'endpoint': endpoint,
                'ip_address': ip_address,
                'severity': 'medium'
            }
        )


def setup_logging(config: LogConfig) -> Dict[str, Any]:
    """Setup comprehensive logging configuration."""
    
    # Create log directory if needed
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config.audit_log_file:
        audit_path = Path(config.audit_log_file)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatters
    if config.format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if config.output in ["console", "both"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        if config.enable_privacy_filtering:
            console_handler.addFilter(PrivacyFilter())
        
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.output in ["file", "both"] and config.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        
        if config.enable_privacy_filtering:
            file_handler.addFilter(PrivacyFilter())
        
        root_logger.addHandler(file_handler)
    
    # Setup specialized loggers
    loggers = {}
    
    if config.enable_audit_logging:
        audit_logger = AuditLogger(config)
        loggers['audit'] = audit_logger
    
    if config.enable_security_logging:
        security_logger = SecurityLogger(config)
        loggers['security'] = security_logger
    
    # Configure specific loggers for libraries
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Log startup message
    startup_logger = logging.getLogger('federated_dp_llm.startup')
    startup_logger.info(
        "Logging system initialized",
        extra={
            'config': asdict(config),
            'loggers_created': list(loggers.keys())
        }
    )
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger with privacy filtering enabled."""
    logger = logging.getLogger(name)
    
    # Ensure privacy filter is applied
    has_privacy_filter = any(
        isinstance(filter, PrivacyFilter) 
        for handler in logger.handlers 
        for filter in handler.filters
    )
    
    if not has_privacy_filter:
        for handler in logger.handlers:
            handler.addFilter(PrivacyFilter())
    
    return logger


def log_performance_metric(metric_name: str, value: float, labels: Dict[str, str] = None):
    """Log performance metric in structured format."""
    perf_logger = logging.getLogger('performance')
    perf_logger.info(
        f"Performance metric: {metric_name}",
        extra={
            'metric_name': metric_name,
            'metric_value': value,
            'labels': labels or {},
            'timestamp': time.time()
        }
    )


class LoggingContextManager:
    """Context manager for adding request context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)