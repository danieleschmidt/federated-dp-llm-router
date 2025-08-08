"""
Structured Logging and Error Handling System

Implements comprehensive logging, error tracking, and observability
for the federated DP-LLM router with security-aware log filtering.
"""

import asyncio
import json
import time
import traceback
import hashlib
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import logging.handlers
from pathlib import Path
import sys
from datetime import datetime, timezone
import threading
import queue


class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    PRIVACY = "PRIVACY"


class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    SECURITY = "security"
    PRIVACY = "privacy"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    FEDERATION = "federation"
    QUANTUM = "quantum"
    COMPLIANCE = "compliance"
    USER_ACTION = "user_action"
    ERROR = "error"


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    component: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    department: Optional[str] = None
    metadata: Dict[str, Any] = None
    error_info: Optional[Dict[str, Any]] = None
    privacy_sensitive: bool = False
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None
    

@dataclass
class ErrorEvent:
    """Detailed error event for tracking."""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    component: str
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    context: Dict[str, Any] = None
    resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_notes: Optional[str] = None
    severity: str = "medium"
    impact: str = "unknown"


class SensitiveDataFilter:
    """Filter to remove sensitive data from logs."""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'password["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',  # Password
            r'token["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',  # Token
            r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',  # API key
            r'secret["\']?\s*[:=]\s*["\']?([^"\'\\s]+)',  # Secret
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        self.replacement_text = "[REDACTED]"
    
    def filter_sensitive_data(self, text: str) -> str:
        """Remove sensitive data from text."""
        import re
        
        filtered_text = text
        for pattern in self.sensitive_patterns:
            filtered_text = re.sub(pattern, self.replacement_text, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data."""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            # Check if key contains sensitive information
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in ['password', 'token', 'secret', 'key', 'auth']):
                sanitized[key] = self.replacement_text
            elif isinstance(value, str):
                sanitized[key] = self.filter_sensitive_data(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_dict(item) if isinstance(item, dict) 
                    else self.filter_sensitive_data(str(item)) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized


class PerformanceTracker:
    """Track performance metrics for logging."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_id: str):
        """Start tracking an operation."""
        with self._lock:
            self.start_times[operation_id] = time.time()
    
    def end_operation(self, operation_id: str) -> Optional[float]:
        """End tracking and return duration."""
        with self._lock:
            start_time = self.start_times.pop(operation_id, None)
            if start_time:
                duration = time.time() - start_time
                
                # Store metric
                metric_name = operation_id.split('_')[0]  # Use first part as metric name
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                
                self.metrics[metric_name].append(duration)
                
                # Keep only recent metrics
                if len(self.metrics[metric_name]) > 1000:
                    self.metrics[metric_name] = self.metrics[metric_name][-1000:]
                
                return duration
        
        return None
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            values = self.metrics.get(metric_name, [])
            
            if not values:
                return {"count": 0}
            
            return {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0],
                "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 1 else values[0]
            }


class StructuredLogger:
    """Enhanced structured logger with security and privacy awareness."""
    
    def __init__(
        self,
        name: str = "federated_dp_llm",
        log_level: LogLevel = LogLevel.INFO,
        log_dir: str = "/var/log/federated_dp_llm",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10,
        enable_console: bool = True,
        enable_audit_log: bool = True,
        enable_security_log: bool = True
    ):
        self.name = name
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_audit_log = enable_audit_log
        self.enable_security_log = enable_security_log
        
        # Components
        self.sensitive_filter = SensitiveDataFilter()
        self.performance_tracker = PerformanceTracker()
        
        # Storage
        self.log_events: List[LogEvent] = []
        self.error_events: List[ErrorEvent] = []
        self.correlation_map: Dict[str, List[str]] = {}
        
        # Async processing
        self.log_queue = queue.Queue()
        self.error_counter = 0
        self._processing = False
        self._lock = threading.Lock()
        
        # Setup loggers
        self._setup_loggers()
        
        # Start background processing
        self._start_background_processing()
    
    def _setup_loggers(self):
        """Setup various loggers for different purposes."""
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main application logger
        self.app_logger = logging.getLogger(f"{self.name}.app")
        self.app_logger.setLevel(getattr(logging, self.log_level.value))
        
        # Security logger
        if self.enable_security_log:
            self.security_logger = logging.getLogger(f"{self.name}.security")
            self.security_logger.setLevel(logging.WARNING)
            
            security_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "security.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            security_formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            security_handler.setFormatter(security_formatter)
            self.security_logger.addHandler(security_handler)
        
        # Audit logger
        if self.enable_audit_log:
            self.audit_logger = logging.getLogger(f"{self.name}.audit")
            self.audit_logger.setLevel(logging.INFO)
            
            audit_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "audit.log",
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            audit_formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            audit_handler.setFormatter(audit_formatter)
            self.audit_logger.addHandler(audit_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger(f"{self.name}.performance")
        self.perf_logger.setLevel(logging.INFO)
        
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        self.perf_logger.addHandler(perf_handler)
        
        # Error logger
        self.error_logger = logging.getLogger(f"{self.name}.error")
        self.error_logger.setLevel(logging.ERROR)
        
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "error.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_formatter = logging.Formatter(
            '%(asctime)s - ERROR - %(levelname)s - %(message)s\n%(stack_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Console logger
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            
            # Add to all loggers
            self.app_logger.addHandler(console_handler)
            if hasattr(self, 'security_logger'):
                self.security_logger.addHandler(console_handler)
            if hasattr(self, 'audit_logger'):
                self.audit_logger.addHandler(console_handler)
    
    def _start_background_processing(self):
        """Start background log processing."""
        if not self._processing:
            self._processing = True
            threading.Thread(target=self._process_logs, daemon=True).start()
    
    def _process_logs(self):
        """Background log processing thread."""
        while self._processing:
            try:
                # Process queued log events
                while not self.log_queue.empty():
                    try:
                        log_event = self.log_queue.get_nowait()
                        self._write_log_event(log_event)
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Error processing log event: {e}")
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                print(f"Error in log processing thread: {e}")
                time.sleep(1)
    
    def _write_log_event(self, event: LogEvent):
        """Write log event to appropriate logger."""
        # Filter sensitive data
        filtered_message = self.sensitive_filter.filter_sensitive_data(event.message)
        filtered_metadata = self.sensitive_filter.sanitize_dict(event.metadata or {})
        
        # Create log message
        log_data = {
            "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
            "level": event.level.value,
            "category": event.category.value,
            "component": event.component,
            "message": filtered_message,
            "request_id": event.request_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "department": event.department,
            "correlation_id": event.correlation_id,
            "metadata": filtered_metadata
        }
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        log_message = json.dumps(log_data, separators=(',', ':'))
        
        # Route to appropriate logger
        if event.category == LogCategory.SECURITY or event.level == LogLevel.SECURITY:
            if hasattr(self, 'security_logger'):
                self.security_logger.warning(log_message)
        elif event.category == LogCategory.AUDIT or event.level == LogLevel.AUDIT:
            if hasattr(self, 'audit_logger'):
                self.audit_logger.info(log_message)
        elif event.category == LogCategory.PERFORMANCE:
            self.perf_logger.info(log_message)
        elif event.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.error_logger.error(log_message)
        else:
            # Default to app logger
            level_map = {
                LogLevel.TRACE: logging.DEBUG,
                LogLevel.DEBUG: logging.DEBUG,
                LogLevel.INFO: logging.INFO,
                LogLevel.WARNING: logging.WARNING,
                LogLevel.ERROR: logging.ERROR,
                LogLevel.CRITICAL: logging.CRITICAL,
                LogLevel.PRIVACY: logging.WARNING
            }
            self.app_logger.log(level_map.get(event.level, logging.INFO), log_message)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        component: str = "unknown",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        department: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_info: Optional[Exception] = None,
        privacy_sensitive: bool = False,
        correlation_id: Optional[str] = None
    ):
        """Log an event."""
        # Create log event
        event = LogEvent(
            timestamp=time.time(),
            level=level,
            category=category,
            message=message,
            component=component,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            department=department,
            metadata=metadata,
            privacy_sensitive=privacy_sensitive,
            correlation_id=correlation_id
        )
        
        # Add error information if provided
        if error_info:
            event.error_info = {
                "type": type(error_info).__name__,
                "message": str(error_info),
                "args": error_info.args
            }
            event.stack_trace = traceback.format_exc()
        
        # Store in memory
        with self._lock:
            self.log_events.append(event)
            
            # Keep only recent events
            if len(self.log_events) > 10000:
                self.log_events = self.log_events[-10000:]
            
            # Add to correlation map
            if correlation_id:
                if correlation_id not in self.correlation_map:
                    self.correlation_map[correlation_id] = []
                self.correlation_map[correlation_id].append(event.request_id or f"event_{int(event.timestamp)}")
        
        # Queue for background processing
        try:
            self.log_queue.put_nowait(event)
        except queue.Full:
            # If queue is full, write directly
            self._write_log_event(event)
    
    def log_error(
        self,
        error: Exception,
        component: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "medium",
        impact: str = "unknown"
    ) -> str:
        """Log and track an error event."""
        self.error_counter += 1
        error_id = f"error_{self.error_counter}_{int(time.time())}"
        
        # Create error event
        error_event = ErrorEvent(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            component=component,
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id,
            context=context,
            severity=severity,
            impact=impact
        )
        
        # Store error event
        with self._lock:
            self.error_events.append(error_event)
            
            # Keep only recent errors
            if len(self.error_events) > 1000:
                self.error_events = self.error_events[-1000:]
        
        # Log the error
        self.log(
            level=LogLevel.ERROR,
            message=f"Error {error_id}: {str(error)}",
            category=LogCategory.ERROR,
            component=component,
            user_id=user_id,
            request_id=request_id,
            metadata={
                "error_id": error_id,
                "error_type": type(error).__name__,
                "severity": severity,
                "impact": impact,
                "context": context
            },
            error_info=error
        )
        
        return error_id
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        component: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics."""
        self.log(
            level=LogLevel.INFO,
            message=f"Operation '{operation}' completed in {duration:.3f}s",
            category=LogCategory.PERFORMANCE,
            component=component,
            metadata={
                "operation": operation,
                "duration_ms": duration * 1000,
                "success": success,
                **(metadata or {})
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        message: str,
        component: str,
        user_id: Optional[str] = None,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event."""
        self.log(
            level=LogLevel.SECURITY,
            message=f"SECURITY: {event_type} - {message}",
            category=LogCategory.SECURITY,
            component=component,
            user_id=user_id,
            metadata={
                "event_type": event_type,
                "severity": severity,
                **(metadata or {})
            }
        )
    
    def log_privacy_event(
        self,
        event_type: str,
        message: str,
        component: str,
        user_id: Optional[str] = None,
        epsilon_spent: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log privacy-related event."""
        privacy_metadata = {"event_type": event_type}
        if epsilon_spent is not None:
            privacy_metadata["epsilon_spent"] = epsilon_spent
        if metadata:
            privacy_metadata.update(metadata)
        
        self.log(
            level=LogLevel.PRIVACY,
            message=f"PRIVACY: {event_type} - {message}",
            category=LogCategory.PRIVACY,
            component=component,
            user_id=user_id,
            metadata=privacy_metadata,
            privacy_sensitive=True
        )
    
    def log_audit_event(
        self,
        action: str,
        resource: str,
        component: str,
        user_id: Optional[str] = None,
        department: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log audit event."""
        self.log(
            level=LogLevel.AUDIT,
            message=f"AUDIT: {action} on {resource} - {'SUCCESS' if success else 'FAILED'}",
            category=LogCategory.AUDIT,
            component=component,
            user_id=user_id,
            department=department,
            metadata={
                "action": action,
                "resource": resource,
                "success": success,
                **(metadata or {})
            }
        )
    
    def start_operation_tracking(self, operation_id: str):
        """Start tracking an operation's performance."""
        self.performance_tracker.start_operation(operation_id)
    
    def end_operation_tracking(
        self,
        operation_id: str,
        component: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """End operation tracking and log performance."""
        duration = self.performance_tracker.end_operation(operation_id)
        if duration is not None:
            self.log_performance(operation_id, duration, component, success, metadata)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        with self._lock:
            errors = self.error_events.copy()
        
        if not errors:
            return {"total_errors": 0}
        
        # Count by severity
        severity_counts = {}
        for error in errors:
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        
        # Count by component
        component_counts = {}
        for error in errors:
            component_counts[error.component] = component_counts.get(error.component, 0) + 1
        
        # Recent errors (last hour)
        recent_cutoff = time.time() - 3600
        recent_errors = [e for e in errors if e.timestamp > recent_cutoff]
        
        return {
            "total_errors": len(errors),
            "recent_errors_1h": len(recent_errors),
            "resolved_errors": len([e for e in errors if e.resolved]),
            "unresolved_errors": len([e for e in errors if not e.resolved]),
            "by_severity": severity_counts,
            "by_component": component_counts,
            "recent_error_rate": len(recent_errors) / 60.0,  # per minute
            "most_recent": {
                "error_id": errors[-1].error_id,
                "timestamp": errors[-1].timestamp,
                "message": errors[-1].error_message,
                "component": errors[-1].component
            } if errors else None
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        return {
            metric: self.performance_tracker.get_stats(metric)
            for metric in self.performance_tracker.metrics.keys()
        }
    
    def search_logs(
        self,
        query: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        component: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search log events."""
        with self._lock:
            events = self.log_events.copy()
        
        # Apply filters
        filtered_events = []
        for event in events:
            # Time filter
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            
            # Level filter
            if level and event.level != level:
                continue
            
            # Category filter
            if category and event.category != category:
                continue
            
            # Component filter
            if component and event.component != component:
                continue
            
            # User filter
            if user_id and event.user_id != user_id:
                continue
            
            # Text search
            if query.lower() in event.message.lower():
                filtered_events.append(event)
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        limited_events = filtered_events[:limit]
        
        # Convert to dict format
        return [asdict(event) for event in limited_events]
    
    def get_correlation_chain(self, correlation_id: str) -> List[str]:
        """Get all events related to a correlation ID."""
        with self._lock:
            return self.correlation_map.get(correlation_id, [])
    
    def shutdown(self):
        """Shutdown the logger."""
        self._processing = False
        
        # Process remaining queued events
        while not self.log_queue.empty():
            try:
                event = self.log_queue.get_nowait()
                self._write_log_event(event)
            except queue.Empty:
                break
        
        # Close handlers
        for logger in [self.app_logger, getattr(self, 'security_logger', None), 
                      getattr(self, 'audit_logger', None), self.perf_logger, self.error_logger]:
            if logger:
                for handler in logger.handlers:
                    handler.close()


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger


def setup_logger(
    log_level: LogLevel = LogLevel.INFO,
    log_dir: str = "/var/log/federated_dp_llm",
    enable_console: bool = True
) -> StructuredLogger:
    """Setup global logger with specific configuration."""
    global _global_logger
    _global_logger = StructuredLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console
    )
    return _global_logger


# Convenience functions
def log_info(message: str, component: str = "unknown", **kwargs):
    """Log info message."""
    get_logger().log(LogLevel.INFO, message, component=component, **kwargs)


def log_warning(message: str, component: str = "unknown", **kwargs):
    """Log warning message."""
    get_logger().log(LogLevel.WARNING, message, component=component, **kwargs)


def log_error(error: Exception, component: str = "unknown", **kwargs) -> str:
    """Log error and return error ID."""
    return get_logger().log_error(error, component, **kwargs)


def log_security(event_type: str, message: str, component: str = "unknown", **kwargs):
    """Log security event."""
    get_logger().log_security_event(event_type, message, component, **kwargs)


def log_audit(action: str, resource: str, component: str = "unknown", **kwargs):
    """Log audit event."""
    get_logger().log_audit_event(action, resource, component, **kwargs)


def log_performance(operation: str, duration: float, component: str = "unknown", **kwargs):
    """Log performance metric."""
    get_logger().log_performance(operation, duration, component, **kwargs)