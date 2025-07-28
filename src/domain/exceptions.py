"""
Custom exception classes for the file processing system.
Provides structured error handling with contextual information.
"""

from typing import Optional, Dict, Any


class FileProcessingError(Exception):
    """Base exception for file processing errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class FileValidationError(FileProcessingError):
    """Raised when file validation fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 line_number: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        if file_path:
            context['file_path'] = file_path
        if line_number:
            context['line_number'] = line_number
        super().__init__(message, context)
        self.file_path = file_path
        self.line_number = line_number


class FileFormatError(FileProcessingError):
    """Raised when file format is invalid or unsupported."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 expected_format: Optional[str] = None, actual_format: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        context = context or {}
        if file_path:
            context['file_path'] = file_path
        if expected_format:
            context['expected_format'] = expected_format
        if actual_format:
            context['actual_format'] = actual_format
        super().__init__(message, context)
        self.file_path = file_path
        self.expected_format = expected_format
        self.actual_format = actual_format


class FileAccessError(FileProcessingError):
    """Raised when file cannot be accessed or read."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        if file_path:
            context['file_path'] = file_path
        if operation:
            context['operation'] = operation
        super().__init__(message, context)
        self.file_path = file_path
        self.operation = operation


class MatchingError(Exception):
    """Base exception for matching operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class AlgorithmError(MatchingError):
    """Raised when matching algorithm encounters an error."""
    
    def __init__(self, message: str, algorithm_name: Optional[str] = None,
                 record1: Optional[Dict[str, Any]] = None, record2: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None):
        context = context or {}
        if algorithm_name:
            context['algorithm'] = algorithm_name
        if record1:
            context['record1_sample'] = str(record1)[:100]  # Truncate for logging
        if record2:
            context['record2_sample'] = str(record2)[:100]  # Truncate for logging
        super().__init__(message, context)
        self.algorithm_name = algorithm_name
        self.record1 = record1
        self.record2 = record2


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.config_key = config_key
        self.config_value = config_value
        self.context = context or {}
        if config_key:
            self.context['config_key'] = config_key
        if config_value is not None:
            self.context['config_value'] = str(config_value)
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ValidationError(Exception):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 field_value: Optional[Any] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.field_name = field_name
        self.field_value = field_value
        self.context = context or {}
        if field_name:
            self.context['field_name'] = field_name
        if field_value is not None:
            self.context['field_value'] = str(field_value)
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ResourceError(Exception):
    """Raised when system resources are insufficient or unavailable."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 requested_amount: Optional[Any] = None, available_amount: Optional[Any] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.resource_type = resource_type
        self.requested_amount = requested_amount
        self.available_amount = available_amount
        self.context = context or {}
        if resource_type:
            self.context['resource_type'] = resource_type
        if requested_amount is not None:
            self.context['requested'] = str(requested_amount)
        if available_amount is not None:
            self.context['available'] = str(available_amount)
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class MemoryError(ResourceError):
    """Raised when memory resources are insufficient."""
    
    def __init__(self, message: str, requested_memory: Optional[int] = None,
                 available_memory: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "memory", requested_memory, available_memory, context)


class ProcessingTimeoutError(Exception):
    """Raised when processing operation times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.context = context or {}
        if timeout_seconds:
            self.context['timeout_seconds'] = timeout_seconds
        if operation:
            self.context['operation'] = operation
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class CancellationError(Exception):
    """Raised when operation is cancelled by user or system."""
    
    def __init__(self, message: str, operation_id: Optional[str] = None,
                 reason: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.operation_id = operation_id
        self.reason = reason
        self.context = context or {}
        if operation_id:
            self.context['operation_id'] = operation_id
        if reason:
            self.context['reason'] = reason
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class SecurityError(Exception):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_check: Optional[str] = None,
                 file_path: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.security_check = security_check
        self.file_path = file_path
        self.context = context or {}
        if security_check:
            self.context['security_check'] = security_check
        if file_path:
            self.context['file_path'] = file_path
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ProgressTrackingError(Exception):
    """Raised when progress tracking operations fail."""
    
    def __init__(self, message: str, operation_id: Optional[str] = None,
                 tracking_component: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.operation_id = operation_id
        self.tracking_component = tracking_component
        self.context = context or {}
        if operation_id:
            self.context['operation_id'] = operation_id
        if tracking_component:
            self.context['tracking_component'] = tracking_component
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class MetricsError(Exception):
    """Raised when metrics collection operations fail."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None,
                 metric_type: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.context = context or {}
        if metric_name:
            self.context['metric_name'] = metric_name
        if metric_type:
            self.context['metric_type'] = metric_type
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class HealthCheckError(Exception):
    """Raised when health check operations fail."""
    
    def __init__(self, message: str, check_name: Optional[str] = None,
                 check_type: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.check_name = check_name
        self.check_type = check_type
        self.context = context or {}
        if check_name:
            self.context['check_name'] = check_name
        if check_type:
            self.context['check_type'] = check_type
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class AlertingError(Exception):
    """Raised when alerting operations fail."""
    
    def __init__(self, message: str, alert_id: Optional[str] = None,
                 channel: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.alert_id = alert_id
        self.channel = channel
        self.context = context or {}
        if alert_id:
            self.context['alert_id'] = alert_id
        if channel:
            self.context['channel'] = channel
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ProcessingError(Exception):
    """Raised when general processing operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 stage: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.stage = stage
        self.context = context or {}
        if operation:
            self.context['operation'] = operation
        if stage:
            self.context['stage'] = stage
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message