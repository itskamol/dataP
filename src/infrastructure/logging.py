"""
Structured logging framework with correlation IDs and contextual information.
Implements requirement 2.1: Comprehensive logging with structured format.
"""

import logging
import logging.config
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record):
        record.correlation_id = getattr(threading.current_thread(), 'correlation_id', 'unknown')
        return True


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'unknown'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'correlation_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class LoggerManager:
    """Centralized logger management with correlation ID support."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'structured': {
                    '()': StructuredFormatter,
                },
                'simple': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
                }
            },
            'filters': {
                'correlation_id': {
                    '()': CorrelationIdFilter,
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'filters': ['correlation_id'],
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'structured',
                    'filters': ['correlation_id'],
                    'filename': 'logs/application.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf8'
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'structured',
                    'filters': ['correlation_id'],
                    'filename': 'logs/errors.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf8'
                }
            },
            'loggers': {
                'file_processing': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'matching_engine': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'web_interface': {
                    'level': 'INFO',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        }
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs('logs', exist_ok=True)
        
        logging.config.dictConfig(config)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for setting correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        old_correlation_id = getattr(threading.current_thread(), 'correlation_id', None)
        threading.current_thread().correlation_id = correlation_id
        
        try:
            yield correlation_id
        finally:
            if old_correlation_id is not None:
                threading.current_thread().correlation_id = old_correlation_id
            else:
                delattr(threading.current_thread(), 'correlation_id')


# Global logger manager instance
logger_manager = LoggerManager()

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger."""
    return logger_manager.get_logger(name)

def with_correlation_id(correlation_id: Optional[str] = None):
    """Decorator to add correlation ID to function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with logger_manager.correlation_context(correlation_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator