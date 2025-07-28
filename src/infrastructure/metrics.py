"""
Comprehensive metrics collection system for file processing application.
Provides Prometheus metrics for monitoring application performance and business metrics.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import psutil
import threading
import logging

logger = logging.getLogger(__name__)

# Flask imports for request tracking
try:
    from flask import request
except ImportError:
    # Fallback if Flask is not available
    request = None


class MetricsCollector:
    """Central metrics collector for the file processing system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector with optional custom registry."""
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._setup_system_metrics()
        self._start_background_collection()
    
    def _setup_metrics(self):
        """Set up application-specific metrics."""
        # HTTP request metrics
        self.http_requests_total = Counter(
            'flask_http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'flask_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_request_exceptions = Counter(
            'flask_http_request_exceptions_total',
            'Total number of HTTP request exceptions',
            ['method', 'endpoint', 'exception'],
            registry=self.registry
        )
        
        # File processing metrics
        self.files_processed_total = Counter(
            'file_processing_files_processed_total',
            'Total number of files processed',
            ['file_type', 'status'],
            registry=self.registry
        )
        
        self.file_processing_duration = Histogram(
            'file_processing_duration_seconds',
            'File processing duration in seconds',
            ['file_type', 'size_category'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, float('inf')]
        )
        
        self.file_processing_queue_size = Gauge(
            'file_processing_queue_size',
            'Current number of files in processing queue',
            registry=self.registry
        )
        
        # Matching algorithm metrics
        self.matching_operations_total = Counter(
            'file_processing_matching_operations_total',
            'Total number of matching operations',
            ['algorithm', 'status'],
            registry=self.registry
        )
        
        self.matching_duration = Histogram(
            'file_processing_matching_duration_seconds',
            'Matching operation duration in seconds',
            ['algorithm'],
            registry=self.registry,
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, float('inf')]
        )
        
        self.matching_accuracy = Gauge(
            'file_processing_matching_accuracy',
            'Current matching accuracy score',
            ['algorithm'],
            registry=self.registry
        )
        
        self.records_matched_total = Counter(
            'file_processing_records_matched_total',
            'Total number of records matched',
            ['algorithm', 'confidence_level'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'file_processing_cache_operations_total',
            'Total number of cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'file_processing_cache_hit_ratio',
            'Cache hit ratio',
            registry=self.registry
        )
        
        # Business metrics
        self.active_sessions = Gauge(
            'file_processing_active_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
        
        self.data_volume_processed = Counter(
            'file_processing_data_volume_bytes_total',
            'Total volume of data processed in bytes',
            ['operation'],
            registry=self.registry
        )
        
        # Enhanced business metrics
        self.processing_throughput = Gauge(
            'file_processing_throughput_records_per_second',
            'Current processing throughput in records per second',
            ['operation_type'],
            registry=self.registry
        )
        
        self.error_rate_by_type = Gauge(
            'file_processing_error_rate_by_type',
            'Error rate by error type',
            ['error_type'],
            registry=self.registry
        )
        
        self.user_satisfaction_score = Gauge(
            'file_processing_user_satisfaction_score',
            'User satisfaction score based on completion time and accuracy',
            registry=self.registry
        )
        
        self.resource_utilization = Gauge(
            'file_processing_resource_utilization_percent',
            'Resource utilization percentage',
            ['resource_type'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'file_processing_app_info',
            'Application information',
            registry=self.registry
        )
        
        self.app_info.info({
            'version': '1.0.0',
            'python_version': '3.11',
            'build_date': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def _setup_system_metrics(self):
        """Set up system-level metrics."""
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['mountpoint', 'type'],
            registry=self.registry
        )
        
        self.process_memory_usage = Gauge(
            'process_memory_usage_bytes',
            'Process memory usage in bytes',
            registry=self.registry
        )
        
        self.process_cpu_usage = Gauge(
            'process_cpu_usage_percent',
            'Process CPU usage percentage',
            registry=self.registry
        )
    
    def _start_background_collection(self):
        """Start background thread for system metrics collection."""
        def collect_system_metrics():
            while True:
                try:
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.system_memory_usage.labels(type='total').set(memory.total)
                    self.system_memory_usage.labels(type='available').set(memory.available)
                    self.system_memory_usage.labels(type='used').set(memory.used)
                    
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_cpu_usage.set(cpu_percent)
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.system_disk_usage.labels(mountpoint='/', type='total').set(disk.total)
                    self.system_disk_usage.labels(mountpoint='/', type='used').set(disk.used)
                    self.system_disk_usage.labels(mountpoint='/', type='free').set(disk.free)
                    
                    # Process metrics
                    process = psutil.Process()
                    self.process_memory_usage.set(process.memory_info().rss)
                    self.process_cpu_usage.set(process.cpu_percent())
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_http_exception(self, method: str, endpoint: str, exception: str):
        """Record HTTP request exception."""
        self.http_request_exceptions.labels(
            method=method,
            endpoint=endpoint,
            exception=exception
        ).inc()
    
    def record_file_processed(self, file_type: str, status: str, duration: float, size_bytes: int):
        """Record file processing metrics."""
        self.files_processed_total.labels(
            file_type=file_type,
            status=status
        ).inc()
        
        # Categorize file size
        if size_bytes < 1024 * 1024:  # < 1MB
            size_category = 'small'
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            size_category = 'medium'
        elif size_bytes < 100 * 1024 * 1024:  # < 100MB
            size_category = 'large'
        else:
            size_category = 'very_large'
        
        self.file_processing_duration.labels(
            file_type=file_type,
            size_category=size_category
        ).observe(duration)
        
        self.data_volume_processed.labels(operation='processing').inc(size_bytes)
    
    def record_matching_operation(self, algorithm: str, status: str, duration: float, 
                                accuracy: Optional[float] = None):
        """Record matching operation metrics."""
        self.matching_operations_total.labels(
            algorithm=algorithm,
            status=status
        ).inc()
        
        self.matching_duration.labels(algorithm=algorithm).observe(duration)
        
        if accuracy is not None:
            self.matching_accuracy.labels(algorithm=algorithm).set(accuracy)
    
    def record_records_matched(self, algorithm: str, confidence_score: float):
        """Record matched records with confidence level."""
        if confidence_score >= 0.9:
            confidence_level = 'high'
        elif confidence_score >= 0.7:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        self.records_matched_total.labels(
            algorithm=algorithm,
            confidence_level=confidence_level
        ).inc()
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics."""
        result = 'hit' if hit else 'miss'
        self.cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def update_queue_size(self, size: int):
        """Update processing queue size."""
        self.file_processing_queue_size.set(size)
    
    def update_active_sessions(self, count: int):
        """Update active sessions count."""
        self.active_sessions.set(count)
    
    def update_processing_throughput(self, operation_type: str, records_per_second: float):
        """Update processing throughput metrics."""
        self.processing_throughput.labels(operation_type=operation_type).set(records_per_second)
    
    def update_error_rate_by_type(self, error_type: str, error_rate: float):
        """Update error rate by error type."""
        self.error_rate_by_type.labels(error_type=error_type).set(error_rate)
    
    def update_user_satisfaction_score(self, score: float):
        """Update user satisfaction score (0.0 to 1.0)."""
        self.user_satisfaction_score.set(score)
    
    def update_resource_utilization(self, resource_type: str, utilization_percent: float):
        """Update resource utilization percentage."""
        self.resource_utilization.labels(resource_type=resource_type).set(utilization_percent)
    
    def calculate_cache_hit_ratio(self):
        """Calculate and update cache hit ratio."""
        try:
            # Get current cache metrics
            cache_hits = 0
            cache_misses = 0
            
            # This would typically be calculated from actual cache statistics
            # For now, we'll use a placeholder calculation
            total_operations = cache_hits + cache_misses
            if total_operations > 0:
                hit_ratio = cache_hits / total_operations
                self.cache_hit_ratio.set(hit_ratio)
        except Exception as e:
            logger.error(f"Error calculating cache hit ratio: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track execution time of functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing based on metric name
                if metric_name == 'file_processing':
                    file_type = labels.get('file_type', 'unknown') if labels else 'unknown'
                    size_bytes = labels.get('size_bytes', 0) if labels else 0
                    metrics_collector.record_file_processed(file_type, 'success', duration, size_bytes)
                elif metric_name == 'matching':
                    algorithm = labels.get('algorithm', 'unknown') if labels else 'unknown'
                    accuracy = labels.get('accuracy') if labels else None
                    metrics_collector.record_matching_operation(algorithm, 'success', duration, accuracy)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if metric_name == 'file_processing':
                    file_type = labels.get('file_type', 'unknown') if labels else 'unknown'
                    size_bytes = labels.get('size_bytes', 0) if labels else 0
                    metrics_collector.record_file_processed(file_type, 'error', duration, size_bytes)
                elif metric_name == 'matching':
                    algorithm = labels.get('algorithm', 'unknown') if labels else 'unknown'
                    metrics_collector.record_matching_operation(algorithm, 'error', duration)
                
                raise
        return wrapper
    return decorator


def track_http_requests(app):
    """Flask middleware to track HTTP requests."""
    @app.before_request
    def before_request():
        app.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(app, 'start_time'):
            duration = time.time() - app.start_time
            metrics_collector.record_http_request(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status_code=response.status_code,
                duration=duration
            )
        return response
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        metrics_collector.record_http_exception(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            exception=type(e).__name__
        )
        raise