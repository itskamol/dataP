"""
Monitoring integration module that ties together all observability components.
Provides a unified interface for metrics, tracing, and monitoring.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

from .metrics import metrics_collector, track_time
from .tracing import tracing_manager, trace_function, trace_analyzer
from .monitoring_endpoints import register_monitoring_endpoints

logger = logging.getLogger(__name__)


class MonitoringIntegration:
    """Unified monitoring integration for the file processing system."""
    
    def __init__(self):
        self.metrics_collector = metrics_collector
        self.tracing_manager = tracing_manager
        self.trace_analyzer = trace_analyzer
        self._background_tasks = []
        self._shutdown_event = threading.Event()
    
    def initialize(self, app=None):
        """Initialize monitoring integration."""
        try:
            # Register monitoring endpoints if Flask app is provided
            if app:
                register_monitoring_endpoints(app)
                logger.info("Monitoring endpoints registered with Flask app")
            
            # Start background monitoring tasks
            self._start_background_tasks()
            
            logger.info("Monitoring integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring integration: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Start cache hit ratio calculation task
        cache_task = threading.Thread(
            target=self._cache_metrics_updater,
            daemon=True,
            name="cache-metrics-updater"
        )
        cache_task.start()
        self._background_tasks.append(cache_task)
        
        # Start trace cleanup task
        cleanup_task = threading.Thread(
            target=self._trace_cleanup_task,
            daemon=True,
            name="trace-cleanup"
        )
        cleanup_task.start()
        self._background_tasks.append(cleanup_task)
        
        # Start performance analysis task
        analysis_task = threading.Thread(
            target=self._performance_analysis_task,
            daemon=True,
            name="performance-analysis"
        )
        analysis_task.start()
        self._background_tasks.append(analysis_task)
        
        logger.info(f"Started {len(self._background_tasks)} background monitoring tasks")
    
    def _cache_metrics_updater(self):
        """Background task to update cache metrics."""
        while not self._shutdown_event.is_set():
            try:
                if self.metrics_collector:
                    self.metrics_collector.calculate_cache_hit_ratio()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in cache metrics updater: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _trace_cleanup_task(self):
        """Background task to clean up old traces."""
        while not self._shutdown_event.is_set():
            try:
                if self.tracing_manager:
                    # Clean up traces older than 1 hour
                    self.tracing_manager.cleanup_old_traces(max_age_seconds=3600)
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in trace cleanup task: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _performance_analysis_task(self):
        """Background task to perform performance analysis."""
        while not self._shutdown_event.is_set():
            try:
                if self.trace_analyzer:
                    # Analyze performance bottlenecks
                    analysis = self.trace_analyzer.analyze_performance_bottlenecks()
                    
                    # Log critical bottlenecks
                    critical_bottlenecks = [
                        b for b in analysis.get('bottlenecks', [])
                        if b.get('severity') == 'high'
                    ]
                    
                    if critical_bottlenecks:
                        logger.warning(f"Critical performance bottlenecks detected: {critical_bottlenecks}")
                    
                    # Update user satisfaction score based on performance
                    self._update_user_satisfaction_score(analysis)
                
                time.sleep(120)  # Run every 2 minutes
            except Exception as e:
                logger.error(f"Error in performance analysis task: {e}")
                time.sleep(300)  # Wait longer on error
    
    def _update_user_satisfaction_score(self, analysis: Dict[str, Any]):
        """Update user satisfaction score based on performance analysis."""
        try:
            if not self.metrics_collector:
                return
            
            # Calculate satisfaction score based on various factors
            base_score = 1.0
            
            # Reduce score for bottlenecks
            bottlenecks = analysis.get('bottlenecks', [])
            for bottleneck in bottlenecks:
                if bottleneck.get('severity') == 'high':
                    base_score -= 0.2
                elif bottleneck.get('severity') == 'medium':
                    base_score -= 0.1
            
            # Ensure score is between 0 and 1
            satisfaction_score = max(0.0, min(1.0, base_score))
            
            self.metrics_collector.update_user_satisfaction_score(satisfaction_score)
            
        except Exception as e:
            logger.error(f"Error updating user satisfaction score: {e}")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, operation_type: str = "general", 
                         tags: Optional[Dict[str, Any]] = None):
        """Context manager to monitor an operation with both metrics and tracing."""
        start_time = time.time()
        
        # Start tracing
        with self.tracing_manager.start_span(operation_name, tags) as span:
            span.set_tag("operation_type", operation_type)
            
            try:
                yield span
                
                # Record success metrics
                duration = time.time() - start_time
                if operation_type == "file_processing":
                    file_type = tags.get('file_type', 'unknown') if tags else 'unknown'
                    size_bytes = tags.get('size_bytes', 0) if tags else 0
                    self.metrics_collector.record_file_processed(file_type, 'success', duration, size_bytes)
                elif operation_type == "matching":
                    algorithm = tags.get('algorithm', 'unknown') if tags else 'unknown'
                    accuracy = tags.get('accuracy') if tags else None
                    self.metrics_collector.record_matching_operation(algorithm, 'success', duration, accuracy)
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                if operation_type == "file_processing":
                    file_type = tags.get('file_type', 'unknown') if tags else 'unknown'
                    size_bytes = tags.get('size_bytes', 0) if tags else 0
                    self.metrics_collector.record_file_processed(file_type, 'error', duration, size_bytes)
                elif operation_type == "matching":
                    algorithm = tags.get('algorithm', 'unknown') if tags else 'unknown'
                    self.metrics_collector.record_matching_operation(algorithm, 'error', duration)
                
                # Update error rate metrics
                error_type = type(e).__name__
                self.metrics_collector.update_error_rate_by_type(error_type, 0.1)  # Placeholder rate
                
                raise
    
    def record_business_metric(self, metric_type: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record business-specific metrics."""
        try:
            if not self.metrics_collector:
                return
            
            if metric_type == "throughput":
                operation_type = labels.get('operation_type', 'general') if labels else 'general'
                self.metrics_collector.update_processing_throughput(operation_type, value)
            elif metric_type == "resource_utilization":
                resource_type = labels.get('resource_type', 'general') if labels else 'general'
                self.metrics_collector.update_resource_utilization(resource_type, value)
            elif metric_type == "user_satisfaction":
                self.metrics_collector.update_user_satisfaction_score(value)
            elif metric_type == "error_rate":
                error_type = labels.get('error_type', 'general') if labels else 'general'
                self.metrics_collector.update_error_rate_by_type(error_type, value)
            else:
                logger.warning(f"Unknown business metric type: {metric_type}")
                
        except Exception as e:
            logger.error(f"Error recording business metric {metric_type}: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a comprehensive monitoring summary."""
        try:
            summary = {
                'timestamp': time.time(),
                'metrics_enabled': self.metrics_collector is not None,
                'tracing_enabled': self.tracing_manager is not None,
                'background_tasks': len(self._background_tasks)
            }
            
            # Add trace summary if available
            if self.tracing_manager:
                from .tracing import get_trace_summary
                summary['traces'] = get_trace_summary()
            
            # Add performance analysis if available
            if self.trace_analyzer:
                analysis = self.trace_analyzer.analyze_performance_bottlenecks()
                summary['performance'] = {
                    'bottlenecks_count': len(analysis.get('bottlenecks', [])),
                    'critical_bottlenecks': len([
                        b for b in analysis.get('bottlenecks', [])
                        if b.get('severity') == 'high'
                    ]),
                    'recommendations_count': len(analysis.get('recommendations', []))
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting monitoring summary: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def shutdown(self):
        """Shutdown monitoring integration and cleanup resources."""
        try:
            logger.info("Shutting down monitoring integration...")
            
            # Signal background tasks to stop
            self._shutdown_event.set()
            
            # Wait for background tasks to finish (with timeout)
            for task in self._background_tasks:
                if task.is_alive():
                    task.join(timeout=5.0)
                    if task.is_alive():
                        logger.warning(f"Background task {task.name} did not shutdown gracefully")
            
            # Clean up traces
            if self.tracing_manager:
                self.tracing_manager.cleanup_old_traces(max_age_seconds=0)  # Clean all traces
            
            logger.info("Monitoring integration shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during monitoring integration shutdown: {e}")


# Global monitoring integration instance
monitoring_integration = MonitoringIntegration()


def initialize_monitoring(app=None) -> bool:
    """Initialize monitoring integration."""
    return monitoring_integration.initialize(app)


def monitor_operation(operation_name: str, operation_type: str = "general", 
                     tags: Optional[Dict[str, Any]] = None):
    """Decorator and context manager for monitoring operations."""
    return monitoring_integration.monitor_operation(operation_name, operation_type, tags)


def record_business_metric(metric_type: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record business-specific metrics."""
    monitoring_integration.record_business_metric(metric_type, value, labels)


def get_monitoring_summary() -> Dict[str, Any]:
    """Get comprehensive monitoring summary."""
    return monitoring_integration.get_monitoring_summary()


def shutdown_monitoring():
    """Shutdown monitoring integration."""
    monitoring_integration.shutdown()


# Decorator for easy function monitoring
def monitor_function(operation_name: str = None, operation_type: str = "general", 
                    tags: Optional[Dict[str, Any]] = None):
    """Decorator to monitor function execution."""
    def decorator(func: Callable) -> Callable:
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with monitor_operation(op_name, operation_type, tags) as span:
                span.set_tag("function.name", func.__name__)
                span.set_tag("function.module", func.__module__)
                
                result = func(*args, **kwargs)
                
                # Add result information if it's a simple type
                if isinstance(result, (str, int, float, bool)):
                    span.set_tag("function.result_type", type(result).__name__)
                
                return result
        
        return wrapper
    return decorator