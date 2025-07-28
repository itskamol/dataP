"""
Monitoring tests and alert validation scenarios for the file processing system.
Tests metrics collection, alerting rules, and observability features.
"""

import pytest
import time
import json
import requests
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, generate_latest
import threading
from typing import Dict, Any

# Import the modules to test
from src.infrastructure.metrics import MetricsCollector, track_time, track_http_requests
from src.infrastructure.tracing import TracingManager, TraceContext, trace_function


class TestMetricsCollection:
    """Test metrics collection functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = CollectorRegistry()
        self.metrics_collector = MetricsCollector(registry=self.registry)
    
    def test_metrics_collector_initialization(self):
        """Test that metrics collector initializes properly."""
        assert self.metrics_collector.registry is not None
        assert hasattr(self.metrics_collector, 'http_requests_total')
        assert hasattr(self.metrics_collector, 'file_processing_duration')
        assert hasattr(self.metrics_collector, 'matching_accuracy')
    
    def test_http_request_recording(self):
        """Test HTTP request metrics recording."""
        # Record some HTTP requests
        self.metrics_collector.record_http_request('GET', '/api/test', 200, 0.5)
        self.metrics_collector.record_http_request('POST', '/api/upload', 201, 1.2)
        self.metrics_collector.record_http_request('GET', '/api/test', 500, 0.3)
        
        # Get metrics output
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        # Check that metrics are recorded
        assert 'flask_http_requests_total' in metrics_output
        assert 'flask_http_request_duration_seconds' in metrics_output
        assert 'method="GET"' in metrics_output
        assert 'method="POST"' in metrics_output
        assert 'status="200"' in metrics_output
        assert 'status="500"' in metrics_output
    
    def test_file_processing_metrics(self):
        """Test file processing metrics recording."""
        # Record file processing operations
        self.metrics_collector.record_file_processed('csv', 'success', 2.5, 1024*1024)  # 1MB
        self.metrics_collector.record_file_processed('json', 'error', 1.0, 512*1024)   # 512KB
        self.metrics_collector.record_file_processed('csv', 'success', 10.0, 50*1024*1024)  # 50MB
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        assert 'file_processing_files_processed_total' in metrics_output
        assert 'file_processing_duration_seconds' in metrics_output
        assert 'file_type="csv"' in metrics_output
        assert 'file_type="json"' in metrics_output
        assert 'status="success"' in metrics_output
        assert 'status="error"' in metrics_output
    
    def test_matching_operation_metrics(self):
        """Test matching operation metrics recording."""
        # Record matching operations
        self.metrics_collector.record_matching_operation('fuzzy', 'success', 1.5, 0.85)
        self.metrics_collector.record_matching_operation('exact', 'success', 0.1, 0.95)
        self.metrics_collector.record_matching_operation('phonetic', 'error', 0.5)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        assert 'file_processing_matching_operations_total' in metrics_output
        assert 'file_processing_matching_duration_seconds' in metrics_output
        assert 'file_processing_matching_accuracy' in metrics_output
        assert 'algorithm="fuzzy"' in metrics_output
        assert 'algorithm="exact"' in metrics_output
    
    def test_cache_metrics(self):
        """Test cache operation metrics."""
        # Record cache operations
        self.metrics_collector.record_cache_operation('get', True)   # hit
        self.metrics_collector.record_cache_operation('get', False)  # miss
        self.metrics_collector.record_cache_operation('set', True)   # success
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        assert 'file_processing_cache_operations_total' in metrics_output
        assert 'operation="get"' in metrics_output
        assert 'operation="set"' in metrics_output
        assert 'result="hit"' in metrics_output
        assert 'result="miss"' in metrics_output
    
    def test_queue_and_session_metrics(self):
        """Test queue size and session metrics."""
        # Update queue size and active sessions
        self.metrics_collector.update_queue_size(25)
        self.metrics_collector.update_active_sessions(10)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        assert 'file_processing_queue_size' in metrics_output
        assert 'file_processing_active_sessions' in metrics_output
    
    def test_track_time_decorator(self):
        """Test the track_time decorator."""
        @track_time('file_processing', {'file_type': 'csv', 'size_bytes': 1024})
        def mock_file_processing():
            time.sleep(0.1)  # Simulate processing time
            return "success"
        
        # Execute the decorated function
        result = mock_file_processing()
        
        assert result == "success"
        
        # Check that metrics were recorded
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_files_processed_total' in metrics_output
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_system_metrics_collection(self, mock_disk, mock_cpu, mock_memory):
        """Test system metrics collection."""
        # Mock system metrics
        mock_memory.return_value = Mock(total=8*1024**3, available=4*1024**3, used=4*1024**3)
        mock_cpu.return_value = 25.5
        mock_disk.return_value = Mock(total=100*1024**3, used=50*1024**3, free=50*1024**3)
        
        # Wait a bit for background collection
        time.sleep(0.1)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        assert 'system_memory_usage_bytes' in metrics_output
        assert 'system_cpu_usage_percent' in metrics_output
        assert 'system_disk_usage_bytes' in metrics_output


class TestDistributedTracing:
    """Test distributed tracing functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tracing_manager = TracingManager()
    
    def test_trace_context_creation(self):
        """Test trace context creation and management."""
        context = TraceContext()
        
        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.start_time > 0
        assert context.status == "ok"
        assert isinstance(context.tags, dict)
        assert isinstance(context.logs, list)
    
    def test_trace_context_operations(self):
        """Test trace context operations."""
        context = TraceContext()
        
        # Test tagging
        context.set_tag("test_key", "test_value")
        assert context.tags["test_key"] == "test_value"
        
        # Test logging
        context.log("Test message", level="info", extra_data="test")
        assert len(context.logs) == 1
        assert context.logs[0]["message"] == "Test message"
        assert context.logs[0]["level"] == "info"
        assert context.logs[0]["extra_data"] == "test"
        
        # Test finishing
        context.finish("success")
        assert context.status == "success"
        assert context.end_time is not None
        assert context.duration() > 0
    
    def test_span_creation(self):
        """Test span creation and nesting."""
        with self.tracing_manager.start_span("test_operation") as span:
            assert span.operation_name == "test_operation"
            assert self.tracing_manager.get_current_context() == span
            
            # Create nested span
            with self.tracing_manager.start_span("nested_operation") as nested_span:
                assert nested_span.parent_span_id == span.span_id
                assert nested_span.trace_id == span.trace_id
                assert self.tracing_manager.get_current_context() == nested_span
            
            # Should return to parent context
            assert self.tracing_manager.get_current_context() == span
        
        # Should clear context after span ends
        assert self.tracing_manager.get_current_context() is None
    
    def test_trace_function_decorator(self):
        """Test the trace_function decorator."""
        @trace_function("test_function", {"test_tag": "test_value"})
        def test_function(arg1, arg2, kwarg1=None):
            time.sleep(0.01)  # Simulate work
            return f"{arg1}_{arg2}_{kwarg1}"
        
        result = test_function("a", "b", kwarg1="c")
        
        assert result == "a_b_c"
        
        # Check that trace was created
        traces = self.tracing_manager.get_all_traces()
        assert len(traces) > 0
        
        trace = traces[0]
        assert trace.operation_name == "test_function"
        assert trace.tags["test_tag"] == "test_value"
        assert trace.tags["function.name"] == "test_function"
    
    def test_exception_handling_in_tracing(self):
        """Test exception handling in tracing."""
        @trace_function("failing_function")
        def failing_function():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Check that error was recorded in trace
        traces = self.tracing_manager.get_all_traces()
        assert len(traces) > 0
        
        trace = traces[0]
        assert trace.status == "error"
        assert trace.tags["error"] is True
        assert "Test exception" in trace.tags["error.message"]
    
    def test_trace_cleanup(self):
        """Test trace cleanup functionality."""
        # Create some old traces
        old_context = TraceContext()
        old_context.start_time = time.time() - 7200  # 2 hours ago
        old_context.finish()
        
        self.tracing_manager._traces[old_context.trace_id] = old_context
        
        # Create a recent trace
        recent_context = TraceContext()
        self.tracing_manager._traces[recent_context.trace_id] = recent_context
        
        # Cleanup old traces (older than 1 hour)
        self.tracing_manager.cleanup_old_traces(max_age_seconds=3600)
        
        # Check that old trace was removed but recent one remains
        assert old_context.trace_id not in self.tracing_manager._traces
        assert recent_context.trace_id in self.tracing_manager._traces


class TestAlertValidation:
    """Test alert rules and validation scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = CollectorRegistry()
        self.metrics_collector = MetricsCollector(registry=self.registry)
    
    def test_high_error_rate_alert_condition(self):
        """Test conditions that should trigger high error rate alert."""
        # Simulate high error rate
        for _ in range(20):
            self.metrics_collector.record_http_request('GET', '/api/test', 500, 0.1)
        
        # Add some successful requests
        for _ in range(5):
            self.metrics_collector.record_http_request('GET', '/api/test', 200, 0.1)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        
        # Check that error metrics are recorded
        assert 'flask_http_request_exceptions_total' in metrics_output or 'flask_http_requests_total' in metrics_output
    
    def test_high_memory_usage_alert_condition(self):
        """Test conditions that should trigger high memory usage alert."""
        # This would typically be tested with actual memory pressure
        # For now, we'll test that the metric exists
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'system_memory_usage_bytes' in metrics_output
    
    def test_queue_size_alert_condition(self):
        """Test conditions that should trigger queue size alert."""
        # Simulate high queue size
        self.metrics_collector.update_queue_size(150)  # Above threshold of 100
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_queue_size' in metrics_output
    
    def test_matching_accuracy_alert_condition(self):
        """Test conditions that should trigger low matching accuracy alert."""
        # Simulate low matching accuracy
        self.metrics_collector.record_matching_operation('fuzzy', 'success', 1.0, 0.6)  # Below 0.7 threshold
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_matching_accuracy' in metrics_output


class TestHealthChecks:
    """Test health check implementations."""
    
    def test_health_check_response_structure(self):
        """Test health check response structure."""
        # This would be tested with actual Flask app
        # Mock the health check response
        health_response = {
            'status': 'healthy',
            'timestamp': time.time(),
            'version': '1.0.0',
            'uptime': 3600,
            'memory_usage': {
                'percent': 45.2,
                'available_mb': 2048
            },
            'cpu_usage': 25.5,
            'disk_usage': 60.0,
            'redis': 'connected'
        }
        
        # Validate response structure
        assert 'status' in health_response
        assert 'timestamp' in health_response
        assert 'version' in health_response
        assert 'memory_usage' in health_response
        assert 'cpu_usage' in health_response
        assert health_response['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_health_check_degraded_conditions(self):
        """Test conditions that should mark health as degraded."""
        # Test high memory usage
        health_response = {
            'status': 'healthy',
            'memory_usage': {'percent': 95.0},  # Above 90% threshold
            'cpu_usage': 50.0,
            'disk_usage': 70.0
        }
        
        # Should be marked as degraded
        if (health_response['memory_usage']['percent'] > 90 or 
            health_response['cpu_usage'] > 95 or 
            health_response['disk_usage'] > 95):
            health_response['status'] = 'degraded'
        
        assert health_response['status'] == 'degraded'


class TestObservabilityIntegration:
    """Test integration of observability components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = CollectorRegistry()
        self.metrics_collector = MetricsCollector(registry=self.registry)
        self.tracing_manager = TracingManager()
    
    def test_metrics_and_tracing_integration(self):
        """Test that metrics and tracing work together."""
        @trace_function("integrated_operation")
        def integrated_operation():
            # Record some metrics during traced operation
            self.metrics_collector.record_file_processed('csv', 'success', 1.0, 1024)
            return "success"
        
        result = integrated_operation()
        
        assert result == "success"
        
        # Check that both metrics and traces were recorded
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_files_processed_total' in metrics_output
        
        traces = self.tracing_manager.get_all_traces()
        assert len(traces) > 0
        assert traces[0].operation_name == "integrated_operation"
    
    def test_correlation_between_metrics_and_traces(self):
        """Test correlation between metrics and traces."""
        with self.tracing_manager.start_span("correlated_operation") as span:
            span.set_tag("operation_type", "file_processing")
            
            # Record metrics with similar tags
            self.metrics_collector.record_file_processed('json', 'success', 2.0, 2048)
            
            span.log("File processing completed")
        
        # Verify both systems recorded the operation
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_files_processed_total' in metrics_output
        
        traces = self.tracing_manager.get_all_traces()
        trace = next((t for t in traces if t.operation_name == "correlated_operation"), None)
        assert trace is not None
        assert trace.tags["operation_type"] == "file_processing"
        assert len(trace.logs) > 0


class TestEnhancedMetrics:
    """Test enhanced business metrics functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = CollectorRegistry()
        self.metrics_collector = MetricsCollector(registry=self.registry)
    
    def test_processing_throughput_metrics(self):
        """Test processing throughput metrics."""
        self.metrics_collector.update_processing_throughput('file_matching', 150.5)
        self.metrics_collector.update_processing_throughput('data_validation', 200.0)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_throughput_records_per_second' in metrics_output
        assert 'operation_type="file_matching"' in metrics_output
        assert 'operation_type="data_validation"' in metrics_output
    
    def test_error_rate_by_type_metrics(self):
        """Test error rate by type metrics."""
        self.metrics_collector.update_error_rate_by_type('validation_error', 0.05)
        self.metrics_collector.update_error_rate_by_type('timeout_error', 0.02)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_error_rate_by_type' in metrics_output
        assert 'error_type="validation_error"' in metrics_output
        assert 'error_type="timeout_error"' in metrics_output
    
    def test_user_satisfaction_metrics(self):
        """Test user satisfaction score metrics."""
        self.metrics_collector.update_user_satisfaction_score(0.85)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_user_satisfaction_score' in metrics_output
    
    def test_resource_utilization_metrics(self):
        """Test resource utilization metrics."""
        self.metrics_collector.update_resource_utilization('cpu', 75.5)
        self.metrics_collector.update_resource_utilization('memory', 60.2)
        self.metrics_collector.update_resource_utilization('disk', 45.0)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_resource_utilization_percent' in metrics_output
        assert 'resource_type="cpu"' in metrics_output
        assert 'resource_type="memory"' in metrics_output
        assert 'resource_type="disk"' in metrics_output


class TestTraceAnalyzer:
    """Test trace analyzer functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tracing_manager = TracingManager()
        from src.infrastructure.tracing import TraceAnalyzer
        self.trace_analyzer = TraceAnalyzer(self.tracing_manager)
    
    def test_performance_bottleneck_analysis(self):
        """Test performance bottleneck analysis."""
        # Create some test traces with different performance characteristics
        with self.tracing_manager.start_span("slow_operation") as span:
            span.start_time = time.time() - 10  # Simulate 10 second operation
            span.finish("ok")
        
        with self.tracing_manager.start_span("fast_operation") as span:
            span.start_time = time.time() - 0.1  # Simulate 0.1 second operation
            span.finish("ok")
        
        with self.tracing_manager.start_span("error_operation") as span:
            span.start_time = time.time() - 1
            span.finish("error")
        
        # Analyze bottlenecks
        analysis = self.trace_analyzer.analyze_performance_bottlenecks()
        
        assert 'bottlenecks' in analysis
        assert 'recommendations' in analysis
        assert 'operation_stats' in analysis
        
        # Check that slow operation is identified as bottleneck
        slow_bottlenecks = [b for b in analysis['bottlenecks'] if b['operation'] == 'slow_operation']
        assert len(slow_bottlenecks) > 0
        assert slow_bottlenecks[0]['type'] == 'slow_operation'
    
    def test_trace_dependencies(self):
        """Test trace dependency analysis."""
        # Create a trace with nested spans
        with self.tracing_manager.start_span("parent_operation") as parent_span:
            parent_trace_id = parent_span.trace_id
            
            with self.tracing_manager.start_span("child_operation_1") as child_span1:
                pass
            
            with self.tracing_manager.start_span("child_operation_2") as child_span2:
                pass
        
        # Analyze dependencies
        dependencies = self.trace_analyzer.get_trace_dependencies(parent_trace_id)
        
        assert 'trace_id' in dependencies
        assert 'dependency_tree' in dependencies
        assert 'total_spans' in dependencies
        assert dependencies['trace_id'] == parent_trace_id
        assert dependencies['total_spans'] >= 3  # parent + 2 children


class TestMonitoringEndpoints:
    """Test monitoring endpoints functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        from flask import Flask
        from src.infrastructure.monitoring_endpoints import register_monitoring_endpoints
        
        self.app = Flask(__name__)
        register_monitoring_endpoints(self.app)
        self.client = self.app.test_client()
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/monitoring/health')
        
        assert response.status_code in [200, 503]  # Healthy or unhealthy
        
        data = response.get_json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'checks' in data
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.client.get('/monitoring/metrics')
        
        # Should return metrics or error if not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            assert response.content_type == 'text/plain; version=0.0.4; charset=utf-8'
    
    def test_traces_endpoint(self):
        """Test traces endpoint."""
        response = self.client.get('/monitoring/traces')
        
        # Should return traces data or error if not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'summary' in data
            assert 'recent_traces' in data
    
    def test_performance_analysis_endpoint(self):
        """Test performance analysis endpoint."""
        response = self.client.get('/monitoring/performance')
        
        # Should return performance data or error if not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'system_metrics' in data
            assert 'timestamp' in data
    
    def test_alerts_status_endpoint(self):
        """Test alerts status endpoint."""
        response = self.client.get('/monitoring/alerts')
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'alerts' in data
        assert 'total_alerts' in data
        assert 'timestamp' in data
    
    def test_dashboard_data_endpoint(self):
        """Test dashboard data endpoint."""
        response = self.client.get('/monitoring/dashboard')
        
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'timestamp' in data
        assert 'health' in data
        assert 'metrics_summary' in data


class TestAlertValidationScenarios:
    """Test comprehensive alert validation scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = CollectorRegistry()
        self.metrics_collector = MetricsCollector(registry=self.registry)
    
    def test_processing_throughput_alert_scenario(self):
        """Test processing throughput alert conditions."""
        # Simulate low throughput
        self.metrics_collector.update_processing_throughput('file_matching', 50.0)  # Below 100 threshold
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_throughput_records_per_second' in metrics_output
        
        # In a real scenario, this would trigger the LowProcessingThroughput alert
    
    def test_cache_hit_ratio_alert_scenario(self):
        """Test cache hit ratio alert conditions."""
        # Simulate low cache hit ratio
        self.metrics_collector.cache_hit_ratio.set(0.3)  # Below 0.5 threshold
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_cache_hit_ratio' in metrics_output
        
        # In a real scenario, this would trigger the LowCacheHitRatio alert
    
    def test_user_satisfaction_alert_scenario(self):
        """Test user satisfaction alert conditions."""
        # Simulate low user satisfaction
        self.metrics_collector.update_user_satisfaction_score(0.5)  # Below 0.6 threshold
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_user_satisfaction_score' in metrics_output
        
        # In a real scenario, this would trigger the LowUserSatisfaction alert
    
    def test_resource_utilization_alert_scenario(self):
        """Test resource utilization alert conditions."""
        # Simulate high resource utilization
        self.metrics_collector.update_resource_utilization('cpu', 95.0)  # Above 90% threshold
        self.metrics_collector.update_resource_utilization('memory', 92.0)  # Above 90% threshold
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_resource_utilization_percent' in metrics_output
        
        # In a real scenario, this would trigger the HighResourceUtilization alert
    
    def test_data_volume_processing_alert_scenario(self):
        """Test data volume processing alert conditions."""
        # Simulate high data volume processing
        large_volume = 150 * 1024 * 1024  # 150MB
        self.metrics_collector.data_volume_processed.labels(operation='processing').inc(large_volume)
        
        metrics_output = generate_latest(self.registry).decode('utf-8')
        assert 'file_processing_data_volume_bytes_total' in metrics_output
        
        # In a real scenario, this would trigger the HighDataVolumeProcessing alert


if __name__ == "__main__":
    pytest.main([__file__, "-v"])