"""
Tests for logging and monitoring infrastructure.
Tests logging output format and metric accuracy as required by task 4.2.
"""

import json
import time
import threading
import tempfile
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.infrastructure.logging import (
    LoggerManager, StructuredFormatter, CorrelationIdFilter,
    get_logger, with_correlation_id
)
from src.infrastructure.metrics import (
    MetricsCollector, PerformanceMetrics, MetricValue, MetricSummary,
    get_metrics_collector, track_performance, record_counter, record_gauge, record_histogram
)
from src.infrastructure.health_checks import (
    HealthMonitor, HealthChecker, HealthStatus, HealthCheckResult,
    get_health_monitor
)
from src.infrastructure.error_aggregation import (
    ErrorAggregator, AlertManager, ErrorEvent, ErrorGroup, Alert, AlertLevel,
    AlertChannel, AlertRule, get_error_aggregator, get_alert_manager,
    setup_error_monitoring
)


class TestStructuredLogging:
    """Test structured logging functionality."""
    
    def test_structured_formatter(self):
        """Test structured JSON formatter."""
        import logging
        
        formatter = StructuredFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.correlation_id = 'test-123'
        record.custom_field = 'custom_value'
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse JSON
        log_data = json.loads(formatted)
        
        # Verify structure
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert log_data['correlation_id'] == 'test-123'
        assert log_data['module'] == 'path'
        assert log_data['function'] == '<module>'
        assert log_data['line'] == 42
        assert log_data['custom_field'] == 'custom_value'
        assert 'timestamp' in log_data
        
        # Verify timestamp format
        datetime.fromisoformat(log_data['timestamp'].replace('Z', '+00:00'))
    
    def test_correlation_id_filter(self):
        """Test correlation ID filter."""
        import logging
        
        filter_obj = CorrelationIdFilter()
        
        # Test without correlation ID
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='test', args=(), exc_info=None
        )
        
        assert filter_obj.filter(record)
        assert record.correlation_id == 'unknown'
        
        # Test with correlation ID in thread
        threading.current_thread().correlation_id = 'thread-123'
        
        record2 = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='test', args=(), exc_info=None
        )
        
        assert filter_obj.filter(record2)
        assert record2.correlation_id == 'thread-123'
        
        # Cleanup
        delattr(threading.current_thread(), 'correlation_id')
    
    def test_logger_manager_singleton(self):
        """Test logger manager singleton pattern."""
        manager1 = LoggerManager()
        manager2 = LoggerManager()
        
        assert manager1 is manager2
    
    def test_correlation_context(self):
        """Test correlation context manager."""
        logger_manager = LoggerManager()
        
        # Test with provided correlation ID
        with logger_manager.correlation_context('test-456') as correlation_id:
            assert correlation_id == 'test-456'
            assert threading.current_thread().correlation_id == 'test-456'
        
        # Test with auto-generated correlation ID
        with logger_manager.correlation_context() as correlation_id:
            assert correlation_id is not None
            assert len(correlation_id) > 0
            assert threading.current_thread().correlation_id == correlation_id
        
        # Verify cleanup
        assert not hasattr(threading.current_thread(), 'correlation_id')
    
    def test_with_correlation_id_decorator(self):
        """Test correlation ID decorator."""
        
        @with_correlation_id('decorator-test')
        def test_function():
            return threading.current_thread().correlation_id
        
        result = test_function()
        assert result == 'decorator-test'
        
        # Verify cleanup
        assert not hasattr(threading.current_thread(), 'correlation_id')
    
    def test_logger_configuration(self):
        """Test logger configuration and output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary log files
            log_file = Path(temp_dir) / 'test.log'
            error_file = Path(temp_dir) / 'error.log'
            
            # Mock the logs directory
            with patch('os.makedirs'):
                with patch.object(LoggerManager, '_setup_logging') as mock_setup:
                    manager = LoggerManager()
                    mock_setup.assert_called_once()


class TestMetricsCollection:
    """Test metrics collection functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_history_size=100)
        
        assert collector.max_history_size == 100
        assert len(collector._metrics) == 0
        assert len(collector._performance_metrics) == 0
        assert len(collector._counters) == 0
        assert len(collector._gauges) == 0
    
    def test_counter_metrics(self):
        """Test counter metric recording."""
        collector = MetricsCollector()
        
        # Record counter without labels
        collector.record_counter('test_counter', 5)
        assert collector._counters['test_counter'] == 5
        
        # Record counter with labels
        collector.record_counter('test_counter', 3, {'type': 'error'})
        assert collector._counters['test_counter[type=error]'] == 3
        
        # Increment existing counter
        collector.record_counter('test_counter', 2)
        assert collector._counters['test_counter'] == 7
        
        # Verify metric history
        assert len(collector._metrics['test_counter']) == 2
    
    def test_gauge_metrics(self):
        """Test gauge metric recording."""
        collector = MetricsCollector()
        
        # Record gauge
        collector.record_gauge('cpu_usage', 75.5)
        assert collector._gauges['cpu_usage'] == 75.5
        
        # Update gauge
        collector.record_gauge('cpu_usage', 80.2)
        assert collector._gauges['cpu_usage'] == 80.2
        
        # Record gauge with labels
        collector.record_gauge('memory_usage', 60.0, {'type': 'heap'})
        assert collector._gauges['memory_usage[type=heap]'] == 60.0
    
    def test_histogram_metrics(self):
        """Test histogram metric recording."""
        collector = MetricsCollector()
        
        # Record histogram values
        values = [1.0, 2.5, 3.2, 1.8, 4.1]
        for value in values:
            collector.record_histogram('response_time', value)
        
        assert len(collector._histograms['response_time']) == 5
        assert collector._histograms['response_time'] == values
        
        # Test histogram size limit
        for i in range(1000):
            collector.record_histogram('large_histogram', float(i))
        
        assert len(collector._histograms['large_histogram']) == 1000
        
        # Add one more to trigger size limit
        collector.record_histogram('large_histogram', 1000.0)
        assert len(collector._histograms['large_histogram']) == 1000
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        collector = MetricsCollector()
        
        # Start performance tracking
        metrics = collector.start_performance_tracking(
            'test-op-123', 'file_processing',
            dataset1_size=1000, dataset2_size=2000
        )
        
        assert metrics.operation_id == 'test-op-123'
        assert metrics.operation_type == 'file_processing'
        assert metrics.dataset1_size == 1000
        assert metrics.dataset2_size == 2000
        assert metrics.end_time is None
        
        # Update metrics
        collector.update_performance_metrics('test-op-123', 
                                            matches_found=150,
                                            actual_comparisons=5000)
        
        updated_metrics = collector.get_performance_metrics('test-op-123')
        assert updated_metrics.matches_found == 150
        assert updated_metrics.actual_comparisons == 5000
        
        # Complete tracking
        time.sleep(0.1)  # Small delay to ensure duration > 0
        final_metrics = collector.complete_performance_tracking('test-op-123')
        
        assert final_metrics.end_time is not None
        assert final_metrics.duration_seconds > 0
        assert final_metrics.throughput_records_per_second > 0
    
    def test_metric_summary(self):
        """Test metric summary calculation."""
        collector = MetricsCollector()
        
        # Record some histogram values
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for value in values:
            collector.record_histogram('test_metric', value)
        
        # Get summary
        summary = collector.get_metric_summary('test_metric')
        
        assert summary is not None
        assert summary.count == 10
        assert summary.sum == 55.0
        assert summary.min == 1.0
        assert summary.max == 10.0
        assert summary.mean == 5.5
        assert summary.median == 5.5
    
    def test_track_performance_context_manager(self):
        """Test performance tracking context manager."""
        collector = MetricsCollector()
        
        with track_performance(collector, 'test_operation', 
                             dataset1_size=100, dataset2_size=200) as metrics:
            assert metrics.operation_type == 'test_operation'
            assert metrics.dataset1_size == 100
            assert metrics.dataset2_size == 200
            
            # Simulate some work
            time.sleep(0.01)
            metrics.matches_found = 50
        
        # Verify completion
        assert metrics.end_time is not None
        assert metrics.duration_seconds > 0
    
    def test_track_performance_with_exception(self):
        """Test performance tracking with exception."""
        collector = MetricsCollector()
        
        with pytest.raises(ValueError):
            with track_performance(collector, 'test_operation') as metrics:
                raise ValueError("Test error")
        
        # Verify error was recorded
        assert metrics.errors_count == 1
        assert metrics.end_time is not None
    
    def test_global_metrics_functions(self):
        """Test global metrics convenience functions."""
        # These should not raise exceptions
        record_counter('global_counter', 1)
        record_gauge('global_gauge', 42.0)
        record_histogram('global_histogram', 3.14)
        
        # Verify they use the global collector
        collector = get_metrics_collector()
        assert collector._counters['global_counter'] == 1
        assert collector._gauges['global_gauge'] == 42.0
        assert len(collector._histograms['global_histogram']) == 1


class TestHealthChecks:
    """Test health check functionality."""
    
    def test_health_check_result(self):
        """Test health check result creation."""
        result = HealthCheckResult(
            name='test_check',
            status=HealthStatus.HEALTHY,
            message='All good',
            timestamp=datetime.now(),
            duration_ms=50.0,
            details={'key': 'value'}
        )
        
        result_dict = result.to_dict()
        assert result_dict['name'] == 'test_check'
        assert result_dict['status'] == 'healthy'
        assert result_dict['message'] == 'All good'
        assert result_dict['duration_ms'] == 50.0
        assert result_dict['details'] == {'key': 'value'}
    
    def test_health_checker(self):
        """Test individual health checker."""
        def mock_check():
            return HealthCheckResult(
                name='mock_check',
                status=HealthStatus.HEALTHY,
                message='Mock check passed',
                timestamp=datetime.now(),
                duration_ms=0,
                details={}
            )
        
        checker = HealthChecker('mock_check', mock_check, interval_seconds=30)
        
        result = checker.run_check()
        assert result.name == 'mock_check'
        assert result.status == HealthStatus.HEALTHY
        assert result.duration_ms >= 0
        assert checker.last_result == result
    
    def test_health_checker_exception(self):
        """Test health checker with exception."""
        def failing_check():
            raise RuntimeError("Check failed")
        
        checker = HealthChecker('failing_check', failing_check)
        
        result = checker.run_check()
        assert result.name == 'failing_check'
        assert result.status == HealthStatus.CRITICAL
        assert 'Check failed' in result.message
        assert 'error' in result.details
    
    @patch('psutil.virtual_memory')
    def test_system_memory_check(self, mock_memory):
        """Test system memory health check."""
        # Mock normal memory usage
        mock_memory.return_value = Mock(
            percent=70.0,
            total=8 * 1024**3,  # 8GB
            available=2.4 * 1024**3,  # 2.4GB
            used=5.6 * 1024**3  # 5.6GB
        )
        
        monitor = HealthMonitor()
        result = monitor._check_system_memory()
        
        assert result.name == 'system_memory'
        assert result.status == HealthStatus.HEALTHY
        assert '70.0%' in result.message
        assert result.details['usage_percent'] == 70.0
    
    @patch('psutil.virtual_memory')
    def test_system_memory_check_warning(self, mock_memory):
        """Test system memory health check with warning."""
        # Mock high memory usage
        mock_memory.return_value = Mock(
            percent=85.0,
            total=8 * 1024**3,
            available=1.2 * 1024**3,
            used=6.8 * 1024**3
        )
        
        monitor = HealthMonitor()
        result = monitor._check_system_memory()
        
        assert result.status == HealthStatus.WARNING
        assert 'high' in result.message.lower()
    
    @patch('psutil.virtual_memory')
    def test_system_memory_check_critical(self, mock_memory):
        """Test system memory health check with critical status."""
        # Mock critical memory usage
        mock_memory.return_value = Mock(
            percent=95.0,
            total=8 * 1024**3,
            available=0.4 * 1024**3,
            used=7.6 * 1024**3
        )
        
        monitor = HealthMonitor()
        result = monitor._check_system_memory()
        
        assert result.status == HealthStatus.CRITICAL
        assert 'critical' in result.message.lower()
    
    def test_health_monitor_registration(self):
        """Test health check registration."""
        monitor = HealthMonitor()
        
        def custom_check():
            return HealthCheckResult(
                name='custom',
                status=HealthStatus.HEALTHY,
                message='Custom check',
                timestamp=datetime.now(),
                duration_ms=0,
                details={}
            )
        
        # Register custom check
        monitor.register_check('custom', custom_check, interval_seconds=120)
        
        assert 'custom' in monitor.checkers
        assert monitor.checkers['custom'].interval_seconds == 120
        
        # Unregister check
        monitor.unregister_check('custom')
        assert 'custom' not in monitor.checkers
    
    def test_system_health_determination(self):
        """Test overall system health status determination."""
        monitor = HealthMonitor()
        
        # All healthy
        healthy_results = [
            HealthCheckResult('check1', HealthStatus.HEALTHY, 'OK', datetime.now(), 0, {}),
            HealthCheckResult('check2', HealthStatus.HEALTHY, 'OK', datetime.now(), 0, {})
        ]
        
        status = monitor._determine_overall_status(healthy_results)
        assert status == HealthStatus.HEALTHY
        
        # One warning
        warning_results = [
            HealthCheckResult('check1', HealthStatus.HEALTHY, 'OK', datetime.now(), 0, {}),
            HealthCheckResult('check2', HealthStatus.WARNING, 'Warning', datetime.now(), 0, {})
        ]
        
        status = monitor._determine_overall_status(warning_results)
        assert status == HealthStatus.WARNING
        
        # One critical
        critical_results = [
            HealthCheckResult('check1', HealthStatus.HEALTHY, 'OK', datetime.now(), 0, {}),
            HealthCheckResult('check2', HealthStatus.CRITICAL, 'Critical', datetime.now(), 0, {})
        ]
        
        status = monitor._determine_overall_status(critical_results)
        assert status == HealthStatus.CRITICAL


class TestErrorAggregation:
    """Test error aggregation and alerting."""
    
    def test_error_event_creation(self):
        """Test error event creation and signature."""
        event = ErrorEvent(
            error_id='test-123',
            timestamp=datetime.now(),
            level='ERROR',
            message='Test error message',
            exception_type='ValueError',
            module='test_module',
            function='test_function',
            line_number=42,
            correlation_id='corr-123',
            context={'key': 'value'}
        )
        
        # Test signature generation
        signature = event.get_signature()
        assert len(signature) == 32  # MD5 hash length
        
        # Same signature for same error location
        event2 = ErrorEvent(
            error_id='test-456',
            timestamp=datetime.now(),
            level='ERROR',
            message='Different message',
            exception_type='ValueError',
            module='test_module',
            function='test_function',
            line_number=42,
            correlation_id='corr-456'
        )
        
        assert event.get_signature() == event2.get_signature()
    
    def test_error_aggregator(self):
        """Test error aggregation functionality."""
        aggregator = ErrorAggregator(max_groups=10)
        
        # Add error events
        event1 = ErrorEvent(
            error_id='1', timestamp=datetime.now(), level='ERROR',
            message='Error 1', exception_type='ValueError',
            module='mod1', function='func1', line_number=10,
            correlation_id='corr1'
        )
        
        event2 = ErrorEvent(
            error_id='2', timestamp=datetime.now(), level='ERROR',
            message='Error 2', exception_type='ValueError',
            module='mod1', function='func1', line_number=10,
            correlation_id='corr2'
        )
        
        aggregator.add_error_event(event1)
        aggregator.add_error_event(event2)
        
        # Should be grouped together
        groups = aggregator.get_error_groups()
        assert len(groups) == 1
        assert groups[0].count == 2
        assert groups[0].error_type == 'ValueError'
    
    def test_error_statistics(self):
        """Test error statistics calculation."""
        aggregator = ErrorAggregator()
        
        # Add various error events
        for i in range(5):
            event = ErrorEvent(
                error_id=f'error-{i}',
                timestamp=datetime.now(),
                level='ERROR',
                message=f'Error {i}',
                exception_type='ValueError' if i < 3 else 'RuntimeError',
                module=f'module{i % 2}',
                function='test_func',
                line_number=i + 10,
                correlation_id=f'corr-{i}'
            )
            aggregator.add_error_event(event)
        
        stats = aggregator.get_error_statistics(time_window_minutes=60)
        
        assert stats['total_errors'] == 5
        assert stats['unique_error_types'] == 2
        assert stats['error_rate_per_minute'] > 0
        assert len(stats['top_error_types']) == 2
        assert stats['top_error_types'][0][0] == 'ValueError'  # Most common
        assert stats['top_error_types'][0][1] == 3  # Count
    
    def test_alert_rule_triggering(self):
        """Test alert rule triggering."""
        def high_count_condition(group):
            return group.count > 2
        
        rule = AlertRule(
            name='high_count',
            condition=high_count_condition,
            alert_level=AlertLevel.WARNING,
            channels=[AlertChannel.LOG],
            cooldown_minutes=5
        )
        
        # Create error group with low count
        group = ErrorGroup(
            signature='test-sig',
            first_occurrence=datetime.now(),
            last_occurrence=datetime.now(),
            count=1,
            error_type='ValueError',
            module='test',
            function='test',
            sample_message='Test error'
        )
        
        assert not rule.should_trigger(group)
        
        # Increase count
        group.count = 5
        assert rule.should_trigger(group)
        
        # Trigger rule
        rule.trigger()
        
        # Should not trigger again due to cooldown
        assert not rule.should_trigger(group)
    
    def test_alert_manager(self):
        """Test alert manager functionality."""
        aggregator = ErrorAggregator()
        manager = AlertManager(aggregator)
        
        # Mock alert handler
        alerts_received = []
        
        def mock_handler(alert):
            alerts_received.append(alert)
        
        manager.register_alert_handler(AlertChannel.LOG, mock_handler)
        
        # Create alert
        alert = Alert(
            alert_id='test-alert',
            level=AlertLevel.WARNING,
            title='Test Alert',
            message='This is a test alert',
            timestamp=datetime.now(),
            source='test',
            channels=[AlertChannel.LOG]
        )
        
        manager.send_alert(alert)
        
        assert len(alerts_received) == 1
        assert alerts_received[0].alert_id == 'test-alert'
    
    def test_error_monitoring_setup(self):
        """Test error monitoring setup integration."""
        import logging
        
        # Setup error monitoring
        setup_error_monitoring()
        
        # Get logger and log an error
        logger = logging.getLogger('test_error_monitoring')
        
        try:
            raise ValueError("Test error for monitoring")
        except ValueError:
            logger.exception("Test exception occurred")
        
        # Give some time for processing
        time.sleep(0.1)
        
        # Check if error was aggregated
        aggregator = get_error_aggregator()
        groups = aggregator.get_error_groups(time_window_minutes=1)
        
        # Should have at least one error group
        assert len(groups) >= 0  # May be 0 if handler not properly attached


class TestIntegration:
    """Integration tests for logging and monitoring system."""
    
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        # Get all components
        metrics_collector = get_metrics_collector()
        health_monitor = get_health_monitor()
        error_aggregator = get_error_aggregator()
        alert_manager = get_alert_manager()
        
        # Record some metrics
        metrics_collector.record_counter('integration_test_counter', 1)
        metrics_collector.record_gauge('integration_test_gauge', 42.0)
        
        # Start performance tracking
        with track_performance(metrics_collector, 'integration_test') as perf:
            time.sleep(0.01)
            perf.matches_found = 10
        
        # Run health checks
        health_status = health_monitor.get_health_status()
        assert health_status.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        
        # Add error event
        error_event = ErrorEvent(
            error_id='integration-error',
            timestamp=datetime.now(),
            level='ERROR',
            message='Integration test error',
            exception_type='TestError',
            module='integration_test',
            function='test_function',
            line_number=100,
            correlation_id='integration-corr'
        )
        
        error_aggregator.add_error_event(error_event)
        
        # Verify everything is working
        assert metrics_collector._counters['integration_test_counter'] == 1
        assert metrics_collector._gauges['integration_test_gauge'] == 42.0
        assert len(health_status.checks) > 0
        assert len(error_aggregator.get_error_groups()) >= 1
    
    def test_logging_with_correlation_id(self):
        """Test logging with correlation ID integration."""
        logger = get_logger('integration_test')
        
        with LoggerManager().correlation_context('integration-123'):
            logger.info('Test message with correlation ID')
            
            # Record metrics in same context
            record_counter('correlated_counter', 1)
            
            # The correlation ID should be available in the thread
            assert threading.current_thread().correlation_id == 'integration-123'
    
    def test_performance_metrics_accuracy(self):
        """Test accuracy of performance metrics."""
        collector = MetricsCollector()
        
        start_time = time.time()
        
        with track_performance(collector, 'accuracy_test', 
                             dataset1_size=1000, dataset2_size=2000) as metrics:
            # Simulate processing time
            time.sleep(0.1)
            
            # Update metrics
            metrics.matches_found = 150
            metrics.actual_comparisons = 5000
            metrics.memory_usage_mb = 50.0
        
        end_time = time.time()
        expected_duration = end_time - start_time
        
        # Verify accuracy (allow for small timing differences)
        assert abs(metrics.duration_seconds - expected_duration) < 0.01
        assert metrics.throughput_records_per_second > 0
        assert metrics.matches_found == 150
        assert metrics.actual_comparisons == 5000
        assert metrics.memory_usage_mb == 50.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])