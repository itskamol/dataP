"""
Monitoring endpoints for health checks, metrics, and observability.
Provides comprehensive monitoring capabilities for the file processing system.
"""

import time
import json
import psutil
from typing import Dict, Any, List, Optional
from flask import Blueprint, jsonify, Response, request
from prometheus_client import CONTENT_TYPE_LATEST
import logging

from .metrics import metrics_collector
from .tracing import tracing_manager, trace_analyzer, get_trace_summary

logger = logging.getLogger(__name__)

# Create monitoring blueprint
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')


@monitoring_bp.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'version': '1.0.0',
            'uptime_seconds': time.time() - getattr(health_check, 'start_time', time.time()),
            'checks': {}
        }
        
        # System resource checks
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('/')
        
        health_status['checks']['memory'] = {
            'status': 'healthy' if memory.percent < 90 else 'degraded' if memory.percent < 95 else 'unhealthy',
            'usage_percent': memory.percent,
            'available_mb': memory.available / (1024 * 1024),
            'total_mb': memory.total / (1024 * 1024)
        }
        
        health_status['checks']['cpu'] = {
            'status': 'healthy' if cpu_percent < 80 else 'degraded' if cpu_percent < 95 else 'unhealthy',
            'usage_percent': cpu_percent
        }
        
        health_status['checks']['disk'] = {
            'status': 'healthy' if disk.percent < 85 else 'degraded' if disk.percent < 95 else 'unhealthy',
            'usage_percent': disk.percent,
            'free_gb': disk.free / (1024**3),
            'total_gb': disk.total / (1024**3)
        }
        
        # Application-specific checks
        health_status['checks']['metrics_collection'] = {
            'status': 'healthy' if metrics_collector else 'unhealthy',
            'enabled': metrics_collector is not None
        }
        
        health_status['checks']['tracing'] = {
            'status': 'healthy' if tracing_manager else 'unhealthy',
            'enabled': tracing_manager is not None,
            'active_traces': len(tracing_manager.get_all_traces()) if tracing_manager else 0
        }
        
        # Determine overall status
        check_statuses = [check['status'] for check in health_status['checks'].values()]
        if 'unhealthy' in check_statuses:
            health_status['status'] = 'unhealthy'
        elif 'degraded' in check_statuses:
            health_status['status'] = 'degraded'
        
        # Set appropriate HTTP status code
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': time.time(),
            'error': str(e)
        }), 503


@monitoring_bp.route('/metrics')
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        if not metrics_collector:
            return jsonify({'error': 'Metrics collection not initialized'}), 503
        
        metrics_data = metrics_collector.get_metrics()
        return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return jsonify({'error': f'Failed to generate metrics: {str(e)}'}), 500


@monitoring_bp.route('/traces')
def traces_endpoint():
    """Distributed tracing endpoint for debugging."""
    try:
        if not tracing_manager:
            return jsonify({'error': 'Tracing not initialized'}), 503
        
        # Get query parameters
        limit = request.args.get('limit', 10, type=int)
        trace_id = request.args.get('trace_id')
        
        if trace_id:
            # Get specific trace
            trace = tracing_manager.get_trace(trace_id)
            if not trace:
                return jsonify({'error': 'Trace not found'}), 404
            
            # Get dependency information
            dependencies = trace_analyzer.get_trace_dependencies(trace_id)
            
            return jsonify({
                'trace': trace.to_dict(),
                'dependencies': dependencies
            })
        else:
            # Get trace summary and recent traces
            summary = get_trace_summary()
            traces = tracing_manager.get_all_traces()
            
            # Convert recent traces to serializable format
            recent_traces = []
            for trace in traces[-limit:]:
                recent_traces.append(trace.to_dict())
            
            return jsonify({
                'summary': summary,
                'recent_traces': recent_traces,
                'total_traces': len(traces)
            })
            
    except Exception as e:
        logger.error(f"Failed to get traces: {e}")
        return jsonify({'error': f'Failed to get traces: {str(e)}'}), 500


@monitoring_bp.route('/performance')
def performance_analysis():
    """Performance analysis endpoint."""
    try:
        if not trace_analyzer:
            return jsonify({'error': 'Trace analyzer not available'}), 503
        
        # Get performance bottlenecks
        bottlenecks = trace_analyzer.analyze_performance_bottlenecks()
        
        # Get system performance metrics
        system_metrics = {
            'memory': {
                'usage_percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=0.1),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'disk': {
                'usage_percent': psutil.disk_usage('/').percent,
                'io_stats': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None
            }
        }
        
        # Get application performance metrics
        app_metrics = {}
        if metrics_collector:
            try:
                # This would typically extract specific metrics from the collector
                app_metrics = {
                    'active_sessions': 'Available via /metrics endpoint',
                    'queue_size': 'Available via /metrics endpoint',
                    'processing_throughput': 'Available via /metrics endpoint'
                }
            except Exception as e:
                logger.warning(f"Could not get application metrics: {e}")
        
        return jsonify({
            'bottlenecks': bottlenecks,
            'system_metrics': system_metrics,
            'application_metrics': app_metrics,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return jsonify({'error': f'Performance analysis failed: {str(e)}'}), 500


@monitoring_bp.route('/alerts')
def alerts_status():
    """Get current alert status and thresholds."""
    try:
        alerts = []
        
        # Check system resource alerts
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            alerts.append({
                'type': 'system',
                'severity': 'critical' if memory.percent > 95 else 'warning',
                'message': f'High memory usage: {memory.percent:.1f}%',
                'threshold': 90,
                'current_value': memory.percent,
                'timestamp': time.time()
            })
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 80:
            alerts.append({
                'type': 'system',
                'severity': 'critical' if cpu_percent > 95 else 'warning',
                'message': f'High CPU usage: {cpu_percent:.1f}%',
                'threshold': 80,
                'current_value': cpu_percent,
                'timestamp': time.time()
            })
        
        disk = psutil.disk_usage('/')
        if disk.percent > 85:
            alerts.append({
                'type': 'system',
                'severity': 'critical' if disk.percent > 95 else 'warning',
                'message': f'High disk usage: {disk.percent:.1f}%',
                'threshold': 85,
                'current_value': disk.percent,
                'timestamp': time.time()
            })
        
        # Check application-specific alerts
        if tracing_manager:
            traces = tracing_manager.get_all_traces()
            error_traces = [t for t in traces if t.status == 'error']
            if len(traces) > 0:
                error_rate = len(error_traces) / len(traces)
                if error_rate > 0.1:
                    alerts.append({
                        'type': 'application',
                        'severity': 'critical' if error_rate > 0.2 else 'warning',
                        'message': f'High error rate: {error_rate:.1%}',
                        'threshold': 0.1,
                        'current_value': error_rate,
                        'timestamp': time.time()
                    })
        
        return jsonify({
            'alerts': alerts,
            'total_alerts': len(alerts),
            'critical_alerts': len([a for a in alerts if a['severity'] == 'critical']),
            'warning_alerts': len([a for a in alerts if a['severity'] == 'warning']),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to get alerts status: {e}")
        return jsonify({'error': f'Failed to get alerts status: {str(e)}'}), 500


@monitoring_bp.route('/dashboard')
def dashboard_data():
    """Get dashboard data for monitoring UI."""
    try:
        # Collect all monitoring data for dashboard
        dashboard_data = {
            'timestamp': time.time(),
            'health': {},
            'metrics_summary': {},
            'traces_summary': {},
            'alerts_summary': {}
        }
        
        # Get health status
        try:
            health_response = health_check()
            if health_response[1] == 200:  # Status code
                dashboard_data['health'] = health_response[0].get_json()
        except Exception as e:
            logger.warning(f"Could not get health data for dashboard: {e}")
        
        # Get traces summary
        try:
            if tracing_manager:
                dashboard_data['traces_summary'] = get_trace_summary()
        except Exception as e:
            logger.warning(f"Could not get traces summary for dashboard: {e}")
        
        # Get alerts summary
        try:
            alerts_response = alerts_status()
            dashboard_data['alerts_summary'] = alerts_response.get_json()
        except Exception as e:
            logger.warning(f"Could not get alerts summary for dashboard: {e}")
        
        # Get basic metrics summary
        try:
            system_info = {
                'memory_usage_percent': psutil.virtual_memory().percent,
                'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'uptime_seconds': time.time() - getattr(health_check, 'start_time', time.time())
            }
            dashboard_data['metrics_summary'] = system_info
        except Exception as e:
            logger.warning(f"Could not get metrics summary for dashboard: {e}")
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return jsonify({'error': f'Failed to get dashboard data: {str(e)}'}), 500


# Initialize start time for uptime calculation
health_check.start_time = time.time()


def register_monitoring_endpoints(app):
    """Register monitoring endpoints with Flask app."""
    app.register_blueprint(monitoring_bp)
    
    # Add CORS headers for monitoring endpoints
    @monitoring_bp.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    logger.info("Monitoring endpoints registered successfully")