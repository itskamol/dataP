"""
Distributed tracing implementation for file processing system.
Provides request flow analysis and performance monitoring across services.
"""

import time
import uuid
import threading
import functools
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class TraceContext:
    """Context for distributed tracing."""
    
    def __init__(self, trace_id: str = None, span_id: str = None, parent_span_id: str = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.start_time = time.time()
        self.end_time = None
        self.tags: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.operation_name = ""
        self.status = "ok"
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the trace context."""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the trace context."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def finish(self, status: str = "ok"):
        """Finish the trace context."""
        self.end_time = time.time()
        self.status = status
    
    def duration(self) -> float:
        """Get the duration of the trace."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs
        }


class TracingManager:
    """Manager for distributed tracing operations."""
    
    def __init__(self):
        self._local = threading.local()
        self._traces: Dict[str, TraceContext] = {}
        self._lock = threading.Lock()
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        return getattr(self._local, 'context', None)
    
    def set_current_context(self, context: TraceContext):
        """Set the current trace context."""
        self._local.context = context
        with self._lock:
            self._traces[context.trace_id] = context
    
    def clear_current_context(self):
        """Clear the current trace context."""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    @contextmanager
    def start_span(self, operation_name: str, tags: Optional[Dict[str, Any]] = None):
        """Start a new span within the current trace."""
        current_context = self.get_current_context()
        
        if current_context:
            # Create child span
            span_context = TraceContext(
                trace_id=current_context.trace_id,
                parent_span_id=current_context.span_id
            )
        else:
            # Create root span
            span_context = TraceContext()
        
        span_context.operation_name = operation_name
        
        if tags:
            for key, value in tags.items():
                span_context.set_tag(key, value)
        
        # Set as current context
        previous_context = self.get_current_context()
        self.set_current_context(span_context)
        
        try:
            yield span_context
        except Exception as e:
            span_context.set_tag("error", True)
            span_context.set_tag("error.message", str(e))
            span_context.log(f"Exception occurred: {str(e)}", level="error")
            span_context.finish("error")
            raise
        else:
            span_context.finish("ok")
        finally:
            # Restore previous context
            if previous_context:
                self.set_current_context(previous_context)
            else:
                self.clear_current_context()
    
    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Get a trace by ID."""
        with self._lock:
            return self._traces.get(trace_id)
    
    def get_all_traces(self) -> List[TraceContext]:
        """Get all traces."""
        with self._lock:
            return list(self._traces.values())
    
    def cleanup_old_traces(self, max_age_seconds: int = 3600):
        """Clean up old traces."""
        current_time = time.time()
        with self._lock:
            traces_to_remove = []
            for trace_id, context in self._traces.items():
                if current_time - context.start_time > max_age_seconds:
                    traces_to_remove.append(trace_id)
            
            for trace_id in traces_to_remove:
                del self._traces[trace_id]


# Global tracing manager
tracing_manager = TracingManager()


def trace_function(operation_name: str = None, tags: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracing_manager.start_span(op_name, tags) as span:
                span.set_tag("function.name", func.__name__)
                span.set_tag("function.module", func.__module__)
                
                # Add argument information (be careful with sensitive data)
                if args:
                    span.set_tag("function.args_count", len(args))
                if kwargs:
                    span.set_tag("function.kwargs_count", len(kwargs))
                
                result = func(*args, **kwargs)
                
                # Add result information if it's a simple type
                if isinstance(result, (str, int, float, bool)):
                    span.set_tag("function.result_type", type(result).__name__)
                
                return result
        
        return wrapper
    return decorator


def trace_http_request(app):
    """Flask middleware to trace HTTP requests."""
    @app.before_request
    def before_request():
        from flask import request
        
        # Extract trace context from headers if present
        trace_id = request.headers.get('X-Trace-Id')
        span_id = request.headers.get('X-Span-Id')
        
        if trace_id and span_id:
            context = TraceContext(trace_id=trace_id, parent_span_id=span_id)
        else:
            context = TraceContext()
        
        context.operation_name = f"{request.method} {request.endpoint or request.path}"
        context.set_tag("http.method", request.method)
        context.set_tag("http.url", request.url)
        context.set_tag("http.endpoint", request.endpoint)
        context.set_tag("http.remote_addr", request.remote_addr)
        context.set_tag("http.user_agent", request.headers.get('User-Agent', ''))
        
        tracing_manager.set_current_context(context)
    
    @app.after_request
    def after_request(response):
        from flask import request
        
        # Skip tracing for static files to avoid errors
        if request.endpoint == 'static':
            return response
            
        context = tracing_manager.get_current_context()
        if context:
            context.set_tag("http.status_code", response.status_code)
            
            # Safely get response size without breaking static file serving
            try:
                if hasattr(response, 'content_length') and response.content_length:
                    context.set_tag("http.response_size", response.content_length)
                elif hasattr(response, 'get_data'):
                    # Only call get_data if response is not in direct passthrough mode
                    if not getattr(response, 'direct_passthrough', False):
                        context.set_tag("http.response_size", len(response.get_data()))
            except (RuntimeError, AttributeError):
                # Skip response size tracking if it causes issues
                pass
            
            # Add trace headers to response
            response.headers['X-Trace-Id'] = context.trace_id
            response.headers['X-Span-Id'] = context.span_id
            
            context.finish("ok" if response.status_code < 400 else "error")
            tracing_manager.clear_current_context()
        
        return response
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        context = tracing_manager.get_current_context()
        if context:
            context.set_tag("error", True)
            context.set_tag("error.message", str(e))
            context.set_tag("error.type", type(e).__name__)
            context.log(f"Exception: {str(e)}", level="error")
            context.finish("error")
            tracing_manager.clear_current_context()
        raise


class TraceExporter:
    """Export traces to external systems."""
    
    def __init__(self, export_format: str = "json"):
        self.export_format = export_format
    
    def export_trace(self, trace: TraceContext) -> str:
        """Export a single trace."""
        if self.export_format == "json":
            import json
            return json.dumps(trace.to_dict(), indent=2)
        elif self.export_format == "jaeger":
            return self._export_jaeger_format(trace)
        elif self.export_format == "zipkin":
            return self._export_zipkin_format(trace)
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")
    
    def _export_jaeger_format(self, trace: TraceContext) -> Dict[str, Any]:
        """Export trace in Jaeger format."""
        return {
            "traceID": trace.trace_id,
            "spanID": trace.span_id,
            "parentSpanID": trace.parent_span_id,
            "operationName": trace.operation_name,
            "startTime": int(trace.start_time * 1000000),  # microseconds
            "duration": int(trace.duration() * 1000000),  # microseconds
            "tags": [{"key": k, "value": v} for k, v in trace.tags.items()],
            "logs": [
                {
                    "timestamp": int(log["timestamp"] * 1000000),
                    "fields": [{"key": k, "value": v} for k, v in log.items() if k != "timestamp"]
                }
                for log in trace.logs
            ]
        }
    
    def _export_zipkin_format(self, trace: TraceContext) -> Dict[str, Any]:
        """Export trace in Zipkin format."""
        return {
            "traceId": trace.trace_id,
            "id": trace.span_id,
            "parentId": trace.parent_span_id,
            "name": trace.operation_name,
            "timestamp": int(trace.start_time * 1000000),  # microseconds
            "duration": int(trace.duration() * 1000000),  # microseconds
            "tags": trace.tags,
            "annotations": [
                {
                    "timestamp": int(log["timestamp"] * 1000000),
                    "value": log["message"]
                }
                for log in trace.logs
            ]
        }
    
    def export_traces_batch(self, traces: List[TraceContext]) -> str:
        """Export multiple traces in batch format."""
        if self.export_format == "json":
            import json
            return json.dumps([trace.to_dict() for trace in traces], indent=2)
        elif self.export_format == "jaeger":
            return json.dumps({
                "data": [self._export_jaeger_format(trace) for trace in traces]
            })
        elif self.export_format == "zipkin":
            return json.dumps([self._export_zipkin_format(trace) for trace in traces])
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")


# Trace exporter instance
trace_exporter = TraceExporter()


class TraceAnalyzer:
    """Analyze traces for performance insights and bottlenecks."""
    
    def __init__(self, tracing_manager: TracingManager):
        self.tracing_manager = tracing_manager
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze traces to identify performance bottlenecks."""
        traces = self.tracing_manager.get_all_traces()
        completed_traces = [t for t in traces if t.end_time is not None]
        
        if not completed_traces:
            return {"bottlenecks": [], "recommendations": []}
        
        # Group traces by operation
        operation_stats = {}
        for trace in completed_traces:
            op_name = trace.operation_name
            if op_name not in operation_stats:
                operation_stats[op_name] = {
                    "durations": [],
                    "error_count": 0,
                    "total_count": 0
                }
            
            operation_stats[op_name]["durations"].append(trace.duration())
            operation_stats[op_name]["total_count"] += 1
            if trace.status == "error":
                operation_stats[op_name]["error_count"] += 1
        
        # Identify bottlenecks
        bottlenecks = []
        recommendations = []
        
        for op_name, stats in operation_stats.items():
            durations = stats["durations"]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            error_rate = stats["error_count"] / stats["total_count"]
            
            # Identify slow operations (> 5 seconds average)
            if avg_duration > 5.0:
                bottlenecks.append({
                    "operation": op_name,
                    "type": "slow_operation",
                    "avg_duration": avg_duration,
                    "max_duration": max_duration,
                    "severity": "high" if avg_duration > 10.0 else "medium"
                })
                recommendations.append(f"Optimize {op_name} - average duration {avg_duration:.2f}s")
            
            # Identify high error rate operations (> 10%)
            if error_rate > 0.1:
                bottlenecks.append({
                    "operation": op_name,
                    "type": "high_error_rate",
                    "error_rate": error_rate,
                    "severity": "high" if error_rate > 0.2 else "medium"
                })
                recommendations.append(f"Investigate errors in {op_name} - error rate {error_rate:.1%}")
        
        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "operation_stats": operation_stats
        }
    
    def get_trace_dependencies(self, trace_id: str) -> Dict[str, Any]:
        """Get dependency graph for a specific trace."""
        trace = self.tracing_manager.get_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        # Find all related traces (same trace_id)
        all_traces = self.tracing_manager.get_all_traces()
        related_traces = [t for t in all_traces if t.trace_id == trace_id]
        
        # Build dependency tree
        dependency_tree = self._build_dependency_tree(related_traces)
        
        return {
            "trace_id": trace_id,
            "dependency_tree": dependency_tree,
            "total_spans": len(related_traces),
            "total_duration": max(t.duration() for t in related_traces) if related_traces else 0
        }
    
    def _build_dependency_tree(self, traces: List[TraceContext]) -> Dict[str, Any]:
        """Build a dependency tree from traces."""
        # Create a map of span_id to trace
        span_map = {trace.span_id: trace for trace in traces}
        
        # Find root traces (no parent)
        root_traces = [trace for trace in traces if not trace.parent_span_id]
        
        def build_tree_node(trace: TraceContext) -> Dict[str, Any]:
            children = [
                build_tree_node(child_trace)
                for child_trace in traces
                if child_trace.parent_span_id == trace.span_id
            ]
            
            return {
                "span_id": trace.span_id,
                "operation_name": trace.operation_name,
                "duration": trace.duration(),
                "status": trace.status,
                "tags": trace.tags,
                "children": children
            }
        
        return [build_tree_node(root) for root in root_traces]


def get_trace_summary() -> Dict[str, Any]:
    """Get a summary of current traces."""
    traces = tracing_manager.get_all_traces()
    
    if not traces:
        return {"total_traces": 0, "active_traces": 0}
    
    active_traces = [t for t in traces if t.end_time is None]
    completed_traces = [t for t in traces if t.end_time is not None]
    
    avg_duration = 0
    p95_duration = 0
    p99_duration = 0
    
    if completed_traces:
        durations = sorted([t.duration() for t in completed_traces])
        avg_duration = sum(durations) / len(durations)
        
        # Calculate percentiles
        if len(durations) > 0:
            p95_index = int(0.95 * len(durations))
            p99_index = int(0.99 * len(durations))
            p95_duration = durations[min(p95_index, len(durations) - 1)]
            p99_duration = durations[min(p99_index, len(durations) - 1)]
    
    error_traces = [t for t in completed_traces if t.status == "error"]
    error_rate = len(error_traces) / len(completed_traces) if completed_traces else 0
    
    # Group by operation for detailed stats
    operation_stats = {}
    for trace in completed_traces:
        op_name = trace.operation_name
        if op_name not in operation_stats:
            operation_stats[op_name] = {"count": 0, "avg_duration": 0, "error_count": 0}
        
        operation_stats[op_name]["count"] += 1
        operation_stats[op_name]["avg_duration"] += trace.duration()
        if trace.status == "error":
            operation_stats[op_name]["error_count"] += 1
    
    # Calculate averages
    for op_name, stats in operation_stats.items():
        if stats["count"] > 0:
            stats["avg_duration"] /= stats["count"]
            stats["error_rate"] = stats["error_count"] / stats["count"]
    
    return {
        "total_traces": len(traces),
        "active_traces": len(active_traces),
        "completed_traces": len(completed_traces),
        "error_traces": len(error_traces),
        "error_rate": error_rate,
        "average_duration": avg_duration,
        "p95_duration": p95_duration,
        "p99_duration": p99_duration,
        "operation_stats": operation_stats
    }


# Global trace analyzer
trace_analyzer = TraceAnalyzer(tracing_manager)