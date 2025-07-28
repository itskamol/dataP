"""
Error aggregation and alerting system.
Implements requirements 2.1, 2.4: Error aggregation and alerting mechanisms.
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import hashlib

from src.domain.exceptions import AlertingError
from src.infrastructure.logging import get_logger, with_correlation_id


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class ErrorEvent:
    """Represents an error event."""
    error_id: str
    timestamp: datetime
    level: str
    message: str
    exception_type: str
    module: str
    function: str
    line_number: int
    correlation_id: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def get_signature(self) -> str:
        """Get unique signature for error grouping."""
        signature_data = f"{self.exception_type}:{self.module}:{self.function}:{self.line_number}"
        return hashlib.md5(signature_data.encode()).hexdigest()


@dataclass
class ErrorGroup:
    """Grouped errors with the same signature."""
    signature: str
    first_occurrence: datetime
    last_occurrence: datetime
    count: int
    error_type: str
    module: str
    function: str
    sample_message: str
    recent_events: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_event(self, event: ErrorEvent):
        """Add an error event to this group."""
        self.count += 1
        self.last_occurrence = event.timestamp
        self.recent_events.append(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signature': self.signature,
            'first_occurrence': self.first_occurrence.isoformat(),
            'last_occurrence': self.last_occurrence.isoformat(),
            'count': self.count,
            'error_type': self.error_type,
            'module': self.module,
            'function': self.function,
            'sample_message': self.sample_message,
            'recent_events': [event.to_dict() for event in self.recent_events]
        }


@dataclass
class Alert:
    """Alert to be sent through various channels."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    details: Dict[str, Any] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'details': self.details,
            'channels': [ch.value for ch in self.channels]
        }


@dataclass
class AlertRule:
    """Rule for generating alerts based on error patterns."""
    name: str
    condition: Callable[[ErrorGroup], bool]
    alert_level: AlertLevel
    channels: List[AlertChannel]
    cooldown_minutes: int = 60
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, error_group: ErrorGroup) -> bool:
        """Check if this rule should trigger for the given error group."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if (self.last_triggered and 
            datetime.now() - self.last_triggered < timedelta(minutes=self.cooldown_minutes)):
            return False
        
        return self.condition(error_group)
    
    def trigger(self):
        """Mark this rule as triggered."""
        self.last_triggered = datetime.now()


class ErrorAggregator:
    """Aggregates and analyzes error events."""
    
    def __init__(self, max_groups: int = 1000, max_events_per_group: int = 100):
        self.max_groups = max_groups
        self.max_events_per_group = max_events_per_group
        self._error_groups: Dict[str, ErrorGroup] = {}
        self._recent_events: deque = deque(maxlen=10000)
        self._lock = threading.RLock()
        self.logger = get_logger('error_aggregator')
    
    @with_correlation_id()
    def add_error_event(self, event: ErrorEvent):
        """Add an error event to the aggregator."""
        with self._lock:
            signature = event.get_signature()
            
            if signature not in self._error_groups:
                # Create new error group
                if len(self._error_groups) >= self.max_groups:
                    # Remove oldest group
                    oldest_signature = min(self._error_groups.keys(), 
                                         key=lambda k: self._error_groups[k].first_occurrence)
                    del self._error_groups[oldest_signature]
                
                self._error_groups[signature] = ErrorGroup(
                    signature=signature,
                    first_occurrence=event.timestamp,
                    last_occurrence=event.timestamp,
                    count=0,
                    error_type=event.exception_type,
                    module=event.module,
                    function=event.function,
                    sample_message=event.message
                )
            
            # Add event to group
            error_group = self._error_groups[signature]
            error_group.add_event(event)
            
            # Add to recent events
            self._recent_events.append(event)
            
            self.logger.debug(f"Added error event to group {signature}: {event.message}")
    
    def get_error_groups(self, time_window_minutes: int = 60) -> List[ErrorGroup]:
        """Get error groups within a time window."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            return [
                group for group in self._error_groups.values()
                if group.last_occurrence >= cutoff_time
            ]
    
    def get_error_statistics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get error statistics within a time window."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            recent_events = [
                event for event in self._recent_events
                if event.timestamp >= cutoff_time
            ]
            
            if not recent_events:
                return {
                    'total_errors': 0,
                    'unique_error_types': 0,
                    'error_rate_per_minute': 0.0,
                    'top_error_types': [],
                    'top_modules': []
                }
            
            # Count by error type
            error_type_counts = defaultdict(int)
            module_counts = defaultdict(int)
            
            for event in recent_events:
                error_type_counts[event.exception_type] += 1
                module_counts[event.module] += 1
            
            # Calculate error rate
            error_rate = len(recent_events) / max(time_window_minutes, 1)
            
            return {
                'total_errors': len(recent_events),
                'unique_error_types': len(error_type_counts),
                'error_rate_per_minute': error_rate,
                'top_error_types': sorted(error_type_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:10],
                'top_modules': sorted(module_counts.items(), 
                                    key=lambda x: x[1], reverse=True)[:10],
                'time_window_minutes': time_window_minutes,
                'timestamp': datetime.now().isoformat()
            }
    
    def clear_old_data(self, max_age_hours: int = 24):
        """Clear old error data."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Remove old groups
            old_signatures = [
                signature for signature, group in self._error_groups.items()
                if group.last_occurrence < cutoff_time
            ]
            
            for signature in old_signatures:
                del self._error_groups[signature]
            
            # Clear old recent events (deque handles this automatically)
            self.logger.info(f"Cleared {len(old_signatures)} old error groups")


class AlertManager:
    """Manages alert rules and delivery."""
    
    def __init__(self, error_aggregator: ErrorAggregator):
        self.error_aggregator = error_aggregator
        self._alert_rules: List[AlertRule] = []
        self._alert_handlers: Dict[AlertChannel, Callable[[Alert], None]] = {}
        self._lock = threading.RLock()
        self.logger = get_logger('alert_manager')
        
        # Register default alert handlers
        self._register_default_handlers()
        
        # Register default alert rules
        self._register_default_rules()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._stop_monitoring = threading.Event()
        self._monitoring_thread.start()
    
    def _register_default_handlers(self):
        """Register default alert handlers."""
        self._alert_handlers[AlertChannel.LOG] = self._log_alert_handler
        self._alert_handlers[AlertChannel.CONSOLE] = self._console_alert_handler
    
    def _register_default_rules(self):
        """Register default alert rules."""
        
        # High error rate rule
        self.add_alert_rule(AlertRule(
            name="high_error_rate",
            condition=lambda group: group.count > 10 and 
                     (datetime.now() - group.first_occurrence).total_seconds() < 300,  # 10 errors in 5 minutes
            alert_level=AlertLevel.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=30
        ))
        
        # Critical error rule
        self.add_alert_rule(AlertRule(
            name="critical_errors",
            condition=lambda group: any(event.level == 'CRITICAL' or event.level == 'ERROR' 
                                       for event in group.recent_events),
            alert_level=AlertLevel.ERROR,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            cooldown_minutes=15
        ))
        
        # Repeated error rule
        self.add_alert_rule(AlertRule(
            name="repeated_errors",
            condition=lambda group: group.count > 50,
            alert_level=AlertLevel.WARNING,
            channels=[AlertChannel.LOG],
            cooldown_minutes=60
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self._alert_rules.append(rule)
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        with self._lock:
            self._alert_rules = [rule for rule in self._alert_rules if rule.name != rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def register_alert_handler(self, channel: AlertChannel, handler: Callable[[Alert], None]):
        """Register an alert handler for a channel."""
        with self._lock:
            self._alert_handlers[channel] = handler
            self.logger.info(f"Registered alert handler for channel: {channel.value}")
    
    def send_alert(self, alert: Alert):
        """Send an alert through configured channels."""
        with self._lock:
            for channel in alert.channels:
                if channel in self._alert_handlers:
                    try:
                        self._alert_handlers[channel](alert)
                        self.logger.debug(f"Sent alert {alert.alert_id} via {channel.value}")
                    except Exception as e:
                        self.logger.error(f"Failed to send alert via {channel.value}: {str(e)}")
                else:
                    self.logger.warning(f"No handler registered for alert channel: {channel.value}")
    
    def _monitoring_loop(self):
        """Background monitoring loop to check alert rules."""
        while not self._stop_monitoring.is_set():
            try:
                # Check error groups against alert rules
                error_groups = self.error_aggregator.get_error_groups(time_window_minutes=60)
                
                with self._lock:
                    for group in error_groups:
                        for rule in self._alert_rules:
                            if rule.should_trigger(group):
                                alert = Alert(
                                    alert_id=f"{rule.name}_{group.signature}_{int(time.time())}",
                                    level=rule.alert_level,
                                    title=f"Alert: {rule.name}",
                                    message=f"Error group triggered alert rule '{rule.name}': "
                                           f"{group.error_type} in {group.module}.{group.function} "
                                           f"(occurred {group.count} times)",
                                    timestamp=datetime.now(),
                                    source="error_aggregator",
                                    details={
                                        'rule_name': rule.name,
                                        'error_group': group.to_dict()
                                    },
                                    channels=rule.channels
                                )
                                
                                self.send_alert(alert)
                                rule.trigger()
                
                # Sleep before next check
                self._stop_monitoring.wait(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {str(e)}")
                self._stop_monitoring.wait(60)
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._stop_monitoring.set()
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
    
    def _log_alert_handler(self, alert: Alert):
        """Handle alert by logging it."""
        log_level = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }.get(alert.level, self.logger.info)
        
        log_level(f"ALERT: {alert.title} - {alert.message}",
                 extra={'alert_id': alert.alert_id, 'alert_details': alert.details})
    
    def _console_alert_handler(self, alert: Alert):
        """Handle alert by printing to console."""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"ALERT [{alert.level.value.upper()}] - {timestamp}")
        print(f"Title: {alert.title}")
        print(f"Message: {alert.message}")
        print(f"Source: {alert.source}")
        if alert.details:
            print(f"Details: {json.dumps(alert.details, indent=2)}")
        print(f"{'='*60}\n")


class ErrorLoggingHandler:
    """Custom logging handler that feeds errors to the aggregator."""
    
    def __init__(self, error_aggregator: ErrorAggregator):
        self.error_aggregator = error_aggregator
        self.logger = get_logger('error_logging_handler')
    
    def handle_log_record(self, record):
        """Handle a log record and extract error information."""
        if record.levelno >= 40:  # ERROR and CRITICAL levels
            try:
                error_event = ErrorEvent(
                    error_id=f"{record.module}_{record.funcName}_{record.lineno}_{int(time.time())}",
                    timestamp=datetime.fromtimestamp(record.created),
                    level=record.levelname,
                    message=record.getMessage(),
                    exception_type=getattr(record, 'exc_info', [None, None, None])[0].__name__ 
                                 if getattr(record, 'exc_info', None) and record.exc_info[0] 
                                 else 'UnknownError',
                    module=record.module,
                    function=record.funcName,
                    line_number=record.lineno,
                    correlation_id=getattr(record, 'correlation_id', 'unknown'),
                    stack_trace=record.exc_text if hasattr(record, 'exc_text') else None,
                    context={
                        'pathname': record.pathname,
                        'process': record.process,
                        'thread': record.thread,
                        'thread_name': record.threadName
                    }
                )
                
                self.error_aggregator.add_error_event(error_event)
                
            except Exception as e:
                self.logger.error(f"Failed to process error event: {str(e)}")


# Global instances
_error_aggregator: Optional[ErrorAggregator] = None
_alert_manager: Optional[AlertManager] = None
_error_logging_handler: Optional[ErrorLoggingHandler] = None
_aggregation_lock = threading.Lock()


def get_error_aggregator() -> ErrorAggregator:
    """Get global error aggregator instance."""
    global _error_aggregator
    
    if _error_aggregator is None:
        with _aggregation_lock:
            if _error_aggregator is None:
                _error_aggregator = ErrorAggregator()
    
    return _error_aggregator


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    
    if _alert_manager is None:
        with _aggregation_lock:
            if _alert_manager is None:
                _alert_manager = AlertManager(get_error_aggregator())
    
    return _alert_manager


def get_error_logging_handler() -> ErrorLoggingHandler:
    """Get global error logging handler instance."""
    global _error_logging_handler
    
    if _error_logging_handler is None:
        with _aggregation_lock:
            if _error_logging_handler is None:
                _error_logging_handler = ErrorLoggingHandler(get_error_aggregator())
    
    return _error_logging_handler


def setup_error_monitoring():
    """Set up error monitoring integration with logging system."""
    import logging
    
    # Get the error logging handler
    handler = get_error_logging_handler()
    
    # Create a custom logging handler that feeds to error aggregator
    class ErrorAggregatorHandler(logging.Handler):
        def emit(self, record):
            handler.handle_log_record(record)
    
    # Add the handler to the root logger
    error_handler = ErrorAggregatorHandler()
    error_handler.setLevel(logging.ERROR)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(error_handler)
    
    # Initialize alert manager to start monitoring
    get_alert_manager()
    
    get_logger('error_monitoring').info("Error monitoring system initialized")