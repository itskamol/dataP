"""
Health check system for system monitoring.
Implements requirements 6.4, 8.4: Health check endpoints for system monitoring.
"""

import time
import threading
import sqlite3
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from src.domain.exceptions import HealthCheckError
from src.infrastructure.logging import get_logger, with_correlation_id
from src.infrastructure.metrics import get_metrics_collector
from src.infrastructure.caching import get_cache_manager


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'checks': [check.to_dict() for check in self.checks],
            'summary': self.summary
        }


class HealthChecker:
    """Individual health check implementation."""
    
    def __init__(self, name: str, check_func: Callable[[], HealthCheckResult], 
                 interval_seconds: int = 60, timeout_seconds: int = 30):
        self.name = name
        self.check_func = check_func
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.last_result: Optional[HealthCheckResult] = None
        self.logger = get_logger(f'health_check.{name}')
    
    def run_check(self) -> HealthCheckResult:
        """Run the health check."""
        start_time = time.time()
        
        try:
            result = self.check_func()
            result.duration_ms = (time.time() - start_time) * 1000
            result.timestamp = datetime.now()
            self.last_result = result
            
            self.logger.debug(f"Health check {self.name} completed: {result.status.value}")
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
            self.last_result = result
            
            self.logger.error(f"Health check {self.name} failed: {str(e)}")
            return result


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.checkers: Dict[str, HealthChecker] = {}
        self.logger = get_logger('health_monitor')
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Register default health checks
        self._register_default_checks()
        
        # Start monitoring
        self.start_monitoring()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # System resource checks
        self.register_check("system_memory", self._check_system_memory, interval_seconds=30)
        self.register_check("system_cpu", self._check_system_cpu, interval_seconds=30)
        self.register_check("system_disk", self._check_system_disk, interval_seconds=60)
        
        # Application checks
        self.register_check("database_connection", self._check_database_connection, interval_seconds=60)
        self.register_check("cache_health", self._check_cache_health, interval_seconds=60)
        self.register_check("metrics_collection", self._check_metrics_collection, interval_seconds=60)
        
        # File system checks
        self.register_check("upload_directory", self._check_upload_directory, interval_seconds=120)
        self.register_check("logs_directory", self._check_logs_directory, interval_seconds=120)
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult], 
                      interval_seconds: int = 60, timeout_seconds: int = 30):
        """Register a new health check."""
        with self._lock:
            self.checkers[name] = HealthChecker(name, check_func, interval_seconds, timeout_seconds)
            self.logger.info(f"Registered health check: {name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        with self._lock:
            if name in self.checkers:
                del self.checkers[name]
                self.logger.info(f"Unregistered health check: {name}")
    
    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        with self._lock:
            if name not in self.checkers:
                return None
            
            return self.checkers[name].run_check()
    
    def run_all_checks(self) -> SystemHealth:
        """Run all health checks and return system health status."""
        with self._lock:
            results = []
            
            for checker in self.checkers.values():
                result = checker.run_check()
                results.append(result)
            
            # Determine overall system status
            overall_status = self._determine_overall_status(results)
            
            # Create summary
            summary = self._create_summary(results)
            
            system_health = SystemHealth(
                status=overall_status,
                timestamp=datetime.now(),
                checks=results,
                summary=summary
            )
            
            self.logger.info(f"System health check completed: {overall_status.value}")
            return system_health
    
    def get_health_status(self) -> SystemHealth:
        """Get current health status using cached results."""
        with self._lock:
            results = []
            
            for checker in self.checkers.values():
                if checker.last_result:
                    results.append(checker.last_result)
                else:
                    # Run check if no cached result
                    result = checker.run_check()
                    results.append(result)
            
            overall_status = self._determine_overall_status(results)
            summary = self._create_summary(results)
            
            return SystemHealth(
                status=overall_status,
                timestamp=datetime.now(),
                checks=results,
                summary=summary
            )
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                current_time = time.time()
                
                with self._lock:
                    for checker in self.checkers.values():
                        # Check if it's time to run this check
                        if (not checker.last_result or 
                            (current_time - checker.last_result.timestamp.timestamp()) >= checker.interval_seconds):
                            
                            try:
                                checker.run_check()
                            except Exception as e:
                                self.logger.error(f"Error running health check {checker.name}: {str(e)}")
                
                # Sleep for a short interval before checking again
                self._stop_monitoring.wait(10)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {str(e)}")
                self._stop_monitoring.wait(30)
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from individual check results."""
        if not results:
            return HealthStatus.UNKNOWN
        
        # Count status levels
        status_counts = {status: 0 for status in HealthStatus}
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status based on priority
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            return HealthStatus.WARNING
        elif status_counts[HealthStatus.HEALTHY] == len(results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _create_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Create summary statistics from health check results."""
        if not results:
            return {}
        
        status_counts = {status.value: 0 for status in HealthStatus}
        total_duration = 0
        
        for result in results:
            status_counts[result.status.value] += 1
            total_duration += result.duration_ms
        
        return {
            'total_checks': len(results),
            'status_counts': status_counts,
            'average_duration_ms': total_duration / len(results),
            'last_updated': datetime.now().isoformat()
        }
    
    # Default health check implementations
    
    def _check_system_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        if not PSUTIL_AVAILABLE:
            return HealthCheckResult(
                name="system_memory",
                status=HealthStatus.WARNING,
                message="Memory monitoring unavailable (psutil not installed)",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'psutil_available': False}
            )
        
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {usage_percent:.1f}%"
            elif usage_percent > 80:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_memory",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    'usage_percent': usage_percent,
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_memory",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_system_cpu(self) -> HealthCheckResult:
        """Check system CPU usage."""
        if not PSUTIL_AVAILABLE:
            return HealthCheckResult(
                name="system_cpu",
                status=HealthStatus.WARNING,
                message="CPU monitoring unavailable (psutil not installed)",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'psutil_available': False}
            )
        
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_cpu",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    'usage_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_cpu",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check CPU: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_system_disk(self) -> HealthCheckResult:
        """Check system disk usage."""
        if not PSUTIL_AVAILABLE:
            return HealthCheckResult(
                name="system_disk",
                status=HealthStatus.WARNING,
                message="Disk monitoring unavailable (psutil not installed)",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'psutil_available': False}
            )
        
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {usage_percent:.1f}%"
            elif usage_percent > 80:
                status = HealthStatus.WARNING
                message = f"Disk usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_disk",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    'usage_percent': usage_percent,
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_disk",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_database_connection(self) -> HealthCheckResult:
        """Check database connection health."""
        try:
            # Test SQLite connection
            db_path = "progress.db"
            with sqlite3.connect(db_path, timeout=5) as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()
            
            return HealthCheckResult(
                name="database_connection",
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'database_path': db_path}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_cache_health(self) -> HealthCheckResult:
        """Check cache system health."""
        try:
            cache_manager = get_cache_manager()
            stats = cache_manager.get_overall_stats()
            memory_usage = cache_manager.get_memory_usage()
            
            total_memory_mb = memory_usage['total_bytes'] / (1024 * 1024)
            
            if total_memory_mb > 500:  # 500MB threshold
                status = HealthStatus.WARNING
                message = f"Cache memory usage high: {total_memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Cache system healthy: {total_memory_mb:.1f}MB used"
            
            return HealthCheckResult(
                name="cache_health",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    'memory_usage_mb': total_memory_mb,
                    'similarity_cache_hits': stats['similarity'].hits,
                    'similarity_cache_hit_rate': stats['similarity'].hit_rate,
                    'normalization_cache_hits': stats['normalization'].hits,
                    'normalization_cache_hit_rate': stats['normalization'].hit_rate
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="cache_health",
                status=HealthStatus.WARNING,
                message=f"Cache health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_metrics_collection(self) -> HealthCheckResult:
        """Check metrics collection system health."""
        try:
            metrics_collector = get_metrics_collector()
            system_metrics = metrics_collector.get_system_metrics()
            
            active_operations = system_metrics.get('active_operations', 0)
            total_metrics = system_metrics.get('total_metrics_stored', 0)
            
            if active_operations > 100:
                status = HealthStatus.WARNING
                message = f"High number of active operations: {active_operations}"
            elif total_metrics > 100000:
                status = HealthStatus.WARNING
                message = f"High number of stored metrics: {total_metrics}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Metrics collection healthy: {active_operations} active ops, {total_metrics} metrics"
            
            return HealthCheckResult(
                name="metrics_collection",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,
                details=system_metrics
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="metrics_collection",
                status=HealthStatus.WARNING,
                message=f"Metrics collection check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_upload_directory(self) -> HealthCheckResult:
        """Check upload directory health."""
        try:
            upload_dir = Path("uploads")
            
            if not upload_dir.exists():
                return HealthCheckResult(
                    name="upload_directory",
                    status=HealthStatus.CRITICAL,
                    message="Upload directory does not exist",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    details={'path': str(upload_dir)}
                )
            
            # Check if directory is writable
            test_file = upload_dir / ".health_check"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception:
                return HealthCheckResult(
                    name="upload_directory",
                    status=HealthStatus.CRITICAL,
                    message="Upload directory is not writable",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    details={'path': str(upload_dir)}
                )
            
            # Count files and check disk usage
            file_count = len(list(upload_dir.glob("*")))
            
            return HealthCheckResult(
                name="upload_directory",
                status=HealthStatus.HEALTHY,
                message=f"Upload directory healthy: {file_count} files",
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    'path': str(upload_dir),
                    'file_count': file_count,
                    'exists': True,
                    'writable': True
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="upload_directory",
                status=HealthStatus.CRITICAL,
                message=f"Upload directory check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )
    
    def _check_logs_directory(self) -> HealthCheckResult:
        """Check logs directory health."""
        try:
            logs_dir = Path("logs")
            
            if not logs_dir.exists():
                return HealthCheckResult(
                    name="logs_directory",
                    status=HealthStatus.WARNING,
                    message="Logs directory does not exist",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    details={'path': str(logs_dir)}
                )
            
            # Check log files
            log_files = list(logs_dir.glob("*.log"))
            total_size = sum(f.stat().st_size for f in log_files)
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > 1000:  # 1GB threshold
                status = HealthStatus.WARNING
                message = f"Log files large: {total_size_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Logs directory healthy: {len(log_files)} files, {total_size_mb:.1f}MB"
            
            return HealthCheckResult(
                name="logs_directory",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0,
                details={
                    'path': str(logs_dir),
                    'log_file_count': len(log_files),
                    'total_size_mb': total_size_mb
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="logs_directory",
                status=HealthStatus.WARNING,
                message=f"Logs directory check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=0,
                details={'error': str(e)}
            )


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None
_health_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        with _health_lock:
            if _health_monitor is None:
                _health_monitor = HealthMonitor()
    
    return _health_monitor