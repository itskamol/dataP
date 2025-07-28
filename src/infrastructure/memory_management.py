"""
Memory pooling and garbage collection optimization for performance.
Implements requirement 3.2, 3.3: Memory management and optimization.
"""

import gc
import threading
import time
import weakref
from typing import Dict, List, Optional, Any, TypeVar, Generic, Callable
from dataclasses import dataclass
from collections import deque
import psutil
import os
from contextlib import contextmanager

from src.infrastructure.logging import get_logger


T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    gc_collections: Dict[int, int]
    pool_stats: Dict[str, Dict[str, int]]


class ObjectPool(Generic[T]):
    """Generic object pool for reusing expensive objects."""
    
    def __init__(self, factory: Callable[[], T], max_size: int = 100, 
                 reset_func: Optional[Callable[[T], None]] = None):
        """
        Initialize object pool.
        
        Args:
            factory: Function to create new objects
            max_size: Maximum pool size
            reset_func: Optional function to reset objects before reuse
        """
        self.factory = factory
        self.max_size = max_size
        self.reset_func = reset_func
        self._pool: deque = deque()
        self._lock = threading.RLock()
        self._created_count = 0
        self._reused_count = 0
        self.logger = get_logger('object_pool')
    
    def acquire(self) -> T:
        """Acquire object from pool or create new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._reused_count += 1
                return obj
            else:
                obj = self.factory()
                self._created_count += 1
                return obj
    
    def release(self, obj: T):
        """Release object back to pool."""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object if reset function provided
                if self.reset_func:
                    try:
                        self.reset_func(obj)
                    except Exception as e:
                        self.logger.warning(f"Failed to reset object: {str(e)}")
                        return  # Don't return to pool if reset failed
                
                self._pool.append(obj)
    
    def clear(self):
        """Clear all objects from pool."""
        with self._lock:
            self._pool.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'reuse_rate': (self._reused_count / (self._created_count + self._reused_count) * 100) 
                             if (self._created_count + self._reused_count) > 0 else 0.0
            }


class DataFramePool:
    """Specialized pool for pandas DataFrames."""
    
    def __init__(self, max_size: int = 50):
        import pandas as pd
        
        def create_dataframe():
            return pd.DataFrame()
        
        def reset_dataframe(df):
            df.drop(df.index, inplace=True)
            df.drop(df.columns, axis=1, inplace=True)
        
        self.pool = ObjectPool(create_dataframe, max_size, reset_dataframe)
        self.logger = get_logger('dataframe_pool')
    
    @contextmanager
    def get_dataframe(self):
        """Context manager for acquiring and releasing DataFrames."""
        df = self.pool.acquire()
        try:
            yield df
        finally:
            self.pool.release(df)
    
    def get_stats(self) -> Dict[str, int]:
        """Get DataFrame pool statistics."""
        return self.pool.get_stats()


class StringPool:
    """Pool for frequently used strings to reduce memory fragmentation."""
    
    def __init__(self, max_size: int = 1000):
        self._pool: Dict[str, str] = {}
        self._access_count: Dict[str, int] = {}
        self.max_size = max_size
        self._lock = threading.RLock()
        self.logger = get_logger('string_pool')
    
    def intern(self, string: str) -> str:
        """Intern string to reduce memory usage."""
        with self._lock:
            if string in self._pool:
                self._access_count[string] += 1
                return self._pool[string]
            
            # If pool is full, remove least accessed strings
            if len(self._pool) >= self.max_size:
                self._evict_least_used()
            
            # Add to pool
            self._pool[string] = string
            self._access_count[string] = 1
            return string
    
    def _evict_least_used(self):
        """Evict least used strings from pool."""
        if not self._pool:
            return
        
        # Find strings with lowest access count
        min_access = min(self._access_count.values())
        to_remove = [s for s, count in self._access_count.items() if count == min_access]
        
        # Remove up to 10% of pool size
        remove_count = max(1, len(self._pool) // 10)
        for string in to_remove[:remove_count]:
            del self._pool[string]
            del self._access_count[string]
    
    def clear(self):
        """Clear string pool."""
        with self._lock:
            self._pool.clear()
            self._access_count.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get string pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'total_accesses': sum(self._access_count.values()),
                'unique_strings': len(self._pool)
            }


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Memory usage percentage to trigger warning
            critical_threshold: Memory usage percentage to trigger cleanup
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = get_logger('memory_monitor')
        self._callbacks: List[Callable[[], None]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add callback to be called when memory usage is high."""
        self._callbacks.append(callback)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # GC statistics
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]
        
        return MemoryStats(
            total_memory_mb=memory.total / 1024 / 1024,
            available_memory_mb=memory.available / 1024 / 1024,
            used_memory_mb=memory.used / 1024 / 1024,
            memory_percent=memory.percent,
            process_memory_mb=process_memory,
            gc_collections=gc_stats,
            pool_stats={}  # Will be populated by MemoryManager
        )
    
    def check_memory_usage(self) -> bool:
        """Check memory usage and trigger cleanup if needed."""
        stats = self.get_memory_stats()
        
        if stats.memory_percent >= self.critical_threshold:
            self.logger.warning(f"Critical memory usage: {stats.memory_percent:.1f}%")
            self._trigger_cleanup()
            return True
        elif stats.memory_percent >= self.warning_threshold:
            self.logger.info(f"High memory usage: {stats.memory_percent:.1f}%")
            return False
        
        return False
    
    def _trigger_cleanup(self):
        """Trigger cleanup callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback failed: {str(e)}")
    
    def start_monitoring(self, interval: float = 30.0):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while not self._stop_event.wait(interval):
            try:
                self.check_memory_usage()
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {str(e)}")


class GarbageCollectionOptimizer:
    """Optimize garbage collection for better performance."""
    
    def __init__(self):
        self.logger = get_logger('gc_optimizer')
        self._original_thresholds = gc.get_threshold()
        self._gc_stats = {'collections': 0, 'collected': 0}
    
    def optimize_for_performance(self):
        """Optimize GC settings for performance-critical operations."""
        # Increase thresholds to reduce GC frequency during heavy processing
        gc.set_threshold(2000, 20, 20)  # Increased from default (700, 10, 10)
        self.logger.info("GC optimized for performance")
    
    def optimize_for_memory(self):
        """Optimize GC settings for memory-constrained operations."""
        # Decrease thresholds to collect more frequently
        gc.set_threshold(500, 5, 5)
        self.logger.info("GC optimized for memory")
    
    def restore_defaults(self):
        """Restore original GC thresholds."""
        gc.set_threshold(*self._original_thresholds)
        self.logger.info("GC thresholds restored to defaults")
    
    def force_collection(self) -> int:
        """Force garbage collection and return number of collected objects."""
        collected = gc.collect()
        self._gc_stats['collections'] += 1
        self._gc_stats['collected'] += collected
        self.logger.debug(f"Forced GC collected {collected} objects")
        return collected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GC statistics."""
        return {
            'current_thresholds': gc.get_threshold(),
            'original_thresholds': self._original_thresholds,
            'gc_counts': gc.get_count(),
            'forced_collections': self._gc_stats['collections'],
            'total_collected': self._gc_stats['collected']
        }


class MemoryManager:
    """Central memory management system."""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        self.logger = get_logger('memory_manager')
        
        # Initialize components
        self.monitor = MemoryMonitor(warning_threshold, critical_threshold)
        self.gc_optimizer = GarbageCollectionOptimizer()
        
        # Initialize pools
        self.dataframe_pool = DataFramePool()
        self.string_pool = StringPool()
        
        # Register cleanup callbacks
        self.monitor.add_cleanup_callback(self._cleanup_pools)
        self.monitor.add_cleanup_callback(self._force_gc)
        
        self.logger.info("Memory manager initialized")
    
    def _cleanup_pools(self):
        """Clean up object pools."""
        self.dataframe_pool.pool.clear()
        self.string_pool.clear()
        self.logger.info("Object pools cleared")
    
    def _force_gc(self):
        """Force garbage collection."""
        collected = self.gc_optimizer.force_collection()
        self.logger.info(f"Forced GC collected {collected} objects")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        memory_stats = self.monitor.get_memory_stats()
        
        # Add pool statistics
        memory_stats.pool_stats = {
            'dataframe_pool': self.dataframe_pool.get_stats(),
            'string_pool': self.string_pool.get_stats()
        }
        
        return {
            'memory_stats': memory_stats.__dict__,
            'gc_stats': self.gc_optimizer.get_stats()
        }
    
    def optimize_for_large_dataset(self):
        """Optimize memory settings for large dataset processing."""
        self.gc_optimizer.optimize_for_performance()
        self.logger.info("Memory optimized for large dataset processing")
    
    def optimize_for_memory_constrained(self):
        """Optimize memory settings for memory-constrained environments."""
        self.gc_optimizer.optimize_for_memory()
        self.logger.info("Memory optimized for constrained environment")
    
    def start_monitoring(self, interval: float = 30.0):
        """Start memory monitoring."""
        self.monitor.start_monitoring(interval)
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitor.stop_monitoring()
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        self.logger.warning("Performing emergency memory cleanup")
        self._cleanup_pools()
        self._force_gc()
        
        # Additional cleanup
        import pandas as pd
        if hasattr(pd, '_cache'):
            pd._cache.clear()
    
    @contextmanager
    def memory_optimized_context(self, optimize_for: str = 'performance'):
        """Context manager for memory-optimized operations."""
        if optimize_for == 'performance':
            self.optimize_for_large_dataset()
        else:
            self.optimize_for_memory_constrained()
        
        try:
            yield self
        finally:
            self.gc_optimizer.restore_defaults()


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None
_memory_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    
    if _memory_manager is None:
        with _memory_manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager()
    
    return _memory_manager


def cleanup_memory():
    """Perform global memory cleanup."""
    manager = get_memory_manager()
    manager.emergency_cleanup()