"""
Caching infrastructure for performance optimization.
Implements requirement 3.3: Caching mechanisms for similarity calculations.
"""

import threading
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.domain.exceptions import ResourceError
from src.infrastructure.logging import get_logger


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    memory_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate percentage."""
        return 100.0 - self.hit_rate


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self.logger = get_logger('caching')
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating access order."""
        with self._lock:
            self._stats.total_requests += 1
            
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._stats.hits += 1
                return value
            else:
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache, evicting if necessary."""
        with self._lock:
            # Estimate memory usage
            estimated_size = self._estimate_size(key, value)
            
            # Check if single item exceeds memory limit
            if estimated_size > self.max_memory_bytes:
                self.logger.warning(f"Item too large for cache: {estimated_size} bytes")
                return False
            
            # Remove existing key if present
            if key in self._cache:
                del self._cache[key]
            
            # Evict items if necessary
            while (len(self._cache) >= self.max_size or 
                   self._stats.memory_usage_bytes + estimated_size > self.max_memory_bytes):
                if not self._cache:
                    break
                self._evict_lru()
            
            # Add new item
            self._cache[key] = value
            self._stats.memory_usage_bytes += estimated_size
            return True
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._cache:
            key, value = self._cache.popitem(last=False)
            self._stats.memory_usage_bytes -= self._estimate_size(key, value)
            self._stats.evictions += 1
    
    def _estimate_size(self, key: str, value: Any) -> int:
        """Estimate memory size of key-value pair."""
        try:
            import sys
            return sys.getsizeof(key) + sys.getsizeof(value)
        except:
            # Fallback estimation
            return len(str(key)) + len(str(value)) + 100
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.memory_usage_bytes = 0
            self.logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_requests=self._stats.total_requests,
                memory_usage_bytes=self._stats.memory_usage_bytes
            )


class TTLCache:
    """Time-To-Live cache with automatic expiration."""
    
    def __init__(self, default_ttl_seconds: int = 3600, max_size: int = 10000):
        self.default_ttl = default_ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self.logger = get_logger('caching')
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            self._stats.total_requests += 1
            
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.now() < expiry:
                    self._stats.hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
                    self._stats.memory_usage_bytes -= self._estimate_size(key, value)
            
            self._stats.misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in cache with TTL."""
        with self._lock:
            ttl = ttl_seconds or self.default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl)
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            # Remove existing if present
            if key in self._cache:
                old_value = self._cache[key][0]
                self._stats.memory_usage_bytes -= self._estimate_size(key, old_value)
            
            self._cache[key] = (value, expiry)
            self._stats.memory_usage_bytes += self._estimate_size(key, value)
            return True
    
    def _evict_oldest(self):
        """Evict oldest entry."""
        if self._cache:
            # Find entry with earliest expiry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            value = self._cache[oldest_key][0]
            del self._cache[oldest_key]
            self._stats.memory_usage_bytes -= self._estimate_size(oldest_key, value)
            self._stats.evictions += 1
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                with self._lock:
                    now = datetime.now()
                    expired_keys = [
                        key for key, (_, expiry) in self._cache.items()
                        if now >= expiry
                    ]
                    
                    for key in expired_keys:
                        value = self._cache[key][0]
                        del self._cache[key]
                        self._stats.memory_usage_bytes -= self._estimate_size(key, value)
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
    
    def _estimate_size(self, key: str, value: Any) -> int:
        """Estimate memory size of key-value pair."""
        try:
            import sys
            return sys.getsizeof(key) + sys.getsizeof(value)
        except:
            return len(str(key)) + len(str(value)) + 100
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.memory_usage_bytes = 0
            self.logger.info("TTL cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_requests=self._stats.total_requests,
                memory_usage_bytes=self._stats.memory_usage_bytes
            )


class SimilarityCache:
    """Specialized cache for similarity calculations."""
    
    def __init__(self, max_size: int = 50000, max_memory_mb: int = 200):
        self.lru_cache = LRUCache(max_size, max_memory_mb)
        self.logger = get_logger('caching')
    
    def get_similarity(self, text1: str, text2: str, algorithm: str, 
                      normalization: bool = False) -> Optional[float]:
        """Get cached similarity score."""
        cache_key = self._create_key(text1, text2, algorithm, normalization)
        return self.lru_cache.get(cache_key)
    
    def put_similarity(self, text1: str, text2: str, algorithm: str, 
                      similarity: float, normalization: bool = False) -> bool:
        """Cache similarity score."""
        cache_key = self._create_key(text1, text2, algorithm, normalization)
        return self.lru_cache.put(cache_key, similarity)
    
    def _create_key(self, text1: str, text2: str, algorithm: str, 
                   normalization: bool) -> str:
        """Create cache key for similarity calculation."""
        # Ensure consistent ordering for symmetric operations
        if text1 > text2:
            text1, text2 = text2, text1
        
        norm_suffix = "_norm" if normalization else ""
        return f"{algorithm}{norm_suffix}:{hash(text1)}:{hash(text2)}"
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.lru_cache.get_stats()
    
    def clear(self):
        """Clear similarity cache."""
        self.lru_cache.clear()


class NormalizationCache:
    """Cache for text normalization results."""
    
    def __init__(self, max_size: int = 20000):
        self.ttl_cache = TTLCache(default_ttl_seconds=7200, max_size=max_size)  # 2 hours TTL
        self.logger = get_logger('caching')
    
    def get_normalized(self, text: str) -> Optional[str]:
        """Get cached normalized text."""
        return self.ttl_cache.get(text)
    
    def put_normalized(self, text: str, normalized: str) -> bool:
        """Cache normalized text."""
        return self.ttl_cache.put(text, normalized)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.ttl_cache.get_stats()
    
    def clear(self):
        """Clear normalization cache."""
        self.ttl_cache.clear()


class CacheManager:
    """Central manager for all caching operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize caches
        self.similarity_cache = SimilarityCache(
            max_size=config.get('similarity_cache_size', 50000),
            max_memory_mb=config.get('similarity_cache_memory_mb', 200)
        )
        
        self.normalization_cache = NormalizationCache(
            max_size=config.get('normalization_cache_size', 20000)
        )
        
        self.logger = get_logger('caching')
        self.logger.info("Cache manager initialized")
    
    def get_similarity(self, text1: str, text2: str, algorithm: str, 
                      normalization: bool = False) -> Optional[float]:
        """Get cached similarity score."""
        return self.similarity_cache.get_similarity(text1, text2, algorithm, normalization)
    
    def put_similarity(self, text1: str, text2: str, algorithm: str, 
                      similarity: float, normalization: bool = False) -> bool:
        """Cache similarity score."""
        return self.similarity_cache.put_similarity(text1, text2, algorithm, similarity, normalization)
    
    def get_normalized_text(self, text: str) -> Optional[str]:
        """Get cached normalized text."""
        return self.normalization_cache.get_normalized(text)
    
    def put_normalized_text(self, text: str, normalized: str) -> bool:
        """Cache normalized text."""
        return self.normalization_cache.put_normalized(text, normalized)
    
    def get_overall_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {
            'similarity': self.similarity_cache.get_stats(),
            'normalization': self.normalization_cache.get_stats()
        }
    
    def clear_all(self):
        """Clear all caches."""
        self.similarity_cache.clear()
        self.normalization_cache.clear()
        self.logger.info("All caches cleared")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage for all caches."""
        stats = self.get_overall_stats()
        return {
            'similarity_bytes': stats['similarity'].memory_usage_bytes,
            'normalization_bytes': stats['normalization'].memory_usage_bytes,
            'total_bytes': stats['similarity'].memory_usage_bytes + stats['normalization'].memory_usage_bytes
        }
    
    def optimize_memory(self, target_memory_mb: int):
        """Optimize memory usage by clearing caches if needed."""
        current_usage = self.get_memory_usage()
        target_bytes = target_memory_mb * 1024 * 1024
        
        if current_usage['total_bytes'] > target_bytes:
            self.logger.info(f"Memory usage {current_usage['total_bytes']} exceeds target {target_bytes}, clearing caches")
            
            # Clear normalization cache first (usually smaller impact)
            if current_usage['normalization_bytes'] > 0:
                self.normalization_cache.clear()
            
            # Check if we still need to clear similarity cache
            remaining_usage = self.similarity_cache.get_stats().memory_usage_bytes
            if remaining_usage > target_bytes:
                self.similarity_cache.clear()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        with _cache_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager(config)
    
    return _cache_manager


def clear_global_caches():
    """Clear all global caches."""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear_all()