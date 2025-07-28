"""
LRU caching system for matching algorithms to improve performance.
Implements requirements 3.3, 4.3: Caching for similarity calculations and normalized text.
"""

from typing import Dict, Any, Optional, Tuple, Hashable
from functools import lru_cache
import hashlib
import time
from collections import OrderedDict
import threading

from ...infrastructure.logging import get_logger


class MatchingCache:
    """
    Thread-safe LRU cache for matching operations with performance tracking.
    Supports separate caches for similarity calculations and normalized text.
    """
    
    def __init__(self, similarity_cache_size: int = 10000, 
                 normalization_cache_size: int = 5000):
        """
        Initialize the matching cache.
        
        Args:
            similarity_cache_size: Maximum size of similarity cache
            normalization_cache_size: Maximum size of normalization cache
        """
        self.logger = get_logger('matching.cache')
        
        # Thread locks for thread safety
        self._similarity_lock = threading.RLock()
        self._normalization_lock = threading.RLock()
        
        # LRU caches using OrderedDict
        self._similarity_cache = OrderedDict()
        self._normalization_cache = OrderedDict()
        
        # Cache size limits
        self._similarity_cache_size = similarity_cache_size
        self._normalization_cache_size = normalization_cache_size
        
        # Performance tracking
        self._similarity_hits = 0
        self._similarity_misses = 0
        self._normalization_hits = 0
        self._normalization_misses = 0
        
        # Cache creation time for statistics
        self._created_at = time.time()
    
    def _generate_cache_key(self, *args) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Arguments to create key from
            
        Returns:
            Cache key string
        """
        # Create a string representation of all arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, dict):
                # Sort dict items for consistent keys
                sorted_items = sorted(arg.items())
                key_parts.append(str(sorted_items))
            else:
                key_parts.append(str(arg))
        
        # Create hash for consistent key length
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get_similarity(self, text1: str, text2: str, algorithm: str, 
                      config: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Get cached similarity score.
        
        Args:
            text1: First text
            text2: Second text
            algorithm: Algorithm name
            config: Algorithm configuration
            
        Returns:
            Cached similarity score or None if not found
        """
        cache_key = self._generate_cache_key(text1, text2, algorithm, config or {})
        
        with self._similarity_lock:
            if cache_key in self._similarity_cache:
                # Move to end (most recently used)
                value = self._similarity_cache.pop(cache_key)
                self._similarity_cache[cache_key] = value
                self._similarity_hits += 1
                return value
            else:
                self._similarity_misses += 1
                return None
    
    def set_similarity(self, text1: str, text2: str, algorithm: str, 
                      similarity: float, config: Optional[Dict[str, Any]] = None):
        """
        Cache similarity score.
        
        Args:
            text1: First text
            text2: Second text
            algorithm: Algorithm name
            similarity: Similarity score to cache
            config: Algorithm configuration
        """
        cache_key = self._generate_cache_key(text1, text2, algorithm, config or {})
        
        with self._similarity_lock:
            # Remove oldest items if cache is full
            while len(self._similarity_cache) >= self._similarity_cache_size:
                self._similarity_cache.popitem(last=False)
            
            self._similarity_cache[cache_key] = similarity
    
    def get_normalized_text(self, text: str, algorithm: str, 
                           config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get cached normalized text.
        
        Args:
            text: Original text
            algorithm: Algorithm name
            config: Algorithm configuration
            
        Returns:
            Cached normalized text or None if not found
        """
        cache_key = self._generate_cache_key(text, algorithm, config or {})
        
        with self._normalization_lock:
            if cache_key in self._normalization_cache:
                # Move to end (most recently used)
                value = self._normalization_cache.pop(cache_key)
                self._normalization_cache[cache_key] = value
                self._normalization_hits += 1
                return value
            else:
                self._normalization_misses += 1
                return None
    
    def set_normalized_text(self, text: str, algorithm: str, normalized: str,
                           config: Optional[Dict[str, Any]] = None):
        """
        Cache normalized text.
        
        Args:
            text: Original text
            algorithm: Algorithm name
            normalized: Normalized text to cache
            config: Algorithm configuration
        """
        cache_key = self._generate_cache_key(text, algorithm, config or {})
        
        with self._normalization_lock:
            # Remove oldest items if cache is full
            while len(self._normalization_cache) >= self._normalization_cache_size:
                self._normalization_cache.popitem(last=False)
            
            self._normalization_cache[cache_key] = normalized
    
    def clear_similarity_cache(self):
        """Clear the similarity cache."""
        with self._similarity_lock:
            self._similarity_cache.clear()
            self.logger.info("Similarity cache cleared")
    
    def clear_normalization_cache(self):
        """Clear the normalization cache."""
        with self._normalization_lock:
            self._normalization_cache.clear()
            self.logger.info("Normalization cache cleared")
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.clear_similarity_cache()
        self.clear_normalization_cache()
        self.logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._similarity_lock, self._normalization_lock:
            similarity_total = self._similarity_hits + self._similarity_misses
            normalization_total = self._normalization_hits + self._normalization_misses
            
            similarity_hit_rate = (
                self._similarity_hits / similarity_total * 100 
                if similarity_total > 0 else 0.0
            )
            
            normalization_hit_rate = (
                self._normalization_hits / normalization_total * 100 
                if normalization_total > 0 else 0.0
            )
            
            uptime_hours = (time.time() - self._created_at) / 3600
            
            return {
                'similarity_cache': {
                    'size': len(self._similarity_cache),
                    'max_size': self._similarity_cache_size,
                    'hits': self._similarity_hits,
                    'misses': self._similarity_misses,
                    'hit_rate_percent': similarity_hit_rate,
                    'utilization_percent': len(self._similarity_cache) / self._similarity_cache_size * 100
                },
                'normalization_cache': {
                    'size': len(self._normalization_cache),
                    'max_size': self._normalization_cache_size,
                    'hits': self._normalization_hits,
                    'misses': self._normalization_misses,
                    'hit_rate_percent': normalization_hit_rate,
                    'utilization_percent': len(self._normalization_cache) / self._normalization_cache_size * 100
                },
                'uptime_hours': uptime_hours,
                'created_at': self._created_at
            }
    
    def resize_caches(self, similarity_size: Optional[int] = None, 
                     normalization_size: Optional[int] = None):
        """
        Resize cache limits.
        
        Args:
            similarity_size: New similarity cache size
            normalization_size: New normalization cache size
        """
        if similarity_size is not None:
            with self._similarity_lock:
                self._similarity_cache_size = similarity_size
                # Trim cache if new size is smaller
                while len(self._similarity_cache) > similarity_size:
                    self._similarity_cache.popitem(last=False)
                self.logger.info(f"Similarity cache resized to {similarity_size}")
        
        if normalization_size is not None:
            with self._normalization_lock:
                self._normalization_cache_size = normalization_size
                # Trim cache if new size is smaller
                while len(self._normalization_cache) > normalization_size:
                    self._normalization_cache.popitem(last=False)
                self.logger.info(f"Normalization cache resized to {normalization_size}")


# Global cache instance for use across algorithms
_global_cache = None
_cache_lock = threading.Lock()


def get_global_cache() -> MatchingCache:
    """
    Get the global cache instance (singleton pattern).
    
    Returns:
        Global MatchingCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = MatchingCache()
    
    return _global_cache


def reset_global_cache():
    """Reset the global cache instance."""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is not None:
            _global_cache.clear_all_caches()
        _global_cache = None