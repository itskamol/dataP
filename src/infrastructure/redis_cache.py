"""
Redis-based caching for distributed deployments.
Implements requirement 3.2, 3.3: Advanced caching for distributed systems.
"""

import json
import pickle
import threading
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import zlib

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.infrastructure.logging import get_logger
from src.infrastructure.caching import CacheStats


@dataclass
class RedisConfig:
    """Configuration for Redis connection."""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    key_prefix: str = 'file_processing:'
    compression_threshold: int = 1024  # Compress values larger than 1KB
    default_ttl: int = 3600  # 1 hour default TTL


class RedisCache:
    """Redis-based cache with compression and connection pooling."""
    
    def __init__(self, config: RedisConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        self.config = config
        self.logger = get_logger('redis_cache')
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        # Initialize Redis connection pool
        self._pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            max_connections=config.max_connections,
            retry_on_timeout=config.retry_on_timeout,
            health_check_interval=config.health_check_interval
        )
        
        self._redis = redis.Redis(connection_pool=self._pool)
        self._test_connection()
        
        self.logger.info(f"Redis cache initialized: {config.host}:{config.port}")
    
    def _test_connection(self):
        """Test Redis connection."""
        try:
            self._redis.ping()
            self.logger.info("Redis connection successful")
        except RedisConnectionError as e:
            self.logger.error(f"Redis connection failed: {str(e)}")
            raise
    
    def _create_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize and optionally compress value."""
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, list, dict)):
                serialized = json.dumps(value, ensure_ascii=False).encode('utf-8')
                compression_type = 'json'
            else:
                # Use pickle for complex objects
                serialized = pickle.dumps(value)
                compression_type = 'pickle'
            
            # Compress if value is large enough
            if len(serialized) > self.config.compression_threshold:
                compressed = zlib.compress(serialized)
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized):
                    return b'compressed:' + compression_type.encode() + b':' + compressed
            
            return compression_type.encode() + b':' + serialized
            
        except Exception as e:
            self.logger.error(f"Serialization error: {str(e)}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize and decompress value."""
        try:
            if data.startswith(b'compressed:'):
                # Handle compressed data
                parts = data.split(b':', 2)
                compression_type = parts[1].decode()
                compressed_data = parts[2]
                serialized = zlib.decompress(compressed_data)
            else:
                # Handle uncompressed data
                parts = data.split(b':', 1)
                compression_type = parts[0].decode()
                serialized = parts[1]
            
            if compression_type == 'json':
                return json.loads(serialized.decode('utf-8'))
            elif compression_type == 'pickle':
                return pickle.loads(serialized)
            else:
                raise ValueError(f"Unknown compression type: {compression_type}")
                
        except Exception as e:
            self.logger.error(f"Deserialization error: {str(e)}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        with self._lock:
            self._stats.total_requests += 1
            
            try:
                redis_key = self._create_key(key)
                data = self._redis.get(redis_key)
                
                if data is not None:
                    value = self._deserialize_value(data)
                    self._stats.hits += 1
                    return value
                else:
                    self._stats.misses += 1
                    return None
                    
            except RedisError as e:
                self.logger.error(f"Redis get error: {str(e)}")
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in Redis cache with optional TTL."""
        try:
            redis_key = self._create_key(key)
            serialized_value = self._serialize_value(value)
            ttl = ttl or self.config.default_ttl
            
            result = self._redis.setex(redis_key, ttl, serialized_value)
            
            if result:
                with self._lock:
                    self._stats.memory_usage_bytes += len(serialized_value)
                return True
            else:
                return False
                
        except RedisError as e:
            self.logger.error(f"Redis put error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            redis_key = self._create_key(key)
            result = self._redis.delete(redis_key)
            return result > 0
            
        except RedisError as e:
            self.logger.error(f"Redis delete error: {str(e)}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis_key = self._create_key(key)
            return self._redis.exists(redis_key) > 0
            
        except RedisError as e:
            self.logger.error(f"Redis exists error: {str(e)}")
            return False
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries matching pattern."""
        try:
            if pattern:
                search_pattern = self._create_key(pattern)
            else:
                search_pattern = self._create_key('*')
            
            keys = self._redis.keys(search_pattern)
            if keys:
                self._redis.delete(*keys)
                self.logger.info(f"Cleared {len(keys)} cache entries")
            
            with self._lock:
                self._stats.memory_usage_bytes = 0
                
        except RedisError as e:
            self.logger.error(f"Redis clear error: {str(e)}")
    
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
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            info = self._redis.info()
            return {
                'redis_version': info.get('redis_version'),
                'used_memory': info.get('used_memory'),
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses')
            }
        except RedisError as e:
            self.logger.error(f"Redis info error: {str(e)}")
            return {}
    
    def health_check(self) -> bool:
        """Check Redis health."""
        try:
            self._redis.ping()
            return True
        except RedisError:
            return False


class DistributedSimilarityCache:
    """Distributed similarity cache using Redis."""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis_cache = redis_cache
        self.logger = get_logger('distributed_similarity_cache')
    
    def get_similarity(self, text1: str, text2: str, algorithm: str, 
                      normalization: bool = False) -> Optional[float]:
        """Get cached similarity score."""
        cache_key = self._create_similarity_key(text1, text2, algorithm, normalization)
        return self.redis_cache.get(cache_key)
    
    def put_similarity(self, text1: str, text2: str, algorithm: str, 
                      similarity: float, normalization: bool = False, 
                      ttl: Optional[int] = None) -> bool:
        """Cache similarity score."""
        cache_key = self._create_similarity_key(text1, text2, algorithm, normalization)
        return self.redis_cache.put(cache_key, similarity, ttl)
    
    def _create_similarity_key(self, text1: str, text2: str, algorithm: str, 
                              normalization: bool) -> str:
        """Create cache key for similarity calculation."""
        # Ensure consistent ordering for symmetric operations
        if text1 > text2:
            text1, text2 = text2, text1
        
        # Create hash for consistent key length and avoid special characters
        content = f"{algorithm}:{normalization}:{text1}:{text2}"
        key_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        return f"similarity:{key_hash}"


class DistributedNormalizationCache:
    """Distributed normalization cache using Redis."""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis_cache = redis_cache
        self.logger = get_logger('distributed_normalization_cache')
    
    def get_normalized(self, text: str, algorithm: str) -> Optional[str]:
        """Get cached normalized text."""
        cache_key = self._create_normalization_key(text, algorithm)
        return self.redis_cache.get(cache_key)
    
    def put_normalized(self, text: str, algorithm: str, normalized: str, 
                      ttl: Optional[int] = None) -> bool:
        """Cache normalized text."""
        cache_key = self._create_normalization_key(text, algorithm)
        return self.redis_cache.put(cache_key, normalized, ttl)
    
    def _create_normalization_key(self, text: str, algorithm: str) -> str:
        """Create cache key for normalization."""
        content = f"{algorithm}:{text}"
        key_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"normalization:{key_hash}"


class DistributedCacheManager:
    """Manager for distributed caching operations."""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis_cache = RedisCache(redis_config)
        self.similarity_cache = DistributedSimilarityCache(self.redis_cache)
        self.normalization_cache = DistributedNormalizationCache(self.redis_cache)
        self.logger = get_logger('distributed_cache_manager')
        
        self.logger.info("Distributed cache manager initialized")
    
    def get_similarity(self, text1: str, text2: str, algorithm: str, 
                      normalization: bool = False) -> Optional[float]:
        """Get cached similarity score."""
        return self.similarity_cache.get_similarity(text1, text2, algorithm, normalization)
    
    def put_similarity(self, text1: str, text2: str, algorithm: str, 
                      similarity: float, normalization: bool = False, 
                      ttl: Optional[int] = None) -> bool:
        """Cache similarity score."""
        return self.similarity_cache.put_similarity(text1, text2, algorithm, similarity, normalization, ttl)
    
    def get_normalized_text(self, text: str, algorithm: str) -> Optional[str]:
        """Get cached normalized text."""
        return self.normalization_cache.get_normalized(text, algorithm)
    
    def put_normalized_text(self, text: str, algorithm: str, normalized: str, 
                           ttl: Optional[int] = None) -> bool:
        """Cache normalized text."""
        return self.normalization_cache.put_normalized(text, algorithm, normalized, ttl)
    
    def clear_all(self):
        """Clear all caches."""
        self.redis_cache.clear()
        self.logger.info("All distributed caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = self.redis_cache.get_stats()
        redis_info = self.redis_cache.get_redis_info()
        
        return {
            'cache_stats': asdict(cache_stats),
            'redis_info': redis_info,
            'health': self.redis_cache.health_check()
        }
    
    def health_check(self) -> bool:
        """Check distributed cache health."""
        return self.redis_cache.health_check()


# Global distributed cache manager instance
_distributed_cache_manager: Optional[DistributedCacheManager] = None
_distributed_cache_lock = threading.Lock()


def get_distributed_cache_manager(redis_config: Optional[RedisConfig] = None) -> Optional[DistributedCacheManager]:
    """Get global distributed cache manager instance."""
    global _distributed_cache_manager
    
    if not REDIS_AVAILABLE:
        return None
    
    if _distributed_cache_manager is None and redis_config is not None:
        with _distributed_cache_lock:
            if _distributed_cache_manager is None:
                try:
                    _distributed_cache_manager = DistributedCacheManager(redis_config)
                except Exception as e:
                    logger = get_logger('distributed_cache')
                    logger.error(f"Failed to initialize distributed cache: {str(e)}")
                    return None
    
    return _distributed_cache_manager


def clear_distributed_caches():
    """Clear all distributed caches."""
    global _distributed_cache_manager
    if _distributed_cache_manager is not None:
        _distributed_cache_manager.clear_all()