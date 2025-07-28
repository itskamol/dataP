"""
Message queue system for distributed processing support.
Implements requirement 3.4, 8.3: Distributed processing with message queues.
"""

import json
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import queue
from contextlib import contextmanager

# Optional Redis support for distributed queues
try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.infrastructure.logging import get_logger
from src.domain.exceptions import ProcessingError


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class Message:
    """Message structure for queue operations."""
    id: str
    queue_name: str
    payload: Any
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: float = 0.0
    processed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        data['priority'] = MessagePriority(data['priority'])
        data['status'] = MessageStatus(data['status'])
        return cls(**data)


class InMemoryQueue:
    """In-memory message queue implementation."""
    
    def __init__(self, name: str, max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self.logger = get_logger(f'queue.{name}')
        
        # Priority queues for different priority levels
        self._queues = {
            MessagePriority.CRITICAL: queue.PriorityQueue(),
            MessagePriority.HIGH: queue.PriorityQueue(),
            MessagePriority.NORMAL: queue.PriorityQueue(),
            MessagePriority.LOW: queue.PriorityQueue()
        }
        
        # Message storage
        self._messages: Dict[str, Message] = {}
        self._processing_messages: Dict[str, Message] = {}
        
        # Statistics
        self._stats = {
            'messages_sent': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'current_size': 0
        }
        
        self._lock = threading.RLock()
    
    def put(self, message: Message) -> bool:
        """Put message in queue."""
        with self._lock:
            if len(self._messages) >= self.max_size:
                self.logger.warning(f"Queue {self.name} is full")
                return False
            
            # Store message
            self._messages[message.id] = message
            
            # Add to priority queue
            priority_value = -message.priority.value  # Negative for max-heap behavior
            self._queues[message.priority].put((priority_value, message.created_at, message.id))
            
            self._stats['messages_sent'] += 1
            self._stats['current_size'] = len(self._messages)
            
            self.logger.debug(f"Message {message.id} added to queue {self.name}")
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Get message from queue."""
        # Try queues in priority order
        for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            try:
                _, _, message_id = self._queues[priority].get(block=False)
                
                with self._lock:
                    if message_id in self._messages:
                        message = self._messages.pop(message_id)
                        message.status = MessageStatus.PROCESSING
                        message.processed_at = time.time()
                        
                        # Move to processing
                        self._processing_messages[message_id] = message
                        self._stats['current_size'] = len(self._messages)
                        
                        self.logger.debug(f"Message {message_id} retrieved from queue {self.name}")
                        return message
                
            except queue.Empty:
                continue
        
        # If no messages available and timeout specified, wait
        if timeout and timeout > 0:
            time.sleep(min(timeout, 0.1))
            return self.get(timeout - 0.1) if timeout > 0.1 else None
        
        return None
    
    def ack(self, message_id: str) -> bool:
        """Acknowledge message processing completion."""
        with self._lock:
            if message_id in self._processing_messages:
                message = self._processing_messages.pop(message_id)
                message.status = MessageStatus.COMPLETED
                
                self._stats['messages_processed'] += 1
                
                self.logger.debug(f"Message {message_id} acknowledged")
                return True
            
            return False
    
    def nack(self, message_id: str, requeue: bool = True) -> bool:
        """Negative acknowledge - message processing failed."""
        with self._lock:
            if message_id in self._processing_messages:
                message = self._processing_messages.pop(message_id)
                message.retry_count += 1
                
                if requeue and message.retry_count <= message.max_retries:
                    # Requeue for retry
                    message.status = MessageStatus.RETRY
                    self._messages[message_id] = message
                    
                    priority_value = -message.priority.value
                    self._queues[message.priority].put((priority_value, time.time(), message_id))
                    
                    self._stats['current_size'] = len(self._messages)
                    self.logger.debug(f"Message {message_id} requeued for retry ({message.retry_count}/{message.max_retries})")
                else:
                    # Max retries exceeded
                    message.status = MessageStatus.FAILED
                    self._stats['messages_failed'] += 1
                    self.logger.warning(f"Message {message_id} failed after {message.retry_count} retries")
                
                return True
            
            return False
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._messages)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                'name': self.name,
                'current_size': len(self._messages),
                'processing_size': len(self._processing_messages),
                'max_size': self.max_size,
                **self._stats
            }
    
    def clear(self):
        """Clear all messages from queue."""
        with self._lock:
            self._messages.clear()
            self._processing_messages.clear()
            
            for priority_queue in self._queues.values():
                while not priority_queue.empty():
                    try:
                        priority_queue.get(block=False)
                    except queue.Empty:
                        break
            
            self._stats['current_size'] = 0
            self.logger.info(f"Queue {self.name} cleared")


class RedisQueue:
    """Redis-based distributed message queue."""
    
    def __init__(self, name: str, redis_client: 'redis.Redis', max_size: int = 100000):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for distributed queues")
        
        self.name = name
        self.redis_client = redis_client
        self.max_size = max_size
        self.logger = get_logger(f'redis_queue.{name}')
        
        # Redis keys
        self.queue_key = f"queue:{name}"
        self.processing_key = f"processing:{name}"
        self.stats_key = f"stats:{name}"
        
        # Initialize stats if not exists
        if not self.redis_client.exists(self.stats_key):
            self.redis_client.hset(self.stats_key, mapping={
                'messages_sent': 0,
                'messages_processed': 0,
                'messages_failed': 0
            })
    
    def put(self, message: Message) -> bool:
        """Put message in Redis queue."""
        try:
            # Check queue size
            current_size = self.redis_client.llen(self.queue_key)
            if current_size >= self.max_size:
                self.logger.warning(f"Redis queue {self.name} is full")
                return False
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Add to queue based on priority
            if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
                # High priority - add to front
                self.redis_client.lpush(self.queue_key, message_data)
            else:
                # Normal/low priority - add to back
                self.redis_client.rpush(self.queue_key, message_data)
            
            # Update stats
            self.redis_client.hincrby(self.stats_key, 'messages_sent', 1)
            
            self.logger.debug(f"Message {message.id} added to Redis queue {self.name}")
            return True
        
        except RedisError as e:
            self.logger.error(f"Failed to put message in Redis queue: {str(e)}")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Get message from Redis queue."""
        try:
            # Block pop from queue
            timeout_int = int(timeout) if timeout else 0
            result = self.redis_client.blpop(self.queue_key, timeout=timeout_int)
            
            if result:
                _, message_data = result
                message_dict = json.loads(message_data.decode('utf-8'))
                message = Message.from_dict(message_dict)
                
                message.status = MessageStatus.PROCESSING
                message.processed_at = time.time()
                
                # Move to processing set
                processing_data = json.dumps(message.to_dict())
                self.redis_client.hset(self.processing_key, message.id, processing_data)
                
                self.logger.debug(f"Message {message.id} retrieved from Redis queue {self.name}")
                return message
            
            return None
        
        except RedisError as e:
            self.logger.error(f"Failed to get message from Redis queue: {str(e)}")
            return None
    
    def ack(self, message_id: str) -> bool:
        """Acknowledge message processing completion."""
        try:
            # Remove from processing set
            result = self.redis_client.hdel(self.processing_key, message_id)
            
            if result:
                # Update stats
                self.redis_client.hincrby(self.stats_key, 'messages_processed', 1)
                self.logger.debug(f"Message {message_id} acknowledged in Redis queue")
                return True
            
            return False
        
        except RedisError as e:
            self.logger.error(f"Failed to ack message in Redis queue: {str(e)}")
            return False
    
    def nack(self, message_id: str, requeue: bool = True) -> bool:
        """Negative acknowledge - message processing failed."""
        try:
            # Get message from processing set
            processing_data = self.redis_client.hget(self.processing_key, message_id)
            
            if processing_data:
                message_dict = json.loads(processing_data.decode('utf-8'))
                message = Message.from_dict(message_dict)
                message.retry_count += 1
                
                # Remove from processing
                self.redis_client.hdel(self.processing_key, message_id)
                
                if requeue and message.retry_count <= message.max_retries:
                    # Requeue for retry
                    message.status = MessageStatus.RETRY
                    return self.put(message)
                else:
                    # Max retries exceeded
                    message.status = MessageStatus.FAILED
                    self.redis_client.hincrby(self.stats_key, 'messages_failed', 1)
                    self.logger.warning(f"Message {message_id} failed after {message.retry_count} retries")
                
                return True
            
            return False
        
        except RedisError as e:
            self.logger.error(f"Failed to nack message in Redis queue: {str(e)}")
            return False
    
    def size(self) -> int:
        """Get current queue size."""
        try:
            return self.redis_client.llen(self.queue_key)
        except RedisError:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            stats = self.redis_client.hgetall(self.stats_key)
            processing_size = self.redis_client.hlen(self.processing_key)
            
            return {
                'name': self.name,
                'current_size': self.size(),
                'processing_size': processing_size,
                'max_size': self.max_size,
                'messages_sent': int(stats.get(b'messages_sent', 0)),
                'messages_processed': int(stats.get(b'messages_processed', 0)),
                'messages_failed': int(stats.get(b'messages_failed', 0))
            }
        
        except RedisError as e:
            self.logger.error(f"Failed to get Redis queue stats: {str(e)}")
            return {'error': str(e)}
    
    def clear(self):
        """Clear all messages from Redis queue."""
        try:
            self.redis_client.delete(self.queue_key, self.processing_key)
            self.logger.info(f"Redis queue {self.name} cleared")
        except RedisError as e:
            self.logger.error(f"Failed to clear Redis queue: {str(e)}")


class MessageQueueManager:
    """Manager for message queues with support for different backends."""
    
    def __init__(self, use_redis: bool = False, redis_config: Optional[Dict[str, Any]] = None):
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.logger = get_logger('message_queue_manager')
        
        self._queues: Dict[str, Union[InMemoryQueue, RedisQueue]] = {}
        self._redis_client: Optional['redis.Redis'] = None
        
        if self.use_redis:
            self._init_redis(redis_config or {})
        
        self.logger.info(f"Message queue manager initialized (backend: {'Redis' if self.use_redis else 'In-Memory'})")
    
    def _init_redis(self, redis_config: Dict[str, Any]):
        """Initialize Redis connection."""
        try:
            self._redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                socket_timeout=redis_config.get('socket_timeout', 5.0),
                decode_responses=False
            )
            
            # Test connection
            self._redis_client.ping()
            self.logger.info("Redis connection established")
        
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            self.use_redis = False
    
    def get_queue(self, name: str, max_size: int = 10000) -> Union[InMemoryQueue, RedisQueue]:
        """Get or create a message queue."""
        if name not in self._queues:
            if self.use_redis and self._redis_client:
                self._queues[name] = RedisQueue(name, self._redis_client, max_size)
            else:
                self._queues[name] = InMemoryQueue(name, max_size)
            
            self.logger.info(f"Created queue: {name}")
        
        return self._queues[name]
    
    def send_message(self, queue_name: str, payload: Any, 
                    priority: MessagePriority = MessagePriority.NORMAL,
                    timeout_seconds: Optional[int] = None,
                    max_retries: int = 3) -> str:
        """Send message to queue."""
        message = Message(
            id=str(uuid.uuid4()),
            queue_name=queue_name,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        queue = self.get_queue(queue_name)
        
        if queue.put(message):
            self.logger.debug(f"Message sent to queue {queue_name}: {message.id}")
            return message.id
        else:
            raise ProcessingError(f"Failed to send message to queue {queue_name}")
    
    def create_worker(self, queue_name: str, handler: Callable[[Any], Any],
                     worker_id: Optional[str] = None) -> 'MessageWorker':
        """Create a message worker for processing queue messages."""
        return MessageWorker(
            queue=self.get_queue(queue_name),
            handler=handler,
            worker_id=worker_id or f"worker_{uuid.uuid4().hex[:8]}",
            logger=self.logger
        )
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all queues."""
        return {name: queue.get_stats() for name, queue in self._queues.items()}
    
    def cleanup(self):
        """Clean up resources."""
        for queue in self._queues.values():
            if hasattr(queue, 'clear'):
                queue.clear()
        
        self._queues.clear()
        
        if self._redis_client:
            self._redis_client.close()
        
        self.logger.info("Message queue manager cleaned up")


class MessageWorker:
    """Worker for processing messages from a queue."""
    
    def __init__(self, queue: Union[InMemoryQueue, RedisQueue], 
                 handler: Callable[[Any], Any], worker_id: str, logger):
        self.queue = queue
        self.handler = handler
        self.worker_id = worker_id
        self.logger = logger
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'start_time': None
        }
    
    def start(self):
        """Start the worker thread."""
        if self._running:
            return
        
        self._running = True
        self._stats['start_time'] = time.time()
        
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        
        self.logger.info(f"Worker {self.worker_id} started for queue {self.queue.name}")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self):
        """Main worker loop."""
        while self._running:
            try:
                # Get message from queue
                message = self.queue.get(timeout=1.0)
                
                if message:
                    try:
                        # Process message
                        result = self.handler(message.payload)
                        
                        # Acknowledge success
                        self.queue.ack(message.id)
                        self._stats['messages_processed'] += 1
                        
                        self.logger.debug(f"Worker {self.worker_id} processed message {message.id}")
                    
                    except Exception as e:
                        # Negative acknowledge
                        self.queue.nack(message.id, requeue=True)
                        self._stats['messages_failed'] += 1
                        
                        self.logger.error(f"Worker {self.worker_id} failed to process message {message.id}: {str(e)}")
            
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {str(e)}")
                time.sleep(1.0)  # Brief pause on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self._stats['start_time'] if self._stats['start_time'] else 0
        
        return {
            'worker_id': self.worker_id,
            'queue_name': self.queue.name,
            'running': self._running,
            'uptime_seconds': uptime,
            **self._stats
        }


# Global message queue manager
_message_queue_manager: Optional[MessageQueueManager] = None
_queue_manager_lock = threading.Lock()


def get_message_queue_manager(use_redis: bool = False, 
                            redis_config: Optional[Dict[str, Any]] = None) -> MessageQueueManager:
    """Get global message queue manager instance."""
    global _message_queue_manager
    
    if _message_queue_manager is None:
        with _queue_manager_lock:
            if _message_queue_manager is None:
                _message_queue_manager = MessageQueueManager(use_redis, redis_config)
    
    return _message_queue_manager


@contextmanager
def message_queue_context(queue_name: str, use_redis: bool = False):
    """Context manager for message queue operations."""
    manager = get_message_queue_manager(use_redis)
    queue = manager.get_queue(queue_name)
    
    try:
        yield queue
    finally:
        # Queue cleanup is handled by manager
        pass