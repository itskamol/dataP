"""
WebSocket-enabled progress tracking service for real-time updates.
Implements requirements 5.3, 5.4, 4.1: Real-time progress updates with WebSocket support.
"""

import json
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict

try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    SocketIO = None
    emit = None

from src.infrastructure.logging import get_logger


@dataclass
class WebSocketProgressUpdate:
    """Progress update for WebSocket transmission."""
    operation_id: str
    status: str
    progress: int
    message: str
    timestamp: str
    current_step: int
    total_steps: int
    elapsed_time: int
    estimated_remaining: Optional[int] = None
    can_cancel: bool = True
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'operation_id': self.operation_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'timestamp': self.timestamp,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'elapsed_time': self.elapsed_time,
            'estimated_remaining': self.estimated_remaining,
            'can_cancel': self.can_cancel,
            'metadata': self.metadata or {}
        }


class WebSocketProgressService:
    """WebSocket-enabled progress tracking service."""
    
    def __init__(self, socketio: Optional[SocketIO] = None):
        """Initialize WebSocket progress service."""
        self.logger = get_logger('websocket_progress')
        self.socketio = socketio
        
        # Progress tracking
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._cancellation_flags: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        
        # Connection tracking
        self._connected_clients: Dict[str, List[str]] = {}  # operation_id -> [session_ids]
        
        if not SOCKETIO_AVAILABLE:
            self.logger.warning("Flask-SocketIO not available, WebSocket features disabled")
    
    def set_socketio(self, socketio: SocketIO):
        """Set the SocketIO instance."""
        self.socketio = socketio
    
    def start_operation(self, operation_id: Optional[str] = None, 
                       total_steps: int = 100,
                       message: str = "Starting operation") -> str:
        """Start a new operation with progress tracking."""
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        with self._lock:
            start_time = datetime.now()
            
            self._operations[operation_id] = {
                'status': 'starting',
                'progress': 0,
                'message': message,
                'current_step': 0,
                'total_steps': total_steps,
                'start_time': start_time,
                'last_update': start_time,
                'step_durations': [],
                'can_cancel': True
            }
            
            # Initialize cancellation flag
            self._cancellation_flags[operation_id] = threading.Event()
            self._connected_clients[operation_id] = []
            
            # Emit initial status
            self._emit_progress_update(operation_id)
            
            self.logger.info(f"Started operation {operation_id} with {total_steps} steps")
            return operation_id
    
    def update_progress(self, operation_id: str, 
                       current_step: int,
                       message: str = "",
                       status: str = "processing",
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update progress for an operation."""
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Attempted to update unknown operation: {operation_id}")
                return False
            
            # Check if operation was cancelled
            if self._cancellation_flags.get(operation_id, threading.Event()).is_set():
                self.logger.info(f"Operation {operation_id} was cancelled, ignoring update")
                return False
            
            operation = self._operations[operation_id]
            now = datetime.now()
            
            # Calculate step duration
            if current_step > operation['current_step']:
                step_duration = (now - operation['last_update']).total_seconds()
                operation['step_durations'].append(step_duration)
                
                # Keep only recent durations for better estimation
                if len(operation['step_durations']) > 20:
                    operation['step_durations'] = operation['step_durations'][-20:]
            
            # Update operation data
            operation.update({
                'status': status,
                'current_step': current_step,
                'message': message,
                'last_update': now,
                'metadata': metadata or {}
            })
            
            # Calculate progress percentage
            if operation['total_steps'] > 0:
                progress_ratio = current_step / operation['total_steps']
                operation['progress'] = max(0, min(100, int(progress_ratio * 100)))
            else:
                operation['progress'] = 0
            
            # Emit progress update
            self._emit_progress_update(operation_id)
            
            self.logger.debug(f"Updated operation {operation_id}: step {current_step}/{operation['total_steps']}")
            return True
    
    def complete_operation(self, operation_id: str, 
                          message: str = "Operation completed",
                          success: bool = True,
                          error_message: Optional[str] = None) -> bool:
        """Complete an operation."""
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Attempted to complete unknown operation: {operation_id}")
                return False
            
            operation = self._operations[operation_id]
            
            # Update final status
            operation.update({
                'status': 'completed' if success else 'error',
                'progress': 100 if success else operation['progress'],
                'message': message,
                'completed_at': datetime.now(),
                'can_cancel': False,
                'error_message': error_message
            })
            
            # Emit final update
            self._emit_progress_update(operation_id)
            
            # Clean up after delay
            threading.Timer(30.0, self._cleanup_operation, args=[operation_id]).start()
            
            self.logger.info(f"Completed operation {operation_id} with status: {'success' if success else 'error'}")
            return True
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Attempted to cancel unknown operation: {operation_id}")
                return False
            
            operation = self._operations[operation_id]
            
            if not operation.get('can_cancel', True):
                self.logger.warning(f"Operation {operation_id} cannot be cancelled")
                return False
            
            # Set cancellation flag
            if operation_id in self._cancellation_flags:
                self._cancellation_flags[operation_id].set()
            
            # Update status
            operation.update({
                'status': 'cancelled',
                'message': 'Operation cancelled by user',
                'completed_at': datetime.now(),
                'can_cancel': False
            })
            
            # Emit cancellation update
            self._emit_progress_update(operation_id)
            
            # Clean up after delay
            threading.Timer(10.0, self._cleanup_operation, args=[operation_id]).start()
            
            self.logger.info(f"Cancelled operation {operation_id}")
            return True
    
    def is_cancelled(self, operation_id: str) -> bool:
        """Check if an operation was cancelled."""
        if operation_id in self._cancellation_flags:
            return self._cancellation_flags[operation_id].is_set()
        return False
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an operation."""
        with self._lock:
            if operation_id not in self._operations:
                return None
            
            operation = self._operations[operation_id].copy()
            
            # Calculate elapsed time
            start_time = operation['start_time']
            elapsed_seconds = int((datetime.now() - start_time).total_seconds())
            operation['elapsed_time'] = elapsed_seconds
            
            # Calculate estimated remaining time
            if (operation['status'] == 'processing' and 
                operation['current_step'] > 0 and 
                operation['step_durations']):
                
                remaining_steps = operation['total_steps'] - operation['current_step']
                avg_duration = sum(operation['step_durations']) / len(operation['step_durations'])
                estimated_remaining = int(remaining_steps * avg_duration)
                operation['estimated_remaining'] = estimated_remaining
            else:
                operation['estimated_remaining'] = None
            
            return operation
    
    def register_client(self, operation_id: str, session_id: str):
        """Register a client for operation updates."""
        with self._lock:
            if operation_id not in self._connected_clients:
                self._connected_clients[operation_id] = []
            
            if session_id not in self._connected_clients[operation_id]:
                self._connected_clients[operation_id].append(session_id)
                self.logger.debug(f"Registered client {session_id} for operation {operation_id}")
                
                # Send current status to new client
                if operation_id in self._operations:
                    self._emit_progress_update(operation_id, target_session=session_id)
    
    def unregister_client(self, operation_id: str, session_id: str):
        """Unregister a client from operation updates."""
        with self._lock:
            if (operation_id in self._connected_clients and 
                session_id in self._connected_clients[operation_id]):
                
                self._connected_clients[operation_id].remove(session_id)
                self.logger.debug(f"Unregistered client {session_id} from operation {operation_id}")
    
    def _emit_progress_update(self, operation_id: str, target_session: Optional[str] = None):
        """Emit progress update via WebSocket."""
        if not self.socketio or not SOCKETIO_AVAILABLE:
            return
        
        operation = self._operations.get(operation_id)
        if not operation:
            return
        
        # Calculate elapsed time
        start_time = operation['start_time']
        elapsed_seconds = int((datetime.now() - start_time).total_seconds())
        
        # Calculate estimated remaining time
        estimated_remaining = None
        if (operation['status'] == 'processing' and 
            operation['current_step'] > 0 and 
            operation['step_durations']):
            
            remaining_steps = operation['total_steps'] - operation['current_step']
            avg_duration = sum(operation['step_durations']) / len(operation['step_durations'])
            estimated_remaining = int(remaining_steps * avg_duration)
        
        # Create progress update
        update = WebSocketProgressUpdate(
            operation_id=operation_id,
            status=operation['status'],
            progress=operation['progress'],
            message=operation['message'],
            timestamp=datetime.now().isoformat(),
            current_step=operation['current_step'],
            total_steps=operation['total_steps'],
            elapsed_time=elapsed_seconds,
            estimated_remaining=estimated_remaining,
            can_cancel=operation.get('can_cancel', True),
            metadata=operation.get('metadata', {})
        )
        
        # Emit to specific session or all clients for this operation
        if target_session:
            self.socketio.emit('progress_update', update.to_dict(), 
                             room=target_session, namespace='/progress')
        else:
            # Emit to all registered clients for this operation
            clients = self._connected_clients.get(operation_id, [])
            for session_id in clients:
                self.socketio.emit('progress_update', update.to_dict(), 
                                 room=session_id, namespace='/progress')
    
    def _cleanup_operation(self, operation_id: str):
        """Clean up completed operation data."""
        with self._lock:
            if operation_id in self._operations:
                del self._operations[operation_id]
            
            if operation_id in self._cancellation_flags:
                del self._cancellation_flags[operation_id]
            
            if operation_id in self._connected_clients:
                del self._connected_clients[operation_id]
            
            self.logger.debug(f"Cleaned up operation {operation_id}")
    
    def list_operations(self) -> List[Dict[str, Any]]:
        """List all current operations."""
        with self._lock:
            operations = []
            for operation_id, operation in self._operations.items():
                status = self.get_operation_status(operation_id)
                if status:
                    operations.append(status)
            return operations
    
    def get_connected_clients_count(self, operation_id: str) -> int:
        """Get number of connected clients for an operation."""
        with self._lock:
            return len(self._connected_clients.get(operation_id, []))


# Global instance
websocket_progress_service = WebSocketProgressService()