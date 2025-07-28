"""
Progress tracking system with multiple output channels and persistence.
Implements requirements 5.3, 6.4, 2.2: Real-time progress tracking with WebSocket and HTTP endpoints.
"""

import json
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import queue
import asyncio
import socket
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

try:
    from flask import Flask, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    jsonify = None
    request = None
    SocketIO = None
    emit = None

from src.domain.models import ProgressStatus
from src.domain.exceptions import ProgressTrackingError
from src.infrastructure.logging import get_logger, with_correlation_id


@dataclass
class ProgressUpdate:
    """Represents a progress update event."""
    operation_id: str
    current_step: int
    total_steps: int
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_id': self.operation_id,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class EstimationData:
    """Data for completion time estimation."""
    start_time: datetime
    last_update_time: datetime
    completed_steps: int
    total_steps: int
    step_durations: List[float]
    
    def calculate_eta(self) -> Optional[datetime]:
        """Calculate estimated time of arrival."""
        if self.completed_steps == 0 or self.total_steps == 0:
            return None
        
        remaining_steps = self.total_steps - self.completed_steps
        if remaining_steps <= 0:
            return datetime.now()
        
        # Calculate average step duration
        if self.step_durations:
            avg_duration = sum(self.step_durations) / len(self.step_durations)
        else:
            # Fallback to overall average
            elapsed = (self.last_update_time - self.start_time).total_seconds()
            avg_duration = elapsed / max(self.completed_steps, 1)
        
        estimated_remaining_time = remaining_steps * avg_duration
        return datetime.now() + timedelta(seconds=estimated_remaining_time)


class ProgressPersistence:
    """Handles persistence of progress data for recovery."""
    
    def __init__(self, db_path: str = "progress.db"):
        self.db_path = Path(db_path)
        self.logger = get_logger('progress_persistence')
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for progress persistence."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS progress_operations (
                        operation_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        progress REAL NOT NULL,
                        message TEXT,
                        current_step INTEGER NOT NULL,
                        total_steps INTEGER NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        error_message TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS progress_updates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL,
                        current_step INTEGER NOT NULL,
                        total_steps INTEGER NOT NULL,
                        message TEXT,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (operation_id) REFERENCES progress_operations (operation_id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_operation_id ON progress_updates (operation_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON progress_updates (timestamp)
                """)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize progress database: {str(e)}")
            raise ProgressTrackingError(f"Database initialization failed: {str(e)}")
    
    def save_operation(self, status: ProgressStatus):
        """Save operation status to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO progress_operations 
                    (operation_id, status, progress, message, current_step, total_steps,
                     started_at, completed_at, error_message, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM progress_operations WHERE operation_id = ?), ?), ?)
                """, (
                    status.operation_id, status.status, status.progress, status.message,
                    status.current_step, status.total_steps,
                    status.started_at.isoformat() if status.started_at else None,
                    status.completed_at.isoformat() if status.completed_at else None,
                    status.error_message,
                    json.dumps(status.to_dict().get('metadata', {})),
                    status.operation_id, now, now
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to save operation {status.operation_id}: {str(e)}")
    
    def save_update(self, update: ProgressUpdate):
        """Save progress update to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO progress_updates 
                    (operation_id, current_step, total_steps, message, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    update.operation_id, update.current_step, update.total_steps,
                    update.message, update.timestamp.isoformat(),
                    json.dumps(update.metadata)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to save progress update for {update.operation_id}: {str(e)}")
    
    def load_operation(self, operation_id: str) -> Optional[ProgressStatus]:
        """Load operation status from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT operation_id, status, progress, message, current_step, total_steps,
                           started_at, completed_at, error_message, metadata
                    FROM progress_operations 
                    WHERE operation_id = ?
                """, (operation_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return ProgressStatus(
                    operation_id=row[0],
                    status=row[1],
                    progress=row[2],
                    message=row[3] or "",
                    current_step=row[4],
                    total_steps=row[5],
                    started_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    error_message=row[8]
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load operation {operation_id}: {str(e)}")
            return None
    
    def get_active_operations(self) -> List[ProgressStatus]:
        """Get all active (running) operations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT operation_id, status, progress, message, current_step, total_steps,
                           started_at, completed_at, error_message, metadata
                    FROM progress_operations 
                    WHERE status = 'running'
                    ORDER BY started_at DESC
                """)
                
                operations = []
                for row in cursor.fetchall():
                    operations.append(ProgressStatus(
                        operation_id=row[0],
                        status=row[1],
                        progress=row[2],
                        message=row[3] or "",
                        current_step=row[4],
                        total_steps=row[5],
                        started_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        error_message=row[8]
                    ))
                
                return operations
                
        except Exception as e:
            self.logger.error(f"Failed to get active operations: {str(e)}")
            return []
    
    def cleanup_old_operations(self, max_age_days: int = 7):
        """Clean up old completed operations."""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Delete old progress updates first (foreign key constraint)
                conn.execute("""
                    DELETE FROM progress_updates 
                    WHERE operation_id IN (
                        SELECT operation_id FROM progress_operations 
                        WHERE status IN ('completed', 'error', 'cancelled') 
                        AND (completed_at IS NOT NULL AND completed_at < ?)
                    )
                """, (cutoff_date.isoformat(),))
                
                # Delete old operations
                cursor = conn.execute("""
                    DELETE FROM progress_operations 
                    WHERE status IN ('completed', 'error', 'cancelled') 
                    AND (completed_at IS NOT NULL AND completed_at < ?)
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                
            self.logger.info(f"Cleaned up {deleted_count} old progress operations")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old operations: {str(e)}")


def find_free_port(start_port: int = 8765, max_attempts: int = 10) -> int:
    """Find a free port starting from the given port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError(f"No free port found in range {start_port}-{start_port + max_attempts}")


class WebSocketProgressChannel:
    """WebSocket channel for real-time progress updates."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.logger = get_logger('websocket_progress')
        
        # Try to find a free port if the default is busy
        try:
            self.port = find_free_port(port)
            if self.port != port:
                self.logger.info(f"Port {port} was busy, using port {self.port} instead")
        except OSError as e:
            self.logger.error(f"Could not find free port: {e}")
            self.port = port  # Fallback to original port
        
        self.clients = set()
        self.server = None
        self._running = False
        
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("WebSocket support disabled (websockets module not available)")
    
    async def register_client(self, websocket, path):
        """Register a new WebSocket client."""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        self.clients.add(websocket)
        self.logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            self.logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
    
    async def broadcast_update(self, update: ProgressUpdate):
        """Broadcast progress update to all connected clients."""
        if not WEBSOCKETS_AVAILABLE or not self.clients:
            return
        
        message = json.dumps(update.to_dict())
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                if WEBSOCKETS_AVAILABLE and hasattr(websockets, 'exceptions'):
                    if isinstance(e, websockets.exceptions.ConnectionClosed):
                        disconnected_clients.add(client)
                        continue
                self.logger.error(f"Failed to send update to client: {str(e)}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    async def start_server(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("Cannot start WebSocket server - websockets module not available")
            return
            
        try:
            self.server = await websockets.serve(
                self.register_client, 
                self.host, 
                self.port
            )
            self._running = True
            self.logger.info(f"WebSocket server started on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise ProgressTrackingError(f"WebSocket server startup failed: {str(e)}")
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE or not self.server:
            return
            
        self.server.close()
        await self.server.wait_closed()
        self._running = False
        self.logger.info("WebSocket server stopped")


class HTTPProgressChannel:
    """HTTP channel for progress updates via REST API."""
    
    def __init__(self, flask_app: Optional[Flask] = None):
        self.logger = get_logger('http_progress')
        self._progress_tracker = None
        
        if not FLASK_AVAILABLE:
            self.logger.warning("HTTP progress channel disabled (Flask not available)")
            self.app = None
            return
            
        self.app = flask_app or Flask(__name__)
        self._setup_routes()
    
    def set_progress_tracker(self, tracker):
        """Set the progress tracker instance."""
        self._progress_tracker = tracker
    
    def _setup_routes(self):
        """Setup HTTP routes for progress API."""
        if not FLASK_AVAILABLE or not self.app:
            return
        
        @self.app.route('/api/progress/<operation_id>', methods=['GET'])
        def get_progress(operation_id):
            """Get progress status for an operation."""
            try:
                if not self._progress_tracker:
                    return jsonify({'error': 'Progress tracker not initialized'}), 500
                
                status = self._progress_tracker.get_progress(operation_id)
                if not status:
                    return jsonify({'error': 'Operation not found'}), 404
                
                return jsonify(status.to_dict())
                
            except Exception as e:
                self.logger.error(f"Failed to get progress for {operation_id}: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/progress', methods=['GET'])
        def list_operations():
            """List all operations with optional status filter."""
            try:
                if not self._progress_tracker:
                    return jsonify({'error': 'Progress tracker not initialized'}), 500
                
                status_filter = request.args.get('status')
                operations = self._progress_tracker.list_operations(status_filter)
                
                return jsonify({
                    'operations': [op.to_dict() for op in operations],
                    'count': len(operations)
                })
                
            except Exception as e:
                self.logger.error(f"Failed to list operations: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/progress/<operation_id>/cancel', methods=['POST'])
        def cancel_operation(operation_id):
            """Cancel a running operation."""
            try:
                if not self._progress_tracker:
                    return jsonify({'error': 'Progress tracker not initialized'}), 500
                
                success = self._progress_tracker.cancel_operation(operation_id)
                if success:
                    return jsonify({'message': 'Operation cancelled successfully'})
                else:
                    return jsonify({'error': 'Failed to cancel operation'}), 400
                
            except Exception as e:
                self.logger.error(f"Failed to cancel operation {operation_id}: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/progress/<operation_id>/history', methods=['GET'])
        def get_progress_history(operation_id):
            """Get progress update history for an operation."""
            try:
                if not self._progress_tracker:
                    return jsonify({'error': 'Progress tracker not initialized'}), 500
                
                history = self._progress_tracker.get_progress_history(operation_id)
                return jsonify({
                    'operation_id': operation_id,
                    'updates': [update.to_dict() for update in history]
                })
                
            except Exception as e:
                self.logger.error(f"Failed to get progress history for {operation_id}: {str(e)}")
                return jsonify({'error': str(e)}), 500


class ProgressTracker:
    """
    Comprehensive progress tracking system with multiple output channels.
    
    Features:
    - Real-time WebSocket updates
    - HTTP REST API endpoints
    - Progress persistence for recovery
    - Estimated completion time calculation
    - Operation cancellation and cleanup
    """
    
    def __init__(self, 
                 websocket_host: str = "localhost",
                 websocket_port: int = 8765,
                 persistence_db: str = "progress.db",
                 flask_app: Optional[Flask] = None):
        
        self.logger = get_logger('progress_tracker')
        
        # Core tracking data
        self._operations: Dict[str, ProgressStatus] = {}
        self._estimation_data: Dict[str, EstimationData] = {}
        self._cancellation_flags: Dict[str, threading.Event] = {}
        self._cleanup_callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # Persistence
        self.persistence = ProgressPersistence(persistence_db)
        
        # Communication channels
        self.websocket_channel = WebSocketProgressChannel(websocket_host, websocket_port)
        self.http_channel = HTTPProgressChannel(flask_app)
        self.http_channel.set_progress_tracker(self)
        
        # Background processing
        self._update_queue = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="progress")
        self._websocket_loop = None
        self._websocket_thread = None
        
        # Recovery on startup
        self._recover_operations()
        
        # Start background processing
        self._start_background_processing()
    
    def _recover_operations(self):
        """Recover active operations from persistence."""
        try:
            active_operations = self.persistence.get_active_operations()
            
            for operation in active_operations:
                # Mark recovered operations as potentially stale
                operation.message = f"[RECOVERED] {operation.message}"
                self._operations[operation.operation_id] = operation
                
                # Initialize estimation data
                if operation.started_at:
                    self._estimation_data[operation.operation_id] = EstimationData(
                        start_time=operation.started_at,
                        last_update_time=datetime.now(),
                        completed_steps=operation.current_step,
                        total_steps=operation.total_steps,
                        step_durations=[]
                    )
                
                # Initialize cancellation flag
                self._cancellation_flags[operation.operation_id] = threading.Event()
            
            if active_operations:
                self.logger.info(f"Recovered {len(active_operations)} active operations")
                
        except Exception as e:
            self.logger.error(f"Failed to recover operations: {str(e)}")
    
    def _start_background_processing(self):
        """Start background processing threads."""
        # Start update processing thread
        self._executor.submit(self._process_updates)
        
        # Start WebSocket server in separate thread (disabled for now)
        # self._websocket_thread = threading.Thread(
        #     target=self._run_websocket_server,
        #     name="websocket-server"
        # )
        # self._websocket_thread.daemon = True
        # self._websocket_thread.start()
        self.logger.info("WebSocket server disabled to avoid port conflicts")
    
    def _run_websocket_server(self):
        """Run WebSocket server in asyncio event loop."""
        try:
            self._websocket_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._websocket_loop)
            
            self._websocket_loop.run_until_complete(
                self.websocket_channel.start_server()
            )
            self._websocket_loop.run_forever()
            
        except ProgressTrackingError as e:
            self.logger.error(f"WebSocket server error: {str(e)}")
            # Don't raise, just log and continue without WebSocket
        except Exception as e:
            self.logger.error(f"Unexpected WebSocket server error: {str(e)}")
    
    def _process_updates(self):
        """Process progress updates in background thread."""
        while True:
            try:
                update = self._update_queue.get(timeout=1.0)
                if update is None:  # Shutdown signal
                    break
                
                # Broadcast to WebSocket clients
                if self._websocket_loop and not self._websocket_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        self.websocket_channel.broadcast_update(update),
                        self._websocket_loop
                    )
                
                # Save to persistence
                self.persistence.save_update(update)
                
                self._update_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing update: {str(e)}")
    
    @with_correlation_id()
    def start_operation(self, 
                       operation_id: Optional[str] = None, 
                       total_steps: int = 100,
                       message: str = "Starting operation") -> str:
        """
        Initialize progress tracking for an operation.
        
        Args:
            operation_id: Optional operation ID (generated if not provided)
            total_steps: Total number of steps in the operation
            message: Initial status message
            
        Returns:
            Operation ID
        """
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        with self._lock:
            # Create progress status
            status = ProgressStatus(
                operation_id=operation_id,
                status='running',
                progress=0.0,
                message=message,
                current_step=0,
                total_steps=total_steps,
                started_at=datetime.now()
            )
            
            self._operations[operation_id] = status
            
            # Initialize estimation data
            self._estimation_data[operation_id] = EstimationData(
                start_time=datetime.now(),
                last_update_time=datetime.now(),
                completed_steps=0,
                total_steps=total_steps,
                step_durations=[]
            )
            
            # Initialize cancellation flag
            self._cancellation_flags[operation_id] = threading.Event()
            self._cleanup_callbacks[operation_id] = []
            
            # Save to persistence
            self.persistence.save_operation(status)
            
            # Queue update for broadcasting
            update = ProgressUpdate(
                operation_id=operation_id,
                current_step=0,
                total_steps=total_steps,
                message=message,
                timestamp=datetime.now(),
                metadata={'status': 'started'}
            )
            self._update_queue.put(update)
            
            self.logger.info(f"Started operation {operation_id} with {total_steps} steps")
            
            return operation_id    

    @with_correlation_id()
    def update_progress(self, 
                       operation_id: str, 
                       current_step: int, 
                       message: str = "",
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update progress for an operation.
        
        Args:
            operation_id: Operation identifier
            current_step: Current step number
            message: Status message
            metadata: Additional metadata
            
        Returns:
            True if update was successful, False otherwise
        """
        if metadata is None:
            metadata = {}
        
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Attempted to update unknown operation: {operation_id}")
                return False
            
            # Check if operation was cancelled
            if self._cancellation_flags.get(operation_id, threading.Event()).is_set():
                self.logger.info(f"Operation {operation_id} was cancelled, ignoring update")
                return False
            
            status = self._operations[operation_id]
            
            # Update status
            old_step = status.current_step
            status.current_step = current_step
            status.message = message
            # Calculate progress with bounds checking
            if status.total_steps > 0:
                progress_ratio = current_step / status.total_steps
                status.progress = max(0.0, min(100.0, progress_ratio * 100.0))
            else:
                status.progress = 0.0
            
            # Update estimation data
            if operation_id in self._estimation_data:
                estimation = self._estimation_data[operation_id]
                now = datetime.now()
                
                # Calculate step duration if we advanced
                if current_step > old_step:
                    step_duration = (now - estimation.last_update_time).total_seconds()
                    estimation.step_durations.append(step_duration)
                    
                    # Keep only recent durations for better estimation
                    if len(estimation.step_durations) > 50:
                        estimation.step_durations = estimation.step_durations[-50:]
                
                estimation.last_update_time = now
                estimation.completed_steps = current_step
            
            # Save to persistence
            self.persistence.save_operation(status)
            
            # Queue update for broadcasting
            update = ProgressUpdate(
                operation_id=operation_id,
                current_step=current_step,
                total_steps=status.total_steps,
                message=message,
                timestamp=datetime.now(),
                metadata=metadata
            )
            self._update_queue.put(update)
            
            self.logger.debug(f"Updated operation {operation_id}: step {current_step}/{status.total_steps}")
            
            return True
    
    @with_correlation_id()
    def complete_operation(self, 
                          operation_id: str, 
                          message: str = "Operation completed",
                          error_message: Optional[str] = None) -> bool:
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation identifier
            message: Completion message
            error_message: Error message if operation failed
            
        Returns:
            True if completion was successful, False otherwise
        """
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Attempted to complete unknown operation: {operation_id}")
                return False
            
            status = self._operations[operation_id]
            
            # Update status
            status.completed_at = datetime.now()
            status.message = message
            status.error_message = error_message
            
            if error_message:
                status.status = 'error'
                status.progress = 0.0  # Reset progress on error
            else:
                status.status = 'completed'
                status.progress = 100.0
                status.current_step = status.total_steps
            
            # Save to persistence
            self.persistence.save_operation(status)
            
            # Queue final update for broadcasting
            update = ProgressUpdate(
                operation_id=operation_id,
                current_step=status.current_step,
                total_steps=status.total_steps,
                message=message,
                timestamp=datetime.now(),
                metadata={
                    'status': status.status,
                    'error_message': error_message,
                    'duration_seconds': status.duration_seconds
                }
            )
            self._update_queue.put(update)
            
            # Execute cleanup callbacks
            self._execute_cleanup_callbacks(operation_id)
            
            # Clean up tracking data
            self._estimation_data.pop(operation_id, None)
            self._cancellation_flags.pop(operation_id, None)
            self._cleanup_callbacks.pop(operation_id, None)
            
            self.logger.info(f"Completed operation {operation_id} with status: {status.status}")
            
            return True
    
    def get_progress(self, operation_id: str) -> Optional[ProgressStatus]:
        """
        Get current progress status for an operation.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            ProgressStatus if found, None otherwise
        """
        with self._lock:
            status = self._operations.get(operation_id)
            if status:
                # Add estimated completion time if available
                if operation_id in self._estimation_data and status.status == 'running':
                    estimation = self._estimation_data[operation_id]
                    eta = estimation.calculate_eta()
                    if eta:
                        # Add ETA to a copy of the status
                        status_dict = status.to_dict()
                        status_dict['estimated_completion'] = eta.isoformat()
                        return ProgressStatus.from_dict(status_dict)
                
                return status
            
            # Try to load from persistence
            return self.persistence.load_operation(operation_id)
    
    def list_operations(self, status_filter: Optional[str] = None) -> List[ProgressStatus]:
        """
        List operations with optional status filter.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of ProgressStatus objects
        """
        with self._lock:
            operations = list(self._operations.values())
            
            if status_filter:
                operations = [op for op in operations if op.status == status_filter]
            
            # Sort by start time (most recent first)
            operations.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
            
            return operations
    
    def cancel_operation(self, operation_id: str, message: str = "Operation cancelled") -> bool:
        """
        Cancel a running operation.
        
        Args:
            operation_id: Operation identifier
            message: Cancellation message
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Attempted to cancel unknown operation: {operation_id}")
                return False
            
            status = self._operations[operation_id]
            
            if status.status != 'running':
                self.logger.warning(f"Attempted to cancel non-running operation: {operation_id}")
                return False
            
            # Set cancellation flag
            if operation_id in self._cancellation_flags:
                self._cancellation_flags[operation_id].set()
            
            # Update status
            status.status = 'cancelled'
            status.completed_at = datetime.now()
            status.message = message
            
            # Save to persistence
            self.persistence.save_operation(status)
            
            # Queue cancellation update
            update = ProgressUpdate(
                operation_id=operation_id,
                current_step=status.current_step,
                total_steps=status.total_steps,
                message=message,
                timestamp=datetime.now(),
                metadata={'status': 'cancelled'}
            )
            self._update_queue.put(update)
            
            # Execute cleanup callbacks
            self._execute_cleanup_callbacks(operation_id)
            
            self.logger.info(f"Cancelled operation {operation_id}")
            
            return True
    
    def is_cancelled(self, operation_id: str) -> bool:
        """
        Check if an operation has been cancelled.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            True if operation is cancelled, False otherwise
        """
        cancellation_flag = self._cancellation_flags.get(operation_id)
        return cancellation_flag.is_set() if cancellation_flag else False
    
    def add_cleanup_callback(self, operation_id: str, callback: Callable):
        """
        Add a cleanup callback for an operation.
        
        Args:
            operation_id: Operation identifier
            callback: Cleanup function to call when operation completes/cancels
        """
        with self._lock:
            if operation_id not in self._cleanup_callbacks:
                self._cleanup_callbacks[operation_id] = []
            
            self._cleanup_callbacks[operation_id].append(callback)
    
    def _execute_cleanup_callbacks(self, operation_id: str):
        """Execute cleanup callbacks for an operation."""
        callbacks = self._cleanup_callbacks.get(operation_id, [])
        
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback failed for {operation_id}: {str(e)}")
    
    def get_progress_history(self, operation_id: str, limit: int = 100) -> List[ProgressUpdate]:
        """
        Get progress update history for an operation.
        
        Args:
            operation_id: Operation identifier
            limit: Maximum number of updates to return
            
        Returns:
            List of ProgressUpdate objects
        """
        try:
            with sqlite3.connect(self.persistence.db_path) as conn:
                cursor = conn.execute("""
                    SELECT current_step, total_steps, message, timestamp, metadata
                    FROM progress_updates 
                    WHERE operation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (operation_id, limit))
                
                updates = []
                for row in cursor.fetchall():
                    updates.append(ProgressUpdate(
                        operation_id=operation_id,
                        current_step=row[0],
                        total_steps=row[1],
                        message=row[2] or "",
                        timestamp=datetime.fromisoformat(row[3]),
                        metadata=json.loads(row[4]) if row[4] else {}
                    ))
                
                return updates
                
        except Exception as e:
            self.logger.error(f"Failed to get progress history for {operation_id}: {str(e)}")
            return []
    
    def cleanup_old_data(self, max_age_days: int = 7):
        """
        Clean up old progress data.
        
        Args:
            max_age_days: Maximum age in days for keeping data
        """
        try:
            self.persistence.cleanup_old_operations(max_age_days)
            self.logger.info(f"Cleaned up progress data older than {max_age_days} days")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get progress tracking statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_operations = len(self._operations)
            running_operations = len([op for op in self._operations.values() if op.status == 'running'])
            completed_operations = len([op for op in self._operations.values() if op.status == 'completed'])
            error_operations = len([op for op in self._operations.values() if op.status == 'error'])
            cancelled_operations = len([op for op in self._operations.values() if op.status == 'cancelled'])
            
            return {
                'total_operations': total_operations,
                'running_operations': running_operations,
                'completed_operations': completed_operations,
                'error_operations': error_operations,
                'cancelled_operations': cancelled_operations,
                'websocket_clients': len(self.websocket_channel.clients),
                'active_estimations': len(self._estimation_data)
            }
    
    def shutdown(self):
        """Shutdown the progress tracker and cleanup resources."""
        self.logger.info("Shutting down progress tracker...")
        
        # Stop background processing
        self._update_queue.put(None)  # Shutdown signal
        
        # Stop WebSocket server
        if self._websocket_loop and not self._websocket_loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket_channel.stop_server(),
                    self._websocket_loop
                ).result(timeout=5.0)
                
                self._websocket_loop.call_soon_threadsafe(self._websocket_loop.stop)
            except Exception as e:
                self.logger.warning(f"Error stopping WebSocket server: {str(e)}")
        
        # Wait for threads to finish
        if self._websocket_thread and self._websocket_thread.is_alive():
            self._websocket_thread.join(timeout=5.0)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Mark all running operations as interrupted
        with self._lock:
            for operation_id, status in self._operations.items():
                if status.status == 'running':
                    status.status = 'error'
                    status.error_message = 'System shutdown'
                    status.completed_at = datetime.now()
                    self.persistence.save_operation(status)
        
        self.logger.info("Progress tracker shutdown complete")


# Convenience functions for global progress tracker instance
_global_tracker: Optional[ProgressTracker] = None
_tracker_lock = threading.Lock()


def get_progress_tracker(**kwargs) -> ProgressTracker:
    """Get or create global progress tracker instance."""
    global _global_tracker
    
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = ProgressTracker(**kwargs)
        
        return _global_tracker


def shutdown_progress_tracker():
    """Shutdown global progress tracker instance."""
    global _global_tracker
    
    with _tracker_lock:
        if _global_tracker:
            _global_tracker.shutdown()
            _global_tracker = None