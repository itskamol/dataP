"""
Unit tests for the ProgressTracker system.
Tests progress tracking accuracy, persistence, WebSocket/HTTP channels, and edge cases.
"""

import pytest
import tempfile
import threading
import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio
import websockets

from src.infrastructure.progress_tracker import (
    ProgressTracker, ProgressPersistence, ProgressUpdate, EstimationData,
    WebSocketProgressChannel, HTTPProgressChannel, get_progress_tracker, shutdown_progress_tracker
)
from src.domain.models import ProgressStatus
from src.domain.exceptions import ProgressTrackingError


class TestProgressPersistence:
    """Test progress persistence functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_progress.db"
        self.persistence = ProgressPersistence(str(self.db_path))
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_database_initialization(self):
        """Test database initialization creates required tables."""
        assert self.db_path.exists()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('progress_operations', 'progress_updates')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        assert 'progress_operations' in tables
        assert 'progress_updates' in tables
    
    def test_save_and_load_operation(self):
        """Test saving and loading operation status."""
        # Create test status
        status = ProgressStatus(
            operation_id="test-op-1",
            status="running",
            progress=50.0,
            message="Test operation",
            current_step=5,
            total_steps=10,
            started_at=datetime.now()
        )
        
        # Save operation
        self.persistence.save_operation(status)
        
        # Load operation
        loaded_status = self.persistence.load_operation("test-op-1")
        
        assert loaded_status is not None
        assert loaded_status.operation_id == status.operation_id
        assert loaded_status.status == status.status
        assert loaded_status.progress == status.progress
        assert loaded_status.message == status.message
        assert loaded_status.current_step == status.current_step
        assert loaded_status.total_steps == status.total_steps
    
    def test_save_progress_update(self):
        """Test saving progress updates."""
        update = ProgressUpdate(
            operation_id="test-op-1",
            current_step=3,
            total_steps=10,
            message="Processing step 3",
            timestamp=datetime.now(),
            metadata={"test": "data"}
        )
        
        # Save update
        self.persistence.save_update(update)
        
        # Verify update was saved
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT operation_id, current_step, message, metadata
                FROM progress_updates 
                WHERE operation_id = ?
            """, (update.operation_id,))
            
            row = cursor.fetchone()
            
        assert row is not None
        assert row[0] == update.operation_id
        assert row[1] == update.current_step
        assert row[2] == update.message
        assert json.loads(row[3]) == update.metadata
    
    def test_get_active_operations(self):
        """Test retrieving active operations."""
        # Create test operations
        running_status = ProgressStatus(
            operation_id="running-op",
            status="running",
            progress=30.0,
            message="Running operation",
            current_step=3,
            total_steps=10,
            started_at=datetime.now()
        )
        
        completed_status = ProgressStatus(
            operation_id="completed-op",
            status="completed",
            progress=100.0,
            message="Completed operation",
            current_step=10,
            total_steps=10,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Save operations
        self.persistence.save_operation(running_status)
        self.persistence.save_operation(completed_status)
        
        # Get active operations
        active_ops = self.persistence.get_active_operations()
        
        assert len(active_ops) == 1
        assert active_ops[0].operation_id == "running-op"
        assert active_ops[0].status == "running"
    
    def test_cleanup_old_operations(self):
        """Test cleanup of old operations."""
        # Create old operation
        old_status = ProgressStatus(
            operation_id="old-op",
            status="completed",
            progress=100.0,
            message="Old operation",
            current_step=10,
            total_steps=10,
            started_at=datetime.now() - timedelta(days=10),
            completed_at=datetime.now() - timedelta(days=10)
        )
        
        # Create recent operation
        recent_status = ProgressStatus(
            operation_id="recent-op",
            status="completed",
            progress=100.0,
            message="Recent operation",
            current_step=10,
            total_steps=10,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Save operations
        self.persistence.save_operation(old_status)
        self.persistence.save_operation(recent_status)
        
        # Cleanup operations older than 7 days
        self.persistence.cleanup_old_operations(max_age_days=7)
        
        # Verify old operation was deleted
        assert self.persistence.load_operation("old-op") is None
        assert self.persistence.load_operation("recent-op") is not None


class TestEstimationData:
    """Test completion time estimation functionality."""
    
    def test_calculate_eta_with_step_durations(self):
        """Test ETA calculation with step duration history."""
        start_time = datetime.now() - timedelta(minutes=5)
        
        estimation = EstimationData(
            start_time=start_time,
            last_update_time=datetime.now(),
            completed_steps=5,
            total_steps=10,
            step_durations=[30.0, 25.0, 35.0, 20.0, 30.0]  # seconds per step
        )
        
        eta = estimation.calculate_eta()
        
        assert eta is not None
        # Should be approximately 5 steps * 28 seconds average = 140 seconds from now
        expected_eta = datetime.now() + timedelta(seconds=140)
        assert abs((eta - expected_eta).total_seconds()) < 60  # Within 1 minute tolerance
    
    def test_calculate_eta_without_step_durations(self):
        """Test ETA calculation without step duration history."""
        start_time = datetime.now() - timedelta(minutes=5)
        
        estimation = EstimationData(
            start_time=start_time,
            last_update_time=datetime.now(),
            completed_steps=5,
            total_steps=10,
            step_durations=[]
        )
        
        eta = estimation.calculate_eta()
        
        assert eta is not None
        # Should use overall average: 5 minutes for 5 steps = 1 minute per step
        # Remaining 5 steps = 5 minutes
        expected_eta = datetime.now() + timedelta(minutes=5)
        assert abs((eta - expected_eta).total_seconds()) < 120  # Within 2 minutes tolerance
    
    def test_calculate_eta_edge_cases(self):
        """Test ETA calculation edge cases."""
        # No completed steps
        estimation = EstimationData(
            start_time=datetime.now(),
            last_update_time=datetime.now(),
            completed_steps=0,
            total_steps=10,
            step_durations=[]
        )
        
        assert estimation.calculate_eta() is None
        
        # All steps completed
        estimation.completed_steps = 10
        eta = estimation.calculate_eta()
        
        assert eta is not None
        assert abs((eta - datetime.now()).total_seconds()) < 5  # Should be now


class TestProgressTracker:
    """Test main ProgressTracker functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_progress.db"
        
        # Create a minimal tracker for testing core functionality
        self.tracker = ProgressTracker.__new__(ProgressTracker)
        self.tracker.logger = get_logger('progress_tracker')
        
        # Core tracking data
        self.tracker._operations = {}
        self.tracker._estimation_data = {}
        self.tracker._cancellation_flags = {}
        self.tracker._cleanup_callbacks = {}
        self.tracker._lock = threading.RLock()
        
        # Persistence
        self.tracker.persistence = ProgressPersistence(str(self.db_path))
        
        # Mock channels to avoid network operations
        self.tracker.websocket_channel = Mock()
        self.tracker.http_channel = Mock()
        
        # Mock background processing
        self.tracker._update_queue = Mock()
        self.tracker._executor = Mock()
        self.tracker._websocket_loop = None
        self.tracker._websocket_thread = None
    
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, 'tracker'):
            self.tracker.shutdown()
        
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_start_operation(self):
        """Test starting a new operation."""
        operation_id = self.tracker.start_operation(
            total_steps=10,
            message="Test operation"
        )
        
        assert operation_id is not None
        assert len(operation_id) > 0
        
        # Verify operation was created
        status = self.tracker.get_progress(operation_id)
        assert status is not None
        assert status.operation_id == operation_id
        assert status.status == "running"
        assert status.progress == 0.0
        assert status.current_step == 0
        assert status.total_steps == 10
        assert status.message == "Test operation"
        assert status.started_at is not None
    
    def test_update_progress(self):
        """Test updating operation progress."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Update progress
        success = self.tracker.update_progress(
            operation_id=operation_id,
            current_step=5,
            message="Halfway done"
        )
        
        assert success is True
        
        # Verify progress was updated
        status = self.tracker.get_progress(operation_id)
        assert status.current_step == 5
        assert status.progress == 50.0
        assert status.message == "Halfway done"
    
    def test_complete_operation_success(self):
        """Test completing an operation successfully."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Complete operation
        success = self.tracker.complete_operation(
            operation_id=operation_id,
            message="Operation completed successfully"
        )
        
        assert success is True
        
        # Verify operation was completed
        status = self.tracker.get_progress(operation_id)
        assert status.status == "completed"
        assert status.progress == 100.0
        assert status.current_step == 10
        assert status.message == "Operation completed successfully"
        assert status.completed_at is not None
        assert status.error_message is None
    
    def test_complete_operation_with_error(self):
        """Test completing an operation with error."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Complete operation with error
        success = self.tracker.complete_operation(
            operation_id=operation_id,
            message="Operation failed",
            error_message="Test error occurred"
        )
        
        assert success is True
        
        # Verify operation was marked as error
        status = self.tracker.get_progress(operation_id)
        assert status.status == "error"
        assert status.progress == 0.0
        assert status.message == "Operation failed"
        assert status.error_message == "Test error occurred"
        assert status.completed_at is not None
    
    def test_cancel_operation(self):
        """Test cancelling a running operation."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Cancel operation
        success = self.tracker.cancel_operation(
            operation_id=operation_id,
            message="Operation cancelled by user"
        )
        
        assert success is True
        
        # Verify operation was cancelled
        status = self.tracker.get_progress(operation_id)
        assert status.status == "cancelled"
        assert status.message == "Operation cancelled by user"
        assert status.completed_at is not None
        
        # Verify cancellation flag is set
        assert self.tracker.is_cancelled(operation_id) is True
    
    def test_cancel_nonexistent_operation(self):
        """Test cancelling a non-existent operation."""
        success = self.tracker.cancel_operation("nonexistent-op")
        assert success is False
    
    def test_cancel_completed_operation(self):
        """Test cancelling an already completed operation."""
        operation_id = self.tracker.start_operation(total_steps=10)
        self.tracker.complete_operation(operation_id)
        
        # Try to cancel completed operation
        success = self.tracker.cancel_operation(operation_id)
        assert success is False
    
    def test_update_cancelled_operation(self):
        """Test updating a cancelled operation."""
        operation_id = self.tracker.start_operation(total_steps=10)
        self.tracker.cancel_operation(operation_id)
        
        # Try to update cancelled operation
        success = self.tracker.update_progress(operation_id, 5)
        assert success is False
    
    def test_list_operations(self):
        """Test listing operations."""
        # Create multiple operations
        op1 = self.tracker.start_operation(total_steps=10, message="Operation 1")
        op2 = self.tracker.start_operation(total_steps=20, message="Operation 2")
        self.tracker.complete_operation(op1)
        
        # List all operations
        all_ops = self.tracker.list_operations()
        assert len(all_ops) == 2
        
        # List only running operations
        running_ops = self.tracker.list_operations(status_filter="running")
        assert len(running_ops) == 1
        assert running_ops[0].operation_id == op2
        
        # List only completed operations
        completed_ops = self.tracker.list_operations(status_filter="completed")
        assert len(completed_ops) == 1
        assert completed_ops[0].operation_id == op1
    
    def test_cleanup_callbacks(self):
        """Test cleanup callback execution."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Add cleanup callbacks
        callback1_called = threading.Event()
        callback2_called = threading.Event()
        
        def cleanup1():
            callback1_called.set()
        
        def cleanup2():
            callback2_called.set()
        
        self.tracker.add_cleanup_callback(operation_id, cleanup1)
        self.tracker.add_cleanup_callback(operation_id, cleanup2)
        
        # Complete operation
        self.tracker.complete_operation(operation_id)
        
        # Wait for callbacks to be called
        assert callback1_called.wait(timeout=1.0)
        assert callback2_called.wait(timeout=1.0)
    
    def test_cleanup_callback_exception_handling(self):
        """Test cleanup callback exception handling."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Add callback that raises exception
        def failing_callback():
            raise Exception("Callback failed")
        
        # Add normal callback
        normal_callback_called = threading.Event()
        
        def normal_callback():
            normal_callback_called.set()
        
        self.tracker.add_cleanup_callback(operation_id, failing_callback)
        self.tracker.add_cleanup_callback(operation_id, normal_callback)
        
        # Complete operation (should not fail despite callback exception)
        success = self.tracker.complete_operation(operation_id)
        assert success is True
        
        # Normal callback should still be called
        assert normal_callback_called.wait(timeout=1.0)
    
    def test_get_statistics(self):
        """Test getting progress tracking statistics."""
        # Create operations with different statuses
        op1 = self.tracker.start_operation(total_steps=10)
        op2 = self.tracker.start_operation(total_steps=20)
        op3 = self.tracker.start_operation(total_steps=30)
        
        self.tracker.complete_operation(op1)
        self.tracker.complete_operation(op2, error_message="Test error")
        self.tracker.cancel_operation(op3)
        
        # Get statistics
        stats = self.tracker.get_statistics()
        
        assert stats['total_operations'] == 3
        assert stats['running_operations'] == 0
        assert stats['completed_operations'] == 1
        assert stats['error_operations'] == 1
        assert stats['cancelled_operations'] == 1
        assert 'websocket_clients' in stats
        assert 'active_estimations' in stats
    
    def test_progress_history(self):
        """Test getting progress history."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Make several progress updates
        for i in range(1, 6):
            self.tracker.update_progress(operation_id, i, f"Step {i}")
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get progress history
        history = self.tracker.get_progress_history(operation_id)
        
        assert len(history) >= 5  # At least 5 updates (plus initial)
        
        # Verify history is in reverse chronological order
        for i in range(len(history) - 1):
            assert history[i].timestamp >= history[i + 1].timestamp
    
    def test_operation_recovery(self):
        """Test operation recovery from persistence."""
        # Create and save an operation
        operation_id = self.tracker.start_operation(total_steps=10)
        self.tracker.update_progress(operation_id, 5)
        
        # Shutdown tracker
        self.tracker.shutdown()
        
        # Create new tracker (should recover operations)
        with patch('src.infrastructure.progress_tracker.WebSocketProgressChannel'), \
             patch('src.infrastructure.progress_tracker.HTTPProgressChannel'):
            new_tracker = ProgressTracker(
                websocket_host="localhost",
                websocket_port=0,
                persistence_db=str(self.db_path)
            )
        
        try:
            # Verify operation was recovered
            recovered_status = new_tracker.get_progress(operation_id)
            assert recovered_status is not None
            assert recovered_status.operation_id == operation_id
            assert recovered_status.current_step == 5
            assert "[RECOVERED]" in recovered_status.message
            
        finally:
            new_tracker.shutdown()


class TestWebSocketProgressChannel:
    """Test WebSocket progress channel functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_server_startup(self):
        """Test WebSocket server startup and shutdown."""
        channel = WebSocketProgressChannel(host="localhost", port=0)
        
        # Start server
        await channel.start_server()
        assert channel._running is True
        
        # Stop server
        await channel.stop_server()
        assert channel._running is False
    
    @pytest.mark.asyncio
    async def test_broadcast_update(self):
        """Test broadcasting updates to WebSocket clients."""
        channel = WebSocketProgressChannel(host="localhost", port=0)
        
        # Mock WebSocket clients
        mock_client1 = Mock()
        mock_client2 = Mock()
        
        channel.clients.add(mock_client1)
        channel.clients.add(mock_client2)
        
        # Create test update
        update = ProgressUpdate(
            operation_id="test-op",
            current_step=5,
            total_steps=10,
            message="Test update",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Broadcast update
        await channel.broadcast_update(update)
        
        # Verify clients received update
        expected_message = json.dumps(update.to_dict())
        mock_client1.send.assert_called_once_with(expected_message)
        mock_client2.send.assert_called_once_with(expected_message)


class TestHTTPProgressChannel:
    """Test HTTP progress channel functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.channel = HTTPProgressChannel()
        self.app = self.channel.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock progress tracker
        self.mock_tracker = Mock()
        self.channel.set_progress_tracker(self.mock_tracker)
    
    def test_get_progress_endpoint(self):
        """Test GET /api/progress/<operation_id> endpoint."""
        # Mock progress status
        mock_status = ProgressStatus(
            operation_id="test-op",
            status="running",
            progress=50.0,
            message="Test operation",
            current_step=5,
            total_steps=10
        )
        
        self.mock_tracker.get_progress.return_value = mock_status
        
        # Make request
        response = self.client.get('/api/progress/test-op')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['operation_id'] == "test-op"
        assert data['progress'] == 50.0
    
    def test_get_progress_not_found(self):
        """Test GET /api/progress/<operation_id> with non-existent operation."""
        self.mock_tracker.get_progress.return_value = None
        
        response = self.client.get('/api/progress/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_list_operations_endpoint(self):
        """Test GET /api/progress endpoint."""
        # Mock operations list
        mock_operations = [
            ProgressStatus(
                operation_id="op1",
                status="running",
                progress=30.0,
                message="Operation 1",
                current_step=3,
                total_steps=10
            ),
            ProgressStatus(
                operation_id="op2",
                status="completed",
                progress=100.0,
                message="Operation 2",
                current_step=10,
                total_steps=10
            )
        ]
        
        self.mock_tracker.list_operations.return_value = mock_operations
        
        # Make request
        response = self.client.get('/api/progress')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 2
        assert len(data['operations']) == 2
    
    def test_cancel_operation_endpoint(self):
        """Test POST /api/progress/<operation_id>/cancel endpoint."""
        self.mock_tracker.cancel_operation.return_value = True
        
        response = self.client.post('/api/progress/test-op/cancel')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'message' in data
        
        self.mock_tracker.cancel_operation.assert_called_once_with("test-op")
    
    def test_cancel_operation_failure(self):
        """Test POST /api/progress/<operation_id>/cancel with failure."""
        self.mock_tracker.cancel_operation.return_value = False
        
        response = self.client.post('/api/progress/test-op/cancel')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_get_progress_history_endpoint(self):
        """Test GET /api/progress/<operation_id>/history endpoint."""
        # Mock progress history
        mock_history = [
            ProgressUpdate(
                operation_id="test-op",
                current_step=3,
                total_steps=10,
                message="Step 3",
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        self.mock_tracker.get_progress_history.return_value = mock_history
        
        response = self.client.get('/api/progress/test-op/history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['operation_id'] == "test-op"
        assert len(data['updates']) == 1


class TestGlobalProgressTracker:
    """Test global progress tracker functions."""
    
    def teardown_method(self):
        """Cleanup global tracker."""
        shutdown_progress_tracker()
    
    def test_get_global_tracker(self):
        """Test getting global progress tracker instance."""
        tracker1 = get_progress_tracker()
        tracker2 = get_progress_tracker()
        
        # Should return same instance
        assert tracker1 is tracker2
    
    def test_shutdown_global_tracker(self):
        """Test shutting down global progress tracker."""
        tracker = get_progress_tracker()
        assert tracker is not None
        
        shutdown_progress_tracker()
        
        # Getting tracker again should create new instance
        new_tracker = get_progress_tracker()
        assert new_tracker is not tracker


class TestProgressTrackerEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_progress.db"
        
        with patch('src.infrastructure.progress_tracker.WebSocketProgressChannel'), \
             patch('src.infrastructure.progress_tracker.HTTPProgressChannel'):
            self.tracker = ProgressTracker(
                websocket_host="localhost",
                websocket_port=0,
                persistence_db=str(self.db_path)
            )
    
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self, 'tracker'):
            self.tracker.shutdown()
        
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_update_nonexistent_operation(self):
        """Test updating a non-existent operation."""
        success = self.tracker.update_progress("nonexistent-op", 5)
        assert success is False
    
    def test_complete_nonexistent_operation(self):
        """Test completing a non-existent operation."""
        success = self.tracker.complete_operation("nonexistent-op")
        assert success is False
    
    def test_get_progress_nonexistent_operation(self):
        """Test getting progress for non-existent operation."""
        status = self.tracker.get_progress("nonexistent-op")
        assert status is None
    
    def test_zero_total_steps(self):
        """Test operation with zero total steps."""
        operation_id = self.tracker.start_operation(total_steps=0)
        
        status = self.tracker.get_progress(operation_id)
        assert status.total_steps == 0
        assert status.progress == 0.0
        
        # Update should still work
        success = self.tracker.update_progress(operation_id, 0)
        assert success is True
    
    def test_negative_step_values(self):
        """Test handling of negative step values."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        # Negative current step should still be accepted
        success = self.tracker.update_progress(operation_id, -1)
        assert success is True
        
        status = self.tracker.get_progress(operation_id)
        assert status.current_step == -1
        # Progress should be 0 for negative values
        assert status.progress <= 0.0
    
    def test_step_exceeding_total(self):
        """Test step value exceeding total steps."""
        operation_id = self.tracker.start_operation(total_steps=10)
        
        success = self.tracker.update_progress(operation_id, 15)
        assert success is True
        
        status = self.tracker.get_progress(operation_id)
        assert status.current_step == 15
        # Progress should be capped at reasonable value
        assert status.progress >= 100.0
    
    def test_concurrent_operations(self):
        """Test handling multiple concurrent operations."""
        num_operations = 10
        operations = []
        
        # Start multiple operations
        for i in range(num_operations):
            op_id = self.tracker.start_operation(
                total_steps=100,
                message=f"Operation {i}"
            )
            operations.append(op_id)
        
        # Update all operations concurrently
        def update_operation(op_id, step):
            self.tracker.update_progress(op_id, step, f"Step {step}")
        
        threads = []
        for i, op_id in enumerate(operations):
            thread = threading.Thread(
                target=update_operation,
                args=(op_id, (i + 1) * 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all updates to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations were updated correctly
        for i, op_id in enumerate(operations):
            status = self.tracker.get_progress(op_id)
            assert status.current_step == (i + 1) * 10
            assert status.message == f"Step {(i + 1) * 10}"
    
    def test_database_corruption_handling(self):
        """Test handling of database corruption."""
        # Corrupt the database file
        with open(self.db_path, 'w') as f:
            f.write("corrupted data")
        
        # Creating new persistence should handle corruption gracefully
        with pytest.raises(ProgressTrackingError):
            ProgressPersistence(str(self.db_path))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])