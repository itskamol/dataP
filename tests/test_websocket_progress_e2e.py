"""
End-to-end tests for WebSocket progress updates and web interface functionality.
Tests requirements 5.3, 5.4, 4.1: Real-time progress updates with WebSocket support.
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import socketio
from flask import Flask
from flask_socketio import SocketIO

from src.web.services.websocket_progress_service import WebSocketProgressService, websocket_progress_service
from web_app import app, socketio as app_socketio


class TestWebSocketProgressE2E:
    """End-to-end tests for WebSocket progress functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def socketio_client(self):
        """Create SocketIO test client."""
        app.config['TESTING'] = True
        client = socketio.test_client(app_socketio, namespace='/progress')
        yield client
        client.disconnect()
    
    @pytest.fixture
    def progress_service(self):
        """Create fresh progress service for testing."""
        service = WebSocketProgressService()
        service.set_socketio(app_socketio)
        return service
    
    def test_websocket_connection_and_disconnection(self, socketio_client):
        """Test WebSocket connection and disconnection."""
        # Test connection
        assert socketio_client.is_connected(namespace='/progress')
        
        # Test disconnection
        socketio_client.disconnect(namespace='/progress')
        assert not socketio_client.is_connected(namespace='/progress')
    
    def test_join_and_leave_operation(self, socketio_client, progress_service):
        """Test joining and leaving operation rooms."""
        operation_id = progress_service.start_operation(
            total_steps=10,
            message="Test operation"
        )
        
        # Test joining operation
        socketio_client.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
        received = socketio_client.get_received(namespace='/progress')
        
        assert len(received) > 0
        join_response = next((r for r in received if r['name'] == 'joined'), None)
        assert join_response is not None
        assert join_response['args'][0]['operation_id'] == operation_id
        
        # Test leaving operation
        socketio_client.emit('leave_operation', {'operation_id': operation_id}, namespace='/progress')
        received = socketio_client.get_received(namespace='/progress')
        
        leave_response = next((r for r in received if r['name'] == 'left'), None)
        assert leave_response is not None
        assert leave_response['args'][0]['operation_id'] == operation_id
    
    def test_progress_updates_via_websocket(self, socketio_client, progress_service):
        """Test real-time progress updates via WebSocket."""
        operation_id = progress_service.start_operation(
            total_steps=100,
            message="Test progress updates"
        )
        
        # Join operation
        socketio_client.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
        socketio_client.get_received(namespace='/progress')  # Clear initial messages
        
        # Send progress updates
        progress_service.update_progress(operation_id, 25, "25% complete", "processing")
        progress_service.update_progress(operation_id, 50, "50% complete", "processing")
        progress_service.update_progress(operation_id, 75, "75% complete", "processing")
        
        # Allow time for WebSocket messages to be sent
        time.sleep(0.1)
        
        # Check received progress updates
        received = socketio_client.get_received(namespace='/progress')
        progress_updates = [r for r in received if r['name'] == 'progress_update']
        
        assert len(progress_updates) >= 3
        
        # Verify progress update structure
        for update in progress_updates:
            data = update['args'][0]
            assert 'operation_id' in data
            assert 'status' in data
            assert 'progress' in data
            assert 'message' in data
            assert 'timestamp' in data
            assert 'current_step' in data
            assert 'total_steps' in data
            assert 'elapsed_time' in data
            assert 'can_cancel' in data
            assert data['operation_id'] == operation_id
    
    def test_operation_completion_via_websocket(self, socketio_client, progress_service):
        """Test operation completion notifications via WebSocket."""
        operation_id = progress_service.start_operation(
            total_steps=10,
            message="Test completion"
        )
        
        # Join operation
        socketio_client.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
        socketio_client.get_received(namespace='/progress')  # Clear initial messages
        
        # Complete operation
        progress_service.complete_operation(
            operation_id=operation_id,
            message="Operation completed successfully",
            success=True
        )
        
        # Allow time for WebSocket message
        time.sleep(0.1)
        
        # Check completion notification
        received = socketio_client.get_received(namespace='/progress')
        progress_updates = [r for r in received if r['name'] == 'progress_update']
        
        assert len(progress_updates) > 0
        completion_update = progress_updates[-1]['args'][0]
        assert completion_update['status'] == 'completed'
        assert completion_update['progress'] == 100
        assert completion_update['can_cancel'] is False
    
    def test_operation_cancellation_via_websocket(self, socketio_client, progress_service):
        """Test operation cancellation via WebSocket."""
        operation_id = progress_service.start_operation(
            total_steps=100,
            message="Test cancellation"
        )
        
        # Join operation
        socketio_client.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
        socketio_client.get_received(namespace='/progress')  # Clear initial messages
        
        # Cancel operation via WebSocket
        socketio_client.emit('cancel_operation', {'operation_id': operation_id}, namespace='/progress')
        
        # Allow time for processing
        time.sleep(0.1)
        
        # Check cancel response and progress update
        received = socketio_client.get_received(namespace='/progress')
        
        cancel_response = next((r for r in received if r['name'] == 'cancel_response'), None)
        assert cancel_response is not None
        assert cancel_response['args'][0]['success'] is True
        assert cancel_response['args'][0]['operation_id'] == operation_id
        
        progress_updates = [r for r in received if r['name'] == 'progress_update']
        if progress_updates:
            cancellation_update = progress_updates[-1]['args'][0]
            assert cancellation_update['status'] == 'cancelled'
            assert cancellation_update['can_cancel'] is False
    
    def test_error_handling_via_websocket(self, socketio_client, progress_service):
        """Test error handling and notifications via WebSocket."""
        operation_id = progress_service.start_operation(
            total_steps=10,
            message="Test error handling"
        )
        
        # Join operation
        socketio_client.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
        socketio_client.get_received(namespace='/progress')  # Clear initial messages
        
        # Complete operation with error
        progress_service.complete_operation(
            operation_id=operation_id,
            message="Operation failed",
            success=False,
            error_message="Test error occurred"
        )
        
        # Allow time for WebSocket message
        time.sleep(0.1)
        
        # Check error notification
        received = socketio_client.get_received(namespace='/progress')
        progress_updates = [r for r in received if r['name'] == 'progress_update']
        
        assert len(progress_updates) > 0
        error_update = progress_updates[-1]['args'][0]
        assert error_update['status'] == 'error'
        assert error_update['can_cancel'] is False
    
    def test_multiple_clients_same_operation(self, progress_service):
        """Test multiple clients receiving updates for the same operation."""
        # Create multiple SocketIO clients
        client1 = socketio.test_client(app_socketio, namespace='/progress')
        client2 = socketio.test_client(app_socketio, namespace='/progress')
        
        try:
            operation_id = progress_service.start_operation(
                total_steps=10,
                message="Multi-client test"
            )
            
            # Both clients join the same operation
            client1.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
            client2.emit('join_operation', {'operation_id': operation_id}, namespace='/progress')
            
            # Clear initial messages
            client1.get_received(namespace='/progress')
            client2.get_received(namespace='/progress')
            
            # Send progress update
            progress_service.update_progress(operation_id, 5, "50% complete", "processing")
            
            # Allow time for WebSocket messages
            time.sleep(0.1)
            
            # Both clients should receive the update
            received1 = client1.get_received(namespace='/progress')
            received2 = client2.get_received(namespace='/progress')
            
            progress_updates1 = [r for r in received1 if r['name'] == 'progress_update']
            progress_updates2 = [r for r in received2 if r['name'] == 'progress_update']
            
            assert len(progress_updates1) > 0
            assert len(progress_updates2) > 0
            
            # Updates should be identical
            assert progress_updates1[0]['args'][0] == progress_updates2[0]['args'][0]
            
        finally:
            client1.disconnect()
            client2.disconnect()
    
    def test_http_fallback_endpoints(self, client):
        """Test HTTP fallback endpoints for non-WebSocket clients."""
        # Test progress status endpoint
        response = client.get('/progress_status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'progress' in data
        assert 'message' in data
        assert 'completed' in data
        assert 'can_cancel' in data
    
    def test_cancel_processing_http_endpoint(self, client):
        """Test HTTP cancellation endpoint."""
        response = client.post('/cancel_processing')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'success' in data
        # Should return False since no active operation
        assert data['success'] is False
    
    @patch('src.web.services.websocket_progress_service.SOCKETIO_AVAILABLE', False)
    def test_graceful_degradation_without_socketio(self):
        """Test graceful degradation when SocketIO is not available."""
        service = WebSocketProgressService()
        
        # Should not raise errors even without SocketIO
        operation_id = service.start_operation(total_steps=10, message="Test without SocketIO")
        assert operation_id is not None
        
        success = service.update_progress(operation_id, 5, "50% complete")
        assert success is True
        
        success = service.complete_operation(operation_id, "Completed")
        assert success is True
    
    def test_connection_retry_mechanism(self, progress_service):
        """Test connection retry and error handling."""
        # This test simulates connection failures and retries
        # In a real scenario, this would test the client-side retry logic
        
        operation_id = progress_service.start_operation(
            total_steps=10,
            message="Connection retry test"
        )
        
        # Simulate multiple rapid updates (stress test)
        for i in range(10):
            progress_service.update_progress(
                operation_id, 
                i + 1, 
                f"Step {i + 1} complete",
                "processing"
            )
        
        # Operation should still be trackable
        status = progress_service.get_operation_status(operation_id)
        assert status is not None
        assert status['current_step'] == 10
    
    def test_estimated_time_calculation(self, progress_service):
        """Test estimated completion time calculation."""
        operation_id = progress_service.start_operation(
            total_steps=100,
            message="ETA test"
        )
        
        # Simulate progress with delays to generate step durations
        for i in range(1, 6):
            time.sleep(0.01)  # Small delay to simulate work
            progress_service.update_progress(
                operation_id,
                i * 10,
                f"{i * 10}% complete",
                "processing"
            )
        
        status = progress_service.get_operation_status(operation_id)
        assert status is not None
        assert status['current_step'] == 50
        
        # Should have estimated remaining time after several updates
        if status.get('estimated_remaining') is not None:
            assert status['estimated_remaining'] > 0
    
    def test_concurrent_operations(self, progress_service):
        """Test handling multiple concurrent operations."""
        # Start multiple operations
        operation_ids = []
        for i in range(3):
            op_id = progress_service.start_operation(
                total_steps=10,
                message=f"Concurrent operation {i + 1}"
            )
            operation_ids.append(op_id)
        
        # Update all operations
        for i, op_id in enumerate(operation_ids):
            progress_service.update_progress(
                op_id,
                5,
                f"Operation {i + 1} at 50%",
                "processing"
            )
        
        # Verify all operations are tracked
        for op_id in operation_ids:
            status = progress_service.get_operation_status(op_id)
            assert status is not None
            assert status['current_step'] == 5
        
        # Complete operations
        for op_id in operation_ids:
            progress_service.complete_operation(op_id, "Completed")
    
    def test_operation_cleanup(self, progress_service):
        """Test automatic cleanup of completed operations."""
        operation_id = progress_service.start_operation(
            total_steps=10,
            message="Cleanup test"
        )
        
        # Complete operation
        progress_service.complete_operation(operation_id, "Completed")
        
        # Operation should still be accessible immediately after completion
        status = progress_service.get_operation_status(operation_id)
        assert status is not None
        assert status['status'] == 'completed'
        
        # Manually trigger cleanup (in real scenario, this happens automatically after delay)
        progress_service._cleanup_operation(operation_id)
        
        # Operation should be cleaned up
        status = progress_service.get_operation_status(operation_id)
        assert status is None


class TestWebInterfaceE2E:
    """End-to-end tests for web interface functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_processing_page_loads(self, client):
        """Test that processing page loads correctly."""
        response = client.get('/processing')
        assert response.status_code == 200
        assert b'Processing Files...' in response.data
        assert b'socket.io' in response.data  # WebSocket client should be included
    
    def test_processing_page_responsive_elements(self, client):
        """Test responsive design elements in processing page."""
        response = client.get('/processing')
        assert response.status_code == 200
        
        # Check for responsive CSS classes and elements
        assert b'processing-status-container' in response.data
        assert b'col-md-4' in response.data  # Responsive grid
        assert b'btn-outline-danger' in response.data  # Cancel button
        assert b'progress-bar' in response.data
        assert b'connectionStatus' in response.data  # Connection status indicator
    
    def test_processing_page_websocket_integration(self, client):
        """Test WebSocket integration in processing page."""
        response = client.get('/processing')
        assert response.status_code == 200
        
        # Check for WebSocket-related JavaScript
        assert b'socket.io' in response.data
        assert b'io(\'/progress\'' in response.data
        assert b'progress_update' in response.data
        assert b'join_operation' in response.data
        assert b'cancel_operation' in response.data
    
    def test_processing_page_error_handling(self, client):
        """Test error handling elements in processing page."""
        response = client.get('/processing')
        assert response.status_code == 200
        
        # Check for error handling elements
        assert b'errorAlert' in response.data
        assert b'cancelledAlert' in response.data
        assert b'retryConnection' in response.data
        assert b'connect_error' in response.data
    
    def test_processing_page_mobile_responsive(self, client):
        """Test mobile responsive elements."""
        response = client.get('/processing')
        assert response.status_code == 200
        
        # Check for mobile-responsive CSS
        response_text = response.data.decode('utf-8')
        assert '@media (max-width: 768px)' in response_text
        assert '@media (max-width: 576px)' in response_text
        assert 'touch-friendly' in response_text
    
    @patch('web_app.websocket_progress_service')
    def test_cancel_processing_integration(self, mock_service, client):
        """Test cancel processing integration."""
        mock_service.cancel_operation.return_value = True
        
        # Mock current operation
        with patch('web_app.current_operation_id', 'test-operation-id'):
            response = client.post('/cancel_processing')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['success'] is True
            mock_service.cancel_operation.assert_called_once_with('test-operation-id')
    
    def test_progress_status_fallback(self, client):
        """Test progress status fallback for non-WebSocket clients."""
        response = client.get('/progress_status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        expected_keys = [
            'status', 'progress', 'message', 'completed', 
            'error', 'elapsed_time', 'estimated_remaining', 'can_cancel'
        ]
        
        for key in expected_keys:
            assert key in data
    
    def test_websocket_namespace_configuration(self):
        """Test WebSocket namespace configuration."""
        # Verify that the progress namespace is properly configured
        assert '/progress' in app_socketio.server.namespace_handlers
    
    @patch('web_app.run_processing_optimized')
    @patch('web_app.websocket_progress_service')
    def test_processing_with_websocket_updates(self, mock_service, mock_processing, client):
        """Test processing integration with WebSocket updates."""
        # Mock the processing function to simulate progress updates
        def mock_processing_func(config, progress_callback=None):
            if progress_callback:
                progress_callback('processing', 25, '25% complete')
                progress_callback('processing', 50, '50% complete')
                progress_callback('processing', 75, '75% complete')
                progress_callback('completed', 100, 'Processing complete')
        
        mock_processing.side_effect = mock_processing_func
        mock_service.start_operation.return_value = 'test-op-id'
        mock_service.is_cancelled.return_value = False
        
        # Create test session data
        session_data = {
            'file1_path': 'test1.csv',
            'file2_path': 'test2.csv',
            'file1_type': 'csv',
            'file2_type': 'csv',
            'file1_delimiter': ',',
            'file2_delimiter': ','
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(session_data)
            
            # Simulate form data
            form_data = {
                'mapping_file1_col_0': 'name',
                'mapping_file2_col_0': 'name',
                'match_type_0': 'fuzzy',
                'output_cols1': ['name', 'id'],
                'output_cols2': ['name', 'code'],
                'output_format': 'json',
                'output_path': 'test_results',
                'threshold': '80'
            }
            
            response = client.post('/process', data=form_data)
            assert response.status_code == 302  # Redirect to processing page
            
            # Verify WebSocket service was called
            mock_service.start_operation.assert_called_once()
            mock_service.update_progress.assert_called()
            mock_service.complete_operation.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])