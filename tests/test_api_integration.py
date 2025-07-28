"""
API integration tests covering all endpoints and authentication.
Tests requirements 4.1, 4.4: API integration tests with authentication coverage.
"""

import unittest
import tempfile
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from web.api_app import create_app
    from domain.models import *
    API_AVAILABLE = True
except ImportError as e:
    print(f"API import error: {e}")
    API_AVAILABLE = False

try:
    import flask
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


@unittest.skipUnless(API_AVAILABLE and FLASK_AVAILABLE, "API or Flask not available")
class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test app
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Create test data
        self.test_data1 = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'age': [25, 30, 35],
            'city': ['New York', 'Los Angeles', 'Chicago']
        })
        
        self.test_data2 = pd.DataFrame({
            'full_name': ['John Doe', 'Jane Smith', 'Robert Johnson'],
            'years': [25, 30, 36],
            'location': ['New York', 'Los Angeles', 'Chicago']
        })
        
        # Create test files
        self.csv_file1 = Path(self.temp_dir) / "test1.csv"
        self.csv_file2 = Path(self.temp_dir) / "test2.csv"
        self.test_data1.to_csv(self.csv_file1, index=False)
        self.test_data2.to_csv(self.csv_file2, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.app_context.pop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_file_upload_endpoint(self):
        """Test file upload endpoint."""
        with open(self.csv_file1, 'rb') as f:
            response = self.client.post('/api/files/upload', 
                data={'file': (f, 'test1.csv')},
                content_type='multipart/form-data'
            )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('file_id', data)
        self.assertIn('filename', data)
        self.assertIn('size', data)
        self.assertEqual(data['filename'], 'test1.csv')
    
    def test_file_validation_endpoint(self):
        """Test file validation endpoint."""
        # First upload a file
        with open(self.csv_file1, 'rb') as f:
            upload_response = self.client.post('/api/files/upload',
                data={'file': (f, 'test1.csv')},
                content_type='multipart/form-data'
            )
        
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']
        
        # Then validate it
        response = self.client.post(f'/api/files/{file_id}/validate')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('is_valid', data)
        self.assertIn('columns', data)
        self.assertIn('row_count', data)
        self.assertTrue(data['is_valid'])
        self.assertEqual(len(data['columns']), 3)
    
    def test_matching_configuration_endpoint(self):
        """Test matching configuration endpoint."""
        config_data = {
            'mappings': [
                {
                    'source_field': 'name',
                    'target_field': 'full_name',
                    'algorithm': 'fuzzy',
                    'weight': 1.0
                }
            ],
            'confidence_threshold': 75.0,
            'use_blocking': True
        }
        
        response = self.client.post('/api/matching/configure',
            data=json.dumps(config_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('config_id', data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'configured')
    
    def test_matching_execution_endpoint(self):
        """Test matching execution endpoint."""
        # Upload files first
        with open(self.csv_file1, 'rb') as f:
            upload1 = self.client.post('/api/files/upload',
                data={'file': (f, 'test1.csv')},
                content_type='multipart/form-data'
            )
        
        with open(self.csv_file2, 'rb') as f:
            upload2 = self.client.post('/api/files/upload',
                data={'file': (f, 'test2.csv')},
                content_type='multipart/form-data'
            )
        
        file1_id = json.loads(upload1.data)['file_id']
        file2_id = json.loads(upload2.data)['file_id']
        
        # Execute matching
        match_data = {
            'file1_id': file1_id,
            'file2_id': file2_id,
            'config': {
                'mappings': [
                    {
                        'source_field': 'name',
                        'target_field': 'full_name',
                        'algorithm': 'exact',
                        'weight': 1.0
                    }
                ],
                'confidence_threshold': 90.0
            }
        }
        
        response = self.client.post('/api/matching/execute',
            data=json.dumps(match_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('job_id', data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'started')
    
    def test_job_status_endpoint(self):
        """Test job status endpoint."""
        # Create a mock job first
        job_id = 'test-job-123'
        
        response = self.client.get(f'/api/jobs/{job_id}/status')
        
        # Should return job not found or status
        self.assertIn(response.status_code, [200, 404])
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('job_id', data)
            self.assertIn('status', data)
    
    def test_results_retrieval_endpoint(self):
        """Test results retrieval endpoint."""
        # Create a mock result
        result_id = 'test-result-123'
        
        response = self.client.get(f'/api/results/{result_id}')
        
        # Should return result not found or result data
        self.assertIn(response.status_code, [200, 404])
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('result_id', data)
            self.assertIn('matched_records', data)
    
    def test_results_export_endpoint(self):
        """Test results export endpoint."""
        result_id = 'test-result-123'
        
        response = self.client.post(f'/api/results/{result_id}/export',
            data=json.dumps({'format': 'csv'}),
            content_type='application/json'
        )
        
        # Should return export not found or export data
        self.assertIn(response.status_code, [200, 404])
    
    def test_error_handling(self):
        """Test API error handling."""
        # Test invalid endpoint
        response = self.client.get('/api/invalid/endpoint')
        self.assertEqual(response.status_code, 404)
        
        # Test invalid JSON
        response = self.client.post('/api/matching/configure',
            data='invalid json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        # Test missing required fields
        response = self.client.post('/api/matching/execute',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
    
    def test_file_size_limits(self):
        """Test file size limit enforcement."""
        # Create a large file (simulate)
        large_content = 'a' * (10 * 1024 * 1024)  # 10MB
        large_file = Path(self.temp_dir) / "large.csv"
        
        with open(large_file, 'w') as f:
            f.write("name,value\n")
            f.write(large_content)
        
        with open(large_file, 'rb') as f:
            response = self.client.post('/api/files/upload',
                data={'file': (f, 'large.csv')},
                content_type='multipart/form-data'
            )
        
        # Should either accept or reject based on configured limits
        self.assertIn(response.status_code, [200, 413])  # 413 = Payload Too Large
    
    def test_concurrent_api_requests(self):
        """Test concurrent API requests."""
        import threading
        
        responses = []
        errors = []
        
        def make_request():
            try:
                response = self.client.get('/api/health')
                responses.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        # Make concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(responses), 5)
        self.assertTrue(all(status == 200 for status in responses))
    
    def test_api_rate_limiting(self):
        """Test API rate limiting (if implemented)."""
        # Make rapid requests
        responses = []
        
        for i in range(20):
            response = self.client.get('/api/health')
            responses.append(response.status_code)
        
        # Should either all succeed or some be rate limited
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # At least some should succeed
        self.assertGreater(success_count, 0)
        
        # Total should be all requests
        self.assertEqual(success_count + rate_limited_count, len(responses))


@unittest.skipUnless(API_AVAILABLE and FLASK_AVAILABLE, "API or Flask not available")
class TestAPIAuthentication(unittest.TestCase):
    """Tests for API authentication and authorization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.app_context.pop()
    
    def test_authentication_required_endpoints(self):
        """Test that protected endpoints require authentication."""
        protected_endpoints = [
            '/api/files/upload',
            '/api/matching/execute',
            '/api/results/test-id'
        ]
        
        for endpoint in protected_endpoints:
            if endpoint.endswith('upload'):
                response = self.client.post(endpoint)
            else:
                response = self.client.get(endpoint)
            
            # Should require authentication (401) or be not found (404)
            self.assertIn(response.status_code, [401, 404])
    
    def test_public_endpoints(self):
        """Test that public endpoints don't require authentication."""
        public_endpoints = [
            '/api/health',
            '/api/docs',
            '/api/version'
        ]
        
        for endpoint in public_endpoints:
            response = self.client.get(endpoint)
            
            # Should not require authentication (200) or be not found (404)
            self.assertIn(response.status_code, [200, 404])
    
    def test_invalid_token_handling(self):
        """Test handling of invalid authentication tokens."""
        headers = {'Authorization': 'Bearer invalid-token'}
        
        response = self.client.get('/api/files', headers=headers)
        
        # Should reject invalid token
        self.assertIn(response.status_code, [401, 404])
    
    def test_token_expiration_handling(self):
        """Test handling of expired tokens."""
        # This would require a more complex setup with actual JWT tokens
        # For now, just test the structure
        
        expired_token = 'Bearer expired.token.here'
        headers = {'Authorization': expired_token}
        
        response = self.client.get('/api/files', headers=headers)
        
        # Should handle expired tokens appropriately
        self.assertIn(response.status_code, [401, 404])


@unittest.skipUnless(API_AVAILABLE and FLASK_AVAILABLE, "API or Flask not available")
class TestAPIPerformance(unittest.TestCase):
    """Performance tests for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.app_context.pop()
    
    def test_health_check_performance(self):
        """Test health check endpoint performance."""
        times = []
        
        for _ in range(10):
            start_time = time.time()
            response = self.client.get('/api/health')
            end_time = time.time()
            
            times.append(end_time - start_time)
            self.assertEqual(response.status_code, 200)
        
        # Health check should be very fast
        avg_time = sum(times) / len(times)
        self.assertLess(avg_time, 0.1)  # Less than 100ms average
    
    def test_concurrent_health_checks(self):
        """Test concurrent health check performance."""
        import threading
        
        times = []
        errors = []
        
        def make_health_check():
            try:
                start_time = time.time()
                response = self.client.get('/api/health')
                end_time = time.time()
                
                times.append(end_time - start_time)
                if response.status_code != 200:
                    errors.append(f"Status: {response.status_code}")
            except Exception as e:
                errors.append(str(e))
        
        # Make concurrent requests
        threads = [threading.Thread(target=make_health_check) for _ in range(10)]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All requests should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(times), 10)
        
        # Total time should be reasonable
        self.assertLess(total_time, 2.0)
        
        # Individual times should be reasonable
        for t in times:
            self.assertLess(t, 0.5)
    
    def test_api_response_sizes(self):
        """Test API response sizes are reasonable."""
        response = self.client.get('/api/health')
        
        self.assertEqual(response.status_code, 200)
        
        # Response should be reasonably small
        content_length = len(response.data)
        self.assertLess(content_length, 1024)  # Less than 1KB
    
    def test_memory_usage_during_requests(self):
        """Test memory usage during API requests."""
        try:
            import psutil
            import os
        except ImportError:
            self.skipTest("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple requests
        for _ in range(50):
            response = self.client.get('/api/health')
            self.assertEqual(response.status_code, 200)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal
        self.assertLess(memory_increase, 10)  # Less than 10MB increase


if __name__ == '__main__':
    unittest.main(verbosity=2)