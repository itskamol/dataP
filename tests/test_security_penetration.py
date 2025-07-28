#!/usr/bin/env python3
"""
Security Penetration Testing Suite
Tests security vulnerabilities and attack vectors
"""

import os
import sys
import json
import time
import tempfile
import unittest
import subprocess
import requests
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class SecurityPenetrationTestSuite(unittest.TestCase):
    """Security penetration testing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up security test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='security_test_')
        cls.project_root = Path(__file__).parent.parent
        
        # Test server details (if running)
        cls.test_server_url = 'http://localhost:5000'
        cls.server_running = False
        
        # Check if test server is running
        try:
            response = requests.get(f'{cls.test_server_url}/health', timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False
        
        print(f"Security test environment: {cls.test_dir}")
        print(f"Test server running: {cls.server_running}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up security test environment"""
        try:
            import shutil
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")

    def test_01_file_upload_security(self):
        """Test file upload security vulnerabilities"""
        print("\n=== Testing File Upload Security ===")
        
        if not self.server_running:
            self.skipTest("Test server not running")
        
        # Test malicious file uploads
        malicious_files = [
            # Script injection
            ('malicious.csv', 'name,script\ntest,<script>alert("xss")</script>\n'),
            # SQL injection attempt
            ('sql_inject.csv', 'name,value\n"test","1\'; DROP TABLE users; --"\n'),
            # Path traversal
            ('path_traversal.csv', 'name,path\ntest,../../../etc/passwd\n'),
            # Large file (DoS attempt)
            ('large_file.csv', 'name,data\n' + 'test,A' * 10000 + '\n' * 1000),
            # Binary file disguised as CSV
            ('binary.csv', b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'),
        ]
        
        for filename, content in malicious_files:
            print(f"Testing malicious file: {filename}")
            
            # Create temporary malicious file
            malicious_file_path = os.path.join(self.test_dir, filename)
            
            if isinstance(content, str):
                with open(malicious_file_path, 'w') as f:
                    f.write(content)
            else:
                with open(malicious_file_path, 'wb') as f:
                    f.write(content)
            
            # Create a normal file for the second upload
            normal_file_path = os.path.join(self.test_dir, 'normal.csv')
            with open(normal_file_path, 'w') as f:
                f.write('id,name\n1,test\n')
            
            try:
                # Attempt to upload malicious file
                with open(malicious_file_path, 'rb') as f1, open(normal_file_path, 'rb') as f2:
                    files = {
                        'file1': (filename, f1, 'text/csv'),
                        'file2': ('normal.csv', f2, 'text/csv')
                    }
                    
                    response = requests.post(
                        f'{self.test_server_url}/upload',
                        files=files,
                        timeout=10
                    )
                
                # Server should reject malicious files or handle them safely
                if response.status_code == 200:
                    print(f"⚠️  {filename}: Upload accepted (potential vulnerability)")
                else:
                    print(f"✅ {filename}: Upload rejected (status: {response.status_code})")
                
            except requests.exceptions.RequestException as e:
                print(f"✅ {filename}: Upload failed (connection error - good)")
            
            # Clean up
            try:
                os.remove(malicious_file_path)
                os.remove(normal_file_path)
            except:
                pass

    def test_02_input_validation_security(self):
        """Test input validation vulnerabilities"""
        print("\n=== Testing Input Validation Security ===")
        
        # Test security module if available
        try:
            from src.infrastructure.security import SecurityManager
            security_manager = SecurityManager()
            
            # Test XSS prevention
            xss_payloads = [
                '<script>alert("xss")</script>',
                'javascript:alert("xss")',
                '<img src=x onerror=alert("xss")>',
                '"><script>alert("xss")</script>',
                "'; DROP TABLE users; --"
            ]
            
            for payload in xss_payloads:
                sanitized = security_manager.sanitize_input(payload)
                
                # Should not contain dangerous patterns
                dangerous_patterns = ['<script>', 'javascript:', 'onerror=', 'DROP TABLE']
                is_safe = not any(pattern.lower() in sanitized.lower() for pattern in dangerous_patterns)
                
                if is_safe:
                    print(f"✅ XSS payload sanitized: {payload[:30]}...")
                else:
                    print(f"❌ XSS payload not sanitized: {payload[:30]}...")
                    self.fail(f"XSS vulnerability: {payload}")
            
            # Test SQL injection prevention
            sql_payloads = [
                "1' OR '1'='1",
                "1; DROP TABLE users; --",
                "1' UNION SELECT * FROM users --",
                "'; EXEC xp_cmdshell('dir'); --"
            ]
            
            for payload in sql_payloads:
                sanitized = security_manager.sanitize_input(payload)
                
                # Should not contain SQL injection patterns
                sql_patterns = ['DROP TABLE', 'UNION SELECT', 'xp_cmdshell', "' OR '"]
                is_safe = not any(pattern.lower() in sanitized.lower() for pattern in sql_patterns)
                
                if is_safe:
                    print(f"✅ SQL injection payload sanitized: {payload[:30]}...")
                else:
                    print(f"❌ SQL injection payload not sanitized: {payload[:30]}...")
                    self.fail(f"SQL injection vulnerability: {payload}")
            
        except ImportError:
            print("⚠️  Security module not available - skipping input validation tests")

    def test_03_authentication_security(self):
        """Test authentication and authorization vulnerabilities"""
        print("\n=== Testing Authentication Security ===")
        
        if not self.server_running:
            self.skipTest("Test server not running")
        
        # Test for authentication bypass
        protected_endpoints = [
            '/admin',
            '/api/admin',
            '/config',
            '/system',
            '/debug'
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = requests.get(f'{self.test_server_url}{endpoint}', timeout=5)
                
                if response.status_code == 200:
                    print(f"❌ {endpoint}: Accessible without authentication")
                elif response.status_code in [401, 403]:
                    print(f"✅ {endpoint}: Properly protected")
                elif response.status_code == 404:
                    print(f"ℹ️  {endpoint}: Not found (expected)")
                else:
                    print(f"⚠️  {endpoint}: Unexpected status {response.status_code}")
                
            except requests.exceptions.RequestException:
                print(f"ℹ️  {endpoint}: Not accessible (expected)")
        
        # Test session management
        try:
            # Test session fixation
            session = requests.Session()
            
            # Get initial session
            response1 = session.get(f'{self.test_server_url}/')
            initial_cookies = session.cookies.get_dict()
            
            # Simulate login (if login endpoint exists)
            login_data = {'username': 'test', 'password': 'test'}
            response2 = session.post(f'{self.test_server_url}/login', data=login_data)
            
            # Check if session ID changed after login
            post_login_cookies = session.cookies.get_dict()
            
            if initial_cookies != post_login_cookies:
                print("✅ Session ID changes after login (good)")
            else:
                print("⚠️  Session ID doesn't change after login")
            
        except requests.exceptions.RequestException:
            print("ℹ️  Session management test skipped (endpoints not available)")

    def test_04_data_exposure_vulnerabilities(self):
        """Test for data exposure vulnerabilities"""
        print("\n=== Testing Data Exposure Vulnerabilities ===")
        
        if not self.server_running:
            self.skipTest("Test server not running")
        
        # Test for sensitive file exposure
        sensitive_files = [
            '/.env',
            '/config.json',
            '/web_config.json',
            '/session.json',
            '/logs/application.log',
            '/logs/errors.log',
            '/.git/config',
            '/backup.sql',
            '/database.db'
        ]
        
        for file_path in sensitive_files:
            try:
                response = requests.get(f'{self.test_server_url}{file_path}', timeout=5)
                
                if response.status_code == 200:
                    print(f"❌ {file_path}: Sensitive file exposed")
                    self.fail(f"Sensitive file exposed: {file_path}")
                elif response.status_code == 403:
                    print(f"✅ {file_path}: Access forbidden (good)")
                elif response.status_code == 404:
                    print(f"✅ {file_path}: Not found (good)")
                else:
                    print(f"⚠️  {file_path}: Unexpected status {response.status_code}")
                
            except requests.exceptions.RequestException:
                print(f"✅ {file_path}: Not accessible (good)")
        
        # Test for directory traversal
        traversal_payloads = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
            '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd',
            '....//....//....//etc/passwd'
        ]
        
        for payload in traversal_payloads:
            try:
                response = requests.get(f'{self.test_server_url}/download/{payload}', timeout=5)
                
                if response.status_code == 200 and ('root:' in response.text or 'localhost' in response.text):
                    print(f"❌ Directory traversal successful: {payload}")
                    self.fail(f"Directory traversal vulnerability: {payload}")
                else:
                    print(f"✅ Directory traversal blocked: {payload}")
                
            except requests.exceptions.RequestException:
                print(f"✅ Directory traversal blocked: {payload}")

    def test_05_denial_of_service_vulnerabilities(self):
        """Test for denial of service vulnerabilities"""
        print("\n=== Testing Denial of Service Vulnerabilities ===")
        
        if not self.server_running:
            self.skipTest("Test server not running")
        
        # Test rate limiting
        print("Testing rate limiting...")
        
        rapid_requests = []
        start_time = time.time()
        
        # Send rapid requests
        for i in range(50):
            try:
                response = requests.get(f'{self.test_server_url}/', timeout=1)
                rapid_requests.append(response.status_code)
            except requests.exceptions.RequestException:
                rapid_requests.append(0)  # Failed request
        
        end_time = time.time()
        
        # Check if rate limiting is in effect
        rate_limited_responses = sum(1 for status in rapid_requests if status == 429)
        failed_requests = sum(1 for status in rapid_requests if status == 0)
        
        if rate_limited_responses > 0:
            print(f"✅ Rate limiting active: {rate_limited_responses} requests limited")
        elif failed_requests > 30:
            print(f"✅ Server protected against rapid requests: {failed_requests} failed")
        else:
            print(f"⚠️  No rate limiting detected: {len(rapid_requests)} requests processed")
        
        # Test large payload handling
        print("Testing large payload handling...")
        
        large_payload = 'A' * (10 * 1024 * 1024)  # 10MB payload
        
        try:
            response = requests.post(
                f'{self.test_server_url}/upload',
                data={'large_field': large_payload},
                timeout=30
            )
            
            if response.status_code == 413:  # Payload Too Large
                print("✅ Large payload rejected (413 Payload Too Large)")
            elif response.status_code == 400:  # Bad Request
                print("✅ Large payload rejected (400 Bad Request)")
            else:
                print(f"⚠️  Large payload accepted: {response.status_code}")
            
        except requests.exceptions.RequestException as e:
            print(f"✅ Large payload rejected (connection error)")

    def test_06_configuration_security(self):
        """Test configuration security"""
        print("\n=== Testing Configuration Security ===")
        
        # Test for hardcoded secrets
        source_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith(('.py', '.js', '.json', '.yaml', '.yml', '.env')):
                    source_files.append(os.path.join(root, file))
        
        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']'
        ]
        
        import re
        
        hardcoded_secrets_found = 0
        
        for file_path in source_files[:50]:  # Limit to first 50 files for performance
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip obvious test/example values
                        if any(test_val in match.lower() for test_val in 
                               ['test', 'example', 'dummy', 'placeholder', 'your_', 'change_me']):
                            continue
                        
                        print(f"⚠️  Potential hardcoded secret in {file_path}: {match[:50]}...")
                        hardcoded_secrets_found += 1
                
            except Exception:
                continue
        
        if hardcoded_secrets_found == 0:
            print("✅ No hardcoded secrets detected")
        else:
            print(f"❌ {hardcoded_secrets_found} potential hardcoded secrets found")
        
        # Test environment variable usage
        env_example_file = self.project_root / '.env.example'
        if env_example_file.exists():
            print("✅ Environment variable template found (.env.example)")
        else:
            print("⚠️  No environment variable template found")

    def test_07_dependency_vulnerabilities(self):
        """Test for known vulnerabilities in dependencies"""
        print("\n=== Testing Dependency Vulnerabilities ===")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.skipTest("requirements.txt not found")
        
        # Check if safety is available for vulnerability scanning
        try:
            result = subprocess.run(['safety', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            safety_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            safety_available = False
        
        if safety_available:
            print("Running safety check for known vulnerabilities...")
            
            safety_result = subprocess.run([
                'safety', 'check', '-r', str(requirements_file)
            ], capture_output=True, text=True, timeout=60)
            
            if safety_result.returncode == 0:
                print("✅ No known vulnerabilities found in dependencies")
            else:
                print(f"❌ Vulnerabilities found in dependencies:")
                print(safety_result.stdout)
                print(safety_result.stderr)
        else:
            print("⚠️  Safety tool not available - install with 'pip install safety'")
        
        # Check for outdated packages
        try:
            pip_result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, timeout=30)
            
            if pip_result.returncode == 0:
                outdated_packages = json.loads(pip_result.stdout)
                
                if outdated_packages:
                    print(f"⚠️  {len(outdated_packages)} outdated packages found:")
                    for pkg in outdated_packages[:5]:  # Show first 5
                        print(f"   - {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
                else:
                    print("✅ All packages are up to date")
            
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            print("⚠️  Could not check for outdated packages")

    def test_08_logging_security(self):
        """Test logging security"""
        print("\n=== Testing Logging Security ===")
        
        # Test that sensitive data is not logged
        try:
            from src.infrastructure.logging import setup_logging
            import logging
            
            # Set up test logging
            test_log_file = os.path.join(self.test_dir, 'security_test.log')
            setup_logging(log_file=test_log_file)
            
            logger = logging.getLogger('security_test')
            
            # Test logging of potentially sensitive data
            sensitive_data = {
                'password': 'secret123',
                'api_key': 'sk-1234567890abcdef',
                'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
                'credit_card': '4111-1111-1111-1111'
            }
            
            for key, value in sensitive_data.items():
                logger.info(f"Processing {key}: {value}")
            
            # Check log file for sensitive data
            if os.path.exists(test_log_file):
                with open(test_log_file, 'r') as f:
                    log_content = f.read()
                
                sensitive_found = []
                for key, value in sensitive_data.items():
                    if value in log_content:
                        sensitive_found.append(key)
                
                if sensitive_found:
                    print(f"❌ Sensitive data found in logs: {sensitive_found}")
                    self.fail(f"Sensitive data logged: {sensitive_found}")
                else:
                    print("✅ No sensitive data found in logs")
            else:
                print("⚠️  Log file not created")
            
        except ImportError:
            print("⚠️  Logging module not available")

    def test_09_error_handling_security(self):
        """Test error handling security"""
        print("\n=== Testing Error Handling Security ===")
        
        if not self.server_running:
            self.skipTest("Test server not running")
        
        # Test error message information disclosure
        error_inducing_requests = [
            ('Invalid JSON', {'Content-Type': 'application/json'}, '{"invalid": json}'),
            ('SQL-like error', {}, {'query': "SELECT * FROM users WHERE id='"}),
            ('Path traversal error', {}, {'file': '../../../etc/passwd'}),
            ('Buffer overflow attempt', {}, {'data': 'A' * 10000})
        ]
        
        for test_name, headers, data in error_inducing_requests:
            try:
                if isinstance(data, dict):
                    response = requests.post(f'{self.test_server_url}/process', 
                                           data=data, headers=headers, timeout=5)
                else:
                    response = requests.post(f'{self.test_server_url}/process',
                                           data=data, headers=headers, timeout=5)
                
                # Check if error messages reveal sensitive information
                sensitive_patterns = [
                    'traceback',
                    'stack trace',
                    'file not found',
                    'permission denied',
                    'database error',
                    'sql error',
                    'internal server error'
                ]
                
                response_text = response.text.lower()
                revealed_info = [pattern for pattern in sensitive_patterns 
                               if pattern in response_text]
                
                if revealed_info:
                    print(f"⚠️  {test_name}: Error reveals information: {revealed_info}")
                else:
                    print(f"✅ {test_name}: Error handled securely")
                
            except requests.exceptions.RequestException:
                print(f"✅ {test_name}: Request properly rejected")

    def test_10_security_headers(self):
        """Test security headers"""
        print("\n=== Testing Security Headers ===")
        
        if not self.server_running:
            self.skipTest("Test server not running")
        
        try:
            response = requests.get(f'{self.test_server_url}/', timeout=5)
            headers = response.headers
            
            # Required security headers
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': None,  # Should be present
                'Content-Security-Policy': None,    # Should be present
                'Referrer-Policy': None            # Should be present
            }
            
            for header, expected_value in security_headers.items():
                if header in headers:
                    actual_value = headers[header]
                    
                    if expected_value is None:
                        print(f"✅ {header}: Present ({actual_value})")
                    elif isinstance(expected_value, list):
                        if actual_value in expected_value:
                            print(f"✅ {header}: Correct ({actual_value})")
                        else:
                            print(f"⚠️  {header}: Unexpected value ({actual_value})")
                    elif actual_value == expected_value:
                        print(f"✅ {header}: Correct ({actual_value})")
                    else:
                        print(f"⚠️  {header}: Unexpected value ({actual_value})")
                else:
                    print(f"❌ {header}: Missing")
            
        except requests.exceptions.RequestException:
            print("⚠️  Could not test security headers (server not accessible)")

def run_security_penetration_tests():
    """Run all security penetration tests"""
    print("="*80)
    print("SECURITY PENETRATION TESTING SUITE")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(SecurityPenetrationTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("SECURITY PENETRATION TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nSECURITY VULNERABILITIES FOUND:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nTEST ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_security_penetration_tests()
    sys.exit(0 if success else 1)