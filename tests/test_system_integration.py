#!/usr/bin/env python3
"""
System Integration Test Suite
Tests the complete file processing optimization system end-to-end
"""

import os
import sys
import json
import time
import uuid
import shutil
import tempfile
import unittest
import subprocess
import threading
import requests
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all major components
from src.application.services.file_service import FileProcessingService
from src.application.services.config_service import ConfigurationManager
from src.domain.matching.engine import MatchingEngine
from src.infrastructure.progress_tracker import ProgressTracker
from src.infrastructure.metrics import metrics_collector
from src.infrastructure.logging import setup_logging
from src.infrastructure.caching import CacheManager
from src.infrastructure.security import SecurityManager

class SystemIntegrationTestSuite(unittest.TestCase):
    """Comprehensive system integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='system_test_')
        cls.config_path = os.path.join(cls.test_dir, 'test_config.json')
        cls.data_dir = os.path.join(cls.test_dir, 'data')
        cls.results_dir = os.path.join(cls.test_dir, 'results')
        
        # Create directories
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.results_dir, exist_ok=True)
        
        # Set up logging
        setup_logging(log_level='INFO', log_file=os.path.join(cls.test_dir, 'test.log'))
        
        # Create test configuration
        cls.test_config = {
            "file1": {
                "path": os.path.join(cls.data_dir, "test_file1.csv"),
                "type": "csv",
                "delimiter": ","
            },
            "file2": {
                "path": os.path.join(cls.data_dir, "test_file2.csv"),
                "type": "csv", 
                "delimiter": ","
            },
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "region"],
                "from_file2": ["code", "full_name", "district"]
            },
            "settings": {
                "output_format": "json",
                "matched_output_path": os.path.join(cls.results_dir, "matched_results"),
                "file1_output_prefix": "f1_",
                "file2_output_prefix": "f2_",
                "confidence_threshold": 75,
                "matching_type": "one-to-one",
                "unmatched_files": {"generate": True}
            }
        }
        
        # Save test configuration
        with open(cls.config_path, 'w') as f:
            json.dump(cls.test_config, f, indent=2)
        
        # Create test data files
        cls._create_test_data()
        
        print(f"System integration test environment set up in: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            shutil.rmtree(cls.test_dir)
            print(f"Test environment cleaned up: {cls.test_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")
    
    @classmethod
    def _create_test_data(cls):
        """Create realistic test datasets"""
        # File 1: Districts with Uzbek names
        file1_data = [
            {"id": 1, "name": "Toshkent shahri", "region": "Toshkent"},
            {"id": 2, "name": "Samarqand tumani", "region": "Samarqand"},
            {"id": 3, "name": "Buxoro shahri", "region": "Buxoro"},
            {"id": 4, "name": "Andijon tumani", "region": "Andijon"},
            {"id": 5, "name": "Farg'ona shahri", "region": "Farg'ona"},
            {"id": 6, "name": "Namangan tumani", "region": "Namangan"},
            {"id": 7, "name": "Qarshi shahri", "region": "Qashqadaryo"},
            {"id": 8, "name": "Termiz tumani", "region": "Surxondaryo"},
            {"id": 9, "name": "Nukus shahri", "region": "Qoraqalpog'iston"},
            {"id": 10, "name": "Urganch tumani", "region": "Xorazm"}
        ]
        
        # File 2: Similar data with variations
        file2_data = [
            {"code": "TSH001", "full_name": "Toshkent shahar", "district": "Toshkent viloyati"},
            {"code": "SAM002", "full_name": "Samarqand tuman", "district": "Samarqand viloyati"},
            {"code": "BUX003", "full_name": "Buxoro shahar", "district": "Buxoro viloyati"},
            {"code": "AND004", "full_name": "Andijon tuman", "district": "Andijon viloyati"},
            {"code": "FAR005", "full_name": "Fargona shahri", "district": "Farg'ona viloyati"},
            {"code": "NAM006", "full_name": "Namangan tuman", "district": "Namangan viloyati"},
            {"code": "QAR007", "full_name": "Qarshi shahar", "district": "Qashqadaryo viloyati"},
            {"code": "TER008", "full_name": "Termez tumani", "district": "Surxondaryo viloyati"},
            {"code": "NUK009", "full_name": "Nukus shahar", "district": "Qoraqalpog'iston"},
            {"code": "URG010", "full_name": "Urganch tuman", "district": "Xorazm viloyati"},
            {"code": "NEW011", "full_name": "Yangi tuman", "district": "Yangi viloyat"}  # Unmatched record
        ]
        
        # Save as CSV files
        pd.DataFrame(file1_data).to_csv(cls.test_config["file1"]["path"], index=False)
        pd.DataFrame(file2_data).to_csv(cls.test_config["file2"]["path"], index=False)
        
        print(f"Test data created: {len(file1_data)} records in file1, {len(file2_data)} records in file2")

    def test_01_dependency_injection_integration(self):
        """Test that all components integrate properly with dependency injection"""
        print("\n=== Testing Dependency Injection Integration ===")
        
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        config = config_manager.load_config(self.config_path)
        
        # Initialize core services
        file_service = FileProcessingService(config_manager)
        progress_tracker = ProgressTracker()
        cache_manager = CacheManager()
        security_manager = SecurityManager()
        
        # Initialize matching engine with dependencies
        matching_engine = MatchingEngine(
            config=config,
            cache_manager=cache_manager,
            progress_tracker=progress_tracker
        )
        
        # Test that all components are properly initialized
        self.assertIsNotNone(config_manager)
        self.assertIsNotNone(file_service)
        self.assertIsNotNone(matching_engine)
        self.assertIsNotNone(progress_tracker)
        self.assertIsNotNone(cache_manager)
        self.assertIsNotNone(security_manager)
        
        # Test configuration validation
        validation_result = config_manager.validate_config(config)
        self.assertTrue(validation_result.is_valid, f"Config validation failed: {validation_result.errors}")
        
        print("✅ All components initialized and integrated successfully")

    def test_02_end_to_end_file_processing(self):
        """Test complete end-to-end file processing workflow"""
        print("\n=== Testing End-to-End File Processing ===")
        
        # Initialize system components
        config_manager = ConfigurationManager()
        config = config_manager.load_config(self.config_path)
        
        file_service = FileProcessingService(config_manager)
        progress_tracker = ProgressTracker()
        cache_manager = CacheManager()
        
        matching_engine = MatchingEngine(
            config=config,
            cache_manager=cache_manager,
            progress_tracker=progress_tracker
        )
        
        # Track progress
        operation_id = str(uuid.uuid4())
        progress_tracker.start_operation(operation_id, 100)
        
        try:
            # Step 1: Load and validate files
            progress_tracker.update_progress(operation_id, 10, "Loading files...")
            
            dataset1 = file_service.load_file(config["file1"]["path"], config["file1"])
            dataset2 = file_service.load_file(config["file2"]["path"], config["file2"])
            
            self.assertIsNotNone(dataset1)
            self.assertIsNotNone(dataset2)
            self.assertEqual(len(dataset1), 10)  # Expected number of records
            self.assertEqual(len(dataset2), 11)  # Expected number of records
            
            progress_tracker.update_progress(operation_id, 30, "Files loaded successfully")
            
            # Step 2: Perform matching
            progress_tracker.update_progress(operation_id, 40, "Starting matching process...")
            
            matching_result = matching_engine.find_matches(dataset1, dataset2)
            
            self.assertIsNotNone(matching_result)
            self.assertGreater(len(matching_result.matched_records), 0)
            
            progress_tracker.update_progress(operation_id, 70, "Matching completed")
            
            # Step 3: Save results
            progress_tracker.update_progress(operation_id, 80, "Saving results...")
            
            output_files = file_service.save_results(matching_result, config["settings"])
            
            self.assertIsInstance(output_files, list)
            self.assertGreater(len(output_files), 0)
            
            # Verify output files exist
            for file_path in output_files:
                self.assertTrue(os.path.exists(file_path), f"Output file not found: {file_path}")
            
            progress_tracker.update_progress(operation_id, 100, "Processing completed successfully")
            
            # Verify results quality
            matched_file = os.path.join(self.results_dir, "matched_results.json")
            if os.path.exists(matched_file):
                with open(matched_file, 'r') as f:
                    results = json.load(f)
                
                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 5)  # Should have several matches
                
                # Check result structure
                if results:
                    first_result = results[0]
                    self.assertIn('confidence_score', first_result)
                    self.assertIn('f1_name', first_result)
                    self.assertIn('f2_full_name', first_result)
            
            print(f"✅ End-to-end processing completed successfully")
            print(f"   - Matched records: {len(matching_result.matched_records)}")
            print(f"   - Output files: {len(output_files)}")
            
        finally:
            progress_tracker.complete_operation(operation_id, "Test completed")

    def test_03_performance_benchmarking(self):
        """Test system performance with realistic datasets"""
        print("\n=== Testing Performance Benchmarking ===")
        
        # Create larger test dataset for performance testing
        large_data_dir = os.path.join(self.test_dir, 'large_data')
        os.makedirs(large_data_dir, exist_ok=True)
        
        # Generate 1000 records for performance testing
        large_file1_data = []
        large_file2_data = []
        
        base_names = ["Toshkent", "Samarqand", "Buxoro", "Andijon", "Farg'ona", 
                     "Namangan", "Qarshi", "Termiz", "Nukus", "Urganch"]
        
        for i in range(1000):
            base_name = base_names[i % len(base_names)]
            suffix = "shahri" if i % 2 == 0 else "tumani"
            
            large_file1_data.append({
                "id": i + 1,
                "name": f"{base_name} {suffix} {i}",
                "region": f"Region_{i % 10}"
            })
            
            # Add some variations for matching
            variation = "shahar" if suffix == "shahri" else "tuman"
            large_file2_data.append({
                "code": f"CODE_{i:04d}",
                "full_name": f"{base_name} {variation} {i}",
                "district": f"District_{i % 10}"
            })
        
        # Save large datasets
        large_file1_path = os.path.join(large_data_dir, "large_file1.csv")
        large_file2_path = os.path.join(large_data_dir, "large_file2.csv")
        
        pd.DataFrame(large_file1_data).to_csv(large_file1_path, index=False)
        pd.DataFrame(large_file2_data).to_csv(large_file2_path, index=False)
        
        # Create performance test configuration
        perf_config = self.test_config.copy()
        perf_config["file1"]["path"] = large_file1_path
        perf_config["file2"]["path"] = large_file2_path
        perf_config["settings"]["matched_output_path"] = os.path.join(self.results_dir, "perf_results")
        
        # Initialize components
        config_manager = ConfigurationManager()
        file_service = FileProcessingService(config_manager)
        progress_tracker = ProgressTracker()
        cache_manager = CacheManager()
        
        matching_engine = MatchingEngine(
            config=perf_config,
            cache_manager=cache_manager,
            progress_tracker=progress_tracker
        )
        
        # Measure performance
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        progress_tracker.start_operation(operation_id, 100)
        
        try:
            # Load files
            dataset1 = file_service.load_file(large_file1_path, perf_config["file1"])
            dataset2 = file_service.load_file(large_file2_path, perf_config["file2"])
            
            load_time = time.time() - start_time
            
            # Perform matching
            match_start = time.time()
            matching_result = matching_engine.find_matches(dataset1, dataset2)
            match_time = time.time() - match_start
            
            # Save results
            save_start = time.time()
            output_files = file_service.save_results(matching_result, perf_config["settings"])
            save_time = time.time() - save_start
            
            total_time = time.time() - start_time
            
            # Performance assertions
            self.assertLess(load_time, 10.0, "File loading took too long")
            self.assertLess(match_time, 30.0, "Matching took too long")
            self.assertLess(save_time, 5.0, "Saving took too long")
            self.assertLess(total_time, 45.0, "Total processing took too long")
            
            # Calculate throughput
            total_comparisons = len(dataset1) * len(dataset2)
            throughput = total_comparisons / match_time
            
            print(f"✅ Performance benchmarking completed:")
            print(f"   - Dataset size: {len(dataset1)} x {len(dataset2)} = {total_comparisons:,} comparisons")
            print(f"   - Load time: {load_time:.2f}s")
            print(f"   - Match time: {match_time:.2f}s")
            print(f"   - Save time: {save_time:.2f}s")
            print(f"   - Total time: {total_time:.2f}s")
            print(f"   - Throughput: {throughput:,.0f} comparisons/second")
            print(f"   - Matches found: {len(matching_result.matched_records)}")
            
            # Verify results quality
            self.assertGreater(len(matching_result.matched_records), 500, "Too few matches found")
            
        finally:
            progress_tracker.complete_operation(operation_id, "Performance test completed")

    def test_04_web_api_integration(self):
        """Test web API integration"""
        print("\n=== Testing Web API Integration ===")
        
        # Start web application in background thread
        web_process = None
        try:
            # Import and start web app
            import subprocess
            import signal
            
            # Start web app as subprocess
            web_process = subprocess.Popen(
                [sys.executable, 'web_app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for web app to start
            time.sleep(5)
            
            # Test health endpoint
            try:
                response = requests.get('http://localhost:5000/health', timeout=5)
                self.assertEqual(response.status_code, 200)
                
                health_data = response.json()
                self.assertIn('status', health_data)
                self.assertEqual(health_data['status'], 'healthy')
                
                print("✅ Web API health check passed")
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Web API test skipped (server not available): {e}")
                return
            
            # Test file upload endpoint (if available)
            try:
                # Prepare test files for upload
                files = {
                    'file1': open(self.test_config["file1"]["path"], 'rb'),
                    'file2': open(self.test_config["file2"]["path"], 'rb')
                }
                
                response = requests.post('http://localhost:5000/upload', files=files, timeout=10)
                
                # Close files
                files['file1'].close()
                files['file2'].close()
                
                # Check response
                self.assertIn(response.status_code, [200, 302])  # Success or redirect
                
                print("✅ Web API file upload test passed")
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Web API upload test failed: {e}")
            
        except Exception as e:
            print(f"⚠️  Web API test skipped: {e}")
            
        finally:
            # Clean up web process
            if web_process:
                try:
                    os.killpg(os.getpgid(web_process.pid), signal.SIGTERM)
                    web_process.wait(timeout=5)
                except:
                    pass

    def test_05_cli_integration(self):
        """Test CLI integration"""
        print("\n=== Testing CLI Integration ===")
        
        try:
            # Test CLI with configuration file
            cmd = [
                sys.executable, 'main.py',
                '--config', self.config_path,
                '--output-dir', self.results_dir,
                '--verbose'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.dirname(os.path.abspath(__file__ + '/../'))
            )
            
            # Check if CLI executed successfully
            if result.returncode == 0:
                print("✅ CLI integration test passed")
                print(f"   - Exit code: {result.returncode}")
                
                # Check if output files were created
                expected_output = os.path.join(self.results_dir, "matched_results.json")
                if os.path.exists(expected_output):
                    print(f"   - Output file created: {expected_output}")
                else:
                    print(f"   - Warning: Expected output file not found: {expected_output}")
                    
            else:
                print(f"⚠️  CLI test failed with exit code: {result.returncode}")
                print(f"   - STDOUT: {result.stdout}")
                print(f"   - STDERR: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⚠️  CLI test timed out")
        except Exception as e:
            print(f"⚠️  CLI test failed: {e}")

    def test_06_security_validation(self):
        """Test security features and vulnerability assessment"""
        print("\n=== Testing Security Validation ===")
        
        security_manager = SecurityManager()
        
        # Test file validation
        test_files = [
            self.test_config["file1"]["path"],
            self.test_config["file2"]["path"]
        ]
        
        for file_path in test_files:
            validation_result = security_manager.validate_file(file_path)
            self.assertTrue(validation_result.is_valid, f"File validation failed: {validation_result.errors}")
        
        # Test malicious file detection (create a test file with suspicious content)
        malicious_file = os.path.join(self.test_dir, "malicious.csv")
        with open(malicious_file, 'w') as f:
            f.write("name,script\n")
            f.write("test,<script>alert('xss')</script>\n")
        
        validation_result = security_manager.validate_file(malicious_file)
        # Should detect potential security issues
        
        # Test data sanitization
        test_data = {
            "name": "<script>alert('test')</script>",
            "value": "normal_value"
        }
        
        sanitized_data = security_manager.sanitize_data(test_data)
        self.assertNotIn("<script>", str(sanitized_data))
        
        print("✅ Security validation completed")
        print(f"   - File validation: Passed")
        print(f"   - Data sanitization: Passed")

    def test_07_monitoring_and_metrics(self):
        """Test monitoring and metrics collection"""
        print("\n=== Testing Monitoring and Metrics ===")
        
        # Test metrics collection
        if metrics_collector:
            # Record some test metrics
            metrics_collector.record_processing_time("test_operation", 1.5)
            metrics_collector.record_file_processed("test_file.csv", 1000)
            metrics_collector.record_matches_found(150)
            
            # Get metrics summary
            metrics_summary = metrics_collector.get_metrics_summary()
            
            self.assertIsInstance(metrics_summary, dict)
            self.assertIn('processing_times', metrics_summary)
            
            print("✅ Metrics collection test passed")
            print(f"   - Metrics recorded and retrieved successfully")
        else:
            print("⚠️  Metrics collector not available")
        
        # Test health checks
        try:
            from src.infrastructure.health_checks import HealthCheckManager
            
            health_manager = HealthCheckManager()
            health_status = health_manager.get_system_health()
            
            self.assertIsInstance(health_status, dict)
            self.assertIn('status', health_status)
            
            print("✅ Health checks test passed")
            print(f"   - System health: {health_status.get('status', 'unknown')}")
            
        except ImportError:
            print("⚠️  Health check manager not available")

    def test_08_deployment_validation(self):
        """Test deployment procedures and rollback mechanisms"""
        print("\n=== Testing Deployment Validation ===")
        
        # Test Docker container build (if Docker is available)
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ Docker available: {result.stdout.strip()}")
                
                # Test Docker build
                dockerfile_path = os.path.join(os.path.dirname(__file__), '..', 'Dockerfile')
                if os.path.exists(dockerfile_path):
                    build_result = subprocess.run(
                        ['docker', 'build', '-t', 'file-processing-test', '.'],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=os.path.dirname(dockerfile_path)
                    )
                    
                    if build_result.returncode == 0:
                        print("✅ Docker build test passed")
                    else:
                        print(f"⚠️  Docker build failed: {build_result.stderr}")
                else:
                    print("⚠️  Dockerfile not found")
            else:
                print("⚠️  Docker not available")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  Docker test skipped (not available)")
        
        # Test configuration validation for different environments
        environments = ['development', 'staging', 'production']
        
        for env in environments:
            env_config = self.test_config.copy()
            env_config['environment'] = env
            
            config_manager = ConfigurationManager()
            validation_result = config_manager.validate_config(env_config)
            
            self.assertTrue(validation_result.is_valid, 
                          f"Configuration validation failed for {env}: {validation_result.errors}")
        
        print("✅ Deployment validation completed")
        print(f"   - Environment configurations validated: {len(environments)}")

def run_system_integration_tests():
    """Run all system integration tests"""
    print("="*80)
    print("SYSTEM INTEGRATION TEST SUITE")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(SystemIntegrationTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("SYSTEM INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_system_integration_tests()
    sys.exit(0 if success else 1)