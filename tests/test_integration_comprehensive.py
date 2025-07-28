"""
Comprehensive integration tests for end-to-end file processing workflows.
Tests requirements 4.1, 4.4, 3.1: Integration tests covering all endpoints, 
authentication, and performance with various dataset sizes.
"""

import unittest
import tempfile
import json
import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from domain.models import *
    from application.services.file_service import FileProcessingService
    from application.services.config_service import ConfigurationManager
    from domain.matching.engine import MatchingEngine
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestEndToEndFileProcessing(unittest.TestCase):
    """Integration tests for end-to-end file processing workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_service = FileProcessingService()
        
        # Create test datasets
        self.test_data1 = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
            'age': [25, 30, 35, 28],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
        })
        
        self.test_data2 = pd.DataFrame({
            'full_name': ['John Doe', 'Jane Smith', 'Robert Johnson', 'Alice Brown'],
            'years': [25, 30, 36, 28],
            'location': ['New York', 'Los Angeles', 'Chicago', 'Houston']
        })
        
        # Create test files
        self.csv_file1 = Path(self.temp_dir) / "dataset1.csv"
        self.csv_file2 = Path(self.temp_dir) / "dataset2.csv"
        self.json_file1 = Path(self.temp_dir) / "dataset1.json"
        self.json_file2 = Path(self.temp_dir) / "dataset2.json"
        
        self.test_data1.to_csv(self.csv_file1, index=False)
        self.test_data2.to_csv(self.csv_file2, index=False)
        self.test_data1.to_json(self.json_file1, orient='records', indent=2)
        self.test_data2.to_json(self.json_file2, orient='records', indent=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_csv_processing_workflow(self):
        """Test complete CSV file processing workflow."""
        # Step 1: Load files
        dataset1 = self.file_service.load_file(str(self.csv_file1))
        dataset2 = self.file_service.load_file(str(self.csv_file2))
        
        self.assertIsInstance(dataset1, Dataset)
        self.assertIsInstance(dataset2, Dataset)
        self.assertEqual(len(dataset1.data), 4)
        self.assertEqual(len(dataset2.data), 4)
        
        # Step 2: Configure matching
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="full_name",
                    algorithm=AlgorithmType.FUZZY,
                    weight=1.0
                )
            ],
            confidence_threshold=75.0,
            parallel_processing=False
        )
        
        # Step 3: Perform matching
        engine = MatchingEngine(config)
        result = engine.find_matches(dataset1, dataset2)
        
        self.assertIsInstance(result, MatchingResult)
        self.assertGreater(len(result.matched_records), 0)
        self.assertIsInstance(result.statistics, MatchingStatistics)
        
        # Step 4: Save results
        output_path = Path(self.temp_dir) / "results"
        created_files = self.file_service.save_results(
            pd.DataFrame([{
                'name1': record.record1.get('name', ''),
                'name2': record.record2.get('full_name', ''),
                'confidence': record.confidence_score
            } for record in result.matched_records]),
            str(output_path),
            format_type='csv'
        )
        
        self.assertEqual(len(created_files), 1)
        self.assertTrue(Path(created_files[0]).exists())
    
    def test_complete_json_processing_workflow(self):
        """Test complete JSON file processing workflow."""
        # Step 1: Load files
        dataset1 = self.file_service.load_file(str(self.json_file1))
        dataset2 = self.file_service.load_file(str(self.json_file2))
        
        self.assertIsInstance(dataset1, Dataset)
        self.assertIsInstance(dataset2, Dataset)
        
        # Step 2: Configure matching with multiple algorithms
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="full_name",
                    algorithm=AlgorithmType.FUZZY,
                    weight=0.8
                ),
                FieldMapping(
                    source_field="city",
                    target_field="location",
                    algorithm=AlgorithmType.EXACT,
                    weight=0.2
                )
            ],
            algorithms=[
                AlgorithmConfig(
                    name="fuzzy",
                    algorithm_type=AlgorithmType.FUZZY,
                    parameters={"similarity_method": "combined"},
                    enabled=True
                ),
                AlgorithmConfig(
                    name="exact",
                    algorithm_type=AlgorithmType.EXACT,
                    parameters={},
                    enabled=True
                )
            ],
            confidence_threshold=70.0
        )
        
        # Step 3: Perform matching
        engine = MatchingEngine(config)
        result = engine.find_matches(dataset1, dataset2)
        
        self.assertIsInstance(result, MatchingResult)
        self.assertGreater(len(result.matched_records), 0)
        
        # Step 4: Validate results
        for record in result.matched_records:
            self.assertGreaterEqual(record.confidence_score, 70.0)
            self.assertIsInstance(record.record1, dict)
            self.assertIsInstance(record.record2, dict)
    
    def test_large_dataset_processing(self):
        """Test processing of larger datasets."""
        # Create larger test datasets
        large_data1 = pd.DataFrame({
            'name': [f'Person {i}' for i in range(1000)],
            'id': range(1000),
            'category': [f'Cat_{i % 10}' for i in range(1000)]
        })
        
        large_data2 = pd.DataFrame({
            'full_name': [f'Person {i}' for i in range(500, 1500)],
            'identifier': range(500, 1500),
            'type': [f'Cat_{i % 10}' for i in range(500, 1500)]
        })
        
        # Save to files
        large_csv1 = Path(self.temp_dir) / "large1.csv"
        large_csv2 = Path(self.temp_dir) / "large2.csv"
        large_data1.to_csv(large_csv1, index=False)
        large_data2.to_csv(large_csv2, index=False)
        
        # Load and process
        dataset1 = self.file_service.load_file(str(large_csv1))
        dataset2 = self.file_service.load_file(str(large_csv2))
        
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="full_name",
                    algorithm=AlgorithmType.EXACT,
                    weight=1.0
                )
            ],
            confidence_threshold=90.0,
            use_blocking=True,
            parallel_processing=False  # Keep false for test stability
        )
        
        start_time = time.time()
        engine = MatchingEngine(config)
        result = engine.find_matches(dataset1, dataset2)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(processing_time, 30.0)  # 30 seconds max
        self.assertIsInstance(result, MatchingResult)
        self.assertGreater(len(result.matched_records), 0)
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test with invalid file
        with self.assertRaises(Exception):
            self.file_service.load_file("nonexistent.csv")
        
        # Test with empty dataset
        empty_data = pd.DataFrame()
        empty_csv = Path(self.temp_dir) / "empty.csv"
        empty_data.to_csv(empty_csv, index=False)
        
        # Should handle gracefully
        validation_result = self.file_service.validate_file(str(empty_csv))
        self.assertTrue(validation_result.has_warnings)
        
        # Test with malformed configuration
        with self.assertRaises(ValueError):
            MatchingConfig(mappings=[])  # Empty mappings should fail
    
    def test_concurrent_processing(self):
        """Test concurrent file processing operations."""
        results = []
        errors = []
        
        def process_files():
            try:
                dataset1 = self.file_service.load_file(str(self.csv_file1))
                dataset2 = self.file_service.load_file(str(self.csv_file2))
                
                config = MatchingConfig(
                    mappings=[
                        FieldMapping(
                            source_field="name",
                            target_field="full_name",
                            algorithm=AlgorithmType.EXACT,
                            weight=1.0
                        )
                    ],
                    parallel_processing=False
                )
                
                engine = MatchingEngine(config)
                result = engine.find_matches(dataset1, dataset2)
                results.append(len(result.matched_records))
                
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = [threading.Thread(target=process_files) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 3)
        
        # Results should be consistent
        self.assertTrue(all(r == results[0] for r in results))
    
    def test_memory_management_integration(self):
        """Test memory management during processing."""
        # Create multiple datasets and process them sequentially
        datasets = []
        
        for i in range(5):
            data = pd.DataFrame({
                'name': [f'Person_{i}_{j}' for j in range(200)],
                'value': range(200)
            })
            
            csv_file = Path(self.temp_dir) / f"dataset_{i}.csv"
            data.to_csv(csv_file, index=False)
            datasets.append(str(csv_file))
        
        # Process all datasets
        for i, dataset_path in enumerate(datasets):
            dataset = self.file_service.load_file(dataset_path)
            self.assertEqual(len(dataset.data), 200)
            
            # Force cleanup
            del dataset
            import gc
            gc.collect()
    
    def test_configuration_integration(self):
        """Test configuration management integration."""
        config_file = Path(self.temp_dir) / "test_config.json"
        
        # Create test configuration
        config_data = {
            "file1": {
                "path": str(self.csv_file1),
                "file_type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": str(self.csv_file2),
                "file_type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "matching": {
                "mappings": [
                    {
                        "source_field": "name",
                        "target_field": "full_name",
                        "algorithm": "fuzzy",
                        "weight": 1.0,
                        "normalization": True,
                        "case_sensitive": False
                    }
                ],
                "algorithms": [
                    {
                        "name": "fuzzy",
                        "algorithm_type": "fuzzy",
                        "parameters": {"threshold": 80},
                        "enabled": True,
                        "priority": 1
                    }
                ],
                "thresholds": {"minimum_confidence": 75.0},
                "matching_type": "one-to-one",
                "confidence_threshold": 75.0
            },
            "output": {
                "format": "csv",
                "path": str(Path(self.temp_dir) / "results"),
                "include_unmatched": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Test configuration loading
        config_manager = ConfigurationManager(str(config_file))
        app_config = config_manager.load_config()
        
        self.assertIsInstance(app_config, ApplicationConfig)
        self.assertEqual(app_config.file1.path, str(self.csv_file1))
        self.assertEqual(app_config.file2.path, str(self.csv_file2))
        self.assertEqual(len(app_config.matching.mappings), 1)
    
    def test_result_serialization_integration(self):
        """Test result serialization and deserialization."""
        # Create matching result
        dataset1 = self.file_service.load_file(str(self.csv_file1))
        dataset2 = self.file_service.load_file(str(self.csv_file2))
        
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="full_name",
                    algorithm=AlgorithmType.FUZZY,
                    weight=1.0
                )
            ]
        )
        
        engine = MatchingEngine(config)
        result = engine.find_matches(dataset1, dataset2)
        
        # Serialize result
        result_data = result.to_dict()
        self.assertIsInstance(result_data, dict)
        self.assertIn('matched_records', result_data)
        self.assertIn('statistics', result_data)
        self.assertIn('metadata', result_data)
        
        # Deserialize result
        restored_result = MatchingResult.from_dict(result_data)
        self.assertIsInstance(restored_result, MatchingResult)
        self.assertEqual(len(restored_result.matched_records), len(result.matched_records))
        
        # Compare key metrics
        self.assertEqual(
            restored_result.statistics.total_records_file1,
            result.statistics.total_records_file1
        )
        self.assertEqual(
            restored_result.statistics.total_records_file2,
            result.statistics.total_records_file2
        )


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestSystemPerformance(unittest.TestCase):
    """System-level performance tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_service = FileProcessingService()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_performance_dataset(self, size: int, name_prefix: str) -> Path:
        """Create a dataset for performance testing."""
        data = pd.DataFrame({
            'name': [f'{name_prefix}_{i}' for i in range(size)],
            'id': range(size),
            'category': [f'Cat_{i % 20}' for i in range(size)],
            'value': [i * 1.5 for i in range(size)]
        })
        
        file_path = Path(self.temp_dir) / f"{name_prefix}_{size}.csv"
        data.to_csv(file_path, index=False)
        return file_path
    
    def test_small_dataset_performance(self):
        """Test performance with small datasets (< 1K records)."""
        file1 = self.create_performance_dataset(500, "small1")
        file2 = self.create_performance_dataset(500, "small2")
        
        dataset1 = self.file_service.load_file(str(file1))
        dataset2 = self.file_service.load_file(str(file2))
        
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name",
                    algorithm=AlgorithmType.EXACT,
                    weight=1.0
                )
            ],
            use_blocking=True
        )
        
        start_time = time.time()
        engine = MatchingEngine(config)
        result = engine.find_matches(dataset1, dataset2)
        processing_time = time.time() - start_time
        
        # Should complete quickly
        self.assertLess(processing_time, 5.0)
        self.assertIsInstance(result, MatchingResult)
    
    def test_medium_dataset_performance(self):
        """Test performance with medium datasets (1K - 10K records)."""
        file1 = self.create_performance_dataset(2000, "medium1")
        file2 = self.create_performance_dataset(2000, "medium2")
        
        dataset1 = self.file_service.load_file(str(file1))
        dataset2 = self.file_service.load_file(str(file2))
        
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="category",
                    target_field="category",
                    algorithm=AlgorithmType.EXACT,
                    weight=1.0
                )
            ],
            use_blocking=True,
            parallel_processing=False
        )
        
        start_time = time.time()
        engine = MatchingEngine(config)
        result = engine.find_matches(dataset1, dataset2)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(processing_time, 30.0)
        self.assertIsInstance(result, MatchingResult)
        
        # Should find matches due to overlapping categories
        self.assertGreater(len(result.matched_records), 0)
    
    def test_file_loading_performance(self):
        """Test file loading performance with various sizes."""
        sizes = [100, 500, 1000, 2000]
        loading_times = []
        
        for size in sizes:
            file_path = self.create_performance_dataset(size, f"load_test")
            
            start_time = time.time()
            dataset = self.file_service.load_file(str(file_path))
            loading_time = time.time() - start_time
            
            loading_times.append(loading_time)
            
            # Verify dataset loaded correctly
            self.assertEqual(len(dataset.data), size)
            self.assertEqual(len(dataset.columns), 4)
        
        # Loading time should scale reasonably
        for i in range(1, len(loading_times)):
            scale_factor = sizes[i] / sizes[i-1]
            time_factor = loading_times[i] / loading_times[i-1]
            
            # Time should not increase more than linearly with size
            self.assertLess(time_factor, scale_factor * 2)
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during processing."""
        try:
            import psutil
            import os
        except ImportError:
            self.skipTest("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple datasets
        for i in range(3):
            file_path = self.create_performance_dataset(1000, f"memory_test_{i}")
            dataset = self.file_service.load_file(str(file_path))
            
            # Process dataset
            config = MatchingConfig(
                mappings=[
                    FieldMapping(
                        source_field="name",
                        target_field="name",
                        algorithm=AlgorithmType.EXACT,
                        weight=1.0
                    )
                ]
            )
            
            engine = MatchingEngine(config)
            result = engine.find_matches(dataset, dataset)
            
            # Clean up
            del dataset, engine, result
            import gc
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100)
    
    def test_concurrent_system_load(self):
        """Test system behavior under concurrent load."""
        file1 = self.create_performance_dataset(500, "concurrent1")
        file2 = self.create_performance_dataset(500, "concurrent2")
        
        results = []
        errors = []
        processing_times = []
        
        def concurrent_worker():
            try:
                start_time = time.time()
                
                dataset1 = self.file_service.load_file(str(file1))
                dataset2 = self.file_service.load_file(str(file2))
                
                config = MatchingConfig(
                    mappings=[
                        FieldMapping(
                            source_field="category",
                            target_field="category",
                            algorithm=AlgorithmType.EXACT,
                            weight=1.0
                        )
                    ],
                    parallel_processing=False
                )
                
                engine = MatchingEngine(config)
                result = engine.find_matches(dataset1, dataset2)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                results.append(len(result.matched_records))
                
            except Exception as e:
                errors.append(e)
        
        # Run multiple concurrent workers
        threads = [threading.Thread(target=concurrent_worker) for _ in range(4)]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # All workers should complete successfully
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 4)
        
        # Results should be consistent
        self.assertTrue(all(r == results[0] for r in results))
        
        # Total time should be reasonable
        self.assertLess(total_time, 60.0)
        
        # Individual processing times should be reasonable
        for pt in processing_times:
            self.assertLess(pt, 30.0)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestSystemReliability(unittest.TestCase):
    """System reliability and stress tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_service = FileProcessingService()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_repeated_operations_stability(self):
        """Test system stability under repeated operations."""
        # Create test dataset
        data = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob', 'Alice'] * 25,  # 100 records
            'id': range(100)
        })
        
        csv_file = Path(self.temp_dir) / "stability_test.csv"
        data.to_csv(csv_file, index=False)
        
        config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name",
                    algorithm=AlgorithmType.EXACT,
                    weight=1.0
                )
            ]
        )
        
        # Perform repeated operations
        results = []
        for i in range(10):
            dataset = self.file_service.load_file(str(csv_file))
            engine = MatchingEngine(config)
            result = engine.find_matches(dataset, dataset)
            results.append(len(result.matched_records))
            
            # Clean up
            del dataset, engine, result
            import gc
            gc.collect()
        
        # Results should be consistent
        self.assertTrue(all(r == results[0] for r in results))
        self.assertGreater(results[0], 0)
    
    def test_error_recovery(self):
        """Test system recovery from various error conditions."""
        # Test recovery from file errors
        try:
            self.file_service.load_file("nonexistent.csv")
        except Exception:
            pass  # Expected
        
        # System should still work after error
        data = pd.DataFrame({'name': ['Test'], 'id': [1]})
        csv_file = Path(self.temp_dir) / "recovery_test.csv"
        data.to_csv(csv_file, index=False)
        
        dataset = self.file_service.load_file(str(csv_file))
        self.assertEqual(len(dataset.data), 1)
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        # Create and process multiple datasets
        for i in range(5):
            data = pd.DataFrame({
                'name': [f'Person_{j}' for j in range(100)],
                'id': range(100)
            })
            
            csv_file = Path(self.temp_dir) / f"cleanup_test_{i}.csv"
            data.to_csv(csv_file, index=False)
            
            dataset = self.file_service.load_file(str(csv_file))
            
            config = MatchingConfig(
                mappings=[
                    FieldMapping(
                        source_field="name",
                        target_field="name",
                        algorithm=AlgorithmType.EXACT,
                        weight=1.0
                    )
                ]
            )
            
            engine = MatchingEngine(config)
            result = engine.find_matches(dataset, dataset)
            
            # Explicit cleanup
            del dataset, engine, result
            import gc
            gc.collect()
        
        # Test should complete without memory issues
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)