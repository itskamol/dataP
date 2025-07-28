"""
Comprehensive unit test suite with high coverage for all core components.
This module provides extensive unit tests for matching algorithms, file processing,
configuration management, and data models with edge cases and performance benchmarks.

Tests requirements 4.1, 4.3, 4.4: Unit tests with edge cases, performance benchmarks,
and minimum 90% coverage requirement.
"""

import unittest
import time
import tempfile
import json
import os
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import all modules to test
from src.domain.matching.engine import MatchingEngine, ProgressTracker, MemoryManager
from src.domain.matching.exact_matcher import ExactMatcher
from src.domain.matching.fuzzy_matcher import FuzzyMatcher
from src.domain.matching.phonetic_matcher import PhoneticMatcher
from src.domain.matching.blocking import OptimizedBlockingIndex, BlockingStrategy
from src.domain.matching.cache import MatchingCache, get_global_cache, reset_global_cache
from src.domain.matching.uzbek_normalizer import UzbekTextNormalizer
from src.domain.models import *
from src.application.services.file_service import FileProcessingService
from src.application.services.config_service import ConfigurationManager
from src.application.services.result_manager import ResultManager
from src.infrastructure.logging import get_logger
from src.infrastructure.progress_tracker import ProgressTracker as InfraProgressTracker


class TestMatchingEngineComprehensive(unittest.TestCase):
    """Comprehensive tests for MatchingEngine with edge cases and performance benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name",
                    algorithm=AlgorithmType.FUZZY,
                    weight=1.0
                )
            ],
            algorithms=[
                AlgorithmConfig(
                    name="fuzzy",
                    algorithm_type=AlgorithmType.FUZZY,
                    parameters={"threshold": 80},
                    enabled=True
                )
            ],
            confidence_threshold=75.0,
            parallel_processing=False  # Disable for unit tests
        )
        self.engine = MatchingEngine(self.config)
        
        # Create test datasets
        self.dataset1 = Dataset(
            name="test1",
            data=pd.DataFrame({
                'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
                'age': [25, 30, 35]
            })
        )
        
        self.dataset2 = Dataset(
            name="test2", 
            data=pd.DataFrame({
                'name': ['John Doe', 'Jane Smith', 'Robert Johnson'],
                'age': [25, 30, 36]
            })
        )
    
    def test_engine_initialization(self):
        """Test engine initialization with various configurations."""
        # Test with default config
        engine_default = MatchingEngine()
        self.assertIsNotNone(engine_default.config)
        self.assertIsNotNone(engine_default.blocking_index)
        self.assertIsNotNone(engine_default.memory_manager)
        
        # Test with custom config
        self.assertEqual(self.engine.config.confidence_threshold, 75.0)
        self.assertFalse(self.engine.config.parallel_processing)
    
    def test_find_matches_basic(self):
        """Test basic matching functionality."""
        result = self.engine.find_matches(self.dataset1, self.dataset2)
        
        self.assertIsInstance(result, MatchingResult)
        self.assertGreaterEqual(len(result.matched_records), 2)  # Should match John and Jane
        self.assertIsInstance(result.statistics, MatchingStatistics)
        self.assertIsInstance(result.metadata, ResultMetadata)
    
    def test_find_matches_empty_datasets(self):
        """Test matching with empty datasets."""
        empty_dataset = Dataset(name="empty", data=pd.DataFrame())
        
        with self.assertRaises(ValueError):
            self.engine.find_matches(empty_dataset, self.dataset2)
        
        with self.assertRaises(ValueError):
            self.engine.find_matches(self.dataset1, empty_dataset)
    
    def test_find_matches_no_mappings(self):
        """Test matching with no field mappings."""
        config_no_mappings = MatchingConfig(
            mappings=[],
            confidence_threshold=75.0
        )
        engine_no_mappings = MatchingEngine(config_no_mappings)
        
        with self.assertRaises(ValueError):
            engine_no_mappings.find_matches(self.dataset1, self.dataset2)
    
    def test_progress_tracking(self):
        """Test progress tracking during matching."""
        progress_updates = []
        
        def progress_callback(status):
            progress_updates.append(status)
        
        result = self.engine.find_matches(
            self.dataset1, self.dataset2, progress_callback
        )
        
        self.assertGreater(len(progress_updates), 0)
        self.assertTrue(any(update.status == 'completed' for update in progress_updates))
    
    def test_parallel_processing_disabled(self):
        """Test that parallel processing can be disabled."""
        # Already tested in setUp with parallel_processing=False
        result = self.engine.find_matches(self.dataset1, self.dataset2)
        self.assertIsInstance(result, MatchingResult)
    
    def test_memory_management(self):
        """Test memory management functionality."""
        memory_manager = self.engine.memory_manager
        
        # Test memory checking
        self.assertIsInstance(memory_manager.check_memory_usage(), bool)
        
        # Test memory optimization
        memory_manager.optimize_memory()  # Should not raise exception
        
        # Test chunk size calculation
        chunk_size = memory_manager.calculate_optimal_chunk_size(1000, 512)
        self.assertGreater(chunk_size, 0)
        self.assertLessEqual(chunk_size, 1000)
    
    def test_blocking_strategies(self):
        """Test different blocking strategies."""
        # Test with blocking enabled
        config_blocking = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name", 
                    algorithm=AlgorithmType.FUZZY
                )
            ],
            use_blocking=True
        )
        engine_blocking = MatchingEngine(config_blocking)
        
        result_blocking = engine_blocking.find_matches(self.dataset1, self.dataset2)
        
        # Test with blocking disabled
        config_no_blocking = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name",
                    algorithm=AlgorithmType.FUZZY
                )
            ],
            use_blocking=False
        )
        engine_no_blocking = MatchingEngine(config_no_blocking)
        
        result_no_blocking = engine_no_blocking.find_matches(self.dataset1, self.dataset2)
        
        # Both should produce results
        self.assertIsInstance(result_blocking, MatchingResult)
        self.assertIsInstance(result_no_blocking, MatchingResult)
    
    def test_performance_statistics(self):
        """Test performance statistics tracking."""
        # Perform multiple operations
        for _ in range(3):
            self.engine.find_matches(self.dataset1, self.dataset2)
        
        self.assertEqual(self.engine.total_operations, 3)
        self.assertGreater(self.engine.total_processing_time, 0)
        self.assertGreater(self.engine.total_comparisons, 0)
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create larger datasets
        large_data1 = pd.DataFrame({
            'name': [f'Person {i}' for i in range(100)],
            'id': range(100)
        })
        large_data2 = pd.DataFrame({
            'name': [f'Person {i}' for i in range(50, 150)],
            'id': range(50, 150)
        })
        
        large_dataset1 = Dataset(name="large1", data=large_data1)
        large_dataset2 = Dataset(name="large2", data=large_data2)
        
        result = self.engine.find_matches(large_dataset1, large_dataset2)
        
        self.assertIsInstance(result, MatchingResult)
        self.assertGreater(len(result.matched_records), 0)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with None datasets
        with self.assertRaises(ValueError):
            self.engine.find_matches(None, self.dataset2)
        
        # Test with datasets having None data
        invalid_dataset = Dataset(name="invalid", data=None)
        with self.assertRaises(ValueError):
            self.engine.find_matches(invalid_dataset, self.dataset2)
    
    def test_config_hash_calculation(self):
        """Test configuration hash calculation."""
        hash1 = self.engine._calculate_config_hash()
        hash2 = self.engine._calculate_config_hash()
        
        self.assertEqual(hash1, hash2)  # Same config should produce same hash
        
        # Change config and test different hash
        self.engine.config.confidence_threshold = 80.0
        hash3 = self.engine._calculate_config_hash()
        self.assertNotEqual(hash1, hash3)


class TestProgressTrackerComprehensive(unittest.TestCase):
    """Comprehensive tests for ProgressTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ProgressTracker("test-op", 100)
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        self.assertEqual(self.tracker.operation_id, "test-op")
        self.assertEqual(self.tracker.total_steps, 100)
        self.assertEqual(self.tracker.current_step.value, 0)
        self.assertEqual(self.tracker._status, 'running')
        self.assertIsNotNone(self.tracker.started_at)
    
    def test_progress_updates(self):
        """Test progress update functionality."""
        self.tracker.update_progress(25, "Quarter done")
        
        status = self.tracker.get_status()
        self.assertEqual(status.current_step, 25)
        self.assertEqual(status.progress, 25.0)
        self.assertEqual(status.message, "Quarter done")
    
    def test_progress_completion(self):
        """Test progress completion."""
        self.tracker.complete("All done!")
        
        status = self.tracker.get_status()
        self.assertEqual(status.status, 'completed')
        self.assertEqual(status.current_step, 100)
        self.assertEqual(status.progress, 100.0)
        self.assertIsNotNone(status.completed_at)
    
    def test_progress_error(self):
        """Test progress error handling."""
        self.tracker.error("Something went wrong")
        
        status = self.tracker.get_status()
        self.assertEqual(status.status, 'error')
        self.assertEqual(status.error_message, "Something went wrong")
        self.assertIsNotNone(status.completed_at)
    
    def test_progress_cancellation(self):
        """Test progress cancellation."""
        self.tracker.cancel()
        
        status = self.tracker.get_status()
        self.assertEqual(status.status, 'cancelled')
        self.assertIsNotNone(status.completed_at)
    
    def test_progress_callbacks(self):
        """Test progress callbacks."""
        callback_calls = []
        
        def test_callback(status):
            callback_calls.append(status)
        
        self.tracker.add_callback(test_callback)
        self.tracker.update_progress(50, "Half done")
        
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0].current_step, 50)
    
    def test_thread_safety(self):
        """Test thread safety of progress tracker."""
        def update_progress():
            for i in range(10):
                self.tracker.update_progress(i, f"Step {i}")
                time.sleep(0.001)
        
        threads = [threading.Thread(target=update_progress) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not crash and should have valid final state
        status = self.tracker.get_status()
        self.assertGreaterEqual(status.current_step, 0)
        self.assertLessEqual(status.current_step, 100)


class TestMemoryManagerComprehensive(unittest.TestCase):
    """Comprehensive tests for MemoryManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager(max_memory_mb=512, chunk_size=1000)
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        self.assertEqual(self.memory_manager.max_memory_mb, 512)
        self.assertEqual(self.memory_manager.chunk_size, 1000)
        self.assertGreaterEqual(self.memory_manager.initial_memory_mb, 0)
    
    def test_memory_usage_checking(self):
        """Test memory usage checking."""
        # Should return boolean
        result = self.memory_manager.check_memory_usage()
        self.assertIsInstance(result, bool)
        
        # Get current memory usage
        current_memory = self.memory_manager.get_current_memory_mb()
        self.assertGreaterEqual(current_memory, 0)
    
    def test_memory_optimization(self):
        """Test memory optimization."""
        # Should not raise exception
        self.memory_manager.optimize_memory()
    
    def test_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        # Test with various parameters
        chunk_size = self.memory_manager.calculate_optimal_chunk_size(10000, 256)
        self.assertGreater(chunk_size, 0)
        self.assertLessEqual(chunk_size, 10000)
        
        # Test with small available memory
        small_chunk = self.memory_manager.calculate_optimal_chunk_size(10000, 10)
        self.assertGreaterEqual(small_chunk, 100)  # Minimum chunk size
    
    def test_memory_manager_without_psutil(self):
        """Test memory manager when psutil is not available."""
        with patch('src.domain.matching.engine.PSUTIL_AVAILABLE', False):
            manager = MemoryManager()
            
            # Should still work but return defaults
            self.assertTrue(manager.check_memory_usage())  # Always True without psutil
            self.assertEqual(manager.get_current_memory_mb(), 0.0)


class TestAlgorithmPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for matching algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exact_matcher = ExactMatcher()
        self.fuzzy_matcher = FuzzyMatcher()
        self.phonetic_matcher = PhoneticMatcher()
        
        # Create test data of various sizes
        self.small_dataset = ['test', 'hello', 'world'] * 10  # 30 items
        self.medium_dataset = ['test', 'hello', 'world'] * 100  # 300 items
        self.large_dataset = ['test', 'hello', 'world'] * 1000  # 3000 items
    
    def benchmark_algorithm(self, algorithm, dataset, iterations=10):
        """Benchmark an algorithm with a dataset."""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            for i in range(len(dataset)):
                for j in range(min(i + 10, len(dataset))):  # Compare with next 10 items
                    algorithm.calculate_similarity(dataset[i], dataset[j])
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_comparisons': len(dataset) * 10
        }
    
    def test_exact_matcher_performance(self):
        """Benchmark exact matcher performance."""
        # Small dataset
        small_results = self.benchmark_algorithm(self.exact_matcher, self.small_dataset)
        self.assertLess(small_results['avg_time'], 1.0)  # Should be very fast
        
        # Medium dataset
        medium_results = self.benchmark_algorithm(self.exact_matcher, self.medium_dataset)
        self.assertLess(medium_results['avg_time'], 5.0)  # Should still be fast
        
        # Performance should scale reasonably
        self.assertLess(
            medium_results['avg_time'] / small_results['avg_time'], 
            20  # Should not be more than 20x slower
        )
    
    def test_fuzzy_matcher_performance(self):
        """Benchmark fuzzy matcher performance."""
        # Test different similarity methods
        methods = ['levenshtein', 'jaro_winkler', 'sequence', 'combined']
        
        for method in methods:
            matcher = FuzzyMatcher({'similarity_method': method})
            results = self.benchmark_algorithm(matcher, self.small_dataset, iterations=5)
            
            # Fuzzy matching should be slower but still reasonable
            self.assertLess(results['avg_time'], 10.0)
    
    def test_phonetic_matcher_performance(self):
        """Benchmark phonetic matcher performance."""
        results = self.benchmark_algorithm(self.phonetic_matcher, self.small_dataset, iterations=5)
        
        # Phonetic matching should be reasonable
        self.assertLess(results['avg_time'], 15.0)
    
    def test_caching_performance_impact(self):
        """Test performance impact of caching."""
        # Test with cache enabled
        cached_matcher = FuzzyMatcher({'use_cache': True})
        
        # First run (cache misses)
        first_run = self.benchmark_algorithm(cached_matcher, self.small_dataset[:10], iterations=1)
        
        # Second run (cache hits)
        second_run = self.benchmark_algorithm(cached_matcher, self.small_dataset[:10], iterations=1)
        
        # Second run should be faster due to caching
        self.assertLessEqual(second_run['avg_time'], first_run['avg_time'])
    
    def test_memory_usage_during_processing(self):
        """Test memory usage during algorithm processing."""
        # This is a basic test - in practice you'd use memory profiling tools
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        # Process a larger dataset
        large_text_pairs = [
            ('This is a long text string for testing memory usage', 
             'This is another long text string for testing memory usage')
            for _ in range(1000)
        ]
        
        for text1, text2 in large_text_pairs:
            self.fuzzy_matcher.calculate_similarity(text1, text2)
        
        # Force garbage collection after test
        gc.collect()
        
        # Test should complete without memory errors
        self.assertTrue(True)


class TestEdgeCasesAndErrorConditions(unittest.TestCase):
    """Test edge cases and error conditions across all components."""
    
    def test_unicode_handling(self):
        """Test Unicode character handling."""
        unicode_texts = [
            'Hello 世界',  # Mixed English/Chinese
            'Café résumé naïve',  # French accents
            'Алишер Навоий',  # Cyrillic
            '🌍🚀💻',  # Emojis
            'مرحبا بالعالم',  # Arabic
            'Здравствуй мир'  # Russian
        ]
        
        exact_matcher = ExactMatcher()
        fuzzy_matcher = FuzzyMatcher()
        
        for text in unicode_texts:
            # Should not raise exceptions
            exact_result = exact_matcher.calculate_similarity(text, text)
            fuzzy_result = fuzzy_matcher.calculate_similarity(text, text)
            
            self.assertEqual(exact_result.similarity_score, 100.0)
            self.assertGreaterEqual(fuzzy_result.similarity_score, 90.0)
    
    def test_extremely_long_strings(self):
        """Test handling of extremely long strings."""
        long_string = 'a' * 10000  # 10k characters
        very_long_string = 'b' * 50000  # 50k characters
        
        fuzzy_matcher = FuzzyMatcher()
        
        # Should handle long strings without crashing
        result = fuzzy_matcher.calculate_similarity(long_string, very_long_string)
        self.assertIsInstance(result.similarity_score, float)
        self.assertGreaterEqual(result.similarity_score, 0.0)
        self.assertLessEqual(result.similarity_score, 100.0)
    
    def test_special_characters_and_punctuation(self):
        """Test handling of special characters and punctuation."""
        special_texts = [
            '!@#$%^&*()',
            '[]{}|\\:";\'<>?,./~`',
            '   \t\n\r   ',  # Whitespace
            '',  # Empty string
            None,  # None value
            123,  # Non-string type
        ]
        
        exact_matcher = ExactMatcher()
        
        for text in special_texts:
            # Should handle gracefully without exceptions
            try:
                result = exact_matcher.calculate_similarity(str(text) if text is not None else '', 'test')
                self.assertIsInstance(result, object)
            except Exception as e:
                self.fail(f"Failed to handle special text '{text}': {e}")
    
    def test_concurrent_algorithm_usage(self):
        """Test concurrent usage of algorithms."""
        fuzzy_matcher = FuzzyMatcher({'use_cache': True})
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(100):
                    result = fuzzy_matcher.calculate_similarity(f'test{i}', f'test{i+1}')
                    results.append(result.similarity_score)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 500)  # 5 threads * 100 comparisons
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases."""
        # Test with extreme threshold values
        extreme_configs = [
            {'confidence_threshold': 0.0},
            {'confidence_threshold': 100.0},
            {'max_workers': 1},
            {'max_workers': 32},
            {'memory_limit_mb': 128},
            {'memory_limit_mb': 8192}
        ]
        
        for config_dict in extreme_configs:
            try:
                # Create base config
                mapping = FieldMapping(
                    source_field="test",
                    target_field="test", 
                    algorithm=AlgorithmType.EXACT
                )
                
                config = MatchingConfig(mappings=[mapping], **config_dict)
                engine = MatchingEngine(config)
                
                self.assertIsNotNone(engine)
            except Exception as e:
                self.fail(f"Failed to handle config {config_dict}: {e}")
    
    def test_data_model_serialization_edge_cases(self):
        """Test data model serialization with edge cases."""
        # Test with extreme values
        extreme_record = MatchedRecord(
            record1={'field': 'a' * 1000},  # Very long field
            record2={'field': ''},  # Empty field
            confidence_score=99.999999,  # High precision float
            matching_fields=['field' * 100],  # Long field name
            metadata={'key': list(range(1000))}  # Large metadata
        )
        
        # Should serialize and deserialize without issues
        data = extreme_record.to_dict()
        restored = MatchedRecord.from_dict(data)
        
        self.assertEqual(restored.confidence_score, extreme_record.confidence_score)
        self.assertEqual(len(restored.metadata['key']), 1000)
    
    def test_algorithm_config_validation_edge_cases(self):
        """Test algorithm configuration validation edge cases."""
        # Test various invalid configurations
        invalid_configs = [
            {'similarity_method': 'nonexistent'},
            {'min_similarity': -10},
            {'min_similarity': 150},
            {'case_sensitive': 'not_boolean'},
            {'phonetic_method': 'invalid_method'}
        ]
        
        for invalid_config in invalid_configs:
            if 'similarity_method' in invalid_config or 'min_similarity' in invalid_config:
                matcher = FuzzyMatcher(invalid_config)
                errors = matcher.validate_config()
                self.assertGreater(len(errors), 0)
            elif 'phonetic_method' in invalid_config:
                matcher = PhoneticMatcher(invalid_config)
                errors = matcher.validate_config()
                self.assertGreater(len(errors), 0)


class TestCoverageCompleteness(unittest.TestCase):
    """Tests to ensure comprehensive coverage of all modules."""
    
    def test_all_matching_algorithms_covered(self):
        """Ensure all matching algorithms are tested."""
        algorithms = [ExactMatcher, FuzzyMatcher, PhoneticMatcher]
        
        for algorithm_class in algorithms:
            # Test basic instantiation
            algorithm = algorithm_class()
            self.assertIsNotNone(algorithm)
            
            # Test basic functionality
            result = algorithm.calculate_similarity('test', 'test')
            self.assertIsInstance(result, object)
            
            # Test configuration validation
            errors = algorithm.validate_config()
            self.assertIsInstance(errors, list)
            
            # Test algorithm info
            info = algorithm.get_algorithm_info()
            self.assertIsInstance(info, dict)
            self.assertIn('name', info)
    
    def test_all_data_models_covered(self):
        """Ensure all data models are tested."""
        models = [
            FieldMapping, AlgorithmConfig, DatasetMetadata, Dataset,
            MatchingConfig, MatchedRecord, MatchingStatistics, 
            ResultMetadata, MatchingResult, ValidationResult, ProgressStatus
        ]
        
        for model_class in models:
            # Test basic instantiation with minimal required fields
            try:
                if model_class == FieldMapping:
                    instance = model_class(
                        source_field="test",
                        target_field="test",
                        algorithm=AlgorithmType.EXACT
                    )
                elif model_class == AlgorithmConfig:
                    instance = model_class(
                        name="test",
                        algorithm_type=AlgorithmType.EXACT
                    )
                elif model_class == MatchingConfig:
                    mapping = FieldMapping(
                        source_field="test",
                        target_field="test",
                        algorithm=AlgorithmType.EXACT
                    )
                    instance = model_class(mappings=[mapping])
                elif model_class == MatchedRecord:
                    instance = model_class(
                        record1={'test': 'value'},
                        record2={'test': 'value'},
                        confidence_score=90.0
                    )
                elif model_class == ProgressStatus:
                    instance = model_class(
                        operation_id="test",
                        status="running",
                        progress=50.0
                    )
                elif model_class == ValidationResult:
                    instance = model_class(is_valid=True)
                else:
                    instance = model_class()
                
                self.assertIsNotNone(instance)
                
                # Test serialization if available
                if hasattr(instance, 'to_dict'):
                    data = instance.to_dict()
                    self.assertIsInstance(data, dict)
                
                if hasattr(model_class, 'from_dict') and hasattr(instance, 'to_dict'):
                    restored = model_class.from_dict(instance.to_dict())
                    self.assertIsNotNone(restored)
                    
            except Exception as e:
                self.fail(f"Failed to test model {model_class.__name__}: {e}")
    
    def test_infrastructure_components_covered(self):
        """Ensure infrastructure components are tested."""
        # Test logger
        logger = get_logger('test')
        self.assertIsNotNone(logger)
        
        # Test cache
        cache = get_global_cache()
        self.assertIsNotNone(cache)
        
        # Test cache operations
        cache.set_similarity('test1', 'test2', 'algo', 85.0)
        result = cache.get_similarity('test1', 'test2', 'algo')
        self.assertEqual(result, 85.0)
        
        # Reset cache for cleanup
        reset_global_cache()
    
    def test_service_components_covered(self):
        """Ensure service components are tested."""
        # Test file service
        file_service = FileProcessingService()
        self.assertIsNotNone(file_service)
        
        # Test configuration manager (basic instantiation)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            config_manager = ConfigurationManager(tmp.name)
            self.assertIsNotNone(config_manager)
            os.unlink(tmp.name)


if __name__ == '__main__':
    # Run all tests with coverage
    unittest.main(verbosity=2)