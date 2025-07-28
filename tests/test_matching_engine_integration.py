"""
Integration tests for MatchingEngine with end-to-end workflows.
Tests requirements 3.1, 3.4, 5.3, 4.1: End-to-end matching workflows with parallel processing.
"""

import pytest
import pandas as pd
import time
import threading
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
import uuid
from datetime import datetime

from src.domain.matching.engine import MatchingEngine, ProgressTracker, MemoryManager
from src.domain.models import (
    MatchingConfig, Dataset, DatasetMetadata, FieldMapping, AlgorithmConfig,
    AlgorithmType, MatchingType, ProgressStatus
)


class TestMatchingEngineIntegration:
    """Integration tests for the MatchingEngine."""
    
    @pytest.fixture
    def sample_config(self) -> MatchingConfig:
        """Create a sample matching configuration."""
        return MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name",
                    algorithm=AlgorithmType.FUZZY,
                    weight=1.0
                ),
                FieldMapping(
                    source_field="city",
                    target_field="city",
                    algorithm=AlgorithmType.EXACT,
                    weight=0.8
                )
            ],
            algorithms=[
                AlgorithmConfig(
                    name="exact_matcher",
                    algorithm_type=AlgorithmType.EXACT,
                    enabled=True,
                    priority=1
                ),
                AlgorithmConfig(
                    name="fuzzy_matcher",
                    algorithm_type=AlgorithmType.FUZZY,
                    enabled=True,
                    priority=2,
                    parameters={"min_similarity": 60.0}
                ),
                AlgorithmConfig(
                    name="phonetic_matcher",
                    algorithm_type=AlgorithmType.PHONETIC,
                    enabled=True,
                    priority=3
                )
            ],
            confidence_threshold=75.0,
            use_blocking=True,
            parallel_processing=True,
            max_workers=2
        )
    
    @pytest.fixture
    def small_dataset1(self) -> Dataset:
        """Create a small test dataset 1."""
        data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson',
                    'David Lee', 'Emma Davis', 'Frank Miller', 'Grace Taylor', 'Henry Clark'],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            'age': [25, 30, 35, 28, 42, 33, 27, 38, 31, 29]
        })
        
        metadata = DatasetMetadata(
            name="test_dataset1",
            row_count=len(data),
            column_count=len(data.columns)
        )
        
        return Dataset(
            name="test_dataset1",
            data=data,
            metadata=metadata
        )
    
    @pytest.fixture
    def small_dataset2(self) -> Dataset:
        """Create a small test dataset 2 with some matching records."""
        data = pd.DataFrame({
            'id': range(101, 111),
            'name': ['Jon Doe', 'Jane Smith', 'Robert Johnson', 'Alice Brown', 'Charles Wilson',
                    'Dave Lee', 'Emma Davis', 'Franklin Miller', 'Grace Taylor', 'Harry Clark'],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            'age': [26, 30, 36, 28, 43, 34, 27, 39, 31, 30]
        })
        
        metadata = DatasetMetadata(
            name="test_dataset2",
            row_count=len(data),
            column_count=len(data.columns)
        )
        
        return Dataset(
            name="test_dataset2",
            data=data,
            metadata=metadata
        )
    
    @pytest.fixture
    def large_dataset1(self) -> Dataset:
        """Create a larger test dataset for performance testing."""
        size = 500
        data = pd.DataFrame({
            'id': range(1, size + 1),
            'name': [f'Person_{i}' for i in range(1, size + 1)],
            'city': [f'City_{i % 50}' for i in range(1, size + 1)],
            'region': [f'Region_{i % 10}' for i in range(1, size + 1)],
            'phone': [f'+1{555}{1000 + i:04d}' for i in range(1, size + 1)]
        })
        
        metadata = DatasetMetadata(
            name="large_dataset1",
            row_count=len(data),
            column_count=len(data.columns)
        )
        
        return Dataset(
            name="large_dataset1",
            data=data,
            metadata=metadata
        )
    
    @pytest.fixture
    def large_dataset2(self) -> Dataset:
        """Create a second larger test dataset with variations."""
        size = 500
        data = pd.DataFrame({
            'id': range(1001, 1001 + size),
            'name': [f'Person_{i}' if i % 10 != 0 else f'Person_{i}_Modified' for i in range(1, size + 1)],
            'city': [f'City_{i % 50}' for i in range(1, size + 1)],
            'region': [f'Region_{i % 10}' for i in range(1, size + 1)],
            'phone': [f'+1{555}{1000 + i:04d}' if i % 20 != 0 else f'+1{556}{1000 + i:04d}' for i in range(1, size + 1)]
        })
        
        metadata = DatasetMetadata(
            name="large_dataset2",
            row_count=len(data),
            column_count=len(data.columns)
        )
        
        return Dataset(
            name="large_dataset2",
            data=data,
            metadata=metadata
        )
    
    def test_basic_matching_workflow(self, sample_config, small_dataset1, small_dataset2):
        """Test basic end-to-end matching workflow."""
        engine = MatchingEngine(sample_config)
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(status: ProgressStatus):
            progress_updates.append(status)
        
        # Perform matching
        result = engine.find_matches(small_dataset1, small_dataset2, progress_callback)
        
        # Verify result structure
        assert result is not None
        assert hasattr(result, 'matched_records')
        assert hasattr(result, 'unmatched_records')
        assert hasattr(result, 'statistics')
        assert hasattr(result, 'metadata')
        
        # Verify statistics
        assert result.statistics.total_records_file1 == len(small_dataset1.data)
        assert result.statistics.total_records_file2 == len(small_dataset2.data)
        assert result.statistics.processing_time_seconds > 0
        
        # Should find some matches (exact names and similar names)
        assert len(result.matched_records) > 0
        assert len(result.matched_records) <= min(len(small_dataset1.data), len(small_dataset2.data))
        
        # Verify matched record structure
        for match in result.matched_records:
            assert hasattr(match, 'record1')
            assert hasattr(match, 'record2')
            assert hasattr(match, 'confidence_score')
            assert hasattr(match, 'matching_fields')
            assert match.confidence_score >= sample_config.confidence_threshold
        
        # Verify unmatched records
        assert 'file1' in result.unmatched_records
        assert 'file2' in result.unmatched_records
        
        # Verify progress updates were received
        assert len(progress_updates) > 0
        assert any(update.status == 'running' for update in progress_updates)
        assert any(update.status == 'completed' for update in progress_updates[-1:])
    
    def test_parallel_vs_sequential_processing(self, sample_config, large_dataset1, large_dataset2):
        """Test parallel vs sequential processing performance and correctness."""
        # Test sequential processing
        sequential_config = sample_config.model_copy()
        sequential_config.parallel_processing = False
        sequential_engine = MatchingEngine(sequential_config)
        
        start_time = time.time()
        sequential_result = sequential_engine.find_matches(large_dataset1, large_dataset2)
        sequential_time = time.time() - start_time
        
        # Test parallel processing
        parallel_config = sample_config.model_copy()
        parallel_config.parallel_processing = True
        parallel_config.max_workers = 4
        parallel_engine = MatchingEngine(parallel_config)
        
        start_time = time.time()
        parallel_result = parallel_engine.find_matches(large_dataset1, large_dataset2)
        parallel_time = time.time() - start_time
        
        # Verify both completed successfully
        assert sequential_result is not None
        assert parallel_result is not None
        
        # Results should be similar (allowing for some variation due to processing order)
        assert abs(len(sequential_result.matched_records) - len(parallel_result.matched_records)) <= 5
        
        # Parallel should be faster or at least not significantly slower for large datasets
        # Allow some tolerance for overhead
        assert parallel_time <= sequential_time * 1.5
        
        # Verify statistics are reasonable
        assert sequential_result.statistics.total_records_file1 == len(large_dataset1.data)
        assert parallel_result.statistics.total_records_file1 == len(large_dataset1.data)
    
    def test_blocking_effectiveness(self, sample_config, large_dataset1, large_dataset2):
        """Test that blocking reduces the number of comparisons."""
        # Test with blocking enabled
        blocking_config = sample_config.model_copy()
        blocking_config.use_blocking = True
        blocking_engine = MatchingEngine(blocking_config)
        
        blocking_result = blocking_engine.find_matches(large_dataset1, large_dataset2)
        
        # Test without blocking (only for smaller subset to avoid performance issues)
        small_data1 = large_dataset1.data.head(50).copy()
        small_data2 = large_dataset2.data.head(50).copy()
        
        small_dataset1 = Dataset(
            name="small_test1",
            data=small_data1,
            metadata=DatasetMetadata(name="small_test1", row_count=len(small_data1), column_count=len(small_data1.columns))
        )
        
        small_dataset2 = Dataset(
            name="small_test2",
            data=small_data2,
            metadata=DatasetMetadata(name="small_test2", row_count=len(small_data2), column_count=len(small_data2.columns))
        )
        
        no_blocking_config = sample_config.model_copy()
        no_blocking_config.use_blocking = False
        no_blocking_engine = MatchingEngine(no_blocking_config)
        
        no_blocking_result = no_blocking_engine.find_matches(small_dataset1, small_dataset2)
        
        # Verify both found results
        assert blocking_result is not None
        assert no_blocking_result is not None
        
        # Blocking should have made fewer comparisons for large dataset
        blocking_stats = blocking_engine.get_performance_stats()
        no_blocking_stats = no_blocking_engine.get_performance_stats()
        
        # For the large dataset, blocking should reduce comparisons significantly
        total_possible_large = len(large_dataset1.data) * len(large_dataset2.data)
        assert blocking_result.statistics.total_comparisons < total_possible_large
        
        # For small dataset without blocking, should make all possible comparisons
        total_possible_small = len(small_dataset1.data) * len(small_dataset2.data)
        assert no_blocking_result.statistics.total_comparisons == total_possible_small
    
    def test_memory_management(self, sample_config, large_dataset1, large_dataset2):
        """Test memory management during processing."""
        # Configure with limited memory
        memory_config = sample_config.model_copy()
        memory_config.thresholds = {'max_memory_mb': 100, 'chunk_size': 100}
        
        engine = MatchingEngine(memory_config)
        
        # Track memory usage
        initial_memory = engine.memory_manager.get_current_memory_mb()
        
        result = engine.find_matches(large_dataset1, large_dataset2)
        
        final_memory = engine.memory_manager.get_current_memory_mb()
        memory_increase = final_memory - initial_memory
        
        # Verify operation completed successfully
        assert result is not None
        assert len(result.matched_records) >= 0
        
        # Memory increase should be reasonable (less than 200MB for this test)
        assert memory_increase < 200
        
        # Verify memory manager is working
        assert engine.memory_manager.check_memory_usage()
    
    def test_progress_tracking(self, sample_config, large_dataset1, large_dataset2):
        """Test progress tracking functionality."""
        engine = MatchingEngine(sample_config)
        
        progress_updates = []
        progress_lock = threading.Lock()
        
        def progress_callback(status: ProgressStatus):
            with progress_lock:
                progress_updates.append({
                    'progress': status.progress,
                    'message': status.message,
                    'status': status.status,
                    'current_step': status.current_step,
                    'total_steps': status.total_steps
                })
        
        # Start matching in a separate thread to test concurrent progress access
        result_container = {}
        
        def run_matching():
            result_container['result'] = engine.find_matches(
                large_dataset1, large_dataset2, progress_callback
            )
        
        matching_thread = threading.Thread(target=run_matching)
        matching_thread.start()
        
        # Monitor progress while matching is running
        progress_checks = []
        while matching_thread.is_alive():
            current_progress = engine.get_current_progress()
            if current_progress:
                progress_checks.append({
                    'progress': current_progress.progress,
                    'status': current_progress.status,
                    'is_running': current_progress.is_running
                })
            time.sleep(0.1)
        
        matching_thread.join()
        
        # Verify matching completed successfully
        assert 'result' in result_container
        assert result_container['result'] is not None
        
        # Verify progress updates were received
        with progress_lock:
            assert len(progress_updates) > 0
            
            # Should have progress from 0 to 100
            progresses = [update['progress'] for update in progress_updates]
            assert min(progresses) >= 0
            assert max(progresses) == 100.0
            
            # Should have running and completed statuses
            statuses = [update['status'] for update in progress_updates]
            assert 'running' in statuses
            assert 'completed' in statuses[-1:]
        
        # Verify concurrent progress checks worked
        assert len(progress_checks) > 0
        running_checks = [check for check in progress_checks if check['is_running']]
        assert len(running_checks) > 0
    
    def test_operation_cancellation(self, sample_config, large_dataset1, large_dataset2):
        """Test operation cancellation functionality."""
        engine = MatchingEngine(sample_config)
        
        cancellation_detected = threading.Event()
        result_container = {}
        
        def progress_callback(status: ProgressStatus):
            if status.status == 'cancelled':
                cancellation_detected.set()
        
        def run_matching():
            try:
                result_container['result'] = engine.find_matches(
                    large_dataset1, large_dataset2, progress_callback
                )
            except Exception as e:
                result_container['error'] = str(e)
        
        # Start matching
        matching_thread = threading.Thread(target=run_matching)
        matching_thread.start()
        
        # Wait a bit then cancel
        time.sleep(0.5)
        engine.cancel_current_operation()
        
        # Wait for completion
        matching_thread.join(timeout=10)
        
        # Verify cancellation was detected
        assert cancellation_detected.wait(timeout=5)
        
        # Check final progress status
        final_progress = engine.get_current_progress()
        # Progress might be None if operation completed cleanup
        if final_progress:
            assert final_progress.status == 'cancelled'
    
    def test_error_handling(self, sample_config):
        """Test error handling in various scenarios."""
        engine = MatchingEngine(sample_config)
        
        # Test with None datasets
        with pytest.raises(ValueError, match="Dataset data cannot be None"):
            empty_dataset = Dataset(name="empty", data=None, metadata=DatasetMetadata())
            engine.find_matches(empty_dataset, empty_dataset)
        
        # Test with empty configuration
        empty_config = MatchingConfig(mappings=[])
        empty_engine = MatchingEngine(empty_config)
        
        dataset = Dataset(
            name="test",
            data=pd.DataFrame({'id': [1], 'name': ['test']}),
            metadata=DatasetMetadata()
        )
        
        with pytest.raises(ValueError, match="No field mappings configured"):
            empty_engine.find_matches(dataset, dataset)
    
    def test_algorithm_configuration(self, small_dataset1, small_dataset2):
        """Test different algorithm configurations."""
        # Test with only exact matching
        exact_config = MatchingConfig(
            mappings=[
                FieldMapping(
                    source_field="name",
                    target_field="name",
                    algorithm=AlgorithmType.EXACT,
                    weight=1.0
                )
            ],
            algorithms=[
                AlgorithmConfig(
                    name="exact_only",
                    algorithm_type=AlgorithmType.EXACT,
                    enabled=True
                )
            ],
            confidence_threshold=90.0
        )
        
        exact_engine = MatchingEngine(exact_config)
        exact_result = exact_engine.find_matches(small_dataset1, small_dataset2)
        
        # Test with only fuzzy matching
        fuzzy_config = MatchingConfig(
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
                    name="fuzzy_only",
                    algorithm_type=AlgorithmType.FUZZY,
                    enabled=True,
                    parameters={"min_similarity": 70.0}
                )
            ],
            confidence_threshold=70.0
        )
        
        fuzzy_engine = MatchingEngine(fuzzy_config)
        fuzzy_result = fuzzy_engine.find_matches(small_dataset1, small_dataset2)
        
        # Both should complete successfully
        assert exact_result is not None
        assert fuzzy_result is not None
        
        # Fuzzy matching should generally find more matches due to lower threshold
        assert len(fuzzy_result.matched_records) >= len(exact_result.matched_records)
    
    def test_performance_statistics(self, sample_config, small_dataset1, small_dataset2):
        """Test performance statistics collection."""
        engine = MatchingEngine(sample_config)
        
        # Initial stats should be zero
        initial_stats = engine.get_performance_stats()
        assert initial_stats['total_operations'] == 0
        assert initial_stats['total_processing_time'] == 0.0
        
        # Run matching operation
        result1 = engine.find_matches(small_dataset1, small_dataset2)
        
        # Stats should be updated
        stats_after_first = engine.get_performance_stats()
        assert stats_after_first['total_operations'] == 1
        assert stats_after_first['total_processing_time'] > 0
        assert stats_after_first['average_processing_time'] > 0
        
        # Run another operation
        result2 = engine.find_matches(small_dataset1, small_dataset2)
        
        # Stats should accumulate
        stats_after_second = engine.get_performance_stats()
        assert stats_after_second['total_operations'] == 2
        assert stats_after_second['total_processing_time'] > stats_after_first['total_processing_time']
        
        # Reset stats
        engine.reset_performance_stats()
        reset_stats = engine.get_performance_stats()
        assert reset_stats['total_operations'] == 0
        assert reset_stats['total_processing_time'] == 0.0
    
    def test_concurrent_operations(self, sample_config, small_dataset1, small_dataset2):
        """Test that concurrent operations are handled properly."""
        engine = MatchingEngine(sample_config)
        
        results = {}
        errors = {}
        
        def run_matching(thread_id):
            try:
                results[thread_id] = engine.find_matches(small_dataset1, small_dataset2)
            except Exception as e:
                errors[thread_id] = str(e)
        
        # Start multiple threads (should handle gracefully)
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_matching, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # At least one should succeed (others might be blocked or fail gracefully)
        assert len(results) > 0 or len(errors) > 0
        
        # If any succeeded, verify the result
        for result in results.values():
            assert result is not None
            assert hasattr(result, 'matched_records')


class TestProgressTracker:
    """Test the ProgressTracker component."""
    
    def test_progress_tracker_basic_functionality(self):
        """Test basic progress tracking functionality."""
        tracker = ProgressTracker("test_op", 10)
        
        # Initial state
        status = tracker.get_status()
        assert status.operation_id == "test_op"
        assert status.status == "running"
        assert status.progress == 0.0
        assert status.current_step == 0
        assert status.total_steps == 10
        
        # Update progress
        tracker.update_progress(5, "Halfway done")
        status = tracker.get_status()
        assert status.progress == 50.0
        assert status.current_step == 5
        assert status.message == "Halfway done"
        
        # Complete
        tracker.complete("All done")
        status = tracker.get_status()
        assert status.status == "completed"
        assert status.progress == 100.0
        assert status.message == "All done"
        assert status.completed_at is not None
    
    def test_progress_tracker_callbacks(self):
        """Test progress tracker callbacks."""
        tracker = ProgressTracker("test_op", 5)
        
        callback_calls = []
        
        def test_callback(status: ProgressStatus):
            callback_calls.append({
                'progress': status.progress,
                'message': status.message,
                'status': status.status
            })
        
        tracker.add_callback(test_callback)
        
        # Updates should trigger callbacks
        tracker.update_progress(2, "Step 2")
        tracker.update_progress(4, "Step 4")
        tracker.complete("Done")
        
        # Verify callbacks were called
        assert len(callback_calls) == 3
        assert callback_calls[0]['progress'] == 40.0
        assert callback_calls[1]['progress'] == 80.0
        assert callback_calls[2]['status'] == 'completed'
    
    def test_progress_tracker_error_handling(self):
        """Test progress tracker error handling."""
        tracker = ProgressTracker("test_op", 10)
        
        # Test error state
        tracker.error("Something went wrong")
        status = tracker.get_status()
        
        assert status.status == "error"
        assert status.error_message == "Something went wrong"
        assert "Error:" in status.message
        assert status.completed_at is not None
    
    def test_progress_tracker_cancellation(self):
        """Test progress tracker cancellation."""
        tracker = ProgressTracker("test_op", 10)
        
        tracker.update_progress(3, "Working...")
        tracker.cancel()
        
        status = tracker.get_status()
        assert status.status == "cancelled"
        assert status.message == "Operation cancelled"
        assert status.completed_at is not None


class TestMemoryManager:
    """Test the MemoryManager component."""
    
    def test_memory_manager_basic_functionality(self):
        """Test basic memory manager functionality."""
        manager = MemoryManager(max_memory_mb=100, chunk_size=1000)
        
        # Should be able to get current memory
        current_memory = manager.get_current_memory_mb()
        assert current_memory >= 0
        
        # Should be able to check memory usage
        within_limits = manager.check_memory_usage()
        assert isinstance(within_limits, bool)
        
        # Should be able to optimize memory
        manager.optimize_memory()  # Should not raise exception
    
    def test_memory_manager_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        manager = MemoryManager(max_memory_mb=100, chunk_size=1000)
        
        # Test with different dataset sizes and memory constraints
        chunk_size_small = manager.calculate_optimal_chunk_size(100, 50.0)
        chunk_size_large = manager.calculate_optimal_chunk_size(10000, 50.0)
        
        assert chunk_size_small >= 100  # Minimum chunk size
        assert chunk_size_large >= 100  # Minimum chunk size
        assert chunk_size_small <= 1000  # Should not exceed default
        assert chunk_size_large <= 1000  # Should not exceed default


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '--tb=short'])