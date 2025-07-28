"""
Integration tests for result storage and retrieval system.
Tests caching, compression, sharing, access control, and persistence.
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.application.services.result_manager import (
    ResultManager, ExportFormat, ExportConfig, AccessControl
)
from src.domain.models import (
    MatchingResult, MatchedRecord, MatchingStatistics, ResultMetadata,
    DatasetMetadata
)
from src.domain.exceptions import FileProcessingError, ValidationError


class TestResultStorageIntegration:
    """Integration test suite for result storage and retrieval system."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        storage_dir = tempfile.mkdtemp()
        temp_dir = tempfile.mkdtemp()
        yield storage_dir, temp_dir
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def result_manager(self, temp_dirs):
        """Create ResultManager instance for testing."""
        storage_dir, temp_dir = temp_dirs
        return ResultManager(
            storage_dir=storage_dir,
            temp_dir=temp_dir,
            max_storage_size_mb=10,
            auto_cleanup_days=7,
            compression_enabled=True,
            cache_size=5
        )
    
    @pytest.fixture
    def sample_matching_result(self):
        """Create sample MatchingResult for testing."""
        matched_records = [
            MatchedRecord(
                record1={'id': '1', 'name': 'John Doe', 'city': 'New York'},
                record2={'id': 'A', 'name': 'John Doe', 'location': 'NYC'},
                confidence_score=95.5,
                matching_fields=['name'],
                metadata={'algorithm': 'fuzzy'},
                created_at=datetime.now()
            ),
            MatchedRecord(
                record1={'id': '2', 'name': 'Jane Smith', 'city': 'Boston'},
                record2={'id': 'B', 'name': 'Jane Smith', 'location': 'Boston'},
                confidence_score=88.2,
                matching_fields=['name', 'city'],
                metadata={'algorithm': 'exact'},
                created_at=datetime.now()
            )
        ]
        
        unmatched_records = {
            'file1': [{'id': '3', 'name': 'Alice Brown', 'city': 'Seattle'}],
            'file2': [{'id': 'C', 'name': 'David Lee', 'location': 'Miami'}]
        }
        
        statistics = MatchingStatistics(
            total_records_file1=3,
            total_records_file2=3,
            total_comparisons=9,
            high_confidence_matches=2,
            low_confidence_matches=0,
            unmatched_file1=1,
            unmatched_file2=1,
            processing_time_seconds=5.2,
            average_confidence=91.85
        )
        
        metadata = ResultMetadata(
            operation_id='test-operation-123',
            created_at=datetime.now(),
            config_hash='abc123',
            file1_metadata=DatasetMetadata(name='test_file1.csv'),
            file2_metadata=DatasetMetadata(name='test_file2.csv'),
            processing_node='test-node',
            version='1.0'
        )
        
        return MatchingResult(
            matched_records=matched_records,
            unmatched_records=unmatched_records,
            statistics=statistics,
            metadata=metadata
        )
    
    def test_result_caching_workflow(self, result_manager, sample_matching_result):
        """Test complete result caching workflow."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # First retrieval should load from disk and cache
        result1 = result_manager.retrieve_results_with_cache(result_id, 'user1')
        assert len(result1.matched_records) == 2
        
        # Verify it's in cache
        cached_result = result_manager._get_from_cache(result_id)
        assert cached_result is not None
        assert len(cached_result.matched_records) == 2
        
        # Second retrieval should come from cache (faster)
        start_time = time.time()
        result2 = result_manager.retrieve_results_with_cache(result_id, 'user1')
        cache_time = time.time() - start_time
        
        assert len(result2.matched_records) == 2
        assert cache_time < 0.1  # Should be very fast from cache
        
        # Verify cache info
        cache_info = result_manager.get_cache_info()
        assert cache_info['cache_size'] == 1
        assert result_id in cache_info['cached_results']
    
    def test_access_control_workflow(self, result_manager, sample_matching_result):
        """Test complete access control workflow."""
        # Store result with owner
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Owner should have access
        assert result_manager.check_access_permission(result_id, 'user1') is True
        result = result_manager.retrieve_results_with_cache(result_id, 'user1')
        assert len(result.matched_records) == 2
        
        # Other users should not have access
        assert result_manager.check_access_permission(result_id, 'user2') is False
        with pytest.raises(ValidationError, match="Access denied"):
            result_manager.retrieve_results_with_cache(result_id, 'user2')
        
        # Share with specific user
        success = result_manager.share_result(result_id, 'user1', ['user2'], read_only=True)
        assert success is True
        
        # Shared user should now have access
        assert result_manager.check_access_permission(result_id, 'user2') is True
        result = result_manager.retrieve_results_with_cache(result_id, 'user2')
        assert len(result.matched_records) == 2
        
        # Make result public
        success = result_manager.make_result_public(result_id, 'user1', public=True)
        assert success is True
        
        # Any user should now have access
        assert result_manager.check_access_permission(result_id, 'user3') is True
        result = result_manager.retrieve_results_with_cache(result_id, 'user3')
        assert len(result.matched_records) == 2
    
    def test_access_control_expiration(self, result_manager, sample_matching_result):
        """Test access control with expiration."""
        # Store result and share with expiration
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Share with expiration in the past
        past_time = datetime.now() - timedelta(hours=1)
        success = result_manager.share_result(
            result_id, 'user1', ['user2'], 
            read_only=True, expires_at=past_time
        )
        assert success is True
        
        # Access should be denied due to expiration
        assert result_manager.check_access_permission(result_id, 'user2') is False
        
        # Share with future expiration
        future_time = datetime.now() + timedelta(hours=1)
        success = result_manager.share_result(
            result_id, 'user1', ['user3'], 
            read_only=True, expires_at=future_time
        )
        assert success is True
        
        # Access should be allowed
        assert result_manager.check_access_permission(result_id, 'user3') is True
    
    def test_result_compression_workflow(self, result_manager, sample_matching_result):
        """Test result compression workflow."""
        # Store result without compression
        result_manager.compression_enabled = False
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Verify it's not compressed
        stored_result = result_manager.results_index[result_id]
        assert stored_result.compressed is False
        assert stored_result.file_path.endswith('.pkl')
        
        original_size = stored_result.size_bytes
        
        # Compress the result
        success = result_manager.compress_result(result_id)
        assert success is True
        
        # Verify compression
        stored_result = result_manager.results_index[result_id]
        assert stored_result.compressed is True
        assert stored_result.file_path.endswith('.pkl.gz')
        assert stored_result.size_bytes < original_size  # Should be smaller
        
        # Verify result can still be retrieved
        result = result_manager.retrieve_results_with_cache(result_id, 'user1')
        assert len(result.matched_records) == 2
        
        # Verify checksum was updated
        assert stored_result.checksum is not None
    
    def test_result_metadata_retrieval(self, result_manager, sample_matching_result):
        """Test result metadata retrieval without loading full data."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Get metadata
        metadata = result_manager.get_result_metadata(result_id, 'user1')
        
        assert metadata is not None
        assert metadata['result_id'] == result_id
        assert metadata['operation_id'] == 'test-operation-123'
        assert 'created_at' in metadata
        assert 'size_bytes' in metadata
        assert metadata['compressed'] is True
        assert metadata['checksum'] is not None
        assert metadata['access_control']['owner_id'] == 'user1'
        assert metadata['access_control']['public'] is False
        
        # Test access control for metadata
        metadata = result_manager.get_result_metadata(result_id, 'user2')
        assert metadata is None  # Should not have access
    
    def test_cache_size_limits(self, result_manager, sample_matching_result):
        """Test cache size limits and LRU eviction."""
        # Store more results than cache size (5)
        result_ids = []
        for i in range(7):
            result_id = result_manager.store_results(sample_matching_result, owner_id=f'user{i}')
            result_ids.append(result_id)
        
        # Cache should only contain last 5 results
        cache_info = result_manager.get_cache_info()
        assert cache_info['cache_size'] == 5
        assert cache_info['max_cache_size'] == 5
        
        # First two results should have been evicted
        assert result_ids[0] not in cache_info['cached_results']
        assert result_ids[1] not in cache_info['cached_results']
        
        # Last 5 should be in cache
        for result_id in result_ids[-5:]:
            assert result_id in cache_info['cached_results']
    
    def test_cache_clear(self, result_manager, sample_matching_result):
        """Test cache clearing functionality."""
        # Store some results
        for i in range(3):
            result_manager.store_results(sample_matching_result, owner_id=f'user{i}')
        
        # Verify cache has items
        cache_info = result_manager.get_cache_info()
        assert cache_info['cache_size'] == 3
        
        # Clear cache
        cleared_count = result_manager.clear_cache()
        assert cleared_count == 3
        
        # Verify cache is empty
        cache_info = result_manager.get_cache_info()
        assert cache_info['cache_size'] == 0
    
    def test_checksum_verification(self, result_manager, sample_matching_result):
        """Test checksum verification during retrieval."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Get stored result info
        stored_result = result_manager.results_index[result_id]
        original_checksum = stored_result.checksum
        assert original_checksum is not None
        
        # Manually corrupt the checksum in index
        stored_result.checksum = 'corrupted_checksum'
        result_manager._save_index()
        
        # Clear cache to force disk read
        result_manager.clear_cache()
        
        # Retrieval should still work but log warning
        with patch.object(result_manager.logger, 'warning') as mock_warning:
            result = result_manager.retrieve_results_with_cache(result_id, 'user1')
            assert len(result.matched_records) == 2
            mock_warning.assert_called_once()
            assert 'Checksum mismatch' in str(mock_warning.call_args)
    
    def test_storage_persistence_across_instances(self, temp_dirs, sample_matching_result):
        """Test that storage persists across manager instances."""
        storage_dir, temp_dir = temp_dirs
        
        # Create first manager and store result
        manager1 = ResultManager(
            storage_dir=storage_dir, 
            temp_dir=temp_dir,
            compression_enabled=True
        )
        result_id = manager1.store_results(sample_matching_result, owner_id='user1')
        
        # Share result
        manager1.share_result(result_id, 'user1', ['user2'])
        manager1.make_result_public(result_id, 'user1', public=True)
        
        # Create second manager (should load existing data)
        manager2 = ResultManager(
            storage_dir=storage_dir, 
            temp_dir=temp_dir,
            compression_enabled=True
        )
        
        # Verify result is available
        assert result_id in manager2.results_index
        
        # Verify access control persisted
        assert manager2.check_access_permission(result_id, 'user1') is True
        assert manager2.check_access_permission(result_id, 'user2') is True
        assert manager2.check_access_permission(result_id, 'user3') is True  # Public
        
        # Verify result can be retrieved
        result = manager2.retrieve_results_with_cache(result_id, 'user1')
        assert len(result.matched_records) == 2
        
        # Verify metadata
        metadata = manager2.get_result_metadata(result_id, 'user1')
        assert metadata is not None
        assert metadata['access_control']['public'] is True
    
    def test_concurrent_access_safety(self, result_manager, sample_matching_result):
        """Test thread safety of cache operations."""
        import threading
        import time
        
        # Store result
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        results = []
        errors = []
        
        def retrieve_result():
            try:
                for _ in range(10):
                    result = result_manager.retrieve_results_with_cache(result_id, 'user1')
                    results.append(len(result.matched_records))
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=retrieve_result)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
        
        # Verify all retrievals were successful
        assert len(results) == 50  # 5 threads * 10 retrievals each
        assert all(count == 2 for count in results)  # All should have 2 matched records
    
    def test_storage_info_with_new_features(self, result_manager, sample_matching_result):
        """Test storage info includes new features."""
        # Store some results with different configurations
        result_id1 = result_manager.store_results(sample_matching_result, owner_id='user1')
        result_id2 = result_manager.store_results(sample_matching_result, owner_id='user2')
        
        # Make one public and share another
        result_manager.make_result_public(result_id1, 'user1', public=True)
        result_manager.share_result(result_id2, 'user2', ['user3'])
        
        # Get storage info
        info = result_manager.get_storage_info()
        
        # Verify new fields are present
        assert 'compressed_results' in info
        assert 'compression_ratio' in info
        assert 'public_results' in info
        assert 'shared_results' in info
        assert 'cache_info' in info
        
        # Verify values
        assert info['total_results'] == 2
        assert info['compressed_results'] == 2  # Both compressed
        assert info['compression_ratio'] == 100.0  # All compressed
        assert info['public_results'] == 1
        assert info['shared_results'] == 1
        
        # Verify cache info is included
        cache_info = info['cache_info']
        assert 'cache_size' in cache_info
        assert 'max_cache_size' in cache_info
    
    def test_error_handling_corrupted_files(self, result_manager, sample_matching_result):
        """Test error handling when result files are corrupted."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Corrupt the file
        stored_result = result_manager.results_index[result_id]
        result_file = Path(stored_result.file_path)
        
        # Write invalid data to file
        with open(result_file, 'wb') as f:
            f.write(b'corrupted data')
        
        # Clear cache to force disk read
        result_manager.clear_cache()
        
        # Retrieval should fail gracefully
        with pytest.raises(FileProcessingError):
            result_manager.retrieve_results_with_cache(result_id, 'user1')
    
    def test_unauthorized_operations(self, result_manager, sample_matching_result):
        """Test unauthorized operations are properly blocked."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result, owner_id='user1')
        
        # Non-owner should not be able to share
        success = result_manager.share_result(result_id, 'user2', ['user3'])
        assert success is False
        
        # Non-owner should not be able to make public
        success = result_manager.make_result_public(result_id, 'user2', public=True)
        assert success is False
        
        # Verify original access control is unchanged
        stored_result = result_manager.results_index[result_id]
        assert stored_result.access_control.owner_id == 'user1'
        assert stored_result.access_control.public is False
        assert len(stored_result.access_control.shared_with) == 0


if __name__ == '__main__':
    pytest.main([__file__])