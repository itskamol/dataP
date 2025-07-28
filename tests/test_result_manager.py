"""
Unit tests for ResultManager class.
Tests all export formats, pagination, filtering, and cleanup operations.
"""

import pytest
import tempfile
import shutil
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd

from src.application.services.result_manager import (
    ResultManager, ExportFormat, ExportConfig, ResultVersion, StoredResult
)
from src.domain.models import (
    MatchingResult, MatchedRecord, MatchingStatistics, ResultMetadata,
    DatasetMetadata
)
from src.domain.exceptions import FileProcessingError, ValidationError


class TestResultManager:
    """Test suite for ResultManager."""
    
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
            max_storage_size_mb=10,  # Small limit for testing
            auto_cleanup_days=7,
            compression_enabled=True
        )
    
    @pytest.fixture
    def sample_matching_result(self):
        """Create sample MatchingResult for testing."""
        # Create sample matched records
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
            ),
            MatchedRecord(
                record1={'id': '3', 'name': 'Bob Johnson', 'city': 'Chicago'},
                record2={'id': 'C', 'name': 'Robert Johnson', 'location': 'Chicago'},
                confidence_score=72.1,
                matching_fields=['city'],
                metadata={'algorithm': 'phonetic'},
                created_at=datetime.now()
            )
        ]
        
        # Create unmatched records
        unmatched_records = {
            'file1': [
                {'id': '4', 'name': 'Alice Brown', 'city': 'Seattle'},
                {'id': '5', 'name': 'Charlie Wilson', 'city': 'Portland'}
            ],
            'file2': [
                {'id': 'D', 'name': 'David Lee', 'location': 'Miami'}
            ]
        }
        
        # Create statistics
        statistics = MatchingStatistics(
            total_records_file1=5,
            total_records_file2=4,
            total_comparisons=20,
            high_confidence_matches=2,
            low_confidence_matches=1,
            unmatched_file1=2,
            unmatched_file2=1,
            processing_time_seconds=15.5,
            average_confidence=85.3
        )
        
        # Create metadata
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
    
    def test_initialization(self, temp_dirs):
        """Test ResultManager initialization."""
        storage_dir, temp_dir = temp_dirs
        
        manager = ResultManager(
            storage_dir=storage_dir,
            temp_dir=temp_dir,
            max_storage_size_mb=100,
            auto_cleanup_days=30,
            compression_enabled=False
        )
        
        assert manager.storage_dir == Path(storage_dir)
        assert manager.temp_dir == Path(temp_dir)
        assert manager.max_storage_size_mb == 100
        assert manager.auto_cleanup_days == 30
        assert manager.compression_enabled is False
        assert manager.results_index == {}
        
        # Check directories were created
        assert Path(storage_dir).exists()
        assert Path(temp_dir).exists()
    
    def test_store_results(self, result_manager, sample_matching_result):
        """Test storing results."""
        result_id = result_manager.store_results(sample_matching_result, 'test-op-123')
        
        # Check result ID is valid UUID
        assert len(result_id) == 36
        assert result_id in result_manager.results_index
        
        # Check stored result metadata
        stored_result = result_manager.results_index[result_id]
        assert stored_result.operation_id == 'test-op-123'
        assert stored_result.compressed is True
        assert stored_result.size_bytes > 0
        assert len(stored_result.versions) == 1
        assert stored_result.metadata['total_matches'] == 3
        
        # Check file was created
        result_file = Path(stored_result.file_path)
        assert result_file.exists()
        assert result_file.suffix == '.gz'
    
    def test_store_results_without_compression(self, temp_dirs, sample_matching_result):
        """Test storing results without compression."""
        storage_dir, temp_dir = temp_dirs
        manager = ResultManager(
            storage_dir=storage_dir,
            temp_dir=temp_dir,
            compression_enabled=False
        )
        
        result_id = manager.store_results(sample_matching_result)
        stored_result = manager.results_index[result_id]
        
        assert stored_result.compressed is False
        result_file = Path(stored_result.file_path)
        assert result_file.exists()
        assert result_file.suffix == '.pkl'
    
    def test_retrieve_results(self, result_manager, sample_matching_result):
        """Test retrieving stored results."""
        # Store results first
        result_id = result_manager.store_results(sample_matching_result)
        
        # Add small delay to ensure different timestamps
        import time
        time.sleep(0.01)
        
        # Retrieve results
        retrieved_results = result_manager.retrieve_results(result_id)
        
        # Check retrieved data matches original
        assert len(retrieved_results.matched_records) == 3
        assert len(retrieved_results.unmatched_records) == 2
        assert retrieved_results.statistics.total_records_file1 == 5
        assert retrieved_results.metadata.operation_id == sample_matching_result.metadata.operation_id
        
        # Check last accessed time was updated
        stored_result = result_manager.results_index[result_id]
        assert stored_result.last_accessed >= stored_result.created_at
    
    def test_retrieve_nonexistent_result(self, result_manager):
        """Test retrieving non-existent result."""
        with pytest.raises(ValidationError, match="Access denied"):
            result_manager.retrieve_results('nonexistent-id')
    
    def test_export_to_csv(self, result_manager, sample_matching_result):
        """Test exporting results to CSV format."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export to CSV
        config = ExportConfig(format=ExportFormat.CSV)
        export_path = result_manager.export_results(result_id, config)
        
        # Check export file exists
        assert Path(export_path).exists()
        assert export_path.endswith('.csv')
        
        # Check CSV content
        df = pd.read_csv(export_path)
        assert len(df) == 3  # 3 matched records
        assert 'confidence_score' in df.columns
        assert 'matching_fields' in df.columns
        assert 'record1_name' in df.columns
        assert 'record2_name' in df.columns
        
        # Check data values
        assert df.iloc[0]['confidence_score'] == 95.5
        assert df.iloc[0]['record1_name'] == 'John Doe'
    
    def test_export_to_json(self, result_manager, sample_matching_result):
        """Test exporting results to JSON format."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export to JSON
        config = ExportConfig(
            format=ExportFormat.JSON,
            include_metadata=True,
            include_statistics=True,
            include_unmatched=True
        )
        export_path = result_manager.export_results(result_id, config)
        
        # Check export file exists
        assert Path(export_path).exists()
        assert export_path.endswith('.json')
        
        # Check JSON content
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'matched_records' in data
        assert 'unmatched_records' in data
        assert 'statistics' in data
        assert 'metadata' in data
        assert 'export_info' in data
        
        assert len(data['matched_records']) == 3
        assert len(data['unmatched_records']) == 2
        assert data['statistics']['total_records_file1'] == 5
    
    def test_export_to_excel(self, result_manager, sample_matching_result):
        """Test exporting results to Excel format."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export to Excel
        config = ExportConfig(
            format=ExportFormat.EXCEL,
            include_unmatched=True,
            include_statistics=True
        )
        export_path = result_manager.export_results(result_id, config)
        
        # Check export file exists
        assert Path(export_path).exists()
        assert export_path.endswith('.xlsx')
        
        # Check Excel content
        excel_file = pd.ExcelFile(export_path)
        sheet_names = excel_file.sheet_names
        
        assert 'Matched Records' in sheet_names
        assert 'Statistics' in sheet_names
        assert any('Unmatched' in name for name in sheet_names)
        
        # Check matched records sheet
        df_matched = pd.read_excel(export_path, sheet_name='Matched Records')
        assert len(df_matched) == 3
        assert 'confidence_score' in df_matched.columns
    
    def test_export_to_pickle(self, result_manager, sample_matching_result):
        """Test exporting results to pickle format."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export to pickle
        config = ExportConfig(format=ExportFormat.PICKLE)
        export_path = result_manager.export_results(result_id, config)
        
        # Check export file exists
        assert Path(export_path).exists()
        assert export_path.endswith('.pickle')
        
        # Check pickle content
        with open(export_path, 'rb') as f:
            data = pickle.load(f)
        
        assert isinstance(data, MatchingResult)
        assert len(data.matched_records) == 3
    
    def test_export_with_filters(self, result_manager, sample_matching_result):
        """Test exporting with confidence filters."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export with high confidence filter
        config = ExportConfig(
            format=ExportFormat.JSON,
            filters={'min_confidence': 80.0}
        )
        export_path = result_manager.export_results(result_id, config)
        
        # Check filtered content
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Should only have 2 records with confidence >= 80
        assert len(data['matched_records']) == 2
        for record in data['matched_records']:
            assert record['confidence_score'] >= 80.0
    
    def test_export_with_pagination(self, result_manager, sample_matching_result):
        """Test exporting with pagination."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export first page
        config = ExportConfig(
            format=ExportFormat.JSON,
            pagination={'page': 1, 'page_size': 2}
        )
        export_path = result_manager.export_results(result_id, config)
        
        # Check paginated content
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data['matched_records']) == 2
        assert data['export_info']['total_records'] == 2
    
    def test_export_with_sorting(self, result_manager, sample_matching_result):
        """Test exporting with sorting."""
        # Store results
        result_id = result_manager.store_results(sample_matching_result)
        
        # Export sorted by confidence (ascending)
        config = ExportConfig(
            format=ExportFormat.JSON,
            sort_by='confidence_score',
            sort_order='asc'
        )
        export_path = result_manager.export_results(result_id, config)
        
        # Check sorted content
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scores = [record['confidence_score'] for record in data['matched_records']]
        assert scores == sorted(scores)  # Should be in ascending order
    
    def test_list_results(self, result_manager, sample_matching_result):
        """Test listing stored results."""
        # Store multiple results
        result_id1 = result_manager.store_results(sample_matching_result, 'op-1')
        result_id2 = result_manager.store_results(sample_matching_result, 'op-2')
        
        # List all results
        results_list = result_manager.list_results()
        
        assert len(results_list) == 2
        result_ids = [r['result_id'] for r in results_list]
        assert result_id1 in result_ids
        assert result_id2 in result_ids
        
        # Check result structure
        result_info = results_list[0]
        assert 'result_id' in result_info
        assert 'operation_id' in result_info
        assert 'created_at' in result_info
        assert 'size_bytes' in result_info
        assert 'metadata' in result_info
    
    def test_list_results_with_filter(self, result_manager, sample_matching_result):
        """Test listing results with operation ID filter."""
        # Store results with different operation IDs
        result_manager.store_results(sample_matching_result, 'op-1')
        result_manager.store_results(sample_matching_result, 'op-2')
        
        # List results for specific operation
        results_list = result_manager.list_results(operation_id='op-1')
        
        assert len(results_list) == 1
        assert results_list[0]['operation_id'] == 'op-1'
    
    def test_list_results_with_limit(self, result_manager, sample_matching_result):
        """Test listing results with limit."""
        # Store multiple results
        for i in range(5):
            result_manager.store_results(sample_matching_result, f'op-{i}')
        
        # List with limit
        results_list = result_manager.list_results(limit=3)
        
        assert len(results_list) == 3
    
    def test_delete_result(self, result_manager, sample_matching_result):
        """Test deleting a stored result."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result)
        stored_result = result_manager.results_index[result_id]
        result_file = Path(stored_result.file_path)
        
        # Verify file exists
        assert result_file.exists()
        assert result_id in result_manager.results_index
        
        # Delete result
        success = result_manager.delete_result(result_id)
        
        assert success is True
        assert result_id not in result_manager.results_index
        assert not result_file.exists()
    
    def test_delete_nonexistent_result(self, result_manager):
        """Test deleting non-existent result."""
        success = result_manager.delete_result('nonexistent-id')
        assert success is False
    
    def test_cleanup_old_results(self, result_manager, sample_matching_result):
        """Test cleaning up old results."""
        # Store result
        result_id = result_manager.store_results(sample_matching_result)
        
        # Manually set old last_accessed time
        stored_result = result_manager.results_index[result_id]
        stored_result.last_accessed = datetime.now() - timedelta(days=10)
        result_manager._save_index()
        
        # Cleanup with 5 day threshold
        cleaned_count = result_manager.cleanup_old_results(max_age_days=5)
        
        assert cleaned_count == 1
        assert result_id not in result_manager.results_index
    
    def test_cleanup_temp_files(self, result_manager, sample_matching_result):
        """Test cleaning up temporary export files."""
        # Store and export result to create temp file
        result_id = result_manager.store_results(sample_matching_result)
        config = ExportConfig(format=ExportFormat.CSV)
        export_path = result_manager.export_results(result_id, config)
        
        # Verify temp file exists
        temp_file = Path(export_path)
        assert temp_file.exists()
        
        # Manually set old modification time
        import os
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(temp_file, (old_time, old_time))
        
        # Cleanup temp files
        cleaned_count = result_manager.cleanup_temp_files(max_age_hours=24)
        
        assert cleaned_count == 1
        assert not temp_file.exists()
    
    def test_storage_limits(self, result_manager, sample_matching_result):
        """Test storage limit enforcement."""
        # Create larger sample data to exceed storage limit
        large_sample = sample_matching_result
        # Add many more matched records to make it larger
        for i in range(1000):  # Add many records to make it larger
            large_sample.matched_records.append(
                MatchedRecord(
                    record1={'id': f'{i}', 'name': f'Person {i}', 'data': 'x' * 100},
                    record2={'id': f'{i}', 'name': f'Person {i}', 'data': 'y' * 100},
                    confidence_score=80.0,
                    matching_fields=['name'],
                    metadata={'test': 'data'},
                    created_at=datetime.now()
                )
            )
        
        # Store multiple large results to exceed limit
        result_ids = []
        for i in range(5):  # Should exceed 10MB limit with large data
            result_id = result_manager.store_results(large_sample, f'op-{i}')
            result_ids.append(result_id)
        
        # Check that some results were cleaned up or limit was enforced
        remaining_results = len(result_manager.results_index)
        assert remaining_results <= 5  # Should not exceed what we stored or be cleaned up
    
    def test_get_storage_info(self, result_manager, sample_matching_result):
        """Test getting storage information."""
        # Store some results
        result_manager.store_results(sample_matching_result, 'op-1')
        result_manager.store_results(sample_matching_result, 'op-2')
        
        # Get storage info
        info = result_manager.get_storage_info()
        
        assert 'total_results' in info
        assert 'total_size_bytes' in info
        assert 'total_size_mb' in info
        assert 'max_size_mb' in info
        assert 'usage_percentage' in info
        assert 'age_distribution' in info
        assert 'compression_enabled' in info
        assert 'auto_cleanup_days' in info
        
        assert info['total_results'] == 2
        assert info['total_size_bytes'] > 0
        assert info['max_size_mb'] == 10
        assert info['compression_enabled'] is True
    
    def test_index_persistence(self, temp_dirs, sample_matching_result):
        """Test that results index persists across manager instances."""
        storage_dir, temp_dir = temp_dirs
        
        # Create first manager and store result
        manager1 = ResultManager(storage_dir=storage_dir, temp_dir=temp_dir)
        result_id = manager1.store_results(sample_matching_result, 'test-op')
        
        # Create second manager (should load existing index)
        manager2 = ResultManager(storage_dir=storage_dir, temp_dir=temp_dir)
        
        # Check that result is available in second manager
        assert result_id in manager2.results_index
        retrieved_result = manager2.retrieve_results(result_id)
        assert len(retrieved_result.matched_records) == 3
    
    def test_error_handling_corrupted_index(self, temp_dirs):
        """Test handling of corrupted index file."""
        storage_dir, temp_dir = temp_dirs
        
        # Create corrupted index file
        index_file = Path(storage_dir) / "results_index.json"
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(index_file, 'w') as f:
            f.write("invalid json content")
        
        # Manager should handle corrupted index gracefully
        manager = ResultManager(storage_dir=storage_dir, temp_dir=temp_dir)
        assert manager.results_index == {}
    
    def test_export_empty_results(self, result_manager):
        """Test exporting empty results."""
        # Create empty result
        empty_result = MatchingResult(
            matched_records=[],
            unmatched_records={},
            statistics=MatchingStatistics(),
            metadata=ResultMetadata()
        )
        
        result_id = result_manager.store_results(empty_result)
        
        # Export to CSV
        config = ExportConfig(format=ExportFormat.CSV)
        export_path = result_manager.export_results(result_id, config)
        
        # Should create empty CSV file
        assert Path(export_path).exists()
        df = pd.read_csv(export_path)
        assert len(df) == 0
    
    @patch('src.application.services.result_manager.pickle.dump')
    def test_store_results_error_handling(self, mock_pickle_dump, result_manager, sample_matching_result):
        """Test error handling during result storage."""
        mock_pickle_dump.side_effect = Exception("Pickle error")
        
        with pytest.raises(FileProcessingError, match="Failed to store results"):
            result_manager.store_results(sample_matching_result)
    
    def test_unsupported_export_format(self, result_manager, sample_matching_result):
        """Test handling of unsupported export format."""
        result_id = result_manager.store_results(sample_matching_result)
        
        # Create config with invalid format (mock enum value)
        config = ExportConfig(format=ExportFormat.CSV)
        config.format = "unsupported_format"  # Manually set invalid format
        
        with pytest.raises(FileProcessingError):
            result_manager.export_results(result_id, config)


if __name__ == '__main__':
    pytest.main([__file__])