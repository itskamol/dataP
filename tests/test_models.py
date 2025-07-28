"""
Unit tests for core data models.
Tests validation, serialization, and edge cases for all data models.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import json
import uuid

from src.domain.models import (
    FieldMapping, AlgorithmConfig, DatasetMetadata, Dataset,
    MatchingConfig, MatchedRecord, MatchingStatistics, ResultMetadata,
    MatchingResult, ValidationResult, ProgressStatus,
    MatchingType, AlgorithmType, FileType
)


class TestFieldMapping:
    """Test cases for FieldMapping model."""
    
    def test_valid_field_mapping_creation(self):
        """Test creating a valid field mapping."""
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY,
            weight=0.8,
            normalization=True,
            case_sensitive=False
        )
        
        assert mapping.source_field == "name"
        assert mapping.target_field == "full_name"
        assert mapping.algorithm == "fuzzy"  # Enum values are converted to strings
        assert mapping.weight == 0.8
        assert mapping.normalization is True
        assert mapping.case_sensitive is False
    
    def test_field_mapping_validation_errors(self):
        """Test validation errors for field mapping."""
        # Test empty source field
        with pytest.raises(ValueError):
            FieldMapping(
                source_field="",
                target_field="name",
                algorithm=AlgorithmType.EXACT
            )
        
        # Test empty target field
        with pytest.raises(ValueError):
            FieldMapping(
                source_field="name",
                target_field="",
                algorithm=AlgorithmType.EXACT
            )
        
        # Test invalid weight (too low)
        with pytest.raises(ValueError):
            FieldMapping(
                source_field="name",
                target_field="full_name",
                algorithm=AlgorithmType.FUZZY,
                weight=0.05
            )
        
        # Test invalid weight (too high)
        with pytest.raises(ValueError):
            FieldMapping(
                source_field="name",
                target_field="full_name",
                algorithm=AlgorithmType.FUZZY,
                weight=1.5
            )
    
    def test_field_mapping_serialization(self):
        """Test serialization and deserialization of field mapping."""
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY,
            weight=0.8
        )
        
        # Test to_dict
        data = mapping.to_dict()
        assert isinstance(data, dict)
        assert data["source_field"] == "name"
        assert data["algorithm"] == "fuzzy"
        
        # Test from_dict
        restored = FieldMapping.from_dict(data)
        assert restored.source_field == mapping.source_field
        assert restored.algorithm == mapping.algorithm
        assert restored.weight == mapping.weight


class TestAlgorithmConfig:
    """Test cases for AlgorithmConfig model."""
    
    def test_valid_algorithm_config_creation(self):
        """Test creating a valid algorithm configuration."""
        config = AlgorithmConfig(
            name="fuzzy_matcher",
            algorithm_type=AlgorithmType.FUZZY,
            parameters={"threshold": 0.8, "method": "levenshtein"},
            enabled=True,
            priority=2
        )
        
        assert config.name == "fuzzy_matcher"
        assert config.algorithm_type == "fuzzy"  # Enum values are converted to strings
        assert config.parameters["threshold"] == 0.8
        assert config.enabled is True
        assert config.priority == 2
    
    def test_algorithm_config_validation_errors(self):
        """Test validation errors for algorithm configuration."""
        # Test empty name
        with pytest.raises(ValueError):
            AlgorithmConfig(
                name="",
                algorithm_type=AlgorithmType.EXACT
            )
        
        # Test invalid priority
        with pytest.raises(ValueError):
            AlgorithmConfig(
                name="test",
                algorithm_type=AlgorithmType.EXACT,
                priority=0
            )
    
    def test_algorithm_config_serialization(self):
        """Test serialization and deserialization of algorithm config."""
        config = AlgorithmConfig(
            name="fuzzy_matcher",
            algorithm_type=AlgorithmType.FUZZY,
            parameters={"threshold": 0.8}
        )
        
        data = config.to_dict()
        restored = AlgorithmConfig.from_dict(data)
        
        assert restored.name == config.name
        assert restored.algorithm_type == config.algorithm_type
        assert restored.parameters == config.parameters


class TestDatasetMetadata:
    """Test cases for DatasetMetadata model."""
    
    def test_valid_dataset_metadata_creation(self):
        """Test creating valid dataset metadata."""
        metadata = DatasetMetadata(
            name="test_dataset",
            file_path="/path/to/file.csv",
            file_type=FileType.CSV,
            delimiter=",",
            encoding="utf-8",
            row_count=1000,
            column_count=5
        )
        
        assert metadata.name == "test_dataset"
        assert metadata.file_type == "csv"  # Enum values are converted to strings
        assert metadata.row_count == 1000
        assert metadata.column_count == 5
        assert isinstance(metadata.id, str)
        assert isinstance(metadata.created_at, datetime)
    
    def test_dataset_metadata_validation_errors(self):
        """Test validation errors for dataset metadata."""
        # Test negative row count
        with pytest.raises(ValueError):
            DatasetMetadata(row_count=-1)
        
        # Test negative column count
        with pytest.raises(ValueError):
            DatasetMetadata(column_count=-1)
    
    def test_dataset_metadata_serialization(self):
        """Test serialization and deserialization of dataset metadata."""
        metadata = DatasetMetadata(
            name="test_dataset",
            file_type=FileType.JSON,
            row_count=500
        )
        
        data = metadata.to_dict()
        restored = DatasetMetadata.from_dict(data)
        
        assert restored.name == metadata.name
        assert restored.file_type == metadata.file_type
        assert restored.row_count == metadata.row_count
        assert restored.created_at == metadata.created_at


class TestDataset:
    """Test cases for Dataset model."""
    
    def test_valid_dataset_creation(self):
        """Test creating a valid dataset."""
        df = pd.DataFrame({
            'name': ['John', 'Jane'],
            'age': [25, 30]
        })
        
        dataset = Dataset(
            name="test_dataset",
            data=df
        )
        
        assert dataset.name == "test_dataset"
        assert len(dataset.columns) == 2
        assert dataset.metadata.row_count == 2
        assert dataset.metadata.column_count == 2
    
    def test_dataset_without_data(self):
        """Test creating dataset without data."""
        dataset = Dataset(name="empty_dataset")
        
        assert dataset.name == "empty_dataset"
        assert len(dataset.columns) == 0
        assert dataset.data is None
        assert dataset.metadata.row_count == 0
    
    def test_dataset_serialization(self):
        """Test serialization and deserialization of dataset."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        dataset = Dataset(name="test", data=df)
        
        # Test serialization without data
        data = dataset.to_dict(include_data=False)
        assert 'data' not in data
        
        # Test serialization with data
        data_with_df = dataset.to_dict(include_data=True)
        assert 'data' in data_with_df
        assert isinstance(data_with_df['data'], list)
        
        # Test deserialization
        restored = Dataset.from_dict(data_with_df)
        assert restored.name == dataset.name
        assert len(restored.data) == len(dataset.data)


class TestMatchingConfig:
    """Test cases for MatchingConfig model."""
    
    def test_valid_matching_config_creation(self):
        """Test creating a valid matching configuration."""
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        config = MatchingConfig(
            mappings=[mapping],
            matching_type=MatchingType.ONE_TO_ONE,
            confidence_threshold=80.0,
            use_blocking=True
        )
        
        assert len(config.mappings) == 1
        assert config.matching_type == "one-to-one"  # Enum values are converted to strings
        assert config.confidence_threshold == 80.0
        assert config.use_blocking is True
    
    def test_matching_config_validation_errors(self):
        """Test validation errors for matching configuration."""
        # Test empty mappings
        with pytest.raises(ValueError):
            MatchingConfig(mappings=[])
        
        # Test invalid confidence threshold (too low)
        with pytest.raises(ValueError):
            mapping = FieldMapping(
                source_field="name",
                target_field="full_name",
                algorithm=AlgorithmType.EXACT
            )
            MatchingConfig(
                mappings=[mapping],
                confidence_threshold=-1
            )
        
        # Test invalid confidence threshold (too high)
        with pytest.raises(ValueError):
            mapping = FieldMapping(
                source_field="name",
                target_field="full_name",
                algorithm=AlgorithmType.EXACT
            )
            MatchingConfig(
                mappings=[mapping],
                confidence_threshold=101
            )
    
    def test_matching_config_serialization(self):
        """Test serialization and deserialization of matching config."""
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        config = MatchingConfig(
            mappings=[mapping],
            confidence_threshold=85.0
        )
        
        data = config.to_dict()
        restored = MatchingConfig.from_dict(data)
        
        assert len(restored.mappings) == len(config.mappings)
        assert restored.confidence_threshold == config.confidence_threshold


class TestMatchedRecord:
    """Test cases for MatchedRecord model."""
    
    def test_valid_matched_record_creation(self):
        """Test creating a valid matched record."""
        record = MatchedRecord(
            record1={"name": "John Doe", "age": 25},
            record2={"full_name": "John Doe", "years": 25},
            confidence_score=95.5,
            matching_fields=["name", "age"]
        )
        
        assert record.record1["name"] == "John Doe"
        assert record.record2["full_name"] == "John Doe"
        assert record.confidence_score == 95.5
        assert "name" in record.matching_fields
        assert isinstance(record.created_at, datetime)
    
    def test_matched_record_validation_errors(self):
        """Test validation errors for matched record."""
        # Test invalid confidence score (too low)
        with pytest.raises(ValueError):
            MatchedRecord(
                record1={"name": "John"},
                record2={"name": "John"},
                confidence_score=-1
            )
        
        # Test invalid confidence score (too high)
        with pytest.raises(ValueError):
            MatchedRecord(
                record1={"name": "John"},
                record2={"name": "John"},
                confidence_score=101
            )
    
    def test_matched_record_serialization(self):
        """Test serialization and deserialization of matched record."""
        record = MatchedRecord(
            record1={"name": "John"},
            record2={"name": "John"},
            confidence_score=90.0
        )
        
        data = record.to_dict()
        restored = MatchedRecord.from_dict(data)
        
        assert restored.record1 == record.record1
        assert restored.confidence_score == record.confidence_score
        assert restored.created_at == record.created_at


class TestMatchingStatistics:
    """Test cases for MatchingStatistics model."""
    
    def test_valid_matching_statistics_creation(self):
        """Test creating valid matching statistics."""
        stats = MatchingStatistics(
            total_records_file1=1000,
            total_records_file2=800,
            total_comparisons=50000,
            high_confidence_matches=600,
            low_confidence_matches=100,
            unmatched_file1=300,
            unmatched_file2=100,
            processing_time_seconds=45.5,
            average_confidence=82.3
        )
        
        assert stats.total_records_file1 == 1000
        assert stats.high_confidence_matches == 600
        assert stats.processing_time_seconds == 45.5
    
    def test_matching_statistics_properties(self):
        """Test computed properties of matching statistics."""
        stats = MatchingStatistics(
            total_records_file1=100,
            total_records_file2=80,
            high_confidence_matches=60,
            low_confidence_matches=20
        )
        
        assert stats.match_rate_file1 == 80.0  # (60+20)/100 * 100
        assert stats.match_rate_file2 == 100.0  # (60+20)/80 * 100
    
    def test_matching_statistics_zero_division(self):
        """Test match rate calculation with zero records."""
        stats = MatchingStatistics(
            total_records_file1=0,
            total_records_file2=0,
            high_confidence_matches=0,
            low_confidence_matches=0
        )
        
        assert stats.match_rate_file1 == 0.0
        assert stats.match_rate_file2 == 0.0
    
    def test_matching_statistics_validation_errors(self):
        """Test validation errors for matching statistics."""
        # Test negative values
        with pytest.raises(ValueError):
            MatchingStatistics(total_records_file1=-1)
        
        with pytest.raises(ValueError):
            MatchingStatistics(processing_time_seconds=-1)
        
        with pytest.raises(ValueError):
            MatchingStatistics(average_confidence=101)
    
    def test_matching_statistics_serialization(self):
        """Test serialization and deserialization of matching statistics."""
        stats = MatchingStatistics(
            total_records_file1=100,
            high_confidence_matches=80,
            processing_time_seconds=30.5
        )
        
        data = stats.to_dict()
        assert 'match_rate_file1' in data  # Computed property included
        
        restored = MatchingStatistics.from_dict(data)
        assert restored.total_records_file1 == stats.total_records_file1
        assert restored.high_confidence_matches == stats.high_confidence_matches


class TestResultMetadata:
    """Test cases for ResultMetadata model."""
    
    def test_valid_result_metadata_creation(self):
        """Test creating valid result metadata."""
        file1_meta = DatasetMetadata(name="file1", row_count=100)
        file2_meta = DatasetMetadata(name="file2", row_count=80)
        
        metadata = ResultMetadata(
            config_hash="abc123",
            file1_metadata=file1_meta,
            file2_metadata=file2_meta,
            processing_node="node-1",
            version="2.0"
        )
        
        assert metadata.config_hash == "abc123"
        assert metadata.file1_metadata.name == "file1"
        assert metadata.processing_node == "node-1"
        assert metadata.version == "2.0"
        assert isinstance(metadata.operation_id, str)
    
    def test_result_metadata_serialization(self):
        """Test serialization and deserialization of result metadata."""
        metadata = ResultMetadata(
            config_hash="test123",
            processing_node="test-node"
        )
        
        data = metadata.to_dict()
        restored = ResultMetadata.from_dict(data)
        
        assert restored.config_hash == metadata.config_hash
        assert restored.processing_node == metadata.processing_node
        assert restored.created_at == metadata.created_at


class TestMatchingResult:
    """Test cases for MatchingResult model."""
    
    def test_valid_matching_result_creation(self):
        """Test creating a valid matching result."""
        matched_record = MatchedRecord(
            record1={"name": "John"},
            record2={"name": "John"},
            confidence_score=90.0
        )
        
        result = MatchingResult(
            matched_records=[matched_record],
            statistics=MatchingStatistics(high_confidence_matches=1)
        )
        
        assert len(result.matched_records) == 1
        assert result.total_matches == 1
        assert isinstance(result.metadata, ResultMetadata)
    
    def test_matching_result_methods(self):
        """Test methods of matching result."""
        result = MatchingResult()
        
        # Test add_matched_record
        record = MatchedRecord(
            record1={"name": "John"},
            record2={"name": "John"},
            confidence_score=85.0
        )
        result.add_matched_record(record)
        assert len(result.matched_records) == 1
        
        # Test add_unmatched_record
        result.add_unmatched_record("file1", {"name": "Jane"})
        assert "file1" in result.unmatched_records
        assert len(result.unmatched_records["file1"]) == 1
    
    def test_matching_result_confidence_filtering(self):
        """Test confidence-based filtering methods."""
        high_conf_record = MatchedRecord(
            record1={"name": "John"},
            record2={"name": "John"},
            confidence_score=90.0
        )
        low_conf_record = MatchedRecord(
            record1={"name": "Jane"},
            record2={"name": "Jane"},
            confidence_score=60.0
        )
        
        result = MatchingResult(
            matched_records=[high_conf_record, low_conf_record]
        )
        
        high_matches = result.get_high_confidence_matches(threshold=75.0)
        low_matches = result.get_low_confidence_matches(threshold=75.0)
        
        assert len(high_matches) == 1
        assert len(low_matches) == 1
        assert high_matches[0].confidence_score == 90.0
        assert low_matches[0].confidence_score == 60.0
    
    def test_matching_result_serialization(self):
        """Test serialization and deserialization of matching result."""
        record = MatchedRecord(
            record1={"name": "John"},
            record2={"name": "John"},
            confidence_score=90.0
        )
        
        result = MatchingResult(matched_records=[record])
        
        data = result.to_dict()
        restored = MatchingResult.from_dict(data)
        
        assert len(restored.matched_records) == len(result.matched_records)
        assert restored.matched_records[0].confidence_score == 90.0


class TestValidationResult:
    """Test cases for ValidationResult model."""
    
    def test_valid_validation_result_creation(self):
        """Test creating a valid validation result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue detected"],
            metadata={"checked_fields": 5}
        )
        
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.metadata["checked_fields"] == 5
    
    def test_validation_result_methods(self):
        """Test methods of validation result."""
        result = ValidationResult(is_valid=True)
        
        # Test add_error
        result.add_error("Critical error")
        assert result.is_valid is False
        assert result.has_errors is True
        assert "Critical error" in result.errors
        
        # Test add_warning
        result.add_warning("Warning message")
        assert result.has_warnings is True
        assert "Warning message" in result.warnings
    
    def test_validation_result_properties(self):
        """Test computed properties of validation result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert result.has_errors is True
        assert result.has_warnings is True
        
        empty_result = ValidationResult(is_valid=True)
        assert empty_result.has_errors is False
        assert empty_result.has_warnings is False
    
    def test_validation_result_serialization(self):
        """Test serialization and deserialization of validation result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Test error"],
            warnings=["Test warning"]
        )
        
        data = result.to_dict()
        assert 'has_errors' in data  # Computed property included
        assert 'has_warnings' in data
        
        restored = ValidationResult.from_dict(data)
        assert restored.is_valid == result.is_valid
        assert restored.errors == result.errors
        assert restored.warnings == result.warnings


class TestProgressStatus:
    """Test cases for ProgressStatus model."""
    
    def test_valid_progress_status_creation(self):
        """Test creating a valid progress status."""
        start_time = datetime.now()
        status = ProgressStatus(
            operation_id="op-123",
            status="running",
            progress=45.5,
            message="Processing records...",
            current_step=45,
            total_steps=100,
            started_at=start_time
        )
        
        assert status.operation_id == "op-123"
        assert status.status == "running"
        assert status.progress == 45.5
        assert status.current_step == 45
        assert status.started_at == start_time
    
    def test_progress_status_validation_errors(self):
        """Test validation errors for progress status."""
        # Test invalid status
        with pytest.raises(ValueError):
            ProgressStatus(
                operation_id="op-123",
                status="invalid_status",
                progress=50.0
            )
        
        # Test invalid progress (too low)
        with pytest.raises(ValueError):
            ProgressStatus(
                operation_id="op-123",
                status="running",
                progress=-1.0
            )
        
        # Test invalid progress (too high)
        with pytest.raises(ValueError):
            ProgressStatus(
                operation_id="op-123",
                status="running",
                progress=101.0
            )
        
        # Test negative current_step
        with pytest.raises(ValueError):
            ProgressStatus(
                operation_id="op-123",
                status="running",
                progress=50.0,
                current_step=-1
            )
    
    def test_progress_status_properties(self):
        """Test computed properties of progress status."""
        # Test running status
        running_status = ProgressStatus(
            operation_id="op-123",
            status="running",
            progress=50.0
        )
        assert running_status.is_running is True
        assert running_status.is_completed is False
        
        # Test completed status
        completed_status = ProgressStatus(
            operation_id="op-123",
            status="completed",
            progress=100.0
        )
        assert completed_status.is_running is False
        assert completed_status.is_completed is True
        
        # Test error status
        error_status = ProgressStatus(
            operation_id="op-123",
            status="error",
            progress=75.0
        )
        assert error_status.is_running is False
        assert error_status.is_completed is True
    
    def test_progress_status_duration_calculation(self):
        """Test duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)
        
        # Test with both start and end times
        status = ProgressStatus(
            operation_id="op-123",
            status="completed",
            progress=100.0,
            started_at=start_time,
            completed_at=end_time
        )
        
        duration = status.duration_seconds
        assert duration is not None
        assert abs(duration - 30.0) < 1.0  # Allow small timing differences
        
        # Test with no start time
        status_no_start = ProgressStatus(
            operation_id="op-123",
            status="idle",
            progress=0.0
        )
        assert status_no_start.duration_seconds is None
    
    def test_progress_status_serialization(self):
        """Test serialization and deserialization of progress status."""
        start_time = datetime.now()
        status = ProgressStatus(
            operation_id="op-123",
            status="running",
            progress=75.0,
            started_at=start_time
        )
        
        data = status.to_dict()
        assert 'is_running' in data  # Computed property included
        assert 'is_completed' in data
        assert 'duration_seconds' in data
        
        restored = ProgressStatus.from_dict(data)
        assert restored.operation_id == status.operation_id
        assert restored.status == status.status
        assert restored.progress == status.progress
        assert restored.started_at == status.started_at


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_enum_serialization(self):
        """Test that enums are properly serialized."""
        mapping = FieldMapping(
            source_field="test",
            target_field="test",
            algorithm=AlgorithmType.FUZZY
        )
        
        data = mapping.to_dict()
        assert data["algorithm"] == "fuzzy"  # Should be string, not enum
    
    def test_datetime_serialization(self):
        """Test that datetime objects are properly serialized."""
        metadata = DatasetMetadata(name="test")
        data = metadata.to_dict()
        
        # In Pydantic V2, datetime objects are not automatically serialized to strings
        # They remain as datetime objects unless explicitly serialized with mode='json'
        assert isinstance(data["created_at"], datetime)
        
        # Test JSON serialization mode
        json_data = metadata.model_dump(mode='json')
        assert isinstance(json_data["created_at"], str)
        assert "T" in json_data["created_at"]  # ISO format indicator
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create a large DataFrame
        large_df = pd.DataFrame({
            'col1': range(10000),
            'col2': [f'value_{i}' for i in range(10000)]
        })
        
        dataset = Dataset(name="large_dataset", data=large_df)
        
        assert dataset.metadata.row_count == 10000
        assert dataset.metadata.column_count == 2
        
        # Test serialization without data (should be fast)
        data = dataset.to_dict(include_data=False)
        assert 'data' not in data
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        # Test with Uzbek text
        mapping = FieldMapping(
            source_field="ism",  # Uzbek for "name"
            target_field="to'liq_ism",  # Uzbek for "full name"
            algorithm=AlgorithmType.FUZZY
        )
        
        record = MatchedRecord(
            record1={"ism": "Алишер Навоий"},
            record2={"to'liq_ism": "Алишер Навоий"},
            confidence_score=95.0
        )
        
        # Should handle Unicode without errors
        mapping_data = mapping.to_dict()
        record_data = record.to_dict()
        
        assert mapping_data["source_field"] == "ism"
        assert record_data["record1"]["ism"] == "Алишер Навоий"
    
    def test_nested_object_serialization(self):
        """Test serialization of nested objects."""
        file1_meta = DatasetMetadata(name="file1", row_count=100)
        file2_meta = DatasetMetadata(name="file2", row_count=80)
        
        result_meta = ResultMetadata(
            file1_metadata=file1_meta,
            file2_metadata=file2_meta
        )
        
        data = result_meta.to_dict()
        restored = ResultMetadata.from_dict(data)
        
        assert restored.file1_metadata.name == "file1"
        assert restored.file2_metadata.name == "file2"
        assert restored.file1_metadata.row_count == 100


if __name__ == "__main__":
    pytest.main([__file__])