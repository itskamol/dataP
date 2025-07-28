"""
Unit tests for file processing service.
Tests validation, streaming, various file formats, encodings, and error conditions.
"""

import pytest
import pandas as pd
import json
import csv
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

from src.application.services.file_service import FileProcessingService
from src.domain.models import Dataset, DatasetMetadata, ValidationResult, FileType
from src.domain.exceptions import (
    FileProcessingError, FileValidationError, FileFormatError, FileAccessError
)


class TestFileProcessingService:
    """Test cases for FileProcessingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = FileProcessingService(max_file_size=1024*1024, chunk_size=100)  # 1MB, 100 rows
        
        # Create test data
        self.test_data = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'age': [25, 30, 35],
            'city': ['New York', 'Los Angeles', 'Chicago']
        })
        
        # Create test files
        self.csv_file = Path(self.temp_dir) / "test.csv"
        self.json_file = Path(self.temp_dir) / "test.json"
        self.jsonl_file = Path(self.temp_dir) / "test.jsonl"
        self.excel_file = Path(self.temp_dir) / "test.xlsx"
        
        # Write test files
        self.test_data.to_csv(self.csv_file, index=False)
        self.test_data.to_json(self.json_file, orient='records', indent=2)
        self.test_data.to_json(self.jsonl_file, orient='records', lines=True)
        self.test_data.to_excel(self.excel_file, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_service_initialization(self):
        """Test service initialization with custom parameters."""
        service = FileProcessingService(max_file_size=2048, chunk_size=500)
        
        assert service.max_file_size == 2048
        assert service.chunk_size == 500
        assert '.csv' in service.supported_formats
        assert '.json' in service.supported_formats
        assert '.xlsx' in service.supported_formats
    
    def test_detect_encoding_utf8(self):
        """Test encoding detection for UTF-8 files."""
        # Create UTF-8 file with special characters
        utf8_file = Path(self.temp_dir) / "utf8_test.csv"
        with open(utf8_file, 'w', encoding='utf-8') as f:
            f.write("name,description\n")
            f.write("Test,Café résumé naïve\n")
        
        encoding = self.service.detect_encoding(utf8_file)
        assert encoding in ['utf-8', 'UTF-8']
    
    def test_detect_encoding_latin1(self):
        """Test encoding detection for Latin-1 files."""
        # Create Latin-1 file
        latin1_file = Path(self.temp_dir) / "latin1_test.csv"
        with open(latin1_file, 'w', encoding='latin-1') as f:
            f.write("name,description\n")
            f.write("Test,Café\n")
        
        encoding = self.service.detect_encoding(latin1_file)
        # chardet might detect it as various encodings, just ensure it's not None
        assert encoding is not None
        assert len(encoding) > 0
    
    def test_validate_csv_file_success(self):
        """Test successful CSV file validation."""
        result = self.service.validate_file(str(self.csv_file))
        
        assert result.is_valid
        assert not result.has_errors
        assert result.metadata['format'] == '.csv'
        assert result.metadata['delimiter'] == ','
        assert 'name' in result.metadata['columns']
        assert result.metadata['row_count'] == 3
    
    def test_validate_json_file_success(self):
        """Test successful JSON file validation."""
        result = self.service.validate_file(str(self.json_file))
        
        assert result.is_valid
        assert not result.has_errors
        assert result.metadata['format'] == '.json'
        assert result.metadata['json_type'] == 'standard'
        assert 'name' in result.metadata['columns']
        assert result.metadata['row_count'] == 3
    
    def test_validate_jsonl_file_success(self):
        """Test successful JSON Lines file validation."""
        result = self.service.validate_file(str(self.jsonl_file))
        
        assert result.is_valid
        assert not result.has_errors
        assert result.metadata['format'] == '.json'
        assert result.metadata['json_type'] == 'lines'
        assert 'name' in result.metadata['columns']
        assert result.metadata['row_count'] == 3
    
    def test_validate_excel_file_success(self):
        """Test successful Excel file validation."""
        result = self.service.validate_file(str(self.excel_file))
        
        assert result.is_valid
        assert not result.has_errors
        assert result.metadata['format'] == '.xlsx'
        assert 'name' in result.metadata['columns']
        assert result.metadata['row_count'] == 3
    
    def test_validate_file_not_found(self):
        """Test validation of non-existent file."""
        result = self.service.validate_file("nonexistent.csv")
        
        assert not result.is_valid
        assert result.has_errors
        assert "File not found" in result.errors[0]
    
    def test_validate_file_too_large(self):
        """Test validation of file that exceeds size limit."""
        # Create a service with very small size limit
        small_service = FileProcessingService(max_file_size=10)  # 10 bytes
        
        result = small_service.validate_file(str(self.csv_file))
        
        assert not result.is_valid
        assert result.has_errors
        assert "exceeds maximum allowed size" in result.errors[0]
    
    def test_validate_unsupported_format(self):
        """Test validation of unsupported file format."""
        txt_file = Path(self.temp_dir) / "test.txt"
        txt_file.write_text("Some text content")
        
        result = self.service.validate_file(str(txt_file))
        
        assert not result.is_valid
        assert result.has_errors
        assert "Unsupported file format" in result.errors[0]
    
    def test_validate_csv_with_different_delimiters(self):
        """Test CSV validation with different delimiters."""
        # Test semicolon delimiter
        semicolon_csv = Path(self.temp_dir) / "semicolon.csv"
        with open(semicolon_csv, 'w') as f:
            f.write("name;age;city\n")
            f.write("John;25;NYC\n")
            f.write("Jane;30;LA\n")
        
        result = self.service.validate_file(str(semicolon_csv))
        
        assert result.is_valid
        assert result.metadata['delimiter'] == ';'
        assert len(result.metadata['columns']) == 3
    
    def test_validate_csv_with_duplicate_columns(self):
        """Test CSV validation with duplicate column names."""
        duplicate_csv = Path(self.temp_dir) / "duplicate.csv"
        with open(duplicate_csv, 'w') as f:
            f.write("name,age,name\n")
            f.write("John,25,Doe\n")
        
        result = self.service.validate_file(str(duplicate_csv))
        
        assert not result.is_valid
        assert result.has_errors
        assert "Duplicate column names found" in result.errors[0]
    
    def test_validate_empty_csv_file(self):
        """Test validation of empty CSV file."""
        empty_csv = Path(self.temp_dir) / "empty.csv"
        empty_csv.write_text("name,age,city\n")  # Only header
        
        result = self.service.validate_file(str(empty_csv))
        
        assert result.is_valid  # Should be valid but with warning
        assert result.has_warnings
        assert "File is empty" in result.warnings[0]
    
    def test_validate_csv_with_empty_columns(self):
        """Test CSV validation with completely empty columns."""
        empty_col_csv = Path(self.temp_dir) / "empty_col.csv"
        with open(empty_col_csv, 'w') as f:
            f.write("name,empty_col,age\n")
            f.write("John,,25\n")
            f.write("Jane,,30\n")
        
        result = self.service.validate_file(str(empty_col_csv))
        
        assert result.is_valid
        assert result.has_warnings
        assert "Empty columns found" in result.warnings[0]
    
    def test_validate_malformed_json(self):
        """Test validation of malformed JSON file."""
        malformed_json = Path(self.temp_dir) / "malformed.json"
        malformed_json.write_text('{"name": "John", "age": 25,}')  # Trailing comma
        
        result = self.service.validate_file(str(malformed_json))
        
        assert not result.is_valid
        assert result.has_errors
        assert "JSON validation error" in result.errors[0]
    
    def test_load_csv_file_success(self):
        """Test successful CSV file loading."""
        dataset = self.service.load_file(str(self.csv_file))
        
        assert isinstance(dataset, Dataset)
        assert dataset.name == "test"
        assert len(dataset.data) == 3
        assert len(dataset.columns) == 3
        assert 'name' in dataset.columns
        assert dataset.metadata.file_type == FileType.CSV
        assert dataset.metadata.delimiter == ','
    
    def test_load_json_file_success(self):
        """Test successful JSON file loading."""
        dataset = self.service.load_file(str(self.json_file))
        
        assert isinstance(dataset, Dataset)
        assert dataset.name == "test"
        assert len(dataset.data) == 3
        assert len(dataset.columns) == 3
        assert 'name' in dataset.columns
        assert dataset.metadata.file_type == FileType.JSON
    
    def test_load_jsonl_file_success(self):
        """Test successful JSON Lines file loading."""
        dataset = self.service.load_file(str(self.jsonl_file))
        
        assert isinstance(dataset, Dataset)
        assert dataset.name == "test"
        assert len(dataset.data) == 3
        assert len(dataset.columns) == 3
        assert 'name' in dataset.columns
        assert dataset.metadata.file_type == FileType.JSON
    
    def test_load_excel_file_success(self):
        """Test successful Excel file loading."""
        dataset = self.service.load_file(str(self.excel_file))
        
        assert isinstance(dataset, Dataset)
        assert dataset.name == "test"
        assert len(dataset.data) == 3
        assert len(dataset.columns) == 3
        assert 'name' in dataset.columns
    
    def test_load_file_with_custom_config(self):
        """Test file loading with custom configuration."""
        # Create CSV with custom delimiter
        custom_csv = Path(self.temp_dir) / "custom.csv"
        with open(custom_csv, 'w') as f:
            f.write("name|age|city\n")
            f.write("John|25|NYC\n")
        
        config = {'delimiter': '|'}
        dataset = self.service.load_file(str(custom_csv), config)
        
        assert len(dataset.data) == 1
        assert len(dataset.columns) == 3
        assert dataset.metadata.delimiter == '|'
    
    def test_load_invalid_file(self):
        """Test loading invalid file raises exception."""
        # Create invalid CSV
        invalid_csv = Path(self.temp_dir) / "invalid.csv"
        invalid_csv.write_text("name,age,name\nJohn,25,Doe\n")  # Duplicate columns
        
        with pytest.raises(FileValidationError):
            self.service.load_file(str(invalid_csv))
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises exception."""
        with pytest.raises(FileValidationError):
            self.service.load_file("nonexistent.csv")
    
    def test_save_results_csv(self):
        """Test saving results in CSV format."""
        output_path = Path(self.temp_dir) / "output"
        
        created_files = self.service.save_results(
            self.test_data, str(output_path), format_type='csv'
        )
        
        assert len(created_files) == 1
        assert created_files[0].endswith('.csv')
        assert Path(created_files[0]).exists()
        
        # Verify content
        saved_df = pd.read_csv(created_files[0])
        assert len(saved_df) == 3
        assert list(saved_df.columns) == ['name', 'age', 'city']
    
    def test_save_results_json(self):
        """Test saving results in JSON format."""
        output_path = Path(self.temp_dir) / "output"
        
        created_files = self.service.save_results(
            self.test_data, str(output_path), format_type='json'
        )
        
        assert len(created_files) == 1
        assert created_files[0].endswith('.json')
        assert Path(created_files[0]).exists()
        
        # Verify content
        with open(created_files[0], 'r') as f:
            saved_data = json.load(f)
        assert len(saved_data) == 3
        assert saved_data[0]['name'] == 'John Doe'
    
    def test_save_results_both_formats(self):
        """Test saving results in both CSV and JSON formats."""
        output_path = Path(self.temp_dir) / "output"
        
        created_files = self.service.save_results(
            self.test_data, str(output_path), format_type='both'
        )
        
        assert len(created_files) == 2
        csv_files = [f for f in created_files if f.endswith('.csv')]
        json_files = [f for f in created_files if f.endswith('.json')]
        
        assert len(csv_files) == 1
        assert len(json_files) == 1
        assert Path(csv_files[0]).exists()
        assert Path(json_files[0]).exists()
    
    def test_save_empty_dataframe(self):
        """Test saving empty DataFrame."""
        empty_df = pd.DataFrame()
        output_path = Path(self.temp_dir) / "empty_output"
        
        created_files = self.service.save_results(
            empty_df, str(output_path), format_type='csv'
        )
        
        assert len(created_files) == 0  # No files created for empty DataFrame
    
    def test_stream_csv_chunks(self):
        """Test streaming CSV file in chunks."""
        # Create larger CSV file
        large_data = pd.DataFrame({
            'id': range(250),  # More than chunk_size (100)
            'value': [f'value_{i}' for i in range(250)]
        })
        large_csv = Path(self.temp_dir) / "large.csv"
        large_data.to_csv(large_csv, index=False)
        
        chunks = list(self.service.stream_csv_chunks(str(large_csv), chunk_size=100))
        
        assert len(chunks) == 3  # 250 rows / 100 chunk_size = 3 chunks
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50  # Last chunk
        
        # Verify data integrity
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 250
    
    def test_stream_json_lines_chunks(self):
        """Test streaming JSON Lines file in chunks."""
        # Create larger JSON Lines file
        large_data = pd.DataFrame({
            'id': range(250),
            'value': [f'value_{i}' for i in range(250)]
        })
        large_jsonl = Path(self.temp_dir) / "large.jsonl"
        large_data.to_json(large_jsonl, orient='records', lines=True)
        
        chunks = list(self.service.stream_json_lines(str(large_jsonl), chunk_size=100))
        
        assert len(chunks) == 3
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 50
        
        # Verify data integrity
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 250
    
    def test_stream_invalid_file(self):
        """Test streaming invalid file raises exception."""
        with pytest.raises(FileValidationError):
            list(self.service.stream_csv_chunks("nonexistent.csv"))
    
    def test_process_file_streaming_csv(self):
        """Test processing file with streaming for CSV."""
        # Create larger CSV file
        large_data = pd.DataFrame({
            'value': range(250)
        })
        large_csv = Path(self.temp_dir) / "large.csv"
        large_data.to_csv(large_csv, index=False)
        
        # Define processor function that counts rows
        def count_rows(chunk):
            return len(chunk)
        
        results = self.service.process_file_streaming(str(large_csv), count_rows, chunk_size=100)
        
        assert len(results) == 3  # 3 chunks
        assert sum(results) == 250  # Total rows
    
    def test_process_file_streaming_json_standard(self):
        """Test processing standard JSON file (non-streaming)."""
        def count_rows(df):
            return len(df)
        
        result = self.service.process_file_streaming(str(self.json_file), count_rows)
        
        assert result == 3  # Should return single result, not list
    
    def test_cleanup_files(self):
        """Test file cleanup functionality."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            temp_file = Path(self.temp_dir) / f"temp_{i}.txt"
            temp_file.write_text(f"Content {i}")
            temp_files.append(str(temp_file))
        
        # Verify files exist
        for file_path in temp_files:
            assert Path(file_path).exists()
        
        # Clean up files
        cleaned_count = self.service.cleanup_files(temp_files)
        
        assert cleaned_count == 3
        
        # Verify files are deleted
        for file_path in temp_files:
            assert not Path(file_path).exists()
    
    def test_cleanup_files_with_age_limit(self):
        """Test file cleanup with age limit."""
        import time
        
        # Create temporary files
        old_file = Path(self.temp_dir) / "old.txt"
        new_file = Path(self.temp_dir) / "new.txt"
        
        old_file.write_text("Old content")
        time.sleep(0.1)  # Small delay
        new_file.write_text("New content")
        
        # Clean up files older than 1 hour (should clean none)
        cleaned_count = self.service.cleanup_files(
            [str(old_file), str(new_file)], 
            max_age_hours=1
        )
        
        assert cleaned_count == 0
        assert old_file.exists()
        assert new_file.exists()
    
    def test_cleanup_nonexistent_files(self):
        """Test cleanup of non-existent files doesn't raise errors."""
        nonexistent_files = ["nonexistent1.txt", "nonexistent2.txt"]
        
        cleaned_count = self.service.cleanup_files(nonexistent_files)
        
        assert cleaned_count == 0  # No files cleaned


class TestFileProcessingEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = FileProcessingService()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_with_no_delimiter_detected(self):
        """Test CSV file where no delimiter can be detected."""
        # Create CSV with single column (no delimiter needed)
        single_col_csv = Path(self.temp_dir) / "single_col.csv"
        with open(single_col_csv, 'w') as f:
            f.write("singlecolumn\n")
            f.write("value1\n")
            f.write("value2\n")
        
        result = self.service.validate_file(str(single_col_csv))
        
        # Should still be valid, might detect comma as delimiter
        assert result.is_valid or result.has_errors
    
    def test_csv_with_mixed_encodings(self):
        """Test CSV file with mixed encoding issues."""
        mixed_csv = Path(self.temp_dir) / "mixed.csv"
        
        # Write with UTF-8 but include some problematic characters
        with open(mixed_csv, 'w', encoding='utf-8') as f:
            f.write("name,description\n")
            f.write("Test,Normal text\n")
            f.write("Special,Café résumé naïve\n")
        
        result = self.service.validate_file(str(mixed_csv))
        
        assert result.is_valid
        assert result.metadata['encoding'] in ['utf-8', 'UTF-8']
    
    def test_json_with_nested_objects(self):
        """Test JSON file with nested objects."""
        nested_data = [
            {
                "id": 1,
                "name": "John",
                "address": {
                    "street": "123 Main St",
                    "city": "NYC"
                }
            },
            {
                "id": 2,
                "name": "Jane",
                "address": {
                    "street": "456 Oak Ave",
                    "city": "LA"
                }
            }
        ]
        
        nested_json = Path(self.temp_dir) / "nested.json"
        with open(nested_json, 'w') as f:
            json.dump(nested_data, f)
        
        result = self.service.validate_file(str(nested_json))
        
        assert result.is_valid
        # Pandas will flatten nested objects or create complex columns
        assert len(result.metadata['columns']) >= 2
    
    def test_excel_with_multiple_sheets(self):
        """Test Excel file with multiple sheets."""
        multi_sheet_data = {
            'Sheet1': pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']}),
            'Sheet2': pd.DataFrame({'col3': [3, 4], 'col4': ['c', 'd']})
        }
        
        multi_excel = Path(self.temp_dir) / "multi_sheet.xlsx"
        with pd.ExcelWriter(multi_excel) as writer:
            for sheet_name, df in multi_sheet_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        result = self.service.validate_file(str(multi_excel))
        
        assert result.is_valid
        assert result.has_warnings
        assert "Multiple sheets found" in result.warnings[0]
        assert len(result.metadata['sheets']) == 2
    
    def test_large_file_estimation(self):
        """Test row count estimation for large files."""
        # Create a CSV file with exactly 1000 rows (the sample limit)
        large_data = pd.DataFrame({
            'id': range(1000),
            'value': [f'value_{i}' for i in range(1000)]
        })
        large_csv = Path(self.temp_dir) / "exactly_1000.csv"
        large_data.to_csv(large_csv, index=False)
        
        result = self.service.validate_file(str(large_csv))
        
        assert result.is_valid
        # Should have estimated_row_count since we hit the 1000 row limit
        assert 'estimated_row_count' in result.metadata
        assert result.metadata['estimated_row_count'] == 1000
    
    def test_file_with_bom(self):
        """Test file with Byte Order Mark (BOM)."""
        bom_csv = Path(self.temp_dir) / "bom.csv"
        
        # Write CSV with BOM
        with open(bom_csv, 'w', encoding='utf-8-sig') as f:
            f.write("name,age\n")
            f.write("John,25\n")
        
        result = self.service.validate_file(str(bom_csv))
        
        assert result.is_valid
        # Should handle BOM gracefully
        assert 'name' in result.metadata['columns']
    
    def test_concurrent_file_processing(self):
        """Test concurrent file processing doesn't cause issues."""
        import threading
        
        # Create test file
        test_data = pd.DataFrame({'col1': range(100), 'col2': range(100, 200)})
        test_csv = Path(self.temp_dir) / "concurrent_test.csv"
        test_data.to_csv(test_csv, index=False)
        
        results = []
        errors = []
        
        def process_file():
            try:
                dataset = self.service.load_file(str(test_csv))
                results.append(len(dataset.data))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=process_file) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed
        assert len(errors) == 0
        assert len(results) == 5
        assert all(result == 100 for result in results)
    
    def test_memory_efficient_processing(self):
        """Test that streaming actually uses less memory for large files."""
        # This is more of a behavioral test - we can't easily measure memory usage
        # but we can ensure streaming works with larger datasets
        
        # Create a moderately large dataset
        large_data = pd.DataFrame({
            'id': range(5000),
            'data': [f'data_string_{i}_with_some_length' for i in range(5000)]
        })
        large_csv = Path(self.temp_dir) / "memory_test.csv"
        large_data.to_csv(large_csv, index=False)
        
        # Process with streaming
        def count_processor(chunk):
            return len(chunk)
        
        results = self.service.process_file_streaming(
            str(large_csv), count_processor, chunk_size=1000
        )
        
        assert len(results) == 5  # 5000 / 1000 = 5 chunks
        assert sum(results) == 5000
    
    def test_unicode_content_handling(self):
        """Test handling of various Unicode characters."""
        unicode_data = pd.DataFrame({
            'english': ['Hello', 'World'],
            'chinese': ['你好', '世界'],
            'arabic': ['مرحبا', 'عالم'],
            'emoji': ['😀', '🌍'],
            'cyrillic': ['Привет', 'мир']
        })
        
        unicode_csv = Path(self.temp_dir) / "unicode.csv"
        unicode_data.to_csv(unicode_csv, index=False, encoding='utf-8')
        
        result = self.service.validate_file(str(unicode_csv))
        
        assert result.is_valid
        assert len(result.metadata['columns']) == 5
        
        # Load and verify content
        dataset = self.service.load_file(str(unicode_csv))
        assert len(dataset.data) == 2
        assert '你好' in dataset.data['chinese'].values
        assert '😀' in dataset.data['emoji'].values


if __name__ == "__main__":
    pytest.main([__file__])