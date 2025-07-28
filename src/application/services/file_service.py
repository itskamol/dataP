"""
File processing service for handling file operations with validation and optimization.
Enhanced with streaming capabilities for memory-efficient processing of large files.
Implements requirements 3.2, 5.1, 7.1, 4.1: File processing with validation and error handling.
"""

from typing import Dict, List, Optional, Union, Any, Iterator, Generator
import os
import json
import csv
import pandas as pd
from pathlib import Path
from io import StringIO
from contextlib import contextmanager

# Optional import for character encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

from ...domain.models import Dataset, DatasetMetadata, ValidationResult, FileType
from ...domain.exceptions import (
    FileProcessingError, FileValidationError, FileFormatError, 
    FileAccessError, ValidationError
)
from ...infrastructure.logging import get_logger
from ...infrastructure.memory_management import get_memory_manager
from ...infrastructure.memory_mapped_files import get_mapped_file_manager
from ...infrastructure.compressed_storage import get_compressed_dataframe_store


class FileProcessingService:
    """Handles all file operations with validation and optimization."""
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024, chunk_size: int = 10000,
                 enable_memory_optimization: bool = True, enable_compression: bool = True):
        self.logger = get_logger('file_processing')
        self.supported_formats = {'.csv', '.json', '.jsonl', '.xlsx', '.xls'}
        self.max_file_size = max_file_size  # 100MB default
        self.chunk_size = chunk_size  # Default chunk size for streaming
        self.supported_encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        # Optimization features
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_compression = enable_compression
        
        # Initialize optimization components
        if self.enable_memory_optimization:
            self.memory_manager = get_memory_manager()
            self.mapped_file_manager = get_mapped_file_manager()
        
        if self.enable_compression:
            self.compressed_store = get_compressed_dataframe_store()
    
    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding using chardet if available, otherwise use utf-8.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding
        """
        if not CHARDET_AVAILABLE:
            self.logger.debug("chardet not available, using utf-8 encoding")
            return 'utf-8'
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                # Fallback to utf-8 if confidence is too low
                if confidence < 0.7:
                    encoding = 'utf-8'
                
                self.logger.debug(f"Detected encoding: {encoding} (confidence: {confidence})")
                return encoding
                
        except Exception as e:
            self.logger.warning(f"Failed to detect encoding, using utf-8: {str(e)}")
            return 'utf-8'
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate file format, structure, and content.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            ValidationResult with detailed validation information
            
        Raises:
            FileValidationError: If validation fails
            FileAccessError: If file cannot be accessed
        """
        validation_result = ValidationResult(is_valid=True)
        
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                validation_result.add_error(f"File not found: {file_path}")
                return validation_result
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                validation_result.add_error(
                    f"File size {file_size} exceeds maximum allowed size {self.max_file_size}"
                )
                return validation_result
            
            # Check file extension
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_formats:
                validation_result.add_error(
                    f"Unsupported file format: {file_extension}. Supported formats: {self.supported_formats}"
                )
                return validation_result
            
            # Detect encoding
            detected_encoding = self.detect_encoding(file_path)
            
            # Store metadata
            validation_result.metadata.update({
                'file_path': str(file_path),
                'file_size': file_size,
                'format': file_extension,
                'encoding': detected_encoding
            })
            
            # Perform format-specific validation
            if file_extension == '.csv':
                self._validate_csv(file_path, validation_result)
            elif file_extension in {'.json', '.jsonl'}:
                self._validate_json(file_path, validation_result)
            elif file_extension in {'.xlsx', '.xls'}:
                self._validate_excel(file_path, validation_result)
            
            self.logger.info(f"File validation completed", extra={
                'file_path': str(file_path),
                'is_valid': validation_result.is_valid,
                'errors_count': len(validation_result.errors),
                'warnings_count': len(validation_result.warnings)
            })
            
            return validation_result
            
        except Exception as e:
            validation_result.add_error(f"Unexpected error during file validation: {str(e)}")
            self.logger.error(f"File validation error: {str(e)}", extra={'file_path': str(file_path)})
            return validation_result
    
    def _validate_csv(self, file_path: Path, validation_result: ValidationResult) -> None:
        """Validate CSV file structure and content."""
        try:
            # Detect encoding
            encoding = validation_result.metadata.get('encoding', 'utf-8')
            
            # Try different delimiters
            delimiters = [',', ';', '\t', '|']
            successful_delimiter = None
            
            for delimiter in delimiters:
                try:
                    df_sample = pd.read_csv(file_path, delimiter=delimiter, nrows=5, encoding=encoding)
                    if len(df_sample.columns) > 1:  # Multiple columns indicate correct delimiter
                        successful_delimiter = delimiter
                        break
                    elif delimiter == ',' and len(df_sample.columns) == 1:
                        # Single column with comma delimiter is acceptable
                        successful_delimiter = delimiter
                        break
                except:
                    continue
            
            if successful_delimiter is None:
                # Default to comma if no delimiter works
                successful_delimiter = ','
            
            # Store delimiter in metadata
            validation_result.metadata['delimiter'] = successful_delimiter
            
            # Read sample for validation (avoid loading entire large files)
            try:
                df_sample = pd.read_csv(file_path, delimiter=successful_delimiter, nrows=1000, encoding=encoding)
                validation_result.metadata['columns'] = df_sample.columns.tolist()
                validation_result.metadata['sample_row_count'] = len(df_sample)
                
                # Estimate total rows for large files
                if len(df_sample) == 1000:  # Hit the limit, estimate total
                    with open(file_path, 'r', encoding=encoding) as f:
                        total_lines = sum(1 for _ in f) - 1  # Subtract header
                    validation_result.metadata['estimated_row_count'] = total_lines
                else:
                    validation_result.metadata['row_count'] = len(df_sample)
                
                # Check for empty file
                if len(df_sample) == 0:
                    validation_result.add_warning('File is empty')
                
                # Check for duplicate columns
                if len(df_sample.columns) != len(set(df_sample.columns)):
                    validation_result.add_error('Duplicate column names found')
                
                # Check for completely empty columns
                empty_columns = df_sample.columns[df_sample.isnull().all()].tolist()
                if empty_columns:
                    validation_result.add_warning(f'Empty columns found: {empty_columns}')
                
            except Exception as e:
                validation_result.add_error(f'Error reading CSV content: {str(e)}')
            
        except Exception as e:
            validation_result.add_error(f'CSV validation error: {str(e)}')
    
    def _validate_json(self, file_path: Path, validation_result: ValidationResult) -> None:
        """Validate JSON file structure and content."""
        try:
            encoding = validation_result.metadata.get('encoding', 'utf-8')
            
            # Try standard JSON first
            json_type = None
            df_sample = None
            
            try:
                # Try standard JSON first (nrows not supported for JSON)
                df_sample = pd.read_json(file_path, encoding=encoding)
                # Limit to first 1000 rows if larger
                if len(df_sample) > 1000:
                    df_sample = df_sample.head(1000)
                json_type = 'standard'
            except ValueError:
                try:
                    # Try JSON Lines format
                    df_sample = pd.read_json(file_path, lines=True, encoding=encoding, nrows=1000)
                    json_type = 'lines'
                except Exception:
                    validation_result.add_error('Could not parse JSON file in standard or lines format')
                    return
            except Exception as e:
                validation_result.add_error(f'Error reading JSON file: {str(e)}')
                return
            
            if df_sample is not None:
                validation_result.metadata['json_type'] = json_type
                validation_result.metadata['columns'] = df_sample.columns.tolist()
                validation_result.metadata['sample_row_count'] = len(df_sample)
                
                # Estimate total rows for large files
                if len(df_sample) == 1000:  # Hit the limit, estimate total
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            if json_type == 'lines':
                                total_lines = sum(1 for _ in f)
                            else:
                                # For standard JSON, we can't easily count records without parsing
                                total_lines = len(df_sample)  # Use sample size as estimate
                        validation_result.metadata['estimated_row_count'] = total_lines
                    except Exception:
                        validation_result.metadata['row_count'] = len(df_sample)
                else:
                    validation_result.metadata['row_count'] = len(df_sample)
                
                # Check for empty file
                if len(df_sample) == 0:
                    validation_result.add_warning('File is empty')
                
                # Check for completely empty columns
                empty_columns = df_sample.columns[df_sample.isnull().all()].tolist()
                if empty_columns:
                    validation_result.add_warning(f'Empty columns found: {empty_columns}')
            
        except Exception as e:
            validation_result.add_error(f'JSON validation error: {str(e)}')
    
    def _validate_excel(self, file_path: Path, validation_result: ValidationResult) -> None:
        """Validate Excel file structure and content."""
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            validation_result.metadata['sheets'] = sheets
            
            # Read first sheet by default (sample only)
            df_sample = pd.read_excel(file_path, sheet_name=0, nrows=1000)
            validation_result.metadata['columns'] = df_sample.columns.tolist()
            validation_result.metadata['sample_row_count'] = len(df_sample)
            
            # Estimate total rows for large files
            if len(df_sample) == 1000:  # Hit the limit, estimate total
                # For Excel, we can't easily count rows without loading, so use sample
                validation_result.metadata['estimated_row_count'] = len(df_sample)
            else:
                validation_result.metadata['row_count'] = len(df_sample)
            
            # Check for empty file
            if len(df_sample) == 0:
                validation_result.add_warning('File is empty')
            
            # Check for multiple sheets
            if len(sheets) > 1:
                validation_result.add_warning(f'Multiple sheets found: {sheets}. Only first sheet will be processed.')
            
            # Check for completely empty columns
            empty_columns = df_sample.columns[df_sample.isnull().all()].tolist()
            if empty_columns:
                validation_result.add_warning(f'Empty columns found: {empty_columns}')
            
        except Exception as e:
            validation_result.add_error(f'Excel validation error: {str(e)}')
    
    def load_file(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Load and validate file with proper error handling.
        
        Args:
            file_path: Path to the file to load
            config: Optional configuration for file loading
            
        Returns:
            Dataset object with loaded data and metadata
            
        Raises:
            FileProcessingError: If loading fails
        """
        try:
            # Validate file first
            validation_result = self.validate_file(file_path)
            if not validation_result.is_valid:
                raise FileValidationError(
                    f"File validation failed: {validation_result.errors}",
                    file_path=file_path,
                    context={'validation_errors': validation_result.errors}
                )
            
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix.lower()
            config = config or {}
            
            # Get encoding from validation result
            encoding = validation_result.metadata.get('encoding', 'utf-8')
            
            # Load based on file type
            if file_extension == '.csv':
                delimiter = config.get('delimiter') or validation_result.metadata.get('delimiter', ',')
                df = pd.read_csv(file_path_obj, delimiter=delimiter, encoding=encoding, dtype=str, low_memory=False)
            
            elif file_extension in {'.json', '.jsonl'}:
                if validation_result.metadata.get('json_type') == 'lines' or file_extension == '.jsonl':
                    df = pd.read_json(file_path_obj, lines=True, encoding=encoding)
                else:
                    df = pd.read_json(file_path_obj, encoding=encoding)
            
            elif file_extension in {'.xlsx', '.xls'}:
                sheet_name = config.get('sheet_name', 0)
                df = pd.read_excel(file_path_obj, sheet_name=sheet_name)
            
            else:
                raise FileFormatError(
                    f"Unsupported file format: {file_extension}",
                    file_path=str(file_path_obj),
                    expected_format=str(self.supported_formats),
                    actual_format=file_extension
                )
            
            # Create dataset metadata
            file_type_map = {'.csv': FileType.CSV, '.json': FileType.JSON, '.jsonl': FileType.JSON, '.xlsx': FileType.CSV, '.xls': FileType.CSV}
            metadata = DatasetMetadata(
                name=file_path_obj.stem,
                file_path=str(file_path_obj),
                file_type=file_type_map.get(file_extension, FileType.CSV),
                delimiter=validation_result.metadata.get('delimiter'),
                encoding=encoding,
                row_count=len(df),
                column_count=len(df.columns)
            )
            
            # Create dataset
            dataset = Dataset(
                name=file_path_obj.stem,
                data=df,
                metadata=metadata
            )
            
            self.logger.info(f"File loaded successfully", extra={
                'file_path': str(file_path_obj),
                'rows': len(df),
                'columns': len(df.columns)
            })
            
            return dataset
            
        except (FileValidationError, FileFormatError, FileAccessError):
            raise
        except Exception as e:
            raise FileProcessingError(
                f"Failed to load file: {str(e)}",
                context={'file_path': file_path}
            )
    
    def save_results(self, df: pd.DataFrame, output_path: str, 
                    format_type: str = 'csv', **kwargs) -> List[str]:
        """
        Save results in specified formats with proper file naming.
        
        Args:
            df: DataFrame to save
            output_path: Base path for output files
            format_type: Output format ('csv', 'json', 'excel', 'both')
            **kwargs: Additional format-specific options
            
        Returns:
            List of created file paths
            
        Raises:
            FileProcessingError: If saving fails
        """
        try:
            if df.empty:
                self.logger.warning(f"Empty DataFrame not saved: {output_path}")
                return []
            
            created_files = []
            output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save in requested format(s)
            if format_type in ['csv', 'both']:
                csv_path = output_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False, **kwargs.get('csv_options', {}))
                created_files.append(str(csv_path))
            
            if format_type in ['json', 'both']:
                json_path = output_path.with_suffix('.json')
                df.to_json(json_path, orient='records', indent=2, 
                          force_ascii=False, **kwargs.get('json_options', {}))
                created_files.append(str(json_path))
            
            if format_type == 'excel':
                excel_path = output_path.with_suffix('.xlsx')
                df.to_excel(excel_path, index=False, **kwargs.get('excel_options', {}))
                created_files.append(str(excel_path))
            
            self.logger.info(f"Results saved successfully", extra={
                'output_files': created_files,
                'rows': len(df),
                'format': format_type
            })
            
            return created_files
            
        except Exception as e:
            raise FileProcessingError(
                f"Failed to save results: {str(e)}",
                context={'output_path': str(output_path), 'format': format_type},
                cause=e
            )
    
    def stream_csv_chunks(self, file_path: str, chunk_size: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream CSV file in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to the CSV file
            chunk_size: Size of each chunk (default: self.chunk_size)
            
        Yields:
            DataFrame chunks
            
        Raises:
            FileProcessingError: If streaming fails
        """
        try:
            # Validate file first
            validation_result = self.validate_file(file_path)
            if not validation_result.is_valid:
                raise FileValidationError(
                    f"File validation failed: {validation_result.errors}",
                    file_path=file_path,
                    context={'validation_errors': validation_result.errors}
                )
            
            chunk_size = chunk_size or self.chunk_size
            file_path_obj = Path(file_path)
            
            # Get parameters from validation
            delimiter = validation_result.metadata.get('delimiter', ',')
            encoding = validation_result.metadata.get('encoding', 'utf-8')
            
            # Stream file in chunks
            chunk_reader = pd.read_csv(
                file_path_obj,
                delimiter=delimiter,
                encoding=encoding,
                chunksize=chunk_size,
                dtype=str,
                low_memory=False
            )
            
            chunk_count = 0
            for chunk in chunk_reader:
                chunk_count += 1
                self.logger.debug(f"Processing chunk {chunk_count} with {len(chunk)} rows")
                yield chunk
            
            self.logger.info(f"Streamed {chunk_count} chunks from {file_path}")
            
        except (FileValidationError, FileFormatError, FileAccessError):
            raise
        except Exception as e:
            raise FileProcessingError(
                f"Failed to stream CSV file: {str(e)}",
                context={'file_path': file_path}
            )
    
    def stream_json_lines(self, file_path: str, chunk_size: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream JSON Lines file in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to the JSON Lines file
            chunk_size: Size of each chunk (default: self.chunk_size)
            
        Yields:
            DataFrame chunks
            
        Raises:
            FileProcessingError: If streaming fails
        """
        try:
            # Validate file first
            validation_result = self.validate_file(file_path)
            if not validation_result.is_valid:
                raise FileValidationError(
                    f"File validation failed: {validation_result.errors}",
                    file_path=file_path,
                    context={'validation_errors': validation_result.errors}
                )
            
            chunk_size = chunk_size or self.chunk_size
            file_path_obj = Path(file_path)
            encoding = validation_result.metadata.get('encoding', 'utf-8')
            
            # Stream JSON Lines file
            chunk_reader = pd.read_json(
                file_path_obj,
                lines=True,
                encoding=encoding,
                chunksize=chunk_size
            )
            
            chunk_count = 0
            for chunk in chunk_reader:
                chunk_count += 1
                self.logger.debug(f"Processing JSON chunk {chunk_count} with {len(chunk)} rows")
                yield chunk
            
            self.logger.info(f"Streamed {chunk_count} chunks from {file_path}")
            
        except (FileValidationError, FileFormatError, FileAccessError):
            raise
        except Exception as e:
            raise FileProcessingError(
                f"Failed to stream JSON Lines file: {str(e)}",
                context={'file_path': file_path}
            )
    
    def process_file_streaming(self, file_path: str, processor_func, chunk_size: Optional[int] = None) -> Any:
        """
        Process file using streaming with a custom processor function.
        
        Args:
            file_path: Path to the file to process
            processor_func: Function to process each chunk (takes DataFrame, returns Any)
            chunk_size: Size of each chunk (default: self.chunk_size)
            
        Returns:
            Result from processor function
            
        Raises:
            FileProcessingError: If processing fails
        """
        try:
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix.lower()
            
            if file_extension == '.csv':
                chunks = self.stream_csv_chunks(file_path, chunk_size)
            elif file_extension == '.json':
                # Check if it's JSON Lines format
                validation_result = self.validate_file(file_path)
                if validation_result.metadata.get('json_type') == 'lines':
                    chunks = self.stream_json_lines(file_path, chunk_size)
                else:
                    # For standard JSON, we can't stream, so load normally
                    dataset = self.load_file(file_path)
                    return processor_func(dataset.data)
            else:
                # For Excel and other formats, load normally
                dataset = self.load_file(file_path)
                return processor_func(dataset.data)
            
            # Process chunks
            results = []
            for chunk in chunks:
                result = processor_func(chunk)
                if result is not None:
                    results.append(result)
            
            return results
            
        except Exception as e:
            raise FileProcessingError(
                f"Failed to process file with streaming: {str(e)}",
                context={'file_path': file_path}
            )
    
    def cleanup_files(self, file_paths: List[str], max_age_hours: Optional[int] = None) -> int:
        """
        Clean up temporary files and old results.
        
        Args:
            file_paths: List of file paths to clean up
            max_age_hours: Only delete files older than this many hours
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    # Check age if specified
                    if max_age_hours is not None:
                        import time
                        file_age_hours = (time.time() - path.stat().st_mtime) / 3600
                        if file_age_hours < max_age_hours:
                            continue
                    
                    path.unlink()
                    cleaned_count += 1
                    self.logger.debug(f"Cleaned up file: {file_path}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to clean up file {file_path}: {str(e)}")
        
        self.logger.info(f"Cleanup completed", extra={'files_cleaned': cleaned_count})
        return cleaned_count  
  
    @contextmanager
    def memory_optimized_processing(self):
        """Context manager for memory-optimized file processing."""
        if self.enable_memory_optimization:
            with self.memory_manager.memory_optimized_context('performance'):
                yield
        else:
            yield
    
    def load_file_optimized(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Load file with memory optimization and compression support.
        
        Args:
            file_path: Path to the file to load
            config: Optional configuration for file loading
            
        Returns:
            Dataset object with loaded data and metadata
        """
        config = config or {}
        use_memory_mapping = config.get('use_memory_mapping', False)
        use_compression = config.get('use_compression', self.enable_compression)
        
        # Check if compressed version exists
        if use_compression and self.enable_compression:
            cache_key = f"file_{Path(file_path).stem}_{hash(file_path)}"
            if self.compressed_store.store.exists(cache_key):
                try:
                    df = self.compressed_store.retrieve_dataframe(cache_key)
                    
                    # Create dataset metadata
                    file_path_obj = Path(file_path)
                    metadata = DatasetMetadata(
                        name=file_path_obj.stem,
                        file_path=str(file_path_obj),
                        file_type=FileType.CSV,  # Default
                        row_count=len(df),
                        column_count=len(df.columns)
                    )
                    
                    dataset = Dataset(
                        name=file_path_obj.stem,
                        data=df,
                        metadata=metadata
                    )
                    
                    self.logger.info(f"Loaded file from compressed cache: {file_path}")
                    return dataset
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load from compressed cache: {str(e)}")
        
        # Use memory mapping for large CSV files
        if use_memory_mapping and self.enable_memory_optimization:
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.csv':
                return self._load_csv_memory_mapped(file_path, config)
        
        # Use memory-optimized context for regular loading
        with self.memory_optimized_processing():
            dataset = self.load_file(file_path, config)
            
            # Store in compressed cache if enabled
            if use_compression and self.enable_compression:
                try:
                    cache_key = f"file_{Path(file_path).stem}_{hash(file_path)}"
                    self.compressed_store.store_dataframe(cache_key, dataset.data)
                    self.logger.info(f"Stored file in compressed cache: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to store in compressed cache: {str(e)}")
            
            return dataset
    
    def _load_csv_memory_mapped(self, file_path: str, config: Dict[str, Any]) -> Dataset:
        """Load CSV file using memory mapping for large files."""
        try:
            with self.mapped_file_manager.get_csv_reader(file_path) as reader:
                # For very large files, we might want to process in chunks
                # For now, load all data but could be optimized further
                chunks = []
                for chunk in reader.iter_chunks(chunk_rows=self.chunk_size):
                    chunks.append(chunk)
                
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.DataFrame(columns=reader.headers)
                
                # Create dataset metadata
                file_path_obj = Path(file_path)
                metadata = DatasetMetadata(
                    name=file_path_obj.stem,
                    file_path=str(file_path_obj),
                    file_type=FileType.CSV,
                    row_count=len(df),
                    column_count=len(df.columns)
                )
                
                dataset = Dataset(
                    name=file_path_obj.stem,
                    data=df,
                    metadata=metadata
                )
                
                self.logger.info(f"Loaded CSV file using memory mapping: {file_path}")
                return dataset
                
        except Exception as e:
            self.logger.warning(f"Memory-mapped loading failed, falling back to regular loading: {str(e)}")
            return self.load_file(file_path, config)
    
    def process_large_file_streaming(self, file_path: str, processor_func, 
                                   chunk_size: Optional[int] = None,
                                   use_memory_optimization: bool = True) -> Any:
        """
        Process large files using streaming with memory optimization.
        
        Args:
            file_path: Path to the file to process
            processor_func: Function to process each chunk
            chunk_size: Size of each chunk
            use_memory_optimization: Whether to use memory optimization
            
        Returns:
            Result from processor function
        """
        if use_memory_optimization and self.enable_memory_optimization:
            with self.memory_optimized_processing():
                return self.process_file_streaming(file_path, processor_func, chunk_size)
        else:
            return self.process_file_streaming(file_path, processor_func, chunk_size)
    
    def save_results_optimized(self, df: pd.DataFrame, output_path: str,
                             format_type: str = 'csv', use_compression: bool = None,
                             **kwargs) -> List[str]:
        """
        Save results with compression and optimization.
        
        Args:
            df: DataFrame to save
            output_path: Base path for output files
            format_type: Output format
            use_compression: Whether to use compression (default: self.enable_compression)
            **kwargs: Additional options
            
        Returns:
            List of created file paths
        """
        use_compression = use_compression if use_compression is not None else self.enable_compression
        
        # Save normally first
        created_files = self.save_results(df, output_path, format_type, **kwargs)
        
        # Also save compressed version if enabled
        if use_compression and self.enable_compression:
            try:
                cache_key = f"result_{Path(output_path).stem}_{hash(output_path)}"
                
                # Store in compressed format
                if len(df) > 10000:  # Use chunked storage for large DataFrames
                    chunk_keys = self.compressed_store.store_chunked_dataframe(cache_key, df)
                    self.logger.info(f"Stored large result in {len(chunk_keys)} compressed chunks")
                else:
                    self.compressed_store.store_dataframe(cache_key, df)
                    self.logger.info(f"Stored result in compressed format")
                
            except Exception as e:
                self.logger.warning(f"Failed to store compressed result: {str(e)}")
        
        return created_files
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about optimization features."""
        stats = {}
        
        if self.enable_memory_optimization:
            stats['memory'] = self.memory_manager.get_comprehensive_stats()
            stats['mapped_files'] = self.mapped_file_manager.get_stats()
        
        if self.enable_compression:
            stats['compression'] = self.compressed_store.store.get_stats()
        
        return stats
    
    def cleanup_optimization_caches(self):
        """Clean up optimization caches and temporary data."""
        if self.enable_memory_optimization:
            self.memory_manager.emergency_cleanup()
            self.mapped_file_manager.close_all()
        
        if self.enable_compression:
            # Clean up old compressed data (older than 24 hours)
            cleaned = self.compressed_store.store.cleanup_old_data(max_age_hours=24)
            self.logger.info(f"Cleaned up {cleaned} old compressed data entries")


class SecureFileService(FileProcessingService):
    """Enhanced file service with security and data protection features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Import security components if available
        try:
            from ...infrastructure.data_protection import DataProtectionManager
            self.data_protection = DataProtectionManager()
            self.enable_data_protection = True
        except ImportError:
            self.data_protection = None
            self.enable_data_protection = False
            self.logger.warning("Data protection features not available")
    
    def load_file_secure(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Load file with security validation and data protection.
        
        Args:
            file_path: Path to the file to load
            config: Optional configuration
            
        Returns:
            Dataset with protected data
        """
        # Load file normally first
        dataset = self.load_file_optimized(file_path, config)
        
        # Apply data protection if enabled
        if self.enable_data_protection and self.data_protection:
            try:
                # Scan for PII and apply protection
                protected_df = self.data_protection.protect_dataframe(dataset.data)
                dataset.data = protected_df
                
                self.logger.info(f"Applied data protection to file: {file_path}")
                
            except Exception as e:
                self.logger.warning(f"Data protection failed: {str(e)}")
        
        return dataset
    
    def validate_file_security(self, file_path: str) -> Dict[str, Any]:
        """
        Perform security validation on file.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Security validation results
        """
        validation_result = {
            'is_secure': True,
            'security_issues': [],
            'recommendations': []
        }
        
        try:
            file_path_obj = Path(file_path)
            
            # Check file size (security consideration)
            file_size = file_path_obj.stat().st_size
            if file_size > 500 * 1024 * 1024:  # 500MB
                validation_result['security_issues'].append('File size exceeds security threshold')
                validation_result['recommendations'].append('Consider processing file in chunks')
            
            # Check file permissions
            if file_path_obj.stat().st_mode & 0o077:  # World or group writable
                validation_result['security_issues'].append('File has insecure permissions')
                validation_result['recommendations'].append('Restrict file permissions')
            
            # Additional security checks could be added here
            # - Virus scanning
            # - Content validation
            # - File type verification
            
            if validation_result['security_issues']:
                validation_result['is_secure'] = False
            
        except Exception as e:
            validation_result['is_secure'] = False
            validation_result['security_issues'].append(f'Security validation failed: {str(e)}')
        
        return validation_result