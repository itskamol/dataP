"""
Memory-mapped file access for very large datasets.
Implements requirement 3.2: Memory-efficient processing of large files.
"""

import mmap
import os
import threading
from typing import Dict, List, Optional, Any, Iterator, Union, BinaryIO
from pathlib import Path
import struct
from dataclasses import dataclass
from contextlib import contextmanager
import json
import csv
from io import StringIO

import pandas as pd
import numpy as np

from src.infrastructure.logging import get_logger
from src.domain.exceptions import FileProcessingError, FileAccessError


@dataclass
class MappedFileInfo:
    """Information about a memory-mapped file."""
    file_path: str
    file_size: int
    map_size: int
    access_mode: str
    is_compressed: bool
    encoding: str
    created_at: float


class MemoryMappedFile:
    """Memory-mapped file wrapper with safe access patterns."""
    
    def __init__(self, file_path: str, mode: str = 'r', encoding: str = 'utf-8'):
        """
        Initialize memory-mapped file.
        
        Args:
            file_path: Path to the file
            mode: File access mode ('r', 'r+', 'w+')
            encoding: Text encoding for text files
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self.encoding = encoding
        self.logger = get_logger('memory_mapped_file')
        
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        self._lock = threading.RLock()
        self._is_open = False
        
        # File information
        self.file_size = 0
        self.map_size = 0
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self):
        """Open file and create memory mapping."""
        with self._lock:
            if self._is_open:
                return
            
            try:
                # Determine file access mode
                if self.mode == 'r':
                    file_mode = 'rb'
                    map_access = mmap.ACCESS_READ
                elif self.mode == 'r+':
                    file_mode = 'r+b'
                    map_access = mmap.ACCESS_WRITE
                elif self.mode == 'w+':
                    file_mode = 'w+b'
                    map_access = mmap.ACCESS_WRITE
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
                
                # Open file
                self._file = open(self.file_path, file_mode)
                
                # Get file size
                self._file.seek(0, 2)  # Seek to end
                self.file_size = self._file.tell()
                self._file.seek(0)  # Seek back to beginning
                
                # Create memory mapping if file is not empty
                if self.file_size > 0:
                    self._mmap = mmap.mmap(self._file.fileno(), 0, access=map_access)
                    self.map_size = len(self._mmap)
                else:
                    self._mmap = None
                    self.map_size = 0
                
                self._is_open = True
                self.logger.debug(f"Memory-mapped file opened: {self.file_path} ({self.file_size} bytes)")
                
            except Exception as e:
                self._cleanup()
                raise FileAccessError(
                    f"Failed to open memory-mapped file: {str(e)}",
                    file_path=str(self.file_path)
                )
    
    def close(self):
        """Close memory mapping and file."""
        with self._lock:
            self._cleanup()
            self._is_open = False
            self.logger.debug(f"Memory-mapped file closed: {self.file_path}")
    
    def _cleanup(self):
        """Clean up resources."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        if self._file:
            self._file.close()
            self._file = None
    
    def read(self, size: int = -1, offset: int = 0) -> bytes:
        """Read bytes from memory-mapped file."""
        with self._lock:
            if not self._is_open or not self._mmap:
                raise FileAccessError("File is not open or empty")
            
            if offset < 0 or offset >= self.map_size:
                raise ValueError(f"Invalid offset: {offset}")
            
            if size == -1:
                size = self.map_size - offset
            
            end_pos = min(offset + size, self.map_size)
            return self._mmap[offset:end_pos]
    
    def read_text(self, size: int = -1, offset: int = 0) -> str:
        """Read text from memory-mapped file."""
        data = self.read(size, offset)
        return data.decode(self.encoding)
    
    def readline(self, offset: int = 0) -> tuple[bytes, int]:
        """
        Read a line from memory-mapped file.
        
        Returns:
            Tuple of (line_bytes, next_offset)
        """
        with self._lock:
            if not self._is_open or not self._mmap:
                raise FileAccessError("File is not open or empty")
            
            if offset >= self.map_size:
                return b'', offset
            
            # Find next newline
            newline_pos = self._mmap.find(b'\n', offset)
            if newline_pos == -1:
                # No newline found, read to end
                line = self._mmap[offset:]
                next_offset = self.map_size
            else:
                # Include newline in result
                line = self._mmap[offset:newline_pos + 1]
                next_offset = newline_pos + 1
            
            return line, next_offset
    
    def readline_text(self, offset: int = 0) -> tuple[str, int]:
        """Read a text line from memory-mapped file."""
        line_bytes, next_offset = self.readline(offset)
        line_text = line_bytes.decode(self.encoding).rstrip('\r\n')
        return line_text, next_offset
    
    def find_pattern(self, pattern: bytes, start: int = 0, end: Optional[int] = None) -> int:
        """Find pattern in memory-mapped file."""
        with self._lock:
            if not self._is_open or not self._mmap:
                raise FileAccessError("File is not open or empty")
            
            end = end or self.map_size
            return self._mmap.find(pattern, start, end)
    
    def get_info(self) -> MappedFileInfo:
        """Get information about the memory-mapped file."""
        return MappedFileInfo(
            file_path=str(self.file_path),
            file_size=self.file_size,
            map_size=self.map_size,
            access_mode=self.mode,
            is_compressed=False,  # TODO: Add compression detection
            encoding=self.encoding,
            created_at=os.path.getctime(self.file_path)
        )


class MappedCSVReader:
    """Memory-mapped CSV reader for large files."""
    
    def __init__(self, file_path: str, delimiter: str = ',', encoding: str = 'utf-8',
                 chunk_size: int = 1024 * 1024):  # 1MB chunks
        """
        Initialize mapped CSV reader.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            encoding: Text encoding
            chunk_size: Size of chunks to process at once
        """
        self.file_path = file_path
        self.delimiter = delimiter
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.logger = get_logger('mapped_csv_reader')
        
        self._mapped_file: Optional[MemoryMappedFile] = None
        self._headers: Optional[List[str]] = None
        self._header_offset = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self):
        """Open CSV file for memory-mapped reading."""
        self._mapped_file = MemoryMappedFile(self.file_path, 'r', self.encoding)
        self._mapped_file.open()
        
        # Read headers
        self._read_headers()
        
        self.logger.info(f"Opened mapped CSV: {self.file_path} with {len(self._headers)} columns")
    
    def close(self):
        """Close memory-mapped CSV file."""
        if self._mapped_file:
            self._mapped_file.close()
            self._mapped_file = None
    
    def _read_headers(self):
        """Read CSV headers from the file."""
        if not self._mapped_file:
            raise FileAccessError("File is not open")
        
        # Read first line for headers
        header_line, self._header_offset = self._mapped_file.readline_text(0)
        
        # Parse headers
        csv_reader = csv.reader([header_line], delimiter=self.delimiter)
        self._headers = next(csv_reader)
    
    @property
    def headers(self) -> List[str]:
        """Get CSV headers."""
        if self._headers is None:
            raise FileAccessError("Headers not available - file may not be open")
        return self._headers
    
    def read_chunk(self, offset: int, max_rows: Optional[int] = None) -> tuple[pd.DataFrame, int]:
        """
        Read a chunk of CSV data.
        
        Args:
            offset: Byte offset to start reading from
            max_rows: Maximum number of rows to read
            
        Returns:
            Tuple of (DataFrame, next_offset)
        """
        if not self._mapped_file:
            raise FileAccessError("File is not open")
        
        rows = []
        current_offset = offset
        rows_read = 0
        
        # Skip header if starting from beginning
        if offset == 0:
            current_offset = self._header_offset
        
        while current_offset < self._mapped_file.map_size:
            if max_rows and rows_read >= max_rows:
                break
            
            # Read line
            line, next_offset = self._mapped_file.readline_text(current_offset)
            
            if not line.strip():  # Skip empty lines
                current_offset = next_offset
                continue
            
            # Parse CSV line
            try:
                csv_reader = csv.reader([line], delimiter=self.delimiter)
                row = next(csv_reader)
                
                # Ensure row has same number of columns as headers
                while len(row) < len(self._headers):
                    row.append('')
                
                rows.append(row[:len(self._headers)])  # Truncate if too many columns
                rows_read += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to parse CSV line at offset {current_offset}: {str(e)}")
            
            current_offset = next_offset
        
        # Create DataFrame
        if rows:
            df = pd.DataFrame(rows, columns=self._headers)
        else:
            df = pd.DataFrame(columns=self._headers)
        
        return df, current_offset
    
    def iter_chunks(self, chunk_rows: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Iterate over CSV file in chunks.
        
        Args:
            chunk_rows: Number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        offset = 0
        chunk_count = 0
        
        while offset < self._mapped_file.map_size:
            df, next_offset = self.read_chunk(offset, chunk_rows)
            
            if df.empty:
                break
            
            chunk_count += 1
            self.logger.debug(f"Read chunk {chunk_count} with {len(df)} rows")
            yield df
            
            offset = next_offset
        
        self.logger.info(f"Processed {chunk_count} chunks from {self.file_path}")
    
    def count_rows(self) -> int:
        """Count total number of rows in CSV file."""
        if not self._mapped_file:
            raise FileAccessError("File is not open")
        
        row_count = 0
        offset = self._header_offset  # Skip header
        
        while offset < self._mapped_file.map_size:
            line, next_offset = self._mapped_file.readline_text(offset)
            if line.strip():  # Count non-empty lines
                row_count += 1
            offset = next_offset
        
        return row_count
    
    def search_pattern(self, pattern: str, column: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for pattern in CSV file.
        
        Args:
            pattern: Pattern to search for
            column: Specific column to search in (None for all columns)
            
        Returns:
            List of matching rows with metadata
        """
        matches = []
        
        for chunk_df in self.iter_chunks():
            if column and column in chunk_df.columns:
                # Search in specific column
                mask = chunk_df[column].str.contains(pattern, na=False, regex=False)
                matching_rows = chunk_df[mask]
            else:
                # Search in all columns
                mask = chunk_df.astype(str).apply(
                    lambda x: x.str.contains(pattern, na=False, regex=False)
                ).any(axis=1)
                matching_rows = chunk_df[mask]
            
            # Add matches with metadata
            for idx, row in matching_rows.iterrows():
                matches.append({
                    'row_data': row.to_dict(),
                    'row_index': idx,
                    'matched_column': column
                })
        
        return matches


class MappedFileManager:
    """Manager for memory-mapped files with caching and resource management."""
    
    def __init__(self, max_open_files: int = 10):
        """
        Initialize mapped file manager.
        
        Args:
            max_open_files: Maximum number of files to keep open simultaneously
        """
        self.max_open_files = max_open_files
        self.logger = get_logger('mapped_file_manager')
        
        self._open_files: Dict[str, MemoryMappedFile] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
    
    @contextmanager
    def get_mapped_file(self, file_path: str, mode: str = 'r', encoding: str = 'utf-8'):
        """
        Get memory-mapped file with automatic resource management.
        
        Args:
            file_path: Path to file
            mode: Access mode
            encoding: Text encoding
            
        Yields:
            MemoryMappedFile instance
        """
        file_key = f"{file_path}:{mode}:{encoding}"
        
        with self._lock:
            # Check if file is already open
            if file_key in self._open_files:
                mapped_file = self._open_files[file_key]
                # Update access order
                self._access_order.remove(file_key)
                self._access_order.append(file_key)
            else:
                # Open new file
                mapped_file = MemoryMappedFile(file_path, mode, encoding)
                mapped_file.open()
                
                # Add to cache
                self._open_files[file_key] = mapped_file
                self._access_order.append(file_key)
                
                # Evict oldest files if necessary
                self._evict_if_necessary()
        
        try:
            yield mapped_file
        finally:
            # File remains open in cache for reuse
            pass
    
    @contextmanager
    def get_csv_reader(self, file_path: str, delimiter: str = ',', encoding: str = 'utf-8'):
        """
        Get memory-mapped CSV reader.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            encoding: Text encoding
            
        Yields:
            MappedCSVReader instance
        """
        reader = MappedCSVReader(file_path, delimiter, encoding)
        try:
            reader.open()
            yield reader
        finally:
            reader.close()
    
    def _evict_if_necessary(self):
        """Evict oldest files if cache is full."""
        while len(self._open_files) > self.max_open_files:
            oldest_key = self._access_order.pop(0)
            oldest_file = self._open_files.pop(oldest_key)
            oldest_file.close()
            self.logger.debug(f"Evicted mapped file: {oldest_key}")
    
    def close_all(self):
        """Close all open memory-mapped files."""
        with self._lock:
            for mapped_file in self._open_files.values():
                mapped_file.close()
            
            self._open_files.clear()
            self._access_order.clear()
            
            self.logger.info("All memory-mapped files closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about open files."""
        with self._lock:
            total_size = sum(f.map_size for f in self._open_files.values())
            
            return {
                'open_files_count': len(self._open_files),
                'max_open_files': self.max_open_files,
                'total_mapped_size': total_size,
                'open_files': [f.get_info().__dict__ for f in self._open_files.values()]
            }


# Global mapped file manager instance
_mapped_file_manager: Optional[MappedFileManager] = None
_manager_lock = threading.Lock()


def get_mapped_file_manager() -> MappedFileManager:
    """Get global mapped file manager instance."""
    global _mapped_file_manager
    
    if _mapped_file_manager is None:
        with _manager_lock:
            if _mapped_file_manager is None:
                _mapped_file_manager = MappedFileManager()
    
    return _mapped_file_manager