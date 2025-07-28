"""
Compressed data storage and retrieval mechanisms for performance optimization.
Implements requirement 3.2, 3.3: Compressed data storage for memory efficiency.
"""

import gzip
import bz2
import lzma
import zlib
import pickle
import json
import threading
from typing import Dict, List, Optional, Any, Union, BinaryIO, TextIO
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
from contextlib import contextmanager

import pandas as pd
import numpy as np

from src.infrastructure.logging import get_logger
from src.domain.exceptions import FileProcessingError, FileAccessError


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZLIB = "zlib"


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    compression_type: CompressionType


class CompressedDataStore:
    """Store for compressed data with automatic compression selection."""
    
    def __init__(self, storage_dir: str, default_compression: CompressionType = CompressionType.GZIP):
        """
        Initialize compressed data store.
        
        Args:
            storage_dir: Directory to store compressed data
            default_compression: Default compression type
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.default_compression = default_compression
        self.logger = get_logger('compressed_storage')
        
        self._compression_stats: Dict[str, CompressionStats] = {}
        self._lock = threading.RLock()
        
        # Compression handlers
        self._compressors = {
            CompressionType.GZIP: self._gzip_compress,
            CompressionType.BZIP2: self._bzip2_compress,
            CompressionType.LZMA: self._lzma_compress,
            CompressionType.ZLIB: self._zlib_compress,
            CompressionType.NONE: self._no_compress
        }
        
        self._decompressors = {
            CompressionType.GZIP: self._gzip_decompress,
            CompressionType.BZIP2: self._bzip2_decompress,
            CompressionType.LZMA: self._lzma_decompress,
            CompressionType.ZLIB: self._zlib_decompress,
            CompressionType.NONE: self._no_decompress
        }
    
    def _gzip_compress(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data, compresslevel=6)
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)
    
    def _bzip2_compress(self, data: bytes) -> bytes:
        """Compress data using bzip2."""
        return bz2.compress(data, compresslevel=6)
    
    def _bzip2_decompress(self, data: bytes) -> bytes:
        """Decompress bzip2 data."""
        return bz2.decompress(data)
    
    def _lzma_compress(self, data: bytes) -> bytes:
        """Compress data using LZMA."""
        return lzma.compress(data, preset=6)
    
    def _lzma_decompress(self, data: bytes) -> bytes:
        """Decompress LZMA data."""
        return lzma.decompress(data)
    
    def _zlib_compress(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data, level=6)
    
    def _zlib_decompress(self, data: bytes) -> bytes:
        """Decompress zlib data."""
        return zlib.decompress(data)
    
    def _no_compress(self, data: bytes) -> bytes:
        """No compression."""
        return data
    
    def _no_decompress(self, data: bytes) -> bytes:
        """No decompression."""
        return data
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False).encode('utf-8')
        elif isinstance(data, pd.DataFrame):
            return data.to_pickle()
        else:
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes, data_type: str) -> Any:
        """Deserialize data from bytes."""
        if data_type == 'bytes':
            return data
        elif data_type == 'str':
            return data.decode('utf-8')
        elif data_type == 'json':
            return json.loads(data.decode('utf-8'))
        elif data_type == 'dataframe':
            return pd.read_pickle(data)
        else:
            return pickle.loads(data)
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect data type for serialization."""
        if isinstance(data, bytes):
            return 'bytes'
        elif isinstance(data, str):
            return 'str'
        elif isinstance(data, (dict, list)):
            return 'json'
        elif isinstance(data, pd.DataFrame):
            return 'dataframe'
        else:
            return 'pickle'
    
    def _choose_best_compression(self, data: bytes) -> CompressionType:
        """Choose best compression type based on data characteristics."""
        if len(data) < 1024:  # Small data, don't compress
            return CompressionType.NONE
        
        # Test different compression types on a sample
        sample_size = min(len(data), 10000)  # Test on first 10KB
        sample = data[:sample_size]
        
        best_compression = self.default_compression
        best_ratio = 1.0
        
        for compression_type in [CompressionType.GZIP, CompressionType.ZLIB, CompressionType.LZMA]:
            try:
                start_time = time.time()
                compressed = self._compressors[compression_type](sample)
                compression_time = time.time() - start_time
                
                ratio = len(compressed) / len(sample)
                
                # Consider both compression ratio and speed
                score = ratio + (compression_time * 0.1)  # Penalize slow compression
                
                if score < best_ratio:
                    best_ratio = score
                    best_compression = compression_type
                    
            except Exception as e:
                self.logger.warning(f"Compression test failed for {compression_type}: {str(e)}")
        
        return best_compression
    
    def store(self, key: str, data: Any, compression_type: Optional[CompressionType] = None) -> str:
        """
        Store data with compression.
        
        Args:
            key: Storage key
            data: Data to store
            compression_type: Compression type (auto-detect if None)
            
        Returns:
            File path where data was stored
        """
        try:
            # Serialize data
            data_type = self._detect_data_type(data)
            serialized_data = self._serialize_data(data)
            original_size = len(serialized_data)
            
            # Choose compression type
            if compression_type is None:
                compression_type = self._choose_best_compression(serialized_data)
            
            # Compress data
            start_time = time.time()
            compressed_data = self._compressors[compression_type](serialized_data)
            compression_time = time.time() - start_time
            
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Create file path
            file_extension = f".{compression_type.value}" if compression_type != CompressionType.NONE else ""
            file_path = self.storage_dir / f"{key}{file_extension}"
            
            # Write compressed data
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Store metadata
            metadata = {
                'data_type': data_type,
                'compression_type': compression_type.value,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'compression_time': compression_time
            }
            
            metadata_path = self.storage_dir / f"{key}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Update statistics
            with self._lock:
                self._compression_stats[key] = CompressionStats(
                    original_size=original_size,
                    compressed_size=compressed_size,
                    compression_ratio=compression_ratio,
                    compression_time=compression_time,
                    decompression_time=0.0,
                    compression_type=compression_type
                )
            
            self.logger.info(f"Stored compressed data: {key} "
                           f"({original_size} -> {compressed_size} bytes, "
                           f"ratio: {compression_ratio:.2f})")
            
            return str(file_path)
            
        except Exception as e:
            raise FileProcessingError(
                f"Failed to store compressed data: {str(e)}",
                context={'key': key, 'compression_type': compression_type}
            )
    
    def retrieve(self, key: str) -> Any:
        """
        Retrieve and decompress data.
        
        Args:
            key: Storage key
            
        Returns:
            Decompressed data
        """
        try:
            # Load metadata
            metadata_path = self.storage_dir / f"{key}.meta"
            if not metadata_path.exists():
                raise FileAccessError(f"Metadata not found for key: {key}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            data_type = metadata['data_type']
            compression_type = CompressionType(metadata['compression_type'])
            
            # Find data file
            file_extension = f".{compression_type.value}" if compression_type != CompressionType.NONE else ""
            file_path = self.storage_dir / f"{key}{file_extension}"
            
            if not file_path.exists():
                raise FileAccessError(f"Data file not found: {file_path}")
            
            # Read compressed data
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress data
            start_time = time.time()
            decompressed_data = self._decompressors[compression_type](compressed_data)
            decompression_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                if key in self._compression_stats:
                    self._compression_stats[key].decompression_time = decompression_time
            
            # Deserialize data
            data = self._deserialize_data(decompressed_data, data_type)
            
            self.logger.debug(f"Retrieved compressed data: {key} "
                            f"(decompression time: {decompression_time:.3f}s)")
            
            return data
            
        except Exception as e:
            raise FileProcessingError(
                f"Failed to retrieve compressed data: {str(e)}",
                context={'key': key}
            )
    
    def exists(self, key: str) -> bool:
        """Check if key exists in storage."""
        metadata_path = self.storage_dir / f"{key}.meta"
        return metadata_path.exists()
    
    def delete(self, key: str) -> bool:
        """Delete stored data."""
        try:
            deleted = False
            
            # Delete metadata
            metadata_path = self.storage_dir / f"{key}.meta"
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True
            
            # Delete data files (try all possible extensions)
            for compression_type in CompressionType:
                if compression_type == CompressionType.NONE:
                    file_path = self.storage_dir / key
                else:
                    file_path = self.storage_dir / f"{key}.{compression_type.value}"
                
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
            
            # Remove from statistics
            with self._lock:
                self._compression_stats.pop(key, None)
            
            if deleted:
                self.logger.info(f"Deleted compressed data: {key}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete compressed data {key}: {str(e)}")
            return False
    
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        keys = []
        for metadata_file in self.storage_dir.glob("*.meta"):
            keys.append(metadata_file.stem)
        return sorted(keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        with self._lock:
            total_original = sum(stats.original_size for stats in self._compression_stats.values())
            total_compressed = sum(stats.compressed_size for stats in self._compression_stats.values())
            
            compression_types = {}
            for stats in self._compression_stats.values():
                comp_type = stats.compression_type.value
                if comp_type not in compression_types:
                    compression_types[comp_type] = {'count': 0, 'total_ratio': 0.0}
                compression_types[comp_type]['count'] += 1
                compression_types[comp_type]['total_ratio'] += stats.compression_ratio
            
            # Calculate average ratios
            for comp_type in compression_types:
                count = compression_types[comp_type]['count']
                compression_types[comp_type]['avg_ratio'] = compression_types[comp_type]['total_ratio'] / count
            
            return {
                'total_keys': len(self._compression_stats),
                'total_original_size': total_original,
                'total_compressed_size': total_compressed,
                'overall_compression_ratio': total_compressed / total_original if total_original > 0 else 1.0,
                'space_saved_bytes': total_original - total_compressed,
                'space_saved_percent': ((total_original - total_compressed) / total_original * 100) if total_original > 0 else 0.0,
                'compression_types': compression_types
            }
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> int:
        """Clean up old compressed data."""
        cleaned_count = 0
        current_time = time.time()
        
        for key in self.list_keys():
            try:
                metadata_path = self.storage_dir / f"{key}.meta"
                file_age_hours = (current_time - metadata_path.stat().st_mtime) / 3600
                
                if file_age_hours > max_age_hours:
                    if self.delete(key):
                        cleaned_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Failed to check age of {key}: {str(e)}")
        
        self.logger.info(f"Cleaned up {cleaned_count} old compressed data entries")
        return cleaned_count


class CompressedDataFrameStore:
    """Specialized store for compressed pandas DataFrames."""
    
    def __init__(self, storage_dir: str):
        """Initialize compressed DataFrame store."""
        self.store = CompressedDataStore(storage_dir, CompressionType.GZIP)
        self.logger = get_logger('compressed_dataframe_store')
    
    def store_dataframe(self, key: str, df: pd.DataFrame, 
                       compression_type: Optional[CompressionType] = None) -> str:
        """Store DataFrame with optimized compression."""
        # Optimize DataFrame before compression
        optimized_df = self._optimize_dataframe(df)
        return self.store.store(key, optimized_df, compression_type)
    
    def retrieve_dataframe(self, key: str) -> pd.DataFrame:
        """Retrieve compressed DataFrame."""
        return self.store.retrieve(key)
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for better compression."""
        optimized_df = df.copy()
        
        # Convert object columns to category if beneficial
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
        
        # Downcast numeric types
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def store_chunked_dataframe(self, key_prefix: str, df: pd.DataFrame, 
                               chunk_size: int = 10000) -> List[str]:
        """Store large DataFrame in compressed chunks."""
        chunk_keys = []
        
        for i, chunk in enumerate(df.groupby(df.index // chunk_size)):
            chunk_key = f"{key_prefix}_chunk_{i}"
            chunk_df = chunk[1]  # Get the DataFrame from groupby result
            
            file_path = self.store_dataframe(chunk_key, chunk_df)
            chunk_keys.append(chunk_key)
        
        # Store chunk metadata
        metadata = {
            'chunk_keys': chunk_keys,
            'total_rows': len(df),
            'chunk_size': chunk_size,
            'columns': df.columns.tolist()
        }
        
        self.store.store(f"{key_prefix}_metadata", metadata)
        
        self.logger.info(f"Stored DataFrame in {len(chunk_keys)} compressed chunks: {key_prefix}")
        return chunk_keys
    
    def retrieve_chunked_dataframe(self, key_prefix: str) -> pd.DataFrame:
        """Retrieve DataFrame from compressed chunks."""
        # Load metadata
        metadata = self.store.retrieve(f"{key_prefix}_metadata")
        chunk_keys = metadata['chunk_keys']
        
        # Load and concatenate chunks
        chunks = []
        for chunk_key in chunk_keys:
            chunk_df = self.retrieve_dataframe(chunk_key)
            chunks.append(chunk_df)
        
        result_df = pd.concat(chunks, ignore_index=True)
        
        self.logger.info(f"Retrieved DataFrame from {len(chunk_keys)} compressed chunks: {key_prefix}")
        return result_df


# Global compressed storage instances
_compressed_store: Optional[CompressedDataStore] = None
_compressed_df_store: Optional[CompressedDataFrameStore] = None
_storage_lock = threading.Lock()


def get_compressed_store(storage_dir: str = "compressed_data") -> CompressedDataStore:
    """Get global compressed data store instance."""
    global _compressed_store
    
    if _compressed_store is None:
        with _storage_lock:
            if _compressed_store is None:
                _compressed_store = CompressedDataStore(storage_dir)
    
    return _compressed_store


def get_compressed_dataframe_store(storage_dir: str = "compressed_dataframes") -> CompressedDataFrameStore:
    """Get global compressed DataFrame store instance."""
    global _compressed_df_store
    
    if _compressed_df_store is None:
        with _storage_lock:
            if _compressed_df_store is None:
                _compressed_df_store = CompressedDataFrameStore(storage_dir)
    
    return _compressed_df_store