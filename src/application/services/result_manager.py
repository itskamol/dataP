"""
Result management service for handling result storage, export, and lifecycle.
Implements requirements 5.4, 7.3, 4.1: Result export, storage, and cleanup.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import json
import uuid
import shutil
import gzip
import pickle
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import pandas as pd

from ...domain.models import MatchingResult, MatchedRecord, ValidationResult
from ...domain.exceptions import FileProcessingError, ValidationError
from ...infrastructure.logging import get_logger


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PICKLE = "pickle"


@dataclass
class ExportConfig:
    """Configuration for result export."""
    format: ExportFormat
    include_metadata: bool = True
    include_unmatched: bool = True
    include_statistics: bool = True
    pagination: Optional[Dict[str, int]] = None  # {'page': 1, 'page_size': 1000}
    filters: Optional[Dict[str, Any]] = None  # {'min_confidence': 75.0}
    sort_by: Optional[str] = None  # 'confidence_score'
    sort_order: str = 'desc'  # 'asc' or 'desc'


@dataclass
class ResultVersion:
    """Version information for stored results."""
    version_id: str
    created_at: datetime
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessControl:
    """Access control information for results."""
    owner_id: str
    shared_with: List[str] = field(default_factory=list)  # List of user IDs
    public: bool = False
    read_only: bool = False
    expires_at: Optional[datetime] = None


@dataclass
class StoredResult:
    """Information about a stored result."""
    result_id: str
    operation_id: str
    created_at: datetime
    last_accessed: datetime
    file_path: str
    compressed: bool
    size_bytes: int
    versions: List[ResultVersion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_control: Optional[AccessControl] = None
    checksum: Optional[str] = None


class ResultManager:
    """Manages result storage, export, and lifecycle."""
    
    def __init__(self, 
                 storage_dir: str = "data/results",
                 temp_dir: str = "data/temp",
                 max_storage_size_mb: int = 1000,
                 auto_cleanup_days: int = 30,
                 compression_enabled: bool = True,
                 cache_size: int = 100):
        """
        Initialize ResultManager.
        
        Args:
            storage_dir: Directory for storing results
            temp_dir: Directory for temporary files
            max_storage_size_mb: Maximum storage size in MB
            auto_cleanup_days: Days after which to auto-cleanup results
            compression_enabled: Whether to compress stored results
            cache_size: Maximum number of results to keep in memory cache
        """
        self.logger = get_logger('result_manager')
        self.storage_dir = Path(storage_dir)
        self.temp_dir = Path(temp_dir)
        self.max_storage_size_mb = max_storage_size_mb
        self.auto_cleanup_days = auto_cleanup_days
        self.compression_enabled = compression_enabled
        self.cache_size = cache_size
        
        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for tracking stored results
        self.index_file = self.storage_dir / "results_index.json"
        
        # In-memory cache for frequently accessed results
        self._result_cache = OrderedDict()
        self._cache_lock = threading.RLock()
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the results index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    self.results_index = {}
                    for result_id, data in index_data.items():
                        # Handle access control deserialization
                        if 'access_control' in data and data['access_control']:
                            ac_data = data['access_control']
                            if ac_data['expires_at']:
                                ac_data['expires_at'] = datetime.fromisoformat(ac_data['expires_at'])
                            data['access_control'] = AccessControl(**ac_data)
                        
                        # Handle datetime fields
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
                        
                        # Handle versions
                        versions = []
                        for v_data in data.get('versions', []):
                            v_data['created_at'] = datetime.fromisoformat(v_data['created_at'])
                            versions.append(ResultVersion(**v_data))
                        data['versions'] = versions
                        
                        self.results_index[result_id] = StoredResult(**data)
            else:
                self.results_index = {}
        except Exception as e:
            self.logger.error(f"Failed to load results index: {str(e)}")
            self.results_index = {}
    
    def _save_index(self) -> None:
        """Save the results index to disk."""
        try:
            index_data = {
                result_id: {
                    'result_id': result.result_id,
                    'operation_id': result.operation_id,
                    'created_at': result.created_at.isoformat(),
                    'last_accessed': result.last_accessed.isoformat(),
                    'file_path': result.file_path,
                    'compressed': result.compressed,
                    'size_bytes': result.size_bytes,
                    'versions': [
                        {
                            'version_id': v.version_id,
                            'created_at': v.created_at.isoformat(),
                            'description': v.description,
                            'metadata': v.metadata
                        }
                        for v in result.versions
                    ],
                    'metadata': result.metadata,
                    'access_control': {
                        'owner_id': result.access_control.owner_id,
                        'shared_with': result.access_control.shared_with,
                        'public': result.access_control.public,
                        'read_only': result.access_control.read_only,
                        'expires_at': result.access_control.expires_at.isoformat() if result.access_control.expires_at else None
                    } if result.access_control else None,
                    'checksum': result.checksum
                }
                for result_id, result in self.results_index.items()
            }
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save results index: {str(e)}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _add_to_cache(self, result_id: str, results: MatchingResult) -> None:
        """Add results to in-memory cache."""
        with self._cache_lock:
            # Remove oldest item if cache is full
            if len(self._result_cache) >= self.cache_size:
                self._result_cache.popitem(last=False)
            
            self._result_cache[result_id] = results
            # Move to end (most recently used)
            self._result_cache.move_to_end(result_id)
    
    def _get_from_cache(self, result_id: str) -> Optional[MatchingResult]:
        """Get results from in-memory cache."""
        with self._cache_lock:
            if result_id in self._result_cache:
                # Move to end (most recently used)
                self._result_cache.move_to_end(result_id)
                return self._result_cache[result_id]
            return None
    
    def _remove_from_cache(self, result_id: str) -> None:
        """Remove results from in-memory cache."""
        with self._cache_lock:
            self._result_cache.pop(result_id, None)
    
    def store_results(self, results: MatchingResult, operation_id: Optional[str] = None, 
                     owner_id: str = "system") -> str:
        """
        Store results with unique identifier and metadata.
        
        Args:
            results: MatchingResult to store
            operation_id: Optional operation ID for tracking
            owner_id: Owner ID for access control
            
        Returns:
            Unique result identifier
        """
        try:
            result_id = str(uuid.uuid4())
            operation_id = operation_id or results.metadata.operation_id
            
            # Create storage path
            result_file = self.storage_dir / f"{result_id}.pkl"
            if self.compression_enabled:
                result_file = result_file.with_suffix('.pkl.gz')
            
            # Store the result
            if self.compression_enabled:
                with gzip.open(result_file, 'wb') as f:
                    pickle.dump(results, f)
            else:
                with open(result_file, 'wb') as f:
                    pickle.dump(results, f)
            
            # Calculate file size and checksum
            file_size = result_file.stat().st_size
            checksum = self._calculate_checksum(result_file)
            
            # Create version info
            version = ResultVersion(
                version_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                description="Initial version",
                metadata={'total_matches': results.total_matches}
            )
            
            # Create access control
            access_control = AccessControl(
                owner_id=owner_id,
                shared_with=[],
                public=False,
                read_only=False
            )
            
            # Create stored result entry
            stored_result = StoredResult(
                result_id=result_id,
                operation_id=operation_id,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                file_path=str(result_file),
                compressed=self.compression_enabled,
                size_bytes=file_size,
                versions=[version],
                metadata={
                    'total_matches': results.total_matches,
                    'file1_name': results.metadata.file1_metadata.name if results.metadata.file1_metadata else '',
                    'file2_name': results.metadata.file2_metadata.name if results.metadata.file2_metadata else '',
                    'processing_time': results.statistics.processing_time_seconds
                },
                access_control=access_control,
                checksum=checksum
            )
            
            # Add to index and cache
            self.results_index[result_id] = stored_result
            self._add_to_cache(result_id, results)
            self._save_index()
            
            self.logger.info(f"Results stored successfully", extra={
                'result_id': result_id,
                'operation_id': operation_id,
                'file_size': file_size,
                'total_matches': results.total_matches,
                'checksum': checksum
            })
            
            # Check storage limits and cleanup if needed
            self._check_storage_limits()
            
            return result_id
            
        except Exception as e:
            self.logger.error(f"Failed to store results: {str(e)}")
            raise FileProcessingError(f"Failed to store results: {str(e)}")
    
    def retrieve_results(self, result_id: str) -> MatchingResult:
        """
        Retrieve stored results by ID (backward compatibility method).
        
        Args:
            result_id: Unique result identifier
            
        Returns:
            MatchingResult object
        """
        return self.retrieve_results_with_cache(result_id, "system")
    
    def export_results(self, result_id: str, config: ExportConfig) -> str:
        """
        Export results in specified format with filtering and pagination.
        
        Args:
            result_id: Unique result identifier
            config: Export configuration
            
        Returns:
            Path to exported file
        """
        try:
            # Retrieve results
            results = self.retrieve_results(result_id)
            
            # Apply filters and pagination
            filtered_results = self._apply_filters_and_pagination(results, config)
            
            # Generate export filename with proper extension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if config.format == ExportFormat.EXCEL:
                extension = "xlsx"
            else:
                extension = config.format.value
            export_filename = f"export_{result_id}_{timestamp}.{extension}"
            export_path = self.temp_dir / export_filename
            
            # Export based on format
            if config.format == ExportFormat.CSV:
                self._export_to_csv(filtered_results, export_path, config)
            elif config.format == ExportFormat.JSON:
                self._export_to_json(filtered_results, export_path, config)
            elif config.format == ExportFormat.EXCEL:
                self._export_to_excel(filtered_results, export_path, config)
            elif config.format == ExportFormat.PICKLE:
                self._export_to_pickle(filtered_results, export_path, config)
            else:
                raise ValidationError(f"Unsupported export format: {config.format}")
            
            self.logger.info(f"Results exported successfully", extra={
                'result_id': result_id,
                'format': config.format.value,
                'export_path': str(export_path),
                'file_size': export_path.stat().st_size
            })
            
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {str(e)}")
            raise FileProcessingError(f"Failed to export results: {str(e)}")
    
    def _apply_filters_and_pagination(self, results: MatchingResult, config: ExportConfig) -> MatchingResult:
        """Apply filters and pagination to results."""
        filtered_matches = results.matched_records.copy()
        
        # Apply filters
        if config.filters:
            if 'min_confidence' in config.filters:
                min_conf = config.filters['min_confidence']
                filtered_matches = [r for r in filtered_matches if r.confidence_score >= min_conf]
            
            if 'max_confidence' in config.filters:
                max_conf = config.filters['max_confidence']
                filtered_matches = [r for r in filtered_matches if r.confidence_score <= max_conf]
            
            if 'matching_fields' in config.filters:
                required_fields = config.filters['matching_fields']
                filtered_matches = [
                    r for r in filtered_matches 
                    if any(field in r.matching_fields for field in required_fields)
                ]
        
        # Apply sorting
        if config.sort_by:
            reverse = config.sort_order == 'desc'
            if config.sort_by == 'confidence_score':
                filtered_matches.sort(key=lambda x: x.confidence_score, reverse=reverse)
            elif config.sort_by == 'created_at':
                filtered_matches.sort(key=lambda x: x.created_at, reverse=reverse)
        
        # Apply pagination
        if config.pagination:
            page = config.pagination.get('page', 1)
            page_size = config.pagination.get('page_size', len(filtered_matches))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            filtered_matches = filtered_matches[start_idx:end_idx]
        
        # Create filtered result
        filtered_result = MatchingResult(
            matched_records=filtered_matches,
            unmatched_records=results.unmatched_records if config.include_unmatched else {},
            statistics=results.statistics if config.include_statistics else None,
            metadata=results.metadata if config.include_metadata else None
        )
        
        return filtered_result
    
    def _export_to_csv(self, results: MatchingResult, export_path: Path, config: ExportConfig) -> None:
        """Export results to CSV format."""
        # Prepare matched records data
        matched_data = []
        for record in results.matched_records:
            row = {}
            
            # Add record1 fields with prefix
            for key, value in record.record1.items():
                row[f"record1_{key}"] = value
            
            # Add record2 fields with prefix
            for key, value in record.record2.items():
                row[f"record2_{key}"] = value
            
            # Add matching metadata
            row['confidence_score'] = record.confidence_score
            row['matching_fields'] = ', '.join(record.matching_fields)
            row['created_at'] = record.created_at.isoformat()
            
            matched_data.append(row)
        
        # Create DataFrame and save
        if matched_data:
            df = pd.DataFrame(matched_data)
            df.to_csv(export_path, index=False, encoding='utf-8')
        else:
            # Create empty CSV with basic headers
            empty_df = pd.DataFrame(columns=['confidence_score', 'matching_fields', 'created_at'])
            empty_df.to_csv(export_path, index=False, encoding='utf-8')
    
    def _export_to_json(self, results: MatchingResult, export_path: Path, config: ExportConfig) -> None:
        """Export results to JSON format."""
        export_data = {
            'matched_records': [record.to_dict() for record in results.matched_records],
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'total_records': len(results.matched_records),
                'format': 'json'
            }
        }
        
        if config.include_unmatched:
            export_data['unmatched_records'] = results.unmatched_records
        
        if config.include_statistics and results.statistics:
            export_data['statistics'] = results.statistics.to_dict()
        
        if config.include_metadata and results.metadata:
            export_data['metadata'] = results.metadata.to_dict()
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_to_excel(self, results: MatchingResult, export_path: Path, config: ExportConfig) -> None:
        """Export results to Excel format."""
        with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
            # Matched records sheet
            if results.matched_records:
                matched_data = []
                for record in results.matched_records:
                    row = {}
                    
                    # Add record1 fields with prefix
                    for key, value in record.record1.items():
                        row[f"record1_{key}"] = value
                    
                    # Add record2 fields with prefix
                    for key, value in record.record2.items():
                        row[f"record2_{key}"] = value
                    
                    # Add matching metadata
                    row['confidence_score'] = record.confidence_score
                    row['matching_fields'] = ', '.join(record.matching_fields)
                    row['created_at'] = record.created_at.isoformat()
                    
                    matched_data.append(row)
                
                df_matched = pd.DataFrame(matched_data)
                df_matched.to_excel(writer, sheet_name='Matched Records', index=False)
            
            # Unmatched records sheets
            if config.include_unmatched and results.unmatched_records:
                for file_key, unmatched_list in results.unmatched_records.items():
                    if unmatched_list:
                        df_unmatched = pd.DataFrame(unmatched_list)
                        sheet_name = f'Unmatched {file_key}'[:31]  # Excel sheet name limit
                        df_unmatched.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Statistics sheet
            if config.include_statistics and results.statistics:
                stats_data = [
                    {'Metric': key, 'Value': value}
                    for key, value in results.statistics.to_dict().items()
                ]
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
    
    def _export_to_pickle(self, results: MatchingResult, export_path: Path, config: ExportConfig) -> None:
        """Export results to pickle format."""
        with open(export_path, 'wb') as f:
            pickle.dump(results, f)
    
    def list_results(self, limit: Optional[int] = None, 
                    operation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List stored results with metadata.
        
        Args:
            limit: Maximum number of results to return
            operation_id: Filter by operation ID
            
        Returns:
            List of result metadata
        """
        try:
            results_list = []
            
            for result_id, stored_result in self.results_index.items():
                if operation_id and stored_result.operation_id != operation_id:
                    continue
                
                result_info = {
                    'result_id': result_id,
                    'operation_id': stored_result.operation_id,
                    'created_at': stored_result.created_at.isoformat(),
                    'last_accessed': stored_result.last_accessed.isoformat(),
                    'size_bytes': stored_result.size_bytes,
                    'compressed': stored_result.compressed,
                    'versions_count': len(stored_result.versions),
                    'metadata': stored_result.metadata
                }
                results_list.append(result_info)
            
            # Sort by creation time (newest first)
            results_list.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Apply limit
            if limit:
                results_list = results_list[:limit]
            
            return results_list
            
        except Exception as e:
            self.logger.error(f"Failed to list results: {str(e)}")
            raise FileProcessingError(f"Failed to list results: {str(e)}")
    
    def delete_result(self, result_id: str) -> bool:
        """
        Delete a stored result.
        
        Args:
            result_id: Unique result identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            if result_id not in self.results_index:
                return False
            
            stored_result = self.results_index[result_id]
            result_file = Path(stored_result.file_path)
            
            # Delete file if it exists
            if result_file.exists():
                result_file.unlink()
            
            # Remove from index
            del self.results_index[result_id]
            self._save_index()
            
            self.logger.info(f"Result deleted successfully", extra={'result_id': result_id})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete result: {str(e)}")
            return False
    
    def cleanup_old_results(self, max_age_days: Optional[int] = None) -> int:
        """
        Clean up old results based on age.
        
        Args:
            max_age_days: Maximum age in days (default: self.auto_cleanup_days)
            
        Returns:
            Number of results cleaned up
        """
        try:
            max_age_days = max_age_days or self.auto_cleanup_days
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            results_to_delete = []
            for result_id, stored_result in self.results_index.items():
                if stored_result.last_accessed < cutoff_date:
                    results_to_delete.append(result_id)
            
            cleaned_count = 0
            for result_id in results_to_delete:
                if self.delete_result(result_id):
                    cleaned_count += 1
            
            self.logger.info(f"Cleanup completed", extra={
                'cleaned_count': cleaned_count,
                'max_age_days': max_age_days
            })
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old results: {str(e)}")
            return 0
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary export files.
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to delete temp file {file_path}: {str(e)}")
            
            self.logger.info(f"Temp files cleanup completed", extra={
                'cleaned_count': cleaned_count,
                'max_age_hours': max_age_hours
            })
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {str(e)}")
            return 0
    
    def _check_storage_limits(self) -> None:
        """Check storage limits and cleanup if necessary."""
        try:
            # Calculate total storage size
            total_size = 0
            for stored_result in self.results_index.values():
                total_size += stored_result.size_bytes
            
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > self.max_storage_size_mb:
                self.logger.warning(f"Storage limit exceeded: {total_size_mb:.2f}MB > {self.max_storage_size_mb}MB")
                
                # Sort by last accessed time (oldest first)
                sorted_results = sorted(
                    self.results_index.items(),
                    key=lambda x: x[1].last_accessed
                )
                
                # Delete oldest results until under limit
                for result_id, _ in sorted_results:
                    self.delete_result(result_id)
                    
                    # Recalculate size
                    total_size = sum(r.size_bytes for r in self.results_index.values())
                    total_size_mb = total_size / (1024 * 1024)
                    
                    if total_size_mb <= self.max_storage_size_mb * 0.8:  # 80% of limit
                        break
                
                self.logger.info(f"Storage cleanup completed", extra={
                    'final_size_mb': total_size_mb,
                    'remaining_results': len(self.results_index)
                })
            
        except Exception as e:
            self.logger.error(f"Failed to check storage limits: {str(e)}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information and statistics.
        
        Returns:
            Dictionary with storage information
        """
        try:
            total_size = sum(r.size_bytes for r in self.results_index.values())
            total_size_mb = total_size / (1024 * 1024)
            
            # Count by age
            now = datetime.now()
            age_counts = {'1_day': 0, '1_week': 0, '1_month': 0, 'older': 0}
            
            for stored_result in self.results_index.values():
                age = now - stored_result.created_at
                if age.days <= 1:
                    age_counts['1_day'] += 1
                elif age.days <= 7:
                    age_counts['1_week'] += 1
                elif age.days <= 30:
                    age_counts['1_month'] += 1
                else:
                    age_counts['older'] += 1
            
            return {
                'total_results': len(self.results_index),
                'total_size_bytes': total_size,
                'total_size_mb': total_size_mb,
                'max_size_mb': self.max_storage_size_mb,
                'usage_percentage': (total_size_mb / self.max_storage_size_mb) * 100,
                'age_distribution': age_counts,
                'compression_enabled': self.compression_enabled,
                'auto_cleanup_days': self.auto_cleanup_days
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {str(e)}")
            return {}
    
    def retrieve_results_with_cache(self, result_id: str, user_id: str = "system") -> MatchingResult:
        """
        Retrieve stored results by ID with caching and access control.
        
        Args:
            result_id: Unique result identifier
            user_id: User ID for access control
            
        Returns:
            MatchingResult object
        """
        try:
            # Check access permissions
            if not self.check_access_permission(result_id, user_id):
                raise ValidationError(f"Access denied for result: {result_id}")
            
            # Try to get from cache first
            cached_result = self._get_from_cache(result_id)
            if cached_result is not None:
                self.logger.debug(f"Retrieved result from cache", extra={'result_id': result_id})
                return cached_result
            
            # Load from disk
            if result_id not in self.results_index:
                raise ValidationError(f"Result not found: {result_id}")
            
            stored_result = self.results_index[result_id]
            result_file = Path(stored_result.file_path)
            
            if not result_file.exists():
                raise FileProcessingError(f"Result file not found: {result_file}")
            
            # Verify checksum if available
            if stored_result.checksum:
                current_checksum = self._calculate_checksum(result_file)
                if current_checksum != stored_result.checksum:
                    self.logger.warning(f"Checksum mismatch for result {result_id}")
            
            # Load the result
            if stored_result.compressed:
                with gzip.open(result_file, 'rb') as f:
                    results = pickle.load(f)
            else:
                with open(result_file, 'rb') as f:
                    results = pickle.load(f)
            
            # Add to cache
            self._add_to_cache(result_id, results)
            
            # Update last accessed time
            stored_result.last_accessed = datetime.now()
            self._save_index()
            
            self.logger.info(f"Results retrieved successfully", extra={
                'result_id': result_id,
                'total_matches': results.total_matches,
                'user_id': user_id
            })
            
            return results
            
        except (ValidationError, FileProcessingError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve results: {str(e)}")
            raise FileProcessingError(f"Failed to retrieve results: {str(e)}")
    
    def check_access_permission(self, result_id: str, user_id: str) -> bool:
        """
        Check if user has access permission to a result.
        
        Args:
            result_id: Unique result identifier
            user_id: User ID to check
            
        Returns:
            True if user has access
        """
        try:
            if result_id not in self.results_index:
                return False
            
            stored_result = self.results_index[result_id]
            access_control = stored_result.access_control
            
            if not access_control:
                return True  # No access control means public access
            
            # Check if expired
            if access_control.expires_at and datetime.now() > access_control.expires_at:
                return False
            
            # Check access permissions
            if access_control.public:
                return True
            
            if access_control.owner_id == user_id:
                return True
            
            if user_id in access_control.shared_with:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check access permission: {str(e)}")
            return False
    
    def share_result(self, result_id: str, owner_id: str, user_ids: List[str], 
                    read_only: bool = True, expires_at: Optional[datetime] = None) -> bool:
        """
        Share a result with other users.
        
        Args:
            result_id: Unique result identifier
            owner_id: Owner ID (must match current owner)
            user_ids: List of user IDs to share with
            read_only: Whether shared access is read-only
            expires_at: Optional expiration time
            
        Returns:
            True if sharing was successful
        """
        try:
            if result_id not in self.results_index:
                return False
            
            stored_result = self.results_index[result_id]
            
            # Check if user is owner
            if not stored_result.access_control or stored_result.access_control.owner_id != owner_id:
                return False
            
            # Update access control
            stored_result.access_control.shared_with.extend(user_ids)
            stored_result.access_control.shared_with = list(set(stored_result.access_control.shared_with))  # Remove duplicates
            stored_result.access_control.read_only = read_only
            if expires_at:
                stored_result.access_control.expires_at = expires_at
            
            self._save_index()
            
            self.logger.info(f"Result shared successfully", extra={
                'result_id': result_id,
                'owner_id': owner_id,
                'shared_with': user_ids,
                'read_only': read_only
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to share result: {str(e)}")
            return False
    
    def make_result_public(self, result_id: str, owner_id: str, public: bool = True) -> bool:
        """
        Make a result public or private.
        
        Args:
            result_id: Unique result identifier
            owner_id: Owner ID (must match current owner)
            public: Whether to make result public
            
        Returns:
            True if operation was successful
        """
        try:
            if result_id not in self.results_index:
                return False
            
            stored_result = self.results_index[result_id]
            
            # Check if user is owner
            if not stored_result.access_control or stored_result.access_control.owner_id != owner_id:
                return False
            
            # Update access control
            stored_result.access_control.public = public
            self._save_index()
            
            self.logger.info(f"Result publicity changed", extra={
                'result_id': result_id,
                'owner_id': owner_id,
                'public': public
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to change result publicity: {str(e)}")
            return False
    
    def compress_result(self, result_id: str) -> bool:
        """
        Compress a stored result to save space.
        
        Args:
            result_id: Unique result identifier
            
        Returns:
            True if compression was successful
        """
        try:
            if result_id not in self.results_index:
                return False
            
            stored_result = self.results_index[result_id]
            
            # Skip if already compressed
            if stored_result.compressed:
                return True
            
            result_file = Path(stored_result.file_path)
            if not result_file.exists():
                return False
            
            # Load the result
            with open(result_file, 'rb') as f:
                data = f.read()
            
            # Create compressed file
            compressed_file = result_file.with_suffix('.pkl.gz')
            with gzip.open(compressed_file, 'wb') as f:
                f.write(data)
            
            # Update stored result info
            old_size = stored_result.size_bytes
            new_size = compressed_file.stat().st_size
            
            stored_result.file_path = str(compressed_file)
            stored_result.compressed = True
            stored_result.size_bytes = new_size
            stored_result.checksum = self._calculate_checksum(compressed_file)
            
            # Remove original file
            result_file.unlink()
            
            # Remove from cache (will be reloaded when needed)
            self._remove_from_cache(result_id)
            
            self._save_index()
            
            compression_ratio = (old_size - new_size) / old_size * 100
            self.logger.info(f"Result compressed successfully", extra={
                'result_id': result_id,
                'old_size': old_size,
                'new_size': new_size,
                'compression_ratio': f"{compression_ratio:.1f}%"
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compress result: {str(e)}")
            return False
    
    def get_result_metadata(self, result_id: str, user_id: str = "system") -> Optional[Dict[str, Any]]:
        """
        Get metadata for a result without loading the full data.
        
        Args:
            result_id: Unique result identifier
            user_id: User ID for access control
            
        Returns:
            Result metadata or None if not accessible
        """
        try:
            if not self.check_access_permission(result_id, user_id):
                return None
            
            if result_id not in self.results_index:
                return None
            
            stored_result = self.results_index[result_id]
            
            metadata = {
                'result_id': result_id,
                'operation_id': stored_result.operation_id,
                'created_at': stored_result.created_at.isoformat(),
                'last_accessed': stored_result.last_accessed.isoformat(),
                'size_bytes': stored_result.size_bytes,
                'compressed': stored_result.compressed,
                'versions_count': len(stored_result.versions),
                'metadata': stored_result.metadata,
                'checksum': stored_result.checksum,
                'access_control': {
                    'owner_id': stored_result.access_control.owner_id,
                    'public': stored_result.access_control.public,
                    'read_only': stored_result.access_control.read_only,
                    'shared_with_count': len(stored_result.access_control.shared_with),
                    'expires_at': stored_result.access_control.expires_at.isoformat() if stored_result.access_control.expires_at else None
                } if stored_result.access_control else None
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get result metadata: {str(e)}")
            return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the result cache.
        
        Returns:
            Cache information
        """
        with self._cache_lock:
            return {
                'cache_size': len(self._result_cache),
                'max_cache_size': self.cache_size,
                'cached_results': list(self._result_cache.keys()),
                'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
            }
    
    def clear_cache(self) -> int:
        """
        Clear the result cache.
        
        Returns:
            Number of items cleared from cache
        """
        with self._cache_lock:
            cleared_count = len(self._result_cache)
            self._result_cache.clear()
            
            self.logger.info(f"Cache cleared", extra={'cleared_count': cleared_count})
            return cleared_count
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information and statistics.
        
        Returns:
            Dictionary with storage information
        """
        try:
            total_size = sum(r.size_bytes for r in self.results_index.values())
            total_size_mb = total_size / (1024 * 1024)
            
            # Count by age
            now = datetime.now()
            age_counts = {'1_day': 0, '1_week': 0, '1_month': 0, 'older': 0}
            
            # Count by compression
            compressed_count = sum(1 for r in self.results_index.values() if r.compressed)
            
            # Count by access control
            public_count = sum(1 for r in self.results_index.values() 
                             if r.access_control and r.access_control.public)
            shared_count = sum(1 for r in self.results_index.values() 
                             if r.access_control and r.access_control.shared_with)
            
            for stored_result in self.results_index.values():
                age = now - stored_result.created_at
                if age.days <= 1:
                    age_counts['1_day'] += 1
                elif age.days <= 7:
                    age_counts['1_week'] += 1
                elif age.days <= 30:
                    age_counts['1_month'] += 1
                else:
                    age_counts['older'] += 1
            
            return {
                'total_results': len(self.results_index),
                'total_size_bytes': total_size,
                'total_size_mb': total_size_mb,
                'max_size_mb': self.max_storage_size_mb,
                'usage_percentage': (total_size_mb / self.max_storage_size_mb) * 100,
                'age_distribution': age_counts,
                'compression_enabled': self.compression_enabled,
                'compressed_results': compressed_count,
                'compression_ratio': compressed_count / max(len(self.results_index), 1) * 100,
                'public_results': public_count,
                'shared_results': shared_count,
                'auto_cleanup_days': self.auto_cleanup_days,
                'cache_info': self.get_cache_info()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {str(e)}")
            return {}