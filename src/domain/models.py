"""
Core data models for the file processing system.
These models represent the core business entities and value objects.
Enhanced with Pydantic validation and serialization capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import uuid
import json
try:
    from pydantic import BaseModel, Field, model_validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create dummy classes for when pydantic is not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def Field(default=None, **kwargs):
        return default
    
    def model_validator(mode):
        def decorator(func):
            return func
        return decorator
    
    class ConfigDict:
        def __init__(self, **kwargs):
            pass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class MatchingType(Enum):
    """Enumeration of matching types supported by the system."""
    ONE_TO_ONE = "one-to-one"
    ONE_TO_MANY = "one-to-many"
    MANY_TO_ONE = "many-to-one"
    MANY_TO_MANY = "many-to-many"


class AlgorithmType(Enum):
    """Enumeration of matching algorithms."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    PHONETIC = "phonetic"


class FileType(Enum):
    """Supported file types."""
    CSV = "csv"
    JSON = "json"


class FieldMapping(BaseModel):
    """Configuration for mapping between two fields."""
    model_config = ConfigDict(use_enum_values=True)
    
    source_field: str = Field(..., min_length=1, description="Source field name")
    target_field: str = Field(..., min_length=1, description="Target field name")
    algorithm: AlgorithmType = Field(..., description="Matching algorithm to use")
    weight: float = Field(1.0, ge=0.1, le=1.0, description="Weight for this field mapping")
    normalization: bool = Field(False, description="Whether to normalize text before matching")
    case_sensitive: bool = Field(False, description="Whether matching is case sensitive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldMapping':
        """Create instance from dictionary."""
        return cls(**data)


class AlgorithmConfig(BaseModel):
    """Configuration for a specific matching algorithm."""
    model_config = ConfigDict(use_enum_values=True)
    
    name: str = Field(..., min_length=1, description="Algorithm name")
    algorithm_type: AlgorithmType = Field(..., description="Type of matching algorithm")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")
    enabled: bool = Field(True, description="Whether algorithm is enabled")
    priority: int = Field(1, ge=1, description="Algorithm priority (higher = more important)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmConfig':
        """Create instance from dictionary."""
        return cls(**data)


class DatasetMetadata(BaseModel):
    """Metadata for a dataset."""
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique dataset identifier")
    name: str = Field("", description="Dataset name")
    file_path: str = Field("", description="Path to the source file")
    file_type: FileType = Field(FileType.CSV, description="Type of the source file")
    delimiter: Optional[str] = Field(None, description="CSV delimiter character")
    encoding: str = Field("utf-8", description="File encoding")
    row_count: int = Field(0, ge=0, description="Number of rows in dataset")
    column_count: int = Field(0, ge=0, description="Number of columns in dataset")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_modified: datetime = Field(default_factory=datetime.now, description="Last modification timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create instance from dictionary."""
        # Handle datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_modified' in data and isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


class Dataset(BaseModel):
    """Represents a loaded dataset with metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique dataset identifier")
    name: str = Field("", description="Dataset name")
    columns: List[str] = Field(default_factory=list, description="Column names in the dataset")
    data: Optional[Any] = Field(None, description="The actual data as DataFrame (if pandas available)")
    metadata: DatasetMetadata = Field(default_factory=DatasetMetadata, description="Dataset metadata")
    
    @model_validator(mode='after')
    def update_metadata_from_data(self):
        """Update metadata when data is provided."""
        if self.data is not None:
            self.columns = self.data.columns.tolist()
            self.metadata.row_count = len(self.data)
            self.metadata.column_count = len(self.data.columns)
        return self
    
    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.model_dump(exclude={'data'} if not include_data else set())
        if include_data and self.data is not None:
            result['data'] = self.data.to_dict('records')
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dataset':
        """Create instance from dictionary."""
        if 'data' in data and data['data'] is not None:
            if PANDAS_AVAILABLE:
                data['data'] = pd.DataFrame(data['data'])
            else:
                # Keep as raw data if pandas not available
                pass
        if 'metadata' in data:
            data['metadata'] = DatasetMetadata.from_dict(data['metadata'])
        return cls(**data)


class MatchingConfig(BaseModel):
    """Configuration for matching operations."""
    model_config = ConfigDict(use_enum_values=True)
    
    mappings: List[FieldMapping] = Field(default_factory=list, min_length=1, description="Field mappings for matching")
    algorithms: List[AlgorithmConfig] = Field(default_factory=list, description="Algorithm configurations")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Threshold values for different operations")
    matching_type: MatchingType = Field(MatchingType.ONE_TO_ONE, description="Type of matching to perform")
    confidence_threshold: float = Field(75.0, ge=0, le=100, description="Minimum confidence threshold for matches")
    use_blocking: bool = Field(True, description="Whether to use blocking strategies")
    parallel_processing: bool = Field(True, description="Whether to enable parallel processing")
    max_workers: Optional[int] = Field(None, ge=1, description="Maximum number of worker processes")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchingConfig':
        """Create instance from dictionary."""
        # Convert nested objects
        if 'mappings' in data:
            data['mappings'] = [FieldMapping.from_dict(m) if isinstance(m, dict) else m for m in data['mappings']]
        if 'algorithms' in data:
            data['algorithms'] = [AlgorithmConfig.from_dict(a) if isinstance(a, dict) else a for a in data['algorithms']]
        return cls(**data)


class MatchedRecord(BaseModel):
    """A single matched record with confidence score."""
    record1: Dict[str, Any] = Field(..., description="First record in the match")
    record2: Dict[str, Any] = Field(..., description="Second record in the match")
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence score of the match")
    matching_fields: List[str] = Field(default_factory=list, description="Fields that contributed to the match")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the match")
    created_at: datetime = Field(default_factory=datetime.now, description="When the match was created")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchedRecord':
        """Create instance from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class MatchingStatistics(BaseModel):
    """Statistics for a matching operation."""
    total_records_file1: int = Field(0, ge=0, description="Total records in first file")
    total_records_file2: int = Field(0, ge=0, description="Total records in second file")
    total_comparisons: int = Field(0, ge=0, description="Total number of comparisons performed")
    high_confidence_matches: int = Field(0, ge=0, description="Number of high confidence matches")
    low_confidence_matches: int = Field(0, ge=0, description="Number of low confidence matches")
    unmatched_file1: int = Field(0, ge=0, description="Unmatched records from first file")
    unmatched_file2: int = Field(0, ge=0, description="Unmatched records from second file")
    processing_time_seconds: float = Field(0.0, ge=0, description="Processing time in seconds")
    average_confidence: float = Field(0.0, ge=0, le=100, description="Average confidence score")
    
    @property
    def match_rate_file1(self) -> float:
        """Calculate match rate for file 1."""
        if self.total_records_file1 == 0:
            return 0.0
        return (self.high_confidence_matches + self.low_confidence_matches) / self.total_records_file1 * 100
    
    @property
    def match_rate_file2(self) -> float:
        """Calculate match rate for file 2."""
        if self.total_records_file2 == 0:
            return 0.0
        return (self.high_confidence_matches + self.low_confidence_matches) / self.total_records_file2 * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.model_dump()
        result['match_rate_file1'] = self.match_rate_file1
        result['match_rate_file2'] = self.match_rate_file2
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchingStatistics':
        """Create instance from dictionary."""
        # Remove computed properties if present
        data.pop('match_rate_file1', None)
        data.pop('match_rate_file2', None)
        return cls(**data)


class ResultMetadata(BaseModel):
    """Metadata for matching results."""
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique operation identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    config_hash: str = Field("", description="Hash of the configuration used")
    file1_metadata: Optional[DatasetMetadata] = Field(None, description="Metadata for first file")
    file2_metadata: Optional[DatasetMetadata] = Field(None, description="Metadata for second file")
    processing_node: str = Field("", description="Node that processed the operation")
    version: str = Field("1.0", description="Version of the processing system")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultMetadata':
        """Create instance from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'file1_metadata' in data and data['file1_metadata'] is not None:
            data['file1_metadata'] = DatasetMetadata.from_dict(data['file1_metadata'])
        if 'file2_metadata' in data and data['file2_metadata'] is not None:
            data['file2_metadata'] = DatasetMetadata.from_dict(data['file2_metadata'])
        return cls(**data)


class MatchingResult(BaseModel):
    """Results of a matching operation."""
    matched_records: List[MatchedRecord] = Field(default_factory=list, description="List of matched records")
    unmatched_records: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Unmatched records by file")
    statistics: MatchingStatistics = Field(default_factory=MatchingStatistics, description="Matching statistics")
    metadata: ResultMetadata = Field(default_factory=ResultMetadata, description="Result metadata")
    
    def add_matched_record(self, record: MatchedRecord):
        """Add a matched record to the results."""
        self.matched_records.append(record)
        
    def add_unmatched_record(self, file_key: str, record: Dict[str, Any]):
        """Add an unmatched record to the results."""
        if file_key not in self.unmatched_records:
            self.unmatched_records[file_key] = []
        self.unmatched_records[file_key].append(record)
    
    @property
    def total_matches(self) -> int:
        """Get total number of matches."""
        return len(self.matched_records)
    
    def get_high_confidence_matches(self, threshold: float = 75.0) -> List[MatchedRecord]:
        """Get high confidence matches."""
        return [record for record in self.matched_records if record.confidence_score >= threshold]
    
    def get_low_confidence_matches(self, threshold: float = 75.0) -> List[MatchedRecord]:
        """Get low confidence matches."""
        return [record for record in self.matched_records if record.confidence_score < threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchingResult':
        """Create instance from dictionary."""
        # Convert nested objects
        if 'matched_records' in data:
            data['matched_records'] = [MatchedRecord.from_dict(r) if isinstance(r, dict) else r for r in data['matched_records']]
        if 'statistics' in data:
            data['statistics'] = MatchingStatistics.from_dict(data['statistics']) if isinstance(data['statistics'], dict) else data['statistics']
        if 'metadata' in data:
            data['metadata'] = ResultMetadata.from_dict(data['metadata']) if isinstance(data['metadata'], dict) else data['metadata']
        return cls(**data)


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")
    
    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.model_dump()
        result['has_errors'] = self.has_errors
        result['has_warnings'] = self.has_warnings
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create instance from dictionary."""
        # Remove computed properties if present
        data.pop('has_errors', None)
        data.pop('has_warnings', None)
        return cls(**data)


class ProgressStatus(BaseModel):
    """Status of a long-running operation."""
    operation_id: str = Field(..., description="Unique operation identifier")
    status: str = Field(..., pattern="^(idle|running|completed|error|cancelled)$", description="Current operation status")
    progress: float = Field(..., description="Progress percentage")
    message: str = Field("", description="Current status message")
    current_step: int = Field(0, description="Current step number")
    total_steps: int = Field(0, ge=0, description="Total number of steps")
    started_at: Optional[datetime] = Field(None, description="When operation started")
    completed_at: Optional[datetime] = Field(None, description="When operation completed")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    
    @property
    def is_running(self) -> bool:
        """Check if operation is currently running."""
        return self.status == 'running'
    
    @property
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.status in ['completed', 'error', 'cancelled']
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = self.model_dump()
        result['is_running'] = self.is_running
        result['is_completed'] = self.is_completed
        result['duration_seconds'] = self.duration_seconds
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressStatus':
        """Create instance from dictionary."""
        # Handle datetime strings
        if 'started_at' in data and isinstance(data['started_at'], str):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        # Remove computed properties if present
        data.pop('is_running', None)
        data.pop('is_completed', None)
        data.pop('duration_seconds', None)
        return cls(**data)