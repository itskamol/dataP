"""
Web application data models and validation schemas.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class FileUpload:
    """Model for uploaded file information."""
    original_name: str
    unique_name: str
    file_path: str
    file_type: str  # 'csv' or 'json'
    delimiter: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_name': self.original_name,
            'unique_name': self.unique_name,
            'file_path': self.file_path,
            'file_type': self.file_type,
            'delimiter': self.delimiter,
            'upload_timestamp': self.upload_timestamp.isoformat() if self.upload_timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileUpload':
        """Create instance from dictionary."""
        upload_timestamp = None
        if data.get('upload_timestamp'):
            upload_timestamp = datetime.fromisoformat(data['upload_timestamp'])
        
        return cls(
            original_name=data['original_name'],
            unique_name=data['unique_name'],
            file_path=data['file_path'],
            file_type=data['file_type'],
            delimiter=data.get('delimiter'),
            upload_timestamp=upload_timestamp
        )


@dataclass
class FieldMapping:
    """Model for field mapping configuration."""
    file1_col: str
    file2_col: str
    match_type: str = 'exact'
    use_normalization: bool = False
    case_sensitive: bool = False
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file1_col': self.file1_col,
            'file2_col': self.file2_col,
            'match_type': self.match_type,
            'use_normalization': self.use_normalization,
            'case_sensitive': self.case_sensitive,
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldMapping':
        """Create instance from dictionary."""
        return cls(
            file1_col=data['file1_col'],
            file2_col=data['file2_col'],
            match_type=data.get('match_type', 'exact'),
            use_normalization=data.get('use_normalization', False),
            case_sensitive=data.get('case_sensitive', False),
            weight=data.get('weight', 1.0)
        )


@dataclass
class ProcessingConfig:
    """Model for processing configuration."""
    file1: FileUpload
    file2: FileUpload
    mappings: List[FieldMapping]
    output_cols1: List[str]
    output_cols2: List[str]
    output_format: str = 'json'
    output_path: str = 'matched_results'
    prefix1: str = 'f1_'
    prefix2: str = 'f2_'
    threshold: int = 75
    matching_type: str = 'one-to-one'
    generate_unmatched: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file1': self.file1.to_dict(),
            'file2': self.file2.to_dict(),
            'mappings': [mapping.to_dict() for mapping in self.mappings],
            'output_cols1': self.output_cols1,
            'output_cols2': self.output_cols2,
            'output_format': self.output_format,
            'output_path': self.output_path,
            'prefix1': self.prefix1,
            'prefix2': self.prefix2,
            'threshold': self.threshold,
            'matching_type': self.matching_type,
            'generate_unmatched': self.generate_unmatched
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create instance from dictionary."""
        return cls(
            file1=FileUpload.from_dict(data['file1']),
            file2=FileUpload.from_dict(data['file2']),
            mappings=[FieldMapping.from_dict(m) for m in data['mappings']],
            output_cols1=data['output_cols1'],
            output_cols2=data['output_cols2'],
            output_format=data.get('output_format', 'json'),
            output_path=data.get('output_path', 'matched_results'),
            prefix1=data.get('prefix1', 'f1_'),
            prefix2=data.get('prefix2', 'f2_'),
            threshold=data.get('threshold', 75),
            matching_type=data.get('matching_type', 'one-to-one'),
            generate_unmatched=data.get('generate_unmatched', False)
        )


@dataclass
class ProgressStatus:
    """Model for progress tracking."""
    status: str = 'idle'  # idle, starting, processing, completed, error
    progress: int = 0  # 0-100
    message: str = ''
    error: Optional[str] = None
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'error': self.error,
            'completed': self.completed
        }
    
    def update(self, status: str = None, progress: int = None, message: str = None, error: str = None):
        """Update progress status."""
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        if error is not None:
            self.error = error
        self.completed = status == 'completed'


@dataclass
class ResultFile:
    """Model for result file information."""
    name: str
    path: str
    file_type: str  # matched, low_confidence, unmatched_1, unmatched_2
    count: int = 0
    columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'path': self.path,
            'type': self.file_type,
            'count': self.count,
            'columns': self.columns
        }


@dataclass
class SessionData:
    """Model for session data management."""
    file1: Optional[FileUpload] = None
    file2: Optional[FileUpload] = None
    config: Optional[ProcessingConfig] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file1': self.file1.to_dict() if self.file1 else None,
            'file2': self.file2.to_dict() if self.file2 else None,
            'config': self.config.to_dict() if self.config else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create instance from dictionary."""
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        return cls(
            file1=FileUpload.from_dict(data['file1']) if data.get('file1') else None,
            file2=FileUpload.from_dict(data['file2']) if data.get('file2') else None,
            config=ProcessingConfig.from_dict(data['config']) if data.get('config') else None,
            created_at=created_at
        )