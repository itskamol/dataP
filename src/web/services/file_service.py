"""
File handling service for web application.
"""
import os
import json
import uuid
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from src.web.models.web_models import FileUpload, SessionData
from src.domain.exceptions import FileProcessingError, FileValidationError


class FileService:
    """Service for handling file uploads and validation."""
    
    ALLOWED_EXTENSIONS = {'csv', 'json'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    def __init__(self, upload_folder: str):
        """Initialize file service with upload folder."""
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS)
    
    def validate_file_size(self, file: FileStorage) -> None:
        """Validate file size."""
        if hasattr(file, 'content_length') and file.content_length:
            if file.content_length > self.MAX_FILE_SIZE:
                raise FileValidationError(
                    f"File size ({file.content_length} bytes) exceeds maximum allowed size ({self.MAX_FILE_SIZE} bytes)"
                )
    
    def detect_csv_delimiter(self, file_path: str) -> Optional[str]:
        """Detect CSV delimiter automatically."""
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                # Try to read first few rows with this delimiter
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=5)
                # If successful and has more than one column, likely correct
                if len(df.columns) > 1:
                    return delimiter
            except Exception:
                continue
        
        # Default to comma if detection fails
        return ','
    
    def save_uploaded_file(self, file: FileStorage) -> FileUpload:
        """Save uploaded file and return FileUpload model."""
        if not file or not file.filename:
            raise FileValidationError("No file provided")
        
        if not self.is_allowed_file(file.filename):
            raise FileValidationError(
                f"File type not allowed. Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        
        self.validate_file_size(file)
        
        # Generate unique filename
        original_filename = file.filename
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        base_name = secure_filename(original_filename.rsplit('.', 1)[0])
        unique_id = str(uuid.uuid4())
        unique_filename = f"{base_name}_{unique_id}.{file_extension}"
        
        # Save file
        file_path = os.path.join(self.upload_folder, unique_filename)
        file.save(file_path)
        
        # Detect delimiter for CSV files
        delimiter = None
        if file_extension == 'csv':
            delimiter = self.detect_csv_delimiter(file_path)
        
        return FileUpload(
            original_name=original_filename,
            unique_name=unique_filename,
            file_path=file_path,
            file_type=file_extension,
            delimiter=delimiter
        )
    
    def load_dataframe(self, file_upload: FileUpload) -> pd.DataFrame:
        """Load dataframe from FileUpload model."""
        try:
            if file_upload.file_type == 'csv':
                delimiter = file_upload.delimiter or ','
                return pd.read_csv(file_upload.file_path, delimiter=delimiter)
            elif file_upload.file_type == 'json':
                return pd.read_json(file_upload.file_path)
            else:
                raise FileValidationError(f"Unsupported file type: {file_upload.file_type}")
        except Exception as e:
            raise FileProcessingError(f"Error loading file {file_upload.original_name}: {str(e)}")
    
    def get_file_preview(self, file_upload: FileUpload, rows: int = 5) -> Dict[str, Any]:
        """Get file preview with column information."""
        try:
            df = self.load_dataframe(file_upload)
            
            # Convert preview data to ensure JSON serialization
            preview_data = df.head(rows).to_dict('records')
            preview_data = self._convert_numpy_types(preview_data)
            
            return {
                'columns': df.columns.tolist(),
                'preview': preview_data,
                'total_rows': int(len(df)),  # Convert to native Python int
                'file_info': {
                    'name': file_upload.original_name,
                    'type': file_upload.file_type,
                    'delimiter': file_upload.delimiter
                }
            }
        except Exception as e:
            raise FileProcessingError(f"Error getting preview for {file_upload.original_name}: {str(e)}")
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def validate_file_structure(self, file_upload: FileUpload) -> Dict[str, Any]:
        """Validate file structure and return validation results."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            df = self.load_dataframe(file_upload)
            
            # Basic validation
            if df.empty:
                validation_result['errors'].append("File is empty")
                validation_result['valid'] = False
            
            if len(df.columns) == 0:
                validation_result['errors'].append("No columns found")
                validation_result['valid'] = False
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                validation_result['warnings'].append("Duplicate column names found")
            
            # Check for completely empty columns
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                validation_result['warnings'].append(f"Empty columns found: {', '.join(empty_cols)}")
            
            # Add file info - convert numpy/pandas types to native Python types
            validation_result['info'] = {
                'rows': int(len(df)),  # Convert to native Python int
                'columns': int(len(df.columns)),  # Convert to native Python int
                'column_names': df.columns.tolist(),
                'memory_usage': int(df.memory_usage(deep=True).sum())  # Convert to native Python int
            }
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File validation failed: {str(e)}")
        
        return validation_result
    
    def cleanup_file(self, file_upload: FileUpload) -> bool:
        """Clean up uploaded file."""
        try:
            if os.path.exists(file_upload.file_path):
                os.remove(file_upload.file_path)
                return True
        except Exception as e:
            print(f"Error cleaning up file {file_upload.file_path}: {e}")
        return False
    
    def cleanup_files(self, file_uploads: List[FileUpload]) -> int:
        """Clean up multiple files and return count of successfully removed files."""
        removed_count = 0
        for file_upload in file_uploads:
            if self.cleanup_file(file_upload):
                removed_count += 1
        return removed_count


class SessionService:
    """Service for managing session data."""
    
    def __init__(self, session_file: str = 'session.json'):
        """Initialize session service."""
        self.session_file = session_file
    
    def save_session(self, session_data: SessionData) -> None:
        """Save session data to file."""
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise FileProcessingError(f"Error saving session data: {str(e)}")
    
    def load_session(self) -> Optional[SessionData]:
        """Load session data from file."""
        try:
            if not os.path.exists(self.session_file):
                return None
            
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return SessionData.from_dict(data)
        except Exception as e:
            print(f"Error loading session data: {e}")
            return None
    
    def clear_session(self) -> None:
        """Clear session data."""
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
        except Exception as e:
            print(f"Error clearing session: {e}")
    
    def session_exists(self) -> bool:
        """Check if session file exists."""
        return os.path.exists(self.session_file)