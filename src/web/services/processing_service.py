"""
Processing service for web application.
"""
import os
import json
import threading
import time
from typing import Dict, Any, Callable, Optional, List
import pandas as pd

from src.web.models.web_models import ProcessingConfig, ProgressStatus, ResultFile
from src.domain.exceptions import FileProcessingError
# Import will be handled dynamically to avoid circular imports


class ProcessingService:
    """Service for handling file processing operations."""
    
    def __init__(self):
        """Initialize processing service."""
        self.progress_status = ProgressStatus()
        self.processing_thread: Optional[threading.Thread] = None
        self.config_file = 'web_config.json'
    
    def get_progress(self) -> ProgressStatus:
        """Get current progress status."""
        return self.progress_status
    
    def update_progress(self, status: str, progress: int, message: str, error: str = None):
        """Update progress status."""
        self.progress_status.update(status=status, progress=progress, message=message, error=error)
    
    def reset_progress(self):
        """Reset progress to initial state."""
        self.progress_status = ProgressStatus()
    
    def create_processing_config(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Create processing configuration dictionary."""
        # Agar output_cols bo'sh bo'lsa, barcha maydonlarni olish
        output_cols1 = config.output_cols1
        output_cols2 = config.output_cols2
        
        # Agar output_cols1 bo'sh bo'lsa, file1 dan barcha maydonlarni olish
        if not output_cols1:
            try:
                from .file_service import FileService
                file_service = FileService(upload_folder='uploads')
                validation1 = file_service.validate_file_structure(config.file1)
                if validation1.get('valid') and validation1.get('info', {}).get('column_names'):
                    output_cols1 = validation1['info']['column_names']
            except Exception:
                output_cols1 = []
        
        # Agar output_cols2 bo'sh bo'lsa, file2 dan barcha maydonlarni olish
        if not output_cols2:
            try:
                from .file_service import FileService
                file_service = FileService(upload_folder='uploads')
                validation2 = file_service.validate_file_structure(config.file2)
                if validation2.get('valid') and validation2.get('info', {}).get('column_names'):
                    output_cols2 = validation2['info']['column_names']
            except Exception:
                output_cols2 = []
        
        return {
            "file1": {
                "path": config.file1.file_path,
                "type": config.file1.file_type,
                "delimiter": config.file1.delimiter
            },
            "file2": {
                "path": config.file2.file_path,
                "type": config.file2.file_type,
                "delimiter": config.file2.delimiter
            },
            "mapping_fields": [mapping.to_dict() for mapping in config.mappings],
            "output_columns": {
                "from_file1": output_cols1,
                "from_file2": output_cols2
            },
            "settings": {
                "output_format": config.output_format,
                "matched_output_path": config.output_path,
                "file1_output_prefix": config.prefix1,
                "file2_output_prefix": config.prefix2,
                "confidence_threshold": config.threshold,
                "matching_type": config.matching_type,
                "unmatched_files": {
                    "generate": config.generate_unmatched
                }
            }
        }
    
    def save_config(self, config: ProcessingConfig) -> None:
        """Save processing configuration to file."""
        try:
            processing_config = self.create_processing_config(config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(processing_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise FileProcessingError(f"Error saving processing configuration: {str(e)}")
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load processing configuration from file."""
        try:
            if not os.path.exists(self.config_file):
                return None
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processing configuration: {e}")
            return None
    
    def _run_processing_with_progress(self, config_dict: Dict[str, Any]):
        """Run processing in background thread with progress updates."""
        try:
            self.update_progress('starting', 0, 'Initializing processing...')
            
            # Create progress callback
            def progress_callback(status: str, progress: int, message: str):
                self.update_progress(status, progress, message)
            
            self.update_progress('processing', 10, 'Loading processing engine...')
            
            # Dynamic import to avoid circular imports
            try:
                # Try to import from the refactored main
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from main import MainApplication
                
                app = MainApplication()
                app.run_processing_optimized(config_dict, progress_callback=progress_callback)
                
            except ImportError:
                # Fallback to legacy processing if available
                try:
                    from main import run_processing_optimized
                    run_processing_optimized(config_dict, progress_callback=progress_callback)
                except ImportError:
                    raise FileProcessingError("No processing engine available")
            
            self.update_progress('completed', 100, 'Processing completed successfully!')
            
        except Exception as e:
            import traceback
            error_msg = f'Processing failed: {str(e)}'
            self.update_progress('error', 0, error_msg, str(e))
            print(f"Processing error: {traceback.format_exc()}")
    
    def start_processing(self, config: ProcessingConfig) -> bool:
        """Start processing in background thread."""
        try:
            # Check if already processing
            if (self.processing_thread and 
                self.processing_thread.is_alive() and 
                self.progress_status.status in ['starting', 'processing']):
                return False
            
            # Reset progress
            self.reset_progress()
            
            # Save configuration
            self.save_config(config)
            
            # Create processing configuration
            config_dict = self.create_processing_config(config)
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._run_processing_with_progress,
                args=(config_dict,)
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            return True
            
        except Exception as e:
            self.update_progress('error', 0, f'Error starting processing: {str(e)}', str(e))
            return False
    
    def is_processing(self) -> bool:
        """Check if processing is currently running."""
        return (self.processing_thread and 
                self.processing_thread.is_alive() and 
                self.progress_status.status in ['starting', 'processing'])
    
    def get_result_files(self) -> List[ResultFile]:
        """Get information about result files."""
        result_files = []
        
        try:
            config = self.load_config()
            if not config:
                return result_files
            
            output_path = config['settings']['matched_output_path']
            output_format = config['settings']['output_format']
            
            # Agar output_path relative bo'lsa, absolute path yaratish
            import os
            if not os.path.isabs(output_path):
                # Project root directory-ga nisbatan path yaratish
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                output_path = os.path.join(project_root, output_path)
            
            # Define result file types and their display names
            file_types = [
                ('matched', 'Matched Results', f"{output_path}.{output_format}"),
                ('low_confidence', 'Low Confidence Matches', f"{output_path}_low_confidence.{output_format}"),
                ('unmatched_1', 'Unmatched from File 1', f"{output_path}_unmatched_1.{output_format}"),
                ('unmatched_2', 'Unmatched from File 2', f"{output_path}_unmatched_2.{output_format}")
            ]
            
            for file_type, display_name, file_path in file_types:
                if os.path.exists(file_path):
                    try:
                        # Get file info efficiently
                        count, columns = self._get_file_info(file_path, output_format)
                        
                        result_files.append(ResultFile(
                            name=display_name,
                            path=file_path,
                            file_type=file_type,
                            count=count,
                            columns=columns
                        ))
                        
                    except Exception as e:
                        print(f"Error getting info for {file_path}: {e}")
                        # Add file with minimal info
                        result_files.append(ResultFile(
                            name=display_name,
                            path=file_path,
                            file_type=file_type,
                            count=0,
                            columns=[]
                        ))
            
        except Exception as e:
            print(f"Error getting result files: {e}")
        
        return result_files
    
    def _get_file_info(self, file_path: str, output_format: str) -> tuple[int, List[str]]:
        """Get file row count and column information efficiently."""
        try:
            if output_format == 'json':
                # For JSON files, read sample to get columns
                try:
                    # Try JSON Lines format first
                    df_sample = pd.read_json(file_path, lines=True, nrows=1)
                    columns = df_sample.columns.tolist() if not df_sample.empty else []
                except:
                    try:
                        # Try regular JSON format (without nrows)
                        df_sample = pd.read_json(file_path)
                        columns = df_sample.columns.tolist() if not df_sample.empty else []
                    except:
                        columns = []
                
                # Count total rows efficiently
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.jsonl'):
                        count = sum(1 for _ in f)
                    else:
                        data = json.load(f)
                        count = len(data) if isinstance(data, list) else 1
            else:
                # For CSV files
                df_sample = pd.read_csv(file_path, nrows=1)
                columns = df_sample.columns.tolist() if not df_sample.empty else []
                
                # Count rows efficiently (subtract 1 for header)
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f) - 1
            
            return max(0, count), columns
            
        except Exception as e:
            print(f"Error getting file info for {file_path}: {e}")
            return 0, []
    
    def get_paginated_data(self, file_type: str, page: int = 1, per_page: int = 50, 
                          search: str = '', confidence_threshold: float = 80.0) -> Dict[str, Any]:
        """Get paginated data from result files."""
        try:
            config = self.load_config()
            if not config:
                return {'data': [], 'total': 0, 'page': page, 'per_page': per_page}
            
            output_path = config['settings']['matched_output_path']
            output_format = config['settings']['output_format']
            
            # Agar output_path relative bo'lsa, absolute path yaratish
            import os
            if not os.path.isabs(output_path):
                # Project root directory-ga nisbatan path yaratish
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                output_path = os.path.join(project_root, output_path)
            
            # Determine file path based on type
            file_paths = {
                'matched': f"{output_path}.{output_format}",
                'low_confidence': f"{output_path}_low_confidence.{output_format}",
                'unmatched_1': f"{output_path}_unmatched_1.{output_format}",
                'unmatched_2': f"{output_path}_unmatched_2.{output_format}"
            }
            
            file_path = file_paths.get(file_type)
            if not file_path or not os.path.exists(file_path):
                return {'data': [], 'total': 0, 'page': page, 'per_page': per_page}
            
            # Load data efficiently with pagination
            if output_format == 'json':
                try:
                    # Try JSON Lines format first
                    df = pd.read_json(file_path, lines=True)
                except:
                    try:
                        # Try regular JSON format
                        df = pd.read_json(file_path)
                    except:
                        # Fallback: read as text and parse manually
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Handle different JSON structures
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict):
                            # Check if it's a result file with matched_records
                            if 'matched_records' in data:
                                # Extract matched_records and flatten them
                                matched_records = data['matched_records']
                                flattened_records = []
                                
                                for record in matched_records:
                                    if isinstance(record, dict):
                                        # Create flat record
                                        flat_record = {}
                                        
                                        # Add confidence score (rounded to 2 decimal places)
                                        if 'confidence_score' in record:
                                            confidence = record['confidence_score']
                                            if isinstance(confidence, (int, float)):
                                                flat_record['confidence_score'] = round(confidence, 2)
                                            else:
                                                flat_record['confidence_score'] = confidence
                                        
                                        # Add record1 fields with prefix
                                        if 'record1' in record and isinstance(record['record1'], dict):
                                            for key, value in record['record1'].items():
                                                flat_record[f'f1_{key}'] = value
                                        
                                        # Add record2 fields with prefix
                                        if 'record2' in record and isinstance(record['record2'], dict):
                                            for key, value in record['record2'].items():
                                                flat_record[f'f2_{key}'] = value
                                        
                                        # Add matching fields info
                                        if 'matching_fields' in record:
                                            flat_record['matching_fields'] = ', '.join(record['matching_fields'])
                                        
                                        # Add metadata algorithm info
                                        if 'metadata' in record and isinstance(record['metadata'], dict):
                                            if 'algorithm' in record['metadata']:
                                                flat_record['algorithm'] = record['metadata']['algorithm']
                                        
                                        flattened_records.append(flat_record)
                                
                                df = pd.DataFrame(flattened_records)
                            else:
                                # Single dict, wrap in list
                                df = pd.DataFrame([data])
                        else:
                            df = pd.DataFrame()
            else:
                df = pd.read_csv(file_path)
            
            # Apply search filter if provided
            if search:
                # Search across all string columns
                string_cols = df.select_dtypes(include=['object']).columns
                mask = df[string_cols].astype(str).apply(
                    lambda x: x.str.contains(search, case=False, na=False)
                ).any(axis=1)
                df = df[mask]
            
            # Sort by confidence score
            if 'confidence_score' in df.columns:
                df = self._sort_by_confidence_score(df, confidence_threshold)
            
            total = len(df)
            
            # Apply pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_df = df.iloc[start_idx:end_idx]
            
            return {
                'data': paginated_df.to_dict('records'),
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': (total + per_page - 1) // per_page
            }
            
        except Exception as e:
            print(f"Error getting paginated data: {e}")
            return {'data': [], 'total': 0, 'page': page, 'per_page': per_page}
    
    def _sort_by_confidence_score(self, df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
        """
        Sort DataFrame by confidence score.
        - High confidence (>=threshold): ascending order (o'sish tartibida)
        - Low confidence (<threshold): descending order (kamayish tartibida)
        """
        try:
            # Confidence score columnini numeric qilish
            df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
            
            # NaN qiymatlarni 0 bilan almashtirish
            df['confidence_score'] = df['confidence_score'].fillna(0.0)
            
            # High va low confidence recordlarni ajratish
            high_confidence = df[df['confidence_score'] >= threshold].copy()
            low_confidence = df[df['confidence_score'] < threshold].copy()
            
            # High confidence - ascending (o'sish tartibida)
            high_confidence = high_confidence.sort_values('confidence_score', ascending=True)
            
            # Low confidence - descending (kamayish tartibida)
            low_confidence = low_confidence.sort_values('confidence_score', ascending=False)
            
            # Birlashtirib qaytarish: avval high confidence, keyin low confidence
            result = pd.concat([high_confidence, low_confidence], ignore_index=True)
            
            return result
            
        except Exception as e:
            print(f"Error sorting by confidence score: {e}")
            # Xatolik bo'lsa, asl DataFrame ni qaytarish
            return df
    
    def cleanup_results(self) -> int:
        """Clean up result files and return count of removed files."""
        removed_count = 0
        
        try:
            config = self.load_config()
            if config:
                output_path = config['settings']['matched_output_path']
                output_format = config['settings']['output_format']
                
                # Agar output_path relative bo'lsa, absolute path yaratish
                import os
                if not os.path.isabs(output_path):
                    # Project root directory-ga nisbatan path yaratish
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                    output_path = os.path.join(project_root, output_path)
                
                result_files = [
                    f"{output_path}.{output_format}",
                    f"{output_path}_low_confidence.{output_format}",
                    f"{output_path}_unmatched_1.{output_format}",
                    f"{output_path}_unmatched_2.{output_format}"
                ]
                
                for file_path in result_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            removed_count += 1
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
            
            # Clean up config file
            try:
                if os.path.exists(self.config_file):
                    os.remove(self.config_file)
            except Exception as e:
                print(f"Error removing config file: {e}")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return removed_count