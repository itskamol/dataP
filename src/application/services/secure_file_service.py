"""
Secure file processing service that integrates data protection and privacy features.
Extends the base FileProcessingService with PII detection, anonymization, and audit logging.
Implements requirements 7.2, 7.4, 2.1: Data protection, privacy compliance, and audit logging.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import os
from pathlib import Path
import pandas as pd

from .file_service import FileProcessingService
from ...domain.models import Dataset, DatasetMetadata
from ...domain.exceptions import FileProcessingError, SecurityError
from ...infrastructure.data_protection import (
    DataProtectionService, DataProtectionConfig, PIIType, AnonymizationMethod
)
from ...infrastructure.logging import get_logger


class SecureFileProcessingService(FileProcessingService):
    """
    Secure file processing service with integrated data protection features.
    
    This service extends the base FileProcessingService to include:
    - PII detection and anonymization
    - Audit logging for all file operations
    - Secure data disposal
    - Data retention policy enforcement
    """
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024, 
                 chunk_size: int = 10000,
                 data_protection_config: Optional[DataProtectionConfig] = None):
        """
        Initialize secure file processing service.
        
        Args:
            max_file_size: Maximum allowed file size in bytes
            chunk_size: Default chunk size for streaming operations
            data_protection_config: Configuration for data protection features
        """
        super().__init__(max_file_size, chunk_size)
        self.logger = get_logger('secure_file_processing')
        
        # Initialize data protection service
        self.data_protection = DataProtectionService(
            data_protection_config or DataProtectionConfig()
        )
        
        # Track processed files for audit purposes
        self.processed_files: Dict[str, Dict[str, Any]] = {}
    
    def load_file_secure(self, file_path: str, user_id: str, 
                        config: Optional[Dict[str, Any]] = None,
                        anonymize_pii: bool = True) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Load file with integrated data protection features.
        
        Args:
            file_path: Path to the file to load
            user_id: ID of the user performing the operation
            config: Optional configuration for file loading
            anonymize_pii: Whether to automatically anonymize detected PII
            
        Returns:
            Tuple of (Dataset with protected data, protection report)
            
        Raises:
            FileProcessingError: If loading fails
            SecurityError: If data protection fails
        """
        try:
            # Log data access attempt
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='load_secure',
                    success=False  # Will update to True if successful
                )
            
            # Load file using parent class method
            dataset = self.load_file(file_path, config)
            
            # Apply data protection
            protected_df, protection_report = self.data_protection.scan_and_protect_dataframe(
                df=dataset.data,
                file_path=file_path,
                user_id=user_id,
                anonymize=anonymize_pii
            )
            
            # Update dataset with protected data
            dataset.data = protected_df
            
            # Update metadata with protection information
            dataset.metadata.additional_info.update({
                'pii_detected': protection_report.get('pii_detected', False),
                'anonymized': protection_report.get('anonymized', False),
                'protection_processing_time': protection_report.get('processing_time', 0)
            })
            
            # Track processed file
            self.processed_files[file_path] = {
                'user_id': user_id,
                'dataset_name': dataset.name,
                'protection_report': protection_report,
                'processed_at': pd.Timestamp.now()
            }
            
            # Log successful data access
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='load_secure',
                    success=True,
                    details={
                        'rows_processed': len(protected_df),
                        'pii_detected': protection_report.get('pii_detected', False),
                        'anonymized': protection_report.get('anonymized', False)
                    }
                )
            
            self.logger.info(f"Secure file loading completed", extra={
                'file_path': file_path,
                'user_id': user_id,
                'rows': len(protected_df),
                'pii_detected': protection_report.get('pii_detected', False),
                'anonymized': protection_report.get('anonymized', False)
            })
            
            return dataset, protection_report
            
        except Exception as e:
            # Log failed data access
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='load_secure',
                    success=False,
                    details={'error': str(e)}
                )
            
            self.logger.error(f"Secure file loading failed: {str(e)}", extra={
                'file_path': file_path,
                'user_id': user_id
            })
            
            if isinstance(e, SecurityError):
                raise
            else:
                raise FileProcessingError(f"Secure file loading failed: {str(e)}")
    
    def save_results_secure(self, df: pd.DataFrame, output_path: str, user_id: str,
                           format_type: str = 'csv', anonymize_before_export: bool = True,
                           **kwargs) -> Tuple[List[str], Dict[str, Any]]:
        """
        Save results with data protection and audit logging.
        
        Args:
            df: DataFrame to save
            output_path: Base path for output files
            user_id: ID of the user performing the operation
            format_type: Output format ('csv', 'json', 'excel', 'both')
            anonymize_before_export: Whether to anonymize data before export
            **kwargs: Additional format-specific options
            
        Returns:
            Tuple of (list of created file paths, protection report)
            
        Raises:
            FileProcessingError: If saving fails
            SecurityError: If data protection fails
        """
        try:
            protection_report = {}
            export_df = df.copy()
            
            # Apply data protection if requested
            if anonymize_before_export:
                export_df, protection_report = self.data_protection.scan_and_protect_dataframe(
                    df=df,
                    file_path=output_path,
                    user_id=user_id,
                    anonymize=True
                )
            
            # Save using parent class method
            created_files = self.save_results(export_df, output_path, format_type, **kwargs)
            
            # Log data export for each created file
            for file_path in created_files:
                if self.data_protection.audit_logger:
                    self.data_protection.audit_logger.log_data_export(
                        user_id=user_id,
                        source_file=output_path,
                        export_file=file_path,
                        format_type=format_type,
                        anonymized=anonymize_before_export
                    )
            
            self.logger.info(f"Secure results export completed", extra={
                'output_files': created_files,
                'user_id': user_id,
                'rows': len(export_df),
                'anonymized': anonymize_before_export,
                'format': format_type
            })
            
            return created_files, protection_report
            
        except Exception as e:
            self.logger.error(f"Secure results export failed: {str(e)}", extra={
                'output_path': output_path,
                'user_id': user_id,
                'format': format_type
            })
            
            if isinstance(e, SecurityError):
                raise
            else:
                raise FileProcessingError(f"Secure results export failed: {str(e)}")
    
    def process_file_streaming_secure(self, file_path: str, processor_func, user_id: str,
                                     chunk_size: Optional[int] = None,
                                     anonymize_chunks: bool = True) -> Any:
        """
        Process file using streaming with data protection applied to each chunk.
        
        Args:
            file_path: Path to the file to process
            processor_func: Function to process each chunk (takes DataFrame, returns Any)
            user_id: ID of the user performing the operation
            chunk_size: Size of each chunk
            anonymize_chunks: Whether to anonymize each chunk before processing
            
        Returns:
            Result from processor function
            
        Raises:
            FileProcessingError: If processing fails
            SecurityError: If data protection fails
        """
        try:
            # Log data processing start
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='stream_process_secure',
                    success=False
                )
            
            def secure_processor_wrapper(chunk_df: pd.DataFrame) -> Any:
                """Wrapper that applies data protection to each chunk."""
                if anonymize_chunks:
                    protected_chunk, _ = self.data_protection.scan_and_protect_dataframe(
                        df=chunk_df,
                        file_path=file_path,
                        user_id=user_id,
                        anonymize=True
                    )
                    return processor_func(protected_chunk)
                else:
                    return processor_func(chunk_df)
            
            # Process using parent class method with secure wrapper
            results = self.process_file_streaming(file_path, secure_processor_wrapper, chunk_size)
            
            # Log successful processing
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='stream_process_secure',
                    success=True,
                    details={'chunks_processed': len(results) if isinstance(results, list) else 1}
                )
            
            return results
            
        except Exception as e:
            # Log failed processing
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='stream_process_secure',
                    success=False,
                    details={'error': str(e)}
                )
            
            if isinstance(e, SecurityError):
                raise
            else:
                raise FileProcessingError(f"Secure streaming processing failed: {str(e)}")
    
    def secure_file_disposal(self, file_path: str, user_id: str, 
                           disposal_method: str = 'secure_delete') -> bool:
        """
        Securely dispose of sensitive files with audit logging.
        
        Args:
            file_path: Path to file to dispose
            user_id: ID of the user requesting disposal
            disposal_method: Method of disposal ('secure_delete', 'overwrite', 'shred')
            
        Returns:
            True if disposal was successful
        """
        try:
            success = self.data_protection.secure_data_disposal(
                file_path=file_path,
                user_id=user_id,
                method=disposal_method
            )
            
            # Remove from processed files tracking
            if file_path in self.processed_files:
                del self.processed_files[file_path]
            
            self.logger.info(f"Secure file disposal completed", extra={
                'file_path': file_path,
                'user_id': user_id,
                'method': disposal_method,
                'success': success
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Secure file disposal failed: {str(e)}", extra={
                'file_path': file_path,
                'user_id': user_id,
                'method': disposal_method
            })
            return False
    
    def cleanup_expired_files(self, user_id: str = 'system', dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up expired files based on data retention policies.
        
        Args:
            user_id: ID of the user performing cleanup (default: 'system')
            dry_run: If True, only report what would be cleaned up
            
        Returns:
            Cleanup report with details of actions taken
        """
        try:
            cleanup_report = self.data_protection.cleanup_expired_data(dry_run=dry_run)
            
            self.logger.info(f"File cleanup completed", extra={
                'user_id': user_id,
                'dry_run': dry_run,
                'files_checked': cleanup_report.get('total_files_checked', 0),
                'expired_found': cleanup_report.get('expired_files_found', 0)
            })
            
            return cleanup_report
            
        except Exception as e:
            self.logger.error(f"File cleanup failed: {str(e)}", extra={
                'user_id': user_id,
                'dry_run': dry_run
            })
            return {'error': str(e)}
    
    def add_custom_pii_detection_rule(self, pattern: str, pii_type: PIIType, 
                                     confidence: float = 0.8, 
                                     description: str = "") -> bool:
        """
        Add custom PII detection pattern.
        
        Args:
            pattern: Regular expression pattern for PII detection
            pii_type: Type of PII this pattern detects
            confidence: Confidence score for this pattern (0.0-1.0)
            description: Description of what this pattern detects
            
        Returns:
            True if pattern was added successfully
        """
        try:
            self.data_protection.add_custom_pii_pattern(
                pattern=pattern,
                pii_type=pii_type,
                confidence=confidence,
                description=description
            )
            
            self.logger.info(f"Custom PII pattern added", extra={
                'pii_type': pii_type.value,
                'confidence': confidence,
                'description': description
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add custom PII pattern: {str(e)}")
            return False
    
    def create_anonymization_policy(self, pii_types: List[PIIType], 
                                   method: AnonymizationMethod,
                                   parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create custom anonymization policy.
        
        Args:
            pii_types: List of PII types this policy applies to
            method: Anonymization method to use
            parameters: Method-specific parameters
            
        Returns:
            True if policy was created successfully
        """
        try:
            self.data_protection.create_anonymization_policy(
                pii_types=pii_types,
                method=method,
                parameters=parameters
            )
            
            self.logger.info(f"Anonymization policy created", extra={
                'pii_types': [t.value for t in pii_types],
                'method': method.value,
                'parameters': parameters
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create anonymization policy: {str(e)}")
            return False
    
    def get_data_protection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive data protection statistics.
        
        Returns:
            Dictionary containing protection statistics
        """
        try:
            stats = self.data_protection.get_protection_statistics()
            
            # Add file processing statistics
            stats['file_processing'] = {
                'total_files_processed': len(self.processed_files),
                'files_with_pii': sum(1 for f in self.processed_files.values() 
                                    if f.get('protection_report', {}).get('pii_detected', False)),
                'files_anonymized': sum(1 for f in self.processed_files.values() 
                                      if f.get('protection_report', {}).get('anonymized', False))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get protection statistics: {str(e)}")
            return {'error': str(e)}
    
    def validate_file_secure(self, file_path: str, user_id: str) -> Dict[str, Any]:
        """
        Validate file with security checks and audit logging.
        
        Args:
            file_path: Path to file to validate
            user_id: ID of the user performing validation
            
        Returns:
            Validation result with security information
        """
        try:
            # Log validation attempt
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='validate_secure',
                    success=False
                )
            
            # Perform standard validation
            validation_result = self.validate_file(file_path)
            
            # Add security-specific validation
            security_checks = {
                'file_size_check': validation_result.metadata.get('file_size', 0) <= self.max_file_size,
                'format_check': Path(file_path).suffix.lower() in self.supported_formats,
                'encoding_check': validation_result.metadata.get('encoding', 'utf-8') in self.supported_encodings
            }
            
            # Combine results
            result = {
                'is_valid': validation_result.is_valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'metadata': validation_result.metadata,
                'security_checks': security_checks,
                'all_security_checks_passed': all(security_checks.values())
            }
            
            # Log successful validation
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='validate_secure',
                    success=True,
                    details={
                        'is_valid': result['is_valid'],
                        'security_passed': result['all_security_checks_passed']
                    }
                )
            
            return result
            
        except Exception as e:
            # Log failed validation
            if self.data_protection.audit_logger:
                self.data_protection.audit_logger.log_data_access(
                    user_id=user_id,
                    file_path=file_path,
                    operation='validate_secure',
                    success=False,
                    details={'error': str(e)}
                )
            
            self.logger.error(f"Secure file validation failed: {str(e)}", extra={
                'file_path': file_path,
                'user_id': user_id
            })
            
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'metadata': {},
                'security_checks': {},
                'all_security_checks_passed': False
            }
    
    def get_processed_files_summary(self) -> Dict[str, Any]:
        """
        Get summary of all processed files.
        
        Returns:
            Summary of processed files with protection information
        """
        try:
            summary = {
                'total_files': len(self.processed_files),
                'files_by_user': {},
                'files_with_pii': 0,
                'files_anonymized': 0,
                'recent_files': []
            }
            
            for file_path, file_info in self.processed_files.items():
                user_id = file_info.get('user_id', 'unknown')
                protection_report = file_info.get('protection_report', {})
                
                # Count by user
                summary['files_by_user'][user_id] = summary['files_by_user'].get(user_id, 0) + 1
                
                # Count PII and anonymization
                if protection_report.get('pii_detected', False):
                    summary['files_with_pii'] += 1
                
                if protection_report.get('anonymized', False):
                    summary['files_anonymized'] += 1
                
                # Add to recent files (last 10)
                if len(summary['recent_files']) < 10:
                    summary['recent_files'].append({
                        'file_path': file_path,
                        'user_id': user_id,
                        'processed_at': file_info.get('processed_at'),
                        'pii_detected': protection_report.get('pii_detected', False),
                        'anonymized': protection_report.get('anonymized', False)
                    })
            
            # Sort recent files by processing time
            summary['recent_files'].sort(
                key=lambda x: x.get('processed_at', pd.Timestamp.min), 
                reverse=True
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get processed files summary: {str(e)}")
            return {'error': str(e)}