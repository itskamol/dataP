"""
Data protection and privacy features for file processing system.
Implements requirements 7.2, 7.4, 2.1: Data protection, privacy compliance, and audit logging.
"""

import re
import hashlib
import json
import uuid
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import logging
from abc import ABC, abstractmethod

from ..domain.exceptions import ValidationError, SecurityError
from ..infrastructure.logging import get_logger


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


class AnonymizationMethod(Enum):
    """Methods for data anonymization."""
    MASK = "mask"
    HASH = "hash"
    REMOVE = "remove"
    REPLACE = "replace"
    GENERALIZE = "generalize"
    NOISE = "noise"
    PSEUDONYMIZE = "pseudonymize"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class PIIPattern:
    """Pattern definition for PII detection."""
    pii_type: PIIType
    pattern: str
    confidence: float
    description: str
    compiled_pattern: Optional[re.Pattern] = field(init=False, default=None)
    
    def __post_init__(self):
        try:
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            raise ValidationError(f"Invalid regex pattern for {self.pii_type.value}: {e}")


@dataclass
class AnonymizationRule:
    """Rule for data anonymization."""
    pii_type: PIIType
    method: AnonymizationMethod
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1
    
    def __post_init__(self):
        # Validate parameters based on method
        if self.method == AnonymizationMethod.REPLACE and 'replacement' not in self.parameters:
            self.parameters['replacement'] = '[REDACTED]'
        elif self.method == AnonymizationMethod.MASK and 'mask_char' not in self.parameters:
            self.parameters['mask_char'] = '*'
        elif self.method == AnonymizationMethod.GENERALIZE and 'level' not in self.parameters:
            self.parameters['level'] = 1


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    pii_type: PIIType
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    field_name: Optional[str] = None
    row_index: Optional[int] = None


@dataclass
class DataProtectionConfig:
    """Configuration for data protection features."""
    enable_pii_detection: bool = True
    enable_anonymization: bool = True
    enable_audit_logging: bool = True
    pii_patterns: List[PIIPattern] = field(default_factory=list)
    anonymization_rules: List[AnonymizationRule] = field(default_factory=list)
    data_classification: DataClassification = DataClassification.INTERNAL
    retention_days: int = 90
    audit_log_path: Optional[str] = None
    
    def __post_init__(self):
        if not self.pii_patterns:
            self.pii_patterns = self._get_default_pii_patterns()
        if not self.anonymization_rules:
            self.anonymization_rules = self._get_default_anonymization_rules()
    
    def _get_default_pii_patterns(self) -> List[PIIPattern]:
        """Get default PII detection patterns."""
        return [
            PIIPattern(
                pii_type=PIIType.EMAIL,
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                confidence=0.9,
                description="Email address pattern"
            ),
            PIIPattern(
                pii_type=PIIType.PHONE,
                pattern=r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                confidence=0.8,
                description="US phone number pattern"
            ),
            PIIPattern(
                pii_type=PIIType.SSN,
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                confidence=0.95,
                description="US Social Security Number pattern"
            ),
            PIIPattern(
                pii_type=PIIType.CREDIT_CARD,
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                confidence=0.85,
                description="Credit card number pattern"
            ),
            PIIPattern(
                pii_type=PIIType.IP_ADDRESS,
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                confidence=0.7,
                description="IPv4 address pattern"
            ),
            PIIPattern(
                pii_type=PIIType.DATE_OF_BIRTH,
                pattern=r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
                confidence=0.6,
                description="Date of birth pattern (MM/DD/YYYY)"
            )
        ]
    
    def _get_default_anonymization_rules(self) -> List[AnonymizationRule]:
        """Get default anonymization rules."""
        return [
            AnonymizationRule(PIIType.EMAIL, AnonymizationMethod.MASK, {'preserve_domain': True}),
            AnonymizationRule(PIIType.PHONE, AnonymizationMethod.MASK, {'preserve_last_digits': 4}),
            AnonymizationRule(PIIType.SSN, AnonymizationMethod.MASK, {'preserve_last_digits': 4}),
            AnonymizationRule(PIIType.CREDIT_CARD, AnonymizationMethod.MASK, {'preserve_last_digits': 4}),
            AnonymizationRule(PIIType.IP_ADDRESS, AnonymizationMethod.GENERALIZE, {'level': 2}),
            AnonymizationRule(PIIType.NAME, AnonymizationMethod.PSEUDONYMIZE),
            AnonymizationRule(PIIType.ADDRESS, AnonymizationMethod.GENERALIZE, {'level': 1}),
            AnonymizationRule(PIIType.DATE_OF_BIRTH, AnonymizationMethod.GENERALIZE, {'level': 1})
        ]


class PIIDetector:
    """Detector for Personally Identifiable Information."""
    
    def __init__(self, config: DataProtectionConfig):
        self.config = config
        self.logger = get_logger('pii_detector')
        self.patterns = {pattern.pii_type: pattern for pattern in config.pii_patterns}
        
        # Cache for compiled patterns
        self._pattern_cache: Dict[PIIType, re.Pattern] = {}
        
        # Statistics
        self.detection_stats = {
            'total_scans': 0,
            'pii_found': 0,
            'by_type': {pii_type: 0 for pii_type in PIIType}
        }
    
    def detect_pii_in_text(self, text: str, field_name: Optional[str] = None) -> List[PIIDetectionResult]:
        """
        Detect PII in a text string.
        
        Args:
            text: Text to scan for PII
            field_name: Optional field name for context
            
        Returns:
            List of PII detection results
        """
        if not text or not isinstance(text, str):
            return []
        
        results = []
        self.detection_stats['total_scans'] += 1
        
        for pii_type, pattern in self.patterns.items():
            if not pattern.compiled_pattern:
                continue
            
            matches = pattern.compiled_pattern.finditer(text)
            for match in matches:
                result = PIIDetectionResult(
                    pii_type=pii_type,
                    value=match.group(),
                    confidence=pattern.confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    field_name=field_name
                )
                results.append(result)
                
                self.detection_stats['pii_found'] += 1
                self.detection_stats['by_type'][pii_type] += 1
        
        return results
    
    def detect_pii_in_dataframe(self, df: pd.DataFrame) -> Dict[str, List[PIIDetectionResult]]:
        """
        Detect PII in a pandas DataFrame.
        
        Args:
            df: DataFrame to scan
            
        Returns:
            Dictionary mapping column names to PII detection results
        """
        results = {}
        
        for column in df.columns:
            column_results = []
            
            # Skip non-string columns
            if not df[column].dtype == 'object':
                continue
            
            for idx, value in df[column].items():
                if pd.isna(value):
                    continue
                
                pii_results = self.detect_pii_in_text(str(value), field_name=column)
                for result in pii_results:
                    result.row_index = idx
                    column_results.append(result)
            
            if column_results:
                results[column] = column_results
        
        return results
    
    def get_pii_summary(self, pii_results: Dict[str, List[PIIDetectionResult]]) -> Dict[str, Any]:
        """Generate summary of PII detection results."""
        total_pii = sum(len(results) for results in pii_results.values())
        
        pii_by_type = {}
        pii_by_column = {}
        
        for column, results in pii_results.items():
            pii_by_column[column] = len(results)
            
            for result in results:
                pii_type = result.pii_type.value
                pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1
        
        return {
            'total_pii_found': total_pii,
            'columns_with_pii': len(pii_results),
            'pii_by_type': pii_by_type,
            'pii_by_column': pii_by_column,
            'detection_stats': self.detection_stats.copy()
        }


class DataAnonymizer:
    """Data anonymization engine."""
    
    def __init__(self, config: DataProtectionConfig):
        self.config = config
        self.logger = get_logger('data_anonymizer')
        self.rules = {rule.pii_type: rule for rule in config.anonymization_rules if rule.enabled}
        
        # Pseudonymization mapping for consistency
        self._pseudonym_mapping: Dict[str, str] = {}
        
        # Statistics
        self.anonymization_stats = {
            'total_anonymized': 0,
            'by_method': {method: 0 for method in AnonymizationMethod},
            'by_type': {pii_type: 0 for pii_type in PIIType}
        }
    
    def anonymize_text(self, text: str, pii_results: List[PIIDetectionResult]) -> str:
        """
        Anonymize text based on PII detection results.
        
        Args:
            text: Original text
            pii_results: PII detection results
            
        Returns:
            Anonymized text
        """
        if not pii_results:
            return text
        
        # Sort by position (reverse order to maintain positions)
        pii_results = sorted(pii_results, key=lambda x: x.start_pos, reverse=True)
        
        anonymized_text = text
        
        for pii_result in pii_results:
            if pii_result.pii_type not in self.rules:
                continue
            
            rule = self.rules[pii_result.pii_type]
            anonymized_value = self._apply_anonymization_method(
                pii_result.value, rule.method, rule.parameters
            )
            
            # Replace in text
            anonymized_text = (
                anonymized_text[:pii_result.start_pos] +
                anonymized_value +
                anonymized_text[pii_result.end_pos:]
            )
            
            self.anonymization_stats['total_anonymized'] += 1
            self.anonymization_stats['by_method'][rule.method] += 1
            self.anonymization_stats['by_type'][pii_result.pii_type] += 1
        
        return anonymized_text
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                          pii_results: Dict[str, List[PIIDetectionResult]]) -> pd.DataFrame:
        """
        Anonymize DataFrame based on PII detection results.
        
        Args:
            df: Original DataFrame
            pii_results: PII detection results by column
            
        Returns:
            Anonymized DataFrame
        """
        anonymized_df = df.copy()
        
        for column, column_pii_results in pii_results.items():
            if column not in anonymized_df.columns:
                continue
            
            # Group PII results by row
            pii_by_row = {}
            for pii_result in column_pii_results:
                row_idx = pii_result.row_index
                if row_idx not in pii_by_row:
                    pii_by_row[row_idx] = []
                pii_by_row[row_idx].append(pii_result)
            
            # Anonymize each row
            for row_idx, row_pii_results in pii_by_row.items():
                original_value = str(anonymized_df.loc[row_idx, column])
                anonymized_value = self.anonymize_text(original_value, row_pii_results)
                anonymized_df.loc[row_idx, column] = anonymized_value
        
        return anonymized_df
    
    def _apply_anonymization_method(self, value: str, method: AnonymizationMethod, 
                                  parameters: Dict[str, Any]) -> str:
        """Apply specific anonymization method to a value."""
        try:
            if method == AnonymizationMethod.MASK:
                return self._mask_value(value, parameters)
            elif method == AnonymizationMethod.HASH:
                return self._hash_value(value, parameters)
            elif method == AnonymizationMethod.REMOVE:
                return ""
            elif method == AnonymizationMethod.REPLACE:
                return parameters.get('replacement', '[REDACTED]')
            elif method == AnonymizationMethod.GENERALIZE:
                return self._generalize_value(value, parameters)
            elif method == AnonymizationMethod.PSEUDONYMIZE:
                return self._pseudonymize_value(value, parameters)
            elif method == AnonymizationMethod.NOISE:
                return self._add_noise_to_value(value, parameters)
            else:
                return value
                
        except Exception as e:
            self.logger.warning(f"Anonymization failed for value: {e}")
            return parameters.get('fallback', '[ANONYMIZED]')
    
    def _mask_value(self, value: str, parameters: Dict[str, Any]) -> str:
        """Mask a value with specified character."""
        mask_char = parameters.get('mask_char', '*')
        preserve_last_digits = parameters.get('preserve_last_digits', 0)
        preserve_domain = parameters.get('preserve_domain', False)
        
        if preserve_domain and '@' in value:
            # For emails, preserve domain
            local, domain = value.split('@', 1)
            masked_local = mask_char * len(local)
            return f"{masked_local}@{domain}"
        
        if preserve_last_digits > 0 and len(value) > preserve_last_digits:
            masked_part = mask_char * (len(value) - preserve_last_digits)
            preserved_part = value[-preserve_last_digits:]
            return masked_part + preserved_part
        
        return mask_char * len(value)
    
    def _hash_value(self, value: str, parameters: Dict[str, Any]) -> str:
        """Hash a value for anonymization."""
        algorithm = parameters.get('algorithm', 'sha256')
        salt = parameters.get('salt', 'default_salt')
        
        hasher = hashlib.new(algorithm)
        hasher.update((value + salt).encode('utf-8'))
        
        return hasher.hexdigest()[:parameters.get('length', 8)]
    
    def _generalize_value(self, value: str, parameters: Dict[str, Any]) -> str:
        """Generalize a value by reducing precision."""
        level = parameters.get('level', 1)
        
        # For IP addresses
        if re.match(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', value):
            parts = value.split('.')
            if level == 1:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
            elif level == 2:
                return f"{parts[0]}.{parts[1]}.xxx.xxx"
            else:
                return "xxx.xxx.xxx.xxx"
        
        # For dates
        if re.match(r'\d{2}/\d{2}/\d{4}', value):
            if level == 1:
                return value[:7] + "xxxx"  # Keep month/day
            else:
                return "xx/xx/xxxx"
        
        # Default generalization
        return value[:max(1, len(value) - level)] + 'x' * min(level, len(value))
    
    def _pseudonymize_value(self, value: str, parameters: Dict[str, Any]) -> str:
        """Create consistent pseudonym for a value."""
        if value in self._pseudonym_mapping:
            return self._pseudonym_mapping[value]
        
        # Generate pseudonym
        prefix = parameters.get('prefix', 'ANON')
        pseudonym = f"{prefix}_{len(self._pseudonym_mapping) + 1:06d}"
        
        self._pseudonym_mapping[value] = pseudonym
        return pseudonym
    
    def _add_noise_to_value(self, value: str, parameters: Dict[str, Any]) -> str:
        """Add noise to numerical values."""
        try:
            # Only works for numerical values
            num_value = float(value)
            noise_factor = parameters.get('noise_factor', 0.1)
            
            import random
            noise = random.uniform(-noise_factor, noise_factor) * num_value
            noisy_value = num_value + noise
            
            return str(round(noisy_value, 2))
            
        except ValueError:
            # Not a number, return as-is
            return value


class AuditLogger:
    """Audit logging for data access and processing operations."""
    
    def __init__(self, config: DataProtectionConfig):
        self.config = config
        self.logger = get_logger('audit_logger')
        self.audit_log_path = config.audit_log_path
        
        # Audit event types
        self.event_types = {
            'DATA_ACCESS': 'data_access',
            'DATA_PROCESSING': 'data_processing',
            'PII_DETECTION': 'pii_detection',
            'DATA_ANONYMIZATION': 'data_anonymization',
            'DATA_EXPORT': 'data_export',
            'DATA_DELETION': 'data_deletion',
            'SECURITY_EVENT': 'security_event'
        }
    
    def log_data_access(self, user_id: str, file_path: str, operation: str, 
                       success: bool, details: Optional[Dict[str, Any]] = None):
        """Log data access event."""
        self._log_audit_event(
            event_type=self.event_types['DATA_ACCESS'],
            user_id=user_id,
            resource=file_path,
            operation=operation,
            success=success,
            details=details
        )
    
    def log_pii_detection(self, user_id: str, file_path: str, pii_summary: Dict[str, Any]):
        """Log PII detection event."""
        self._log_audit_event(
            event_type=self.event_types['PII_DETECTION'],
            user_id=user_id,
            resource=file_path,
            operation='pii_scan',
            success=True,
            details={
                'pii_found': pii_summary.get('total_pii_found', 0),
                'pii_types': list(pii_summary.get('pii_by_type', {}).keys()),
                'columns_affected': pii_summary.get('columns_with_pii', 0)
            }
        )
    
    def log_data_anonymization(self, user_id: str, file_path: str, 
                             anonymization_stats: Dict[str, Any]):
        """Log data anonymization event."""
        self._log_audit_event(
            event_type=self.event_types['DATA_ANONYMIZATION'],
            user_id=user_id,
            resource=file_path,
            operation='anonymize',
            success=True,
            details={
                'items_anonymized': anonymization_stats.get('total_anonymized', 0),
                'methods_used': [str(method) for method in anonymization_stats.get('by_method', {}).keys()],
                'pii_types_processed': [str(pii_type) for pii_type in anonymization_stats.get('by_type', {}).keys()]
            }
        )
    
    def log_data_export(self, user_id: str, source_file: str, export_file: str, 
                       format_type: str, anonymized: bool):
        """Log data export event."""
        self._log_audit_event(
            event_type=self.event_types['DATA_EXPORT'],
            user_id=user_id,
            resource=source_file,
            operation='export',
            success=True,
            details={
                'export_file': export_file,
                'format': format_type,
                'anonymized': anonymized
            }
        )
    
    def log_data_deletion(self, user_id: str, file_path: str, deletion_method: str, 
                         success: bool):
        """Log data deletion event."""
        self._log_audit_event(
            event_type=self.event_types['DATA_DELETION'],
            user_id=user_id,
            resource=file_path,
            operation='delete',
            success=success,
            details={'deletion_method': deletion_method}
        )
    
    def log_security_event(self, user_id: str, event_description: str, 
                          severity: str, details: Optional[Dict[str, Any]] = None):
        """Log security-related event."""
        self._log_audit_event(
            event_type=self.event_types['SECURITY_EVENT'],
            user_id=user_id,
            resource='system',
            operation='security_check',
            success=True,
            details={
                'event_description': event_description,
                'severity': severity,
                **(details or {})
            }
        )
    
    def _log_audit_event(self, event_type: str, user_id: str, resource: str, 
                        operation: str, success: bool, details: Optional[Dict[str, Any]] = None):
        """Log an audit event."""
        audit_record = {
            'timestamp': datetime.now().isoformat(),
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'operation': operation,
            'success': success,
            'details': details or {}
        }
        
        # Log to structured logger
        self.logger.info(f"Audit event: {event_type}", extra=audit_record)
        
        # Also write to dedicated audit log file if configured
        if self.audit_log_path:
            self._write_to_audit_file(audit_record)
    
    def _write_to_audit_file(self, audit_record: Dict[str, Any]):
        """Write audit record to dedicated audit log file."""
        try:
            audit_file = Path(self.audit_log_path)
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write to audit log file: {e}")
    
    def get_audit_summary(self, start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of audit events for a time period."""
        # This would typically query a database or parse log files
        # For now, return a placeholder summary
        return {
            'total_events': 0,
            'events_by_type': {},
            'events_by_user': {},
            'success_rate': 100.0,
            'time_period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            }
        }


class DataProtectionService:
    """Main service for data protection and privacy features."""
    
    def __init__(self, config: Optional[DataProtectionConfig] = None):
        self.config = config or DataProtectionConfig()
        self.logger = get_logger('data_protection')
        
        # Initialize components
        self.pii_detector = PIIDetector(self.config) if self.config.enable_pii_detection else None
        self.anonymizer = DataAnonymizer(self.config) if self.config.enable_anonymization else None
        self.audit_logger = AuditLogger(self.config) if self.config.enable_audit_logging else None
        
        # Data retention tracking
        self.data_retention_records: Dict[str, Dict[str, Any]] = {}
    
    def scan_and_protect_dataframe(self, df: pd.DataFrame, file_path: str, 
                                  user_id: str, anonymize: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scan DataFrame for PII and optionally anonymize it.
        
        Args:
            df: DataFrame to scan and protect
            file_path: Path to the source file
            user_id: User performing the operation
            anonymize: Whether to anonymize detected PII
            
        Returns:
            Tuple of (protected_dataframe, protection_report)
        """
        try:
            protection_report = {
                'pii_detected': False,
                'pii_summary': {},
                'anonymized': False,
                'anonymization_stats': {},
                'processing_time': 0
            }
            
            start_time = datetime.now()
            
            # Step 1: Detect PII
            pii_results = {}
            if self.pii_detector:
                pii_results = self.pii_detector.detect_pii_in_dataframe(df)
                protection_report['pii_detected'] = len(pii_results) > 0
                protection_report['pii_summary'] = self.pii_detector.get_pii_summary(pii_results)
                
                # Log PII detection
                if self.audit_logger:
                    self.audit_logger.log_pii_detection(user_id, file_path, protection_report['pii_summary'])
            
            # Step 2: Anonymize if requested and PII found
            protected_df = df
            if anonymize and pii_results and self.anonymizer:
                protected_df = self.anonymizer.anonymize_dataframe(df, pii_results)
                protection_report['anonymized'] = True
                protection_report['anonymization_stats'] = self.anonymizer.anonymization_stats.copy()
                
                # Log anonymization
                if self.audit_logger:
                    self.audit_logger.log_data_anonymization(
                        user_id, file_path, protection_report['anonymization_stats']
                    )
            
            # Step 3: Track for data retention
            self._track_data_retention(file_path, user_id, protection_report)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            protection_report['processing_time'] = processing_time
            
            self.logger.info(f"Data protection completed", extra={
                'file_path': file_path,
                'user_id': user_id,
                'pii_detected': protection_report['pii_detected'],
                'anonymized': protection_report['anonymized'],
                'processing_time': processing_time
            })
            
            return protected_df, protection_report
            
        except Exception as e:
            self.logger.error(f"Data protection failed: {e}")
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    user_id, f"Data protection failed: {str(e)}", "high",
                    {'file_path': file_path, 'error': str(e)}
                )
            raise SecurityError(f"Data protection failed: {e}")
    
    def create_anonymization_policy(self, pii_types: List[PIIType], 
                                   method: AnonymizationMethod,
                                   parameters: Optional[Dict[str, Any]] = None) -> AnonymizationRule:
        """Create a new anonymization policy rule."""
        rule = AnonymizationRule(
            pii_type=pii_types[0] if pii_types else PIIType.CUSTOM,
            method=method,
            parameters=parameters or {}
        )
        
        # Add to configuration
        self.config.anonymization_rules.append(rule)
        
        # Update anonymizer if available
        if self.anonymizer:
            self.anonymizer.rules[rule.pii_type] = rule
        
        return rule
    
    def add_custom_pii_pattern(self, pattern: str, pii_type: PIIType, 
                              confidence: float = 0.8, description: str = "") -> PIIPattern:
        """Add a custom PII detection pattern."""
        pii_pattern = PIIPattern(
            pii_type=pii_type,
            pattern=pattern,
            confidence=confidence,
            description=description or f"Custom {pii_type.value} pattern"
        )
        
        # Add to configuration
        self.config.pii_patterns.append(pii_pattern)
        
        # Update detector if available
        if self.pii_detector:
            self.pii_detector.patterns[pii_type] = pii_pattern
        
        return pii_pattern
    
    def _track_data_retention(self, file_path: str, user_id: str, protection_report: Dict[str, Any]):
        """Track data for retention policy compliance."""
        retention_record = {
            'file_path': file_path,
            'user_id': user_id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=self.config.retention_days),
            'pii_detected': protection_report.get('pii_detected', False),
            'anonymized': protection_report.get('anonymized', False),
            'classification': self.config.data_classification.value
        }
        
        self.data_retention_records[file_path] = retention_record
    
    def secure_data_disposal(self, file_path: str, user_id: str, method: str = 'secure_delete') -> bool:
        """
        Securely dispose of sensitive data files.
        
        Args:
            file_path: Path to file to be disposed
            user_id: User requesting disposal
            method: Disposal method ('secure_delete', 'overwrite', 'shred')
            
        Returns:
            True if disposal was successful
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                self.logger.warning(f"File not found for disposal: {file_path}")
                return False
            
            success = False
            
            if method == 'secure_delete':
                # Simple deletion (in production, use secure deletion tools)
                file_path_obj.unlink()
                success = True
                
            elif method == 'overwrite':
                # Overwrite with random data before deletion
                file_size = file_path_obj.stat().st_size
                with open(file_path_obj, 'wb') as f:
                    import os
                    f.write(os.urandom(file_size))
                file_path_obj.unlink()
                success = True
                
            elif method == 'shred':
                # Multiple pass overwrite (simplified version)
                file_size = file_path_obj.stat().st_size
                for _ in range(3):  # 3-pass shred
                    with open(file_path_obj, 'wb') as f:
                        import os
                        f.write(os.urandom(file_size))
                file_path_obj.unlink()
                success = True
            
            # Log disposal
            if self.audit_logger:
                self.audit_logger.log_data_deletion(user_id, file_path, method, success)
            
            # Remove from retention tracking
            if file_path in self.data_retention_records:
                del self.data_retention_records[file_path]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Secure data disposal failed for {file_path}: {e}")
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    user_id, f"Secure disposal failed: {str(e)}", "high",
                    {'file_path': file_path, 'method': method, 'error': str(e)}
                )
            return False
    
    def cleanup_expired_data(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up expired data based on retention policies.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup report with details of actions taken
        """
        now = datetime.now()
        expired_files = []
        
        for file_path, record in self.data_retention_records.items():
            if record['expires_at'] < now:
                expired_files.append(file_path)
        
        cleanup_report = {
            'total_files_checked': len(self.data_retention_records),
            'expired_files_found': len(expired_files),
            'expired_files': expired_files,
            'dry_run': dry_run,
            'cleanup_time': now.isoformat()
        }
        
        if not dry_run:
            # Actually delete expired files
            deleted_count = 0
            for file_path in expired_files:
                try:
                    file_path_obj = Path(file_path)
                    if file_path_obj.exists():
                        file_path_obj.unlink()
                        deleted_count += 1
                    
                    # Remove from tracking
                    del self.data_retention_records[file_path]
                    
                    # Log deletion
                    if self.audit_logger:
                        self.audit_logger.log_data_deletion(
                            'system', file_path, 'retention_policy', True
                        )
                        
                except Exception as e:
                    self.logger.error(f"Failed to delete expired file {file_path}: {e}")
            
            cleanup_report['actually_deleted'] = deleted_count
        
        return cleanup_report
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data protection statistics."""
        stats = {
            'pii_detection': {},
            'anonymization': {},
            'data_retention': {},
            'audit_events': {}
        }
        
        if self.pii_detector:
            stats['pii_detection'] = self.pii_detector.detection_stats.copy()
        
        if self.anonymizer:
            stats['anonymization'] = self.anonymizer.anonymization_stats.copy()
        
        # Data retention stats
        total_tracked = len(self.data_retention_records)
        expired_count = sum(1 for record in self.data_retention_records.values() 
                          if record['expires_at'] < datetime.now())
        
        stats['data_retention'] = {
            'total_files_tracked': total_tracked,
            'expired_files': expired_count,
            'retention_days': self.config.retention_days
        }
        
        if self.audit_logger:
            stats['audit_events'] = self.audit_logger.get_audit_summary()
        
        return stats