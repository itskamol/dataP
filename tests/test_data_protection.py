"""
Comprehensive tests for data protection and privacy features.
Tests requirements 7.2, 7.4, 2.1: Data protection, privacy compliance, and audit logging.
"""

import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.data_protection import (
    PIIType, AnonymizationMethod, DataClassification,
    PIIPattern, AnonymizationRule, PIIDetectionResult, DataProtectionConfig,
    PIIDetector, DataAnonymizer, AuditLogger, DataProtectionService
)
from src.domain.exceptions import ValidationError, SecurityError


class TestPIIPattern:
    """Test PII pattern functionality."""
    
    def test_valid_pattern_creation(self):
        """Test creating a valid PII pattern."""
        pattern = PIIPattern(
            pii_type=PIIType.EMAIL,
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            confidence=0.9,
            description="Email pattern"
        )
        
        assert pattern.pii_type == PIIType.EMAIL
        assert pattern.confidence == 0.9
        assert pattern.compiled_pattern is not None
    
    def test_invalid_pattern_raises_error(self):
        """Test that invalid regex patterns raise ValidationError."""
        with pytest.raises(ValidationError):
            PIIPattern(
                pii_type=PIIType.EMAIL,
                pattern=r'[invalid regex(',
                confidence=0.9,
                description="Invalid pattern"
            )


class TestAnonymizationRule:
    """Test anonymization rule functionality."""
    
    def test_rule_creation_with_defaults(self):
        """Test creating anonymization rule with default parameters."""
        rule = AnonymizationRule(
            pii_type=PIIType.EMAIL,
            method=AnonymizationMethod.MASK
        )
        
        assert rule.pii_type == PIIType.EMAIL
        assert rule.method == AnonymizationMethod.MASK
        assert rule.enabled is True
        assert 'mask_char' in rule.parameters
    
    def test_rule_parameter_validation(self):
        """Test that rule parameters are validated correctly."""
        # Test REPLACE method
        rule = AnonymizationRule(PIIType.EMAIL, AnonymizationMethod.REPLACE)
        assert 'replacement' in rule.parameters
        
        # Test GENERALIZE method
        rule = AnonymizationRule(PIIType.IP_ADDRESS, AnonymizationMethod.GENERALIZE)
        assert 'level' in rule.parameters


class TestDataProtectionConfig:
    """Test data protection configuration."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = DataProtectionConfig()
        
        assert config.enable_pii_detection is True
        assert config.enable_anonymization is True
        assert config.enable_audit_logging is True
        assert len(config.pii_patterns) > 0
        assert len(config.anonymization_rules) > 0
        assert config.retention_days == 90
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = DataProtectionConfig(
            enable_pii_detection=False,
            retention_days=30,
            data_classification=DataClassification.CONFIDENTIAL
        )
        
        assert config.enable_pii_detection is False
        assert config.retention_days == 30
        assert config.data_classification == DataClassification.CONFIDENTIAL


class TestPIIDetector:
    """Test PII detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create a PII detector for testing."""
        config = DataProtectionConfig()
        return PIIDetector(config)
    
    def test_detect_email_in_text(self, detector):
        """Test detecting email addresses in text."""
        text = "Contact us at support@example.com or admin@test.org"
        results = detector.detect_pii_in_text(text)
        
        email_results = [r for r in results if r.pii_type == PIIType.EMAIL]
        assert len(email_results) == 2
        assert "support@example.com" in [r.value for r in email_results]
        assert "admin@test.org" in [r.value for r in email_results]
    
    def test_detect_phone_in_text(self, detector):
        """Test detecting phone numbers in text."""
        text = "Call me at (555) 123-4567 or 555.987.6543"
        results = detector.detect_pii_in_text(text)
        
        phone_results = [r for r in results if r.pii_type == PIIType.PHONE]
        assert len(phone_results) == 2
    
    def test_detect_ssn_in_text(self, detector):
        """Test detecting SSN in text."""
        text = "My SSN is 123-45-6789"
        results = detector.detect_pii_in_text(text)
        
        ssn_results = [r for r in results if r.pii_type == PIIType.SSN]
        assert len(ssn_results) == 1
        assert ssn_results[0].value == "123-45-6789"
    
    def test_detect_pii_in_dataframe(self, detector):
        """Test detecting PII in pandas DataFrame."""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'],
            'email': ['john@example.com', 'jane@test.org'],
            'phone': ['555-123-4567', '(555) 987-6543'],
            'age': [30, 25]
        })
        
        results = detector.detect_pii_in_dataframe(df)
        
        assert 'email' in results
        assert 'phone' in results
        assert 'age' not in results  # Non-string column should be skipped
        
        # Check email detection
        email_results = results['email']
        assert len(email_results) == 2
        assert all(r.pii_type == PIIType.EMAIL for r in email_results)
    
    def test_empty_text_handling(self, detector):
        """Test handling of empty or None text."""
        assert detector.detect_pii_in_text("") == []
        assert detector.detect_pii_in_text(None) == []
    
    def test_pii_summary_generation(self, detector):
        """Test generating PII summary from results."""
        df = pd.DataFrame({
            'email': ['john@example.com', 'jane@test.org'],
            'phone': ['555-123-4567', '(555) 987-6543']
        })
        
        pii_results = detector.detect_pii_in_dataframe(df)
        summary = detector.get_pii_summary(pii_results)
        
        assert summary['total_pii_found'] > 0
        assert summary['columns_with_pii'] == 2
        assert 'email' in summary['pii_by_type']
        assert 'phone' in summary['pii_by_type']


class TestDataAnonymizer:
    """Test data anonymization functionality."""
    
    @pytest.fixture
    def anonymizer(self):
        """Create a data anonymizer for testing."""
        config = DataProtectionConfig()
        return DataAnonymizer(config)
    
    @pytest.fixture
    def sample_pii_results(self):
        """Create sample PII detection results for testing."""
        return [
            PIIDetectionResult(
                pii_type=PIIType.EMAIL,
                value="john@example.com",
                confidence=0.9,
                start_pos=0,
                end_pos=16
            ),
            PIIDetectionResult(
                pii_type=PIIType.PHONE,
                value="555-123-4567",
                confidence=0.8,
                start_pos=20,
                end_pos=32
            )
        ]
    
    def test_mask_email_with_domain_preservation(self, anonymizer):
        """Test masking email while preserving domain."""
        rule = AnonymizationRule(PIIType.EMAIL, AnonymizationMethod.MASK, {'preserve_domain': True})
        anonymizer.rules[PIIType.EMAIL] = rule
        
        result = anonymizer._mask_value("john@example.com", {'preserve_domain': True})
        assert result == "****@example.com"
    
    def test_mask_phone_with_last_digits(self, anonymizer):
        """Test masking phone number while preserving last digits."""
        result = anonymizer._mask_value("555-123-4567", {'preserve_last_digits': 4})
        assert result == "********4567"
    
    def test_hash_value_anonymization(self, anonymizer):
        """Test hash-based anonymization."""
        result = anonymizer._hash_value("sensitive_data", {'algorithm': 'sha256', 'length': 8})
        assert len(result) == 8
        assert result != "sensitive_data"
        
        # Same input should produce same hash
        result2 = anonymizer._hash_value("sensitive_data", {'algorithm': 'sha256', 'length': 8})
        assert result == result2
    
    def test_generalize_ip_address(self, anonymizer):
        """Test IP address generalization."""
        result = anonymizer._generalize_value("192.168.1.100", {'level': 2})
        assert result == "192.168.xxx.xxx"
        
        result = anonymizer._generalize_value("192.168.1.100", {'level': 1})
        assert result == "192.168.1.xxx"
    
    def test_pseudonymize_value_consistency(self, anonymizer):
        """Test that pseudonymization is consistent."""
        result1 = anonymizer._pseudonymize_value("John Doe", {'prefix': 'PERSON'})
        result2 = anonymizer._pseudonymize_value("John Doe", {'prefix': 'PERSON'})
        
        assert result1 == result2
        assert result1.startswith('PERSON_')
    
    def test_anonymize_text_with_multiple_pii(self, anonymizer, sample_pii_results):
        """Test anonymizing text with multiple PII types."""
        text = "john@example.com or 555-123-4567"
        anonymized = anonymizer.anonymize_text(text, sample_pii_results)
        
        assert "john@example.com" not in anonymized
        assert "555-123-4567" not in anonymized
    
    def test_anonymize_dataframe(self, anonymizer):
        """Test anonymizing entire DataFrame."""
        df = pd.DataFrame({
            'email': ['john@example.com', 'jane@test.org'],
            'phone': ['555-123-4567', '(555) 987-6543']
        })
        
        # Create mock PII results
        pii_results = {
            'email': [
                PIIDetectionResult(PIIType.EMAIL, 'john@example.com', 0.9, 0, 16, 'email', 0),
                PIIDetectionResult(PIIType.EMAIL, 'jane@test.org', 0.9, 0, 13, 'email', 1)
            ],
            'phone': [
                PIIDetectionResult(PIIType.PHONE, '555-123-4567', 0.8, 0, 12, 'phone', 0),
                PIIDetectionResult(PIIType.PHONE, '(555) 987-6543', 0.8, 0, 14, 'phone', 1)
            ]
        }
        
        anonymized_df = anonymizer.anonymize_dataframe(df, pii_results)
        
        # Check that original values are not present
        assert 'john@example.com' not in anonymized_df['email'].values
        assert 'jane@test.org' not in anonymized_df['email'].values
    
    def test_noise_addition_to_numerical_values(self, anonymizer):
        """Test adding noise to numerical values."""
        result = anonymizer._add_noise_to_value("100.0", {'noise_factor': 0.1})
        result_float = float(result)
        
        # Result should be within 10% of original
        assert 90.0 <= result_float <= 110.0
        assert result != "100.0"
    
    def test_anonymization_method_fallback(self, anonymizer):
        """Test fallback behavior when anonymization fails."""
        with patch.object(anonymizer, '_mask_value', side_effect=Exception("Test error")):
            result = anonymizer._apply_anonymization_method(
                "test", AnonymizationMethod.MASK, {'fallback': '[ERROR]'}
            )
            assert result == '[ERROR]'


class TestAuditLogger:
    """Test audit logging functionality."""
    
    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataProtectionConfig(audit_log_path=os.path.join(temp_dir, 'audit.log'))
            yield AuditLogger(config)
    
    def test_log_data_access(self, audit_logger):
        """Test logging data access events."""
        with patch.object(audit_logger.logger, 'info') as mock_log:
            audit_logger.log_data_access(
                user_id='user123',
                file_path='/path/to/file.csv',
                operation='read',
                success=True,
                details={'file_size': 1024}
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert 'Audit event: data_access' in call_args[0][0]
            assert call_args[1]['extra']['user_id'] == 'user123'
            assert call_args[1]['extra']['operation'] == 'read'
    
    def test_log_pii_detection(self, audit_logger):
        """Test logging PII detection events."""
        pii_summary = {
            'total_pii_found': 5,
            'pii_by_type': {'email': 2, 'phone': 3},
            'columns_with_pii': 2
        }
        
        with patch.object(audit_logger.logger, 'info') as mock_log:
            audit_logger.log_pii_detection('user123', '/path/to/file.csv', pii_summary)
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['extra']['details']['pii_found'] == 5
    
    def test_log_data_anonymization(self, audit_logger):
        """Test logging data anonymization events."""
        anonymization_stats = {
            'total_anonymized': 10,
            'by_method': {'mask': 5, 'hash': 5},
            'by_type': {'email': 3, 'phone': 7}
        }
        
        with patch.object(audit_logger.logger, 'info') as mock_log:
            audit_logger.log_data_anonymization('user123', '/path/to/file.csv', anonymization_stats)
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['extra']['details']['items_anonymized'] == 10
    
    def test_log_security_event(self, audit_logger):
        """Test logging security events."""
        with patch.object(audit_logger.logger, 'info') as mock_log:
            audit_logger.log_security_event(
                user_id='user123',
                event_description='Suspicious activity detected',
                severity='high',
                details={'ip_address': '192.168.1.100'}
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]['extra']['details']['severity'] == 'high'
    
    def test_audit_file_writing(self, audit_logger):
        """Test writing audit records to file."""
        audit_logger.log_data_access('user123', '/test/file.csv', 'read', True)
        
        # Check that audit file was created and contains data
        audit_file = Path(audit_logger.audit_log_path)
        assert audit_file.exists()
        
        with open(audit_file, 'r') as f:
            content = f.read()
            assert 'user123' in content
            assert 'data_access' in content


class TestDataProtectionService:
    """Test the main data protection service."""
    
    @pytest.fixture
    def protection_service(self):
        """Create a data protection service for testing."""
        config = DataProtectionConfig()
        return DataProtectionService(config)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with PII for testing."""
        return pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.net'],
            'phone': ['555-123-4567', '(555) 987-6543', '555.111.2222'],
            'age': [30, 25, 35],
            'city': ['New York', 'Los Angeles', 'Chicago']
        })
    
    def test_scan_and_protect_dataframe_with_anonymization(self, protection_service, sample_dataframe):
        """Test scanning and protecting DataFrame with anonymization."""
        protected_df, report = protection_service.scan_and_protect_dataframe(
            df=sample_dataframe,
            file_path='/test/data.csv',
            user_id='user123',
            anonymize=True
        )
        
        assert report['pii_detected'] is True
        assert report['anonymized'] is True
        assert report['processing_time'] > 0
        assert 'pii_summary' in report
        assert 'anonymization_stats' in report
        
        # Check that PII was actually anonymized
        assert 'john@example.com' not in protected_df['email'].values
        assert 'jane@test.org' not in protected_df['email'].values
    
    def test_scan_and_protect_dataframe_without_anonymization(self, protection_service, sample_dataframe):
        """Test scanning DataFrame without anonymization."""
        protected_df, report = protection_service.scan_and_protect_dataframe(
            df=sample_dataframe,
            file_path='/test/data.csv',
            user_id='user123',
            anonymize=False
        )
        
        assert report['pii_detected'] is True
        assert report['anonymized'] is False
        
        # Original data should be preserved
        assert 'john@example.com' in protected_df['email'].values
    
    def test_create_anonymization_policy(self, protection_service):
        """Test creating custom anonymization policies."""
        rule = protection_service.create_anonymization_policy(
            pii_types=[PIIType.EMAIL],
            method=AnonymizationMethod.HASH,
            parameters={'algorithm': 'md5', 'length': 10}
        )
        
        assert rule.pii_type == PIIType.EMAIL
        assert rule.method == AnonymizationMethod.HASH
        assert rule.parameters['algorithm'] == 'md5'
        
        # Check that rule was added to configuration
        assert rule in protection_service.config.anonymization_rules
    
    def test_add_custom_pii_pattern(self, protection_service):
        """Test adding custom PII detection patterns."""
        pattern = protection_service.add_custom_pii_pattern(
            pattern=r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',
            pii_type=PIIType.CREDIT_CARD,
            confidence=0.95,
            description="Custom credit card pattern"
        )
        
        assert pattern.pii_type == PIIType.CREDIT_CARD
        assert pattern.confidence == 0.95
        assert pattern.compiled_pattern is not None
        
        # Check that pattern was added to configuration
        assert pattern in protection_service.config.pii_patterns
    
    def test_data_retention_tracking(self, protection_service, sample_dataframe):
        """Test data retention tracking functionality."""
        protected_df, report = protection_service.scan_and_protect_dataframe(
            df=sample_dataframe,
            file_path='/test/data.csv',
            user_id='user123',
            anonymize=True
        )
        
        # Check that data retention record was created
        assert '/test/data.csv' in protection_service.data_retention_records
        
        retention_record = protection_service.data_retention_records['/test/data.csv']
        assert retention_record['user_id'] == 'user123'
        assert retention_record['pii_detected'] is True
        assert retention_record['anonymized'] is True
    
    def test_secure_data_disposal(self, protection_service):
        """Test secure data disposal functionality."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"sensitive data")
            temp_path = temp_file.name
        
        try:
            # Test secure disposal
            success = protection_service.secure_data_disposal(
                file_path=temp_path,
                user_id='user123',
                method='secure_delete'
            )
            
            assert success is True
            assert not Path(temp_path).exists()
            
        finally:
            # Cleanup if file still exists
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    
    def test_cleanup_expired_data(self, protection_service):
        """Test cleanup of expired data."""
        # Add some test retention records
        past_date = datetime.now() - timedelta(days=1)
        future_date = datetime.now() + timedelta(days=1)
        
        protection_service.data_retention_records = {
            '/expired/file1.csv': {
                'expires_at': past_date,
                'user_id': 'user123'
            },
            '/active/file2.csv': {
                'expires_at': future_date,
                'user_id': 'user456'
            }
        }
        
        # Test dry run
        report = protection_service.cleanup_expired_data(dry_run=True)
        
        assert report['total_files_checked'] == 2
        assert report['expired_files_found'] == 1
        assert '/expired/file1.csv' in report['expired_files']
        assert report['dry_run'] is True
    
    def test_protection_statistics(self, protection_service, sample_dataframe):
        """Test getting protection statistics."""
        # Process some data to generate statistics
        protection_service.scan_and_protect_dataframe(
            df=sample_dataframe,
            file_path='/test/data.csv',
            user_id='user123',
            anonymize=True
        )
        
        stats = protection_service.get_protection_statistics()
        
        assert 'pii_detection' in stats
        assert 'anonymization' in stats
        assert 'data_retention' in stats
        assert 'audit_events' in stats
        
        # Check that statistics contain meaningful data
        assert stats['pii_detection']['total_scans'] > 0
        assert stats['anonymization']['total_anonymized'] > 0
    
    def test_error_handling_in_protection(self, protection_service):
        """Test error handling in data protection operations."""
        # Create a DataFrame that will cause an error
        invalid_df = pd.DataFrame({'test': [1, 2, 3]})
        
        # Mock the PII detector to raise an exception
        with patch.object(protection_service.pii_detector, 'detect_pii_in_dataframe', 
                         side_effect=Exception("Test error")):
            with pytest.raises(SecurityError):
                protection_service.scan_and_protect_dataframe(
                    df=invalid_df,
                    file_path='/test/error.csv',
                    user_id='user123',
                    anonymize=True
                )


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    @pytest.fixture
    def full_service(self):
        """Create a fully configured data protection service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataProtectionConfig(
                audit_log_path=os.path.join(temp_dir, 'audit.log'),
                retention_days=30
            )
            yield DataProtectionService(config)
    
    def test_end_to_end_data_protection_workflow(self, full_service):
        """Test complete data protection workflow."""
        # Create test data with various PII types
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.net'],
            'phone': ['555-123-4567', '(555) 987-6543', '555.111.2222'],
            'ssn': ['123-45-6789', '987-65-4321', '456-78-9012'],
            'credit_card': ['4532-1234-5678-9012', '5555-4444-3333-2222', '4111-1111-1111-1111'],
            'ip_address': ['192.168.1.100', '10.0.0.1', '172.16.0.1'],
            'salary': [50000, 75000, 60000]
        })
        
        # Process the data
        protected_df, report = full_service.scan_and_protect_dataframe(
            df=df,
            file_path='/test/sensitive_data.csv',
            user_id='data_analyst_001',
            anonymize=True
        )
        
        # Verify PII detection
        assert report['pii_detected'] is True
        assert report['pii_summary']['total_pii_found'] > 0
        
        # Verify anonymization
        assert report['anonymized'] is True
        assert report['anonymization_stats']['total_anonymized'] > 0
        
        # Verify that sensitive data was anonymized
        original_emails = df['email'].tolist()
        protected_emails = protected_df['email'].tolist()
        
        for original_email in original_emails:
            assert original_email not in protected_emails
        
        # Verify non-sensitive data is preserved
        assert protected_df['customer_id'].tolist() == df['customer_id'].tolist()
        assert protected_df['salary'].tolist() == df['salary'].tolist()
        
        # Verify audit logging
        assert full_service.audit_logger is not None
        audit_file = Path(full_service.audit_logger.audit_log_path)
        assert audit_file.exists()
        
        # Verify data retention tracking
        assert '/test/sensitive_data.csv' in full_service.data_retention_records
    
    def test_custom_pii_patterns_and_rules(self, full_service):
        """Test using custom PII patterns and anonymization rules."""
        # Add custom PII pattern for employee IDs
        full_service.add_custom_pii_pattern(
            pattern=r'EMP\d{6}',
            pii_type=PIIType.CUSTOM,
            confidence=0.95,
            description="Employee ID pattern"
        )
        
        # Add custom anonymization rule
        full_service.create_anonymization_policy(
            pii_types=[PIIType.CUSTOM],
            method=AnonymizationMethod.PSEUDONYMIZE,
            parameters={'prefix': 'EMPLOYEE'}
        )
        
        # Test data with custom PII
        df = pd.DataFrame({
            'employee_id': ['EMP123456', 'EMP789012', 'EMP345678'],
            'department': ['Engineering', 'Marketing', 'Sales']
        })
        
        protected_df, report = full_service.scan_and_protect_dataframe(
            df=df,
            file_path='/test/employee_data.csv',
            user_id='hr_manager_001',
            anonymize=True
        )
        
        # Verify custom PII was detected and anonymized
        assert report['pii_detected'] is True
        assert 'EMP123456' not in protected_df['employee_id'].values
        
        # Check that pseudonyms were generated
        pseudonyms = protected_df['employee_id'].tolist()
        assert all(p.startswith('EMPLOYEE_') for p in pseudonyms)
    
    def test_performance_with_large_dataset(self, full_service):
        """Test performance with larger datasets."""
        # Create a larger dataset
        import random
        
        size = 1000
        df = pd.DataFrame({
            'id': range(size),
            'email': [f'user{i}@example{i%10}.com' for i in range(size)],
            'phone': [f'555-{random.randint(100,999)}-{random.randint(1000,9999)}' for _ in range(size)],
            'data': [f'some data {i}' for i in range(size)]
        })
        
        start_time = datetime.now()
        protected_df, report = full_service.scan_and_protect_dataframe(
            df=df,
            file_path='/test/large_dataset.csv',
            user_id='batch_processor_001',
            anonymize=True
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Verify processing completed successfully
        assert report['pii_detected'] is True
        assert report['anonymized'] is True
        assert len(protected_df) == size
        
        # Performance should be reasonable (less than 30 seconds for 1000 records)
        assert processing_time < 30.0
        
        # Verify statistics
        stats = full_service.get_protection_statistics()
        assert stats['pii_detection']['total_scans'] >= size * 2  # email and phone columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestSecureFileServiceIntegration:
    """Test integration between data protection and file processing services."""
    
    @pytest.fixture
    def secure_file_service(self):
        """Create a secure file processing service for testing."""
        from src.application.services.secure_file_service import SecureFileProcessingService
        from src.infrastructure.data_protection import DataProtectionConfig
        
        config = DataProtectionConfig(
            enable_pii_detection=True,
            enable_anonymization=True,
            enable_audit_logging=True
        )
        return SecureFileProcessingService(data_protection_config=config)
    
    @pytest.fixture
    def sample_csv_with_pii(self):
        """Create a temporary CSV file with PII for testing."""
        import tempfile
        import csv
        
        # Create temporary CSV file with PII
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'email', 'phone', 'ssn', 'age'])
            writer.writerow(['John Doe', 'john@example.com', '555-123-4567', '123-45-6789', '30'])
            writer.writerow(['Jane Smith', 'jane@test.org', '(555) 987-6543', '987-65-4321', '25'])
            writer.writerow(['Bob Johnson', 'bob@company.net', '555.111.2222', '456-78-9012', '35'])
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
    
    def test_load_file_secure_with_anonymization(self, secure_file_service, sample_csv_with_pii):
        """Test secure file loading with PII anonymization."""
        dataset, protection_report = secure_file_service.load_file_secure(
            file_path=sample_csv_with_pii,
            user_id='test_user_001',
            anonymize_pii=True
        )
        
        # Verify dataset was loaded
        assert dataset is not None
        assert len(dataset.data) == 3
        assert 'name' in dataset.data.columns
        assert 'email' in dataset.data.columns
        
        # Verify PII was detected and anonymized
        assert protection_report['pii_detected'] is True
        assert protection_report['anonymized'] is True
        
        # Verify original PII values are not present
        email_values = dataset.data['email'].tolist()
        assert 'john@example.com' not in email_values
        assert 'jane@test.org' not in email_values
        
        # Verify metadata includes protection information
        assert dataset.metadata.additional_info['pii_detected'] is True
        assert dataset.metadata.additional_info['anonymized'] is True
    
    def test_load_file_secure_without_anonymization(self, secure_file_service, sample_csv_with_pii):
        """Test secure file loading without anonymization."""
        dataset, protection_report = secure_file_service.load_file_secure(
            file_path=sample_csv_with_pii,
            user_id='test_user_002',
            anonymize_pii=False
        )
        
        # Verify PII was detected but not anonymized
        assert protection_report['pii_detected'] is True
        assert protection_report['anonymized'] is False
        
        # Verify original PII values are still present
        email_values = dataset.data['email'].tolist()
        assert 'john@example.com' in email_values
        assert 'jane@test.org' in email_values
    
    def test_save_results_secure_with_anonymization(self, secure_file_service):
        """Test secure results saving with anonymization."""
        # Create test DataFrame with PII
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.net'],
            'phone': ['555-123-4567', '(555) 987-6543', '555.111.2222']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'secure_export')
            
            created_files, protection_report = secure_file_service.save_results_secure(
                df=df,
                output_path=output_path,
                user_id='test_user_003',
                format_type='csv',
                anonymize_before_export=True
            )
            
            # Verify files were created
            assert len(created_files) == 1
            assert created_files[0].endswith('.csv')
            assert os.path.exists(created_files[0])
            
            # Verify PII was anonymized in export
            assert protection_report['pii_detected'] is True
            assert protection_report['anonymized'] is True
            
            # Read exported file and verify anonymization
            exported_df = pd.read_csv(created_files[0])
            exported_emails = exported_df['email'].tolist()
            assert 'john@example.com' not in exported_emails
            assert 'jane@test.org' not in exported_emails
    
    def test_secure_file_disposal(self, secure_file_service):
        """Test secure file disposal functionality."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("sensitive data that should be securely disposed")
            temp_path = f.name
        
        # Verify file exists
        assert os.path.exists(temp_path)
        
        # Securely dispose of file
        success = secure_file_service.secure_file_disposal(
            file_path=temp_path,
            user_id='test_user_004',
            disposal_method='secure_delete'
        )
        
        # Verify disposal was successful
        assert success is True
        assert not os.path.exists(temp_path)
    
    def test_custom_pii_detection_rule(self, secure_file_service):
        """Test adding custom PII detection rules."""
        # Add custom pattern for employee IDs
        success = secure_file_service.add_custom_pii_detection_rule(
            pattern=r'EMP\d{6}',
            pii_type=PIIType.CUSTOM,
            confidence=0.95,
            description="Employee ID pattern"
        )
        
        assert success is True
        
        # Test with data containing custom PII
        df = pd.DataFrame({
            'employee_id': ['EMP123456', 'EMP789012'],
            'department': ['Engineering', 'Marketing']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            dataset, protection_report = secure_file_service.load_file_secure(
                file_path=temp_path,
                user_id='test_user_005',
                anonymize_pii=True
            )
            
            # Verify custom PII was detected
            assert protection_report['pii_detected'] is True
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_anonymization_policy_creation(self, secure_file_service):
        """Test creating custom anonymization policies."""
        success = secure_file_service.create_anonymization_policy(
            pii_types=[PIIType.EMAIL],
            method=AnonymizationMethod.HASH,
            parameters={'algorithm': 'sha256', 'length': 10}
        )
        
        assert success is True
    
    def test_data_protection_statistics(self, secure_file_service, sample_csv_with_pii):
        """Test getting data protection statistics."""
        # Process a file to generate statistics
        secure_file_service.load_file_secure(
            file_path=sample_csv_with_pii,
            user_id='test_user_006',
            anonymize_pii=True
        )
        
        stats = secure_file_service.get_data_protection_statistics()
        
        # Verify statistics structure
        assert 'pii_detection' in stats
        assert 'anonymization' in stats
        assert 'data_retention' in stats
        assert 'file_processing' in stats
        
        # Verify file processing statistics
        assert stats['file_processing']['total_files_processed'] >= 1
        assert stats['file_processing']['files_with_pii'] >= 1
        assert stats['file_processing']['files_anonymized'] >= 1
    
    def test_processed_files_summary(self, secure_file_service, sample_csv_with_pii):
        """Test getting processed files summary."""
        # Process multiple files
        secure_file_service.load_file_secure(
            file_path=sample_csv_with_pii,
            user_id='user_a',
            anonymize_pii=True
        )
        
        secure_file_service.load_file_secure(
            file_path=sample_csv_with_pii,
            user_id='user_b',
            anonymize_pii=False
        )
        
        summary = secure_file_service.get_processed_files_summary()
        
        # Verify summary structure
        assert 'total_files' in summary
        assert 'files_by_user' in summary
        assert 'files_with_pii' in summary
        assert 'files_anonymized' in summary
        assert 'recent_files' in summary
        
        # Verify counts
        assert summary['total_files'] >= 2
        assert 'user_a' in summary['files_by_user']
        assert 'user_b' in summary['files_by_user']
    
    def test_validate_file_secure(self, secure_file_service, sample_csv_with_pii):
        """Test secure file validation."""
        result = secure_file_service.validate_file_secure(
            file_path=sample_csv_with_pii,
            user_id='test_user_007'
        )
        
        # Verify validation result structure
        assert 'is_valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'metadata' in result
        assert 'security_checks' in result
        assert 'all_security_checks_passed' in result
        
        # Verify validation passed
        assert result['is_valid'] is True
        assert result['all_security_checks_passed'] is True
    
    def test_cleanup_expired_files(self, secure_file_service):
        """Test cleanup of expired files."""
        cleanup_report = secure_file_service.cleanup_expired_files(
            user_id='system',
            dry_run=True
        )
        
        # Verify cleanup report structure
        assert 'total_files_checked' in cleanup_report or 'error' not in cleanup_report
        assert 'dry_run' in cleanup_report or 'error' not in cleanup_report
    
    def test_error_handling_in_secure_operations(self, secure_file_service):
        """Test error handling in secure file operations."""
        # Test with non-existent file
        with pytest.raises(FileProcessingError):
            secure_file_service.load_file_secure(
                file_path='/non/existent/file.csv',
                user_id='test_user_008',
                anonymize_pii=True
            )
        
        # Test secure disposal of non-existent file
        success = secure_file_service.secure_file_disposal(
            file_path='/non/existent/file.txt',
            user_id='test_user_009'
        )
        assert success is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])