"""
Comprehensive tests for file security and validation features.
Tests requirements 7.1, 7.4, 2.4: File security, data protection, and error handling.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from src.infrastructure.security import (
    FileSecurityValidator, SecureFileManager, FileSecurityConfig,
    SecurityLevel, ThreatType, SecurityThreat
)
from src.application.services.secure_file_service import SecureFileProcessingService
from src.domain.exceptions import SecurityError, FileValidationError


class TestFileSecurityValidator:
    """Test file security validation functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def security_config(self, temp_dir):
        """Create security configuration for tests."""
        return FileSecurityConfig(
            max_file_size=1024 * 1024,  # 1MB for tests
            allowed_extensions={'.csv', '.json', '.txt'},
            allowed_mime_types={'text/csv', 'application/json', 'text/plain'},
            enable_virus_scan=False,  # Disable for tests
            enable_content_scan=True,
            temp_dir=str(temp_dir)
        )
    
    @pytest.fixture
    def validator(self, security_config):
        """Create file security validator."""
        return FileSecurityValidator(security_config)
    
    def test_validate_file_security_valid_file(self, validator, temp_dir):
        """Test security validation of a valid file."""
        # Create a valid CSV file
        test_file = temp_dir / "test.csv"
        test_file.write_text("name,age\nJohn,30\nJane,25")
        
        is_secure, threats = validator.validate_file_security(test_file)
        
        assert is_secure is True
        assert len(threats) == 0
    
    def test_validate_file_security_oversized_file(self, validator, temp_dir):
        """Test security validation of oversized file."""
        # Create a file larger than the limit
        test_file = temp_dir / "large.csv"
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        test_file.write_text(large_content)
        
        is_secure, threats = validator.validate_file_security(test_file)
        
        assert is_secure is False
        assert len(threats) > 0
        assert any(t.threat_type == ThreatType.OVERSIZED_FILE for t in threats)
    
    def test_validate_file_security_invalid_extension(self, validator, temp_dir):
        """Test security validation of file with invalid extension."""
        # Create a file with disallowed extension
        test_file = temp_dir / "test.exe"
        test_file.write_text("some content")
        
        is_secure, threats = validator.validate_file_security(test_file)
        
        assert is_secure is False
        assert len(threats) > 0
        assert any(t.threat_type == ThreatType.INVALID_TYPE for t in threats)
    
    def test_validate_file_security_suspicious_content(self, validator, temp_dir):
        """Test security validation of file with suspicious content."""
        # Create a file with suspicious script content
        test_file = temp_dir / "suspicious.csv"
        test_file.write_text("name,script\nJohn,<script>alert('xss')</script>")
        
        is_secure, threats = validator.validate_file_security(test_file)
        
        assert is_secure is False
        assert len(threats) > 0
        assert any(t.threat_type == ThreatType.SUSPICIOUS_CONTENT for t in threats)
    
    def test_validate_file_security_nonexistent_file(self, validator, temp_dir):
        """Test security validation of nonexistent file."""
        test_file = temp_dir / "nonexistent.csv"
        
        is_secure, threats = validator.validate_file_security(test_file)
        
        assert is_secure is False
        assert len(threats) > 0
        assert any(t.threat_type == ThreatType.INVALID_TYPE for t in threats)
    
    def test_scan_binary_content_executable(self, validator, temp_dir):
        """Test binary content scanning for executable files."""
        # Create a file with PE header (Windows executable)
        test_file = temp_dir / "test.exe"
        test_file.write_bytes(b'MZ\x90\x00' + b'x' * 100)
        
        # Test the binary scanning method directly
        with open(test_file, 'rb') as f:
            content = f.read()
        
        result = validator._scan_binary_content(content, test_file)
        
        assert result is False
        assert len(validator.threats_detected) > 0
        assert any(t.threat_type == ThreatType.MALICIOUS_FILE for t in validator.threats_detected)
    
    def test_get_security_report(self, validator, temp_dir):
        """Test security report generation."""
        # Create files with different threat levels
        test_file1 = temp_dir / "large.csv"
        test_file1.write_text("x" * (2 * 1024 * 1024))  # Oversized
        
        test_file2 = temp_dir / "suspicious.csv"
        test_file2.write_text("name,script\nJohn,<script>alert('xss')</script>")
        
        # Validate files to generate threats
        validator.validate_file_security(test_file1)
        validator.validate_file_security(test_file2)
        
        report = validator.get_security_report()
        
        assert 'total_threats' in report
        assert 'threats_by_type' in report
        assert 'threats_by_severity' in report
        assert 'threats' in report
        assert report['total_threats'] > 0
    
    @patch('subprocess.run')
    def test_clamscan_virus_check_clean(self, mock_run, validator, temp_dir):
        """Test ClamAV virus scanning with clean file."""
        test_file = temp_dir / "clean.txt"
        test_file.write_text("clean content")
        
        # Mock successful ClamAV scan (no virus)
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = validator._clamscan_virus_check(test_file)
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_clamscan_virus_check_infected(self, mock_run, validator, temp_dir):
        """Test ClamAV virus scanning with infected file."""
        test_file = temp_dir / "infected.txt"
        test_file.write_text("infected content")
        
        # Mock ClamAV scan finding virus
        mock_run.return_value = Mock(returncode=1, stdout="FOUND", stderr="")
        
        result = validator._clamscan_virus_check(test_file)
        
        assert result is False
        assert len(validator.threats_detected) > 0
        assert any(t.threat_type == ThreatType.VIRUS_DETECTED for t in validator.threats_detected)


class TestSecureFileManager:
    """Test secure file management functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def security_config(self, temp_dir):
        """Create security configuration for tests."""
        return FileSecurityConfig(
            enable_encryption=True,
            temp_dir=str(temp_dir)
        )
    
    @pytest.fixture
    def file_manager(self, security_config):
        """Create secure file manager."""
        return SecureFileManager(security_config)
    
    def test_create_secure_temp_file(self, file_manager):
        """Test creation of secure temporary files."""
        temp_file = file_manager.create_secure_temp_file(suffix='.csv')
        
        assert temp_file.exists()
        assert temp_file.suffix == '.csv'
        assert temp_file in file_manager.temp_files
        
        # Check file permissions (owner read/write only)
        stat_info = temp_file.stat()
        assert oct(stat_info.st_mode)[-3:] == '600'
    
    def test_encrypt_decrypt_file(self, file_manager, temp_dir):
        """Test file encryption and decryption."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_content = "This is secret content"
        test_file.write_text(test_content)
        
        # Encrypt file
        encrypted_file = file_manager.encrypt_file(test_file)
        
        assert encrypted_file.exists()
        assert encrypted_file != test_file
        
        # Verify encrypted content is different
        encrypted_content = encrypted_file.read_bytes()
        assert encrypted_content != test_content.encode()
        
        # Decrypt file
        decrypted_file = file_manager.decrypt_file(encrypted_file)
        
        assert decrypted_file.exists()
        assert decrypted_file.read_text() == test_content
    
    def test_secure_copy(self, file_manager, temp_dir):
        """Test secure file copying."""
        # Create source file
        source_file = temp_dir / "source.txt"
        source_content = "test content"
        source_file.write_text(source_content)
        
        # Create target path
        target_file = temp_dir / "target.txt"
        
        # Secure copy
        result_file = file_manager.secure_copy(source_file, target_file)
        
        assert result_file == target_file
        assert target_file.exists()
        assert target_file.read_text() == source_content
        
        # Check secure permissions
        stat_info = target_file.stat()
        assert oct(stat_info.st_mode)[-3:] == '600'
    
    def test_secure_delete(self, file_manager, temp_dir):
        """Test secure file deletion."""
        # Create test file
        test_file = temp_dir / "delete_me.txt"
        test_file.write_text("sensitive data")
        
        assert test_file.exists()
        
        # Secure delete
        result = file_manager.secure_delete(test_file)
        
        assert result is True
        assert not test_file.exists()
    
    def test_get_file_hash(self, file_manager, temp_dir):
        """Test file hash calculation."""
        # Create test file
        test_file = temp_dir / "hash_test.txt"
        test_content = "content for hashing"
        test_file.write_text(test_content)
        
        # Calculate hash
        file_hash = file_manager.get_file_hash(test_file)
        
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hash length
        
        # Verify hash is consistent
        file_hash2 = file_manager.get_file_hash(test_file)
        assert file_hash == file_hash2
    
    def test_verify_file_integrity(self, file_manager, temp_dir):
        """Test file integrity verification."""
        # Create test file
        test_file = temp_dir / "integrity_test.txt"
        test_content = "content for integrity check"
        test_file.write_text(test_content)
        
        # Get hash
        expected_hash = file_manager.get_file_hash(test_file)
        
        # Verify integrity
        is_valid = file_manager.verify_file_integrity(test_file, expected_hash)
        assert is_valid is True
        
        # Modify file and verify integrity fails
        test_file.write_text("modified content")
        is_valid = file_manager.verify_file_integrity(test_file, expected_hash)
        assert is_valid is False
    
    def test_cleanup_temp_files(self, file_manager):
        """Test temporary file cleanup."""
        # Create several temp files
        temp_files = []
        for i in range(3):
            temp_file = file_manager.create_secure_temp_file(suffix=f'_{i}.txt')
            temp_file.write_text(f"temp content {i}")
            temp_files.append(temp_file)
        
        assert len(file_manager.temp_files) == 3
        
        # Cleanup all temp files
        cleaned_count = file_manager.cleanup_temp_files()
        
        assert cleaned_count == 3
        assert len(file_manager.temp_files) == 0
        
        # Verify files are deleted
        for temp_file in temp_files:
            assert not temp_file.exists()


class TestSecureFileProcessingService:
    """Test secure file processing service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def security_config(self, temp_dir):
        """Create security configuration for tests."""
        return FileSecurityConfig(
            max_file_size=1024 * 1024,  # 1MB for tests
            enable_virus_scan=False,  # Disable for tests
            temp_dir=str(temp_dir)
        )
    
    @pytest.fixture
    def secure_service(self, security_config):
        """Create secure file processing service."""
        return SecureFileProcessingService(security_config=security_config)
    
    def test_validate_file_security_with_user_quota(self, secure_service, temp_dir):
        """Test file security validation with user quota tracking."""
        # Create valid test file
        test_file = temp_dir / "test.csv"
        test_file.write_text("name,age\nJohn,30")
        
        user_id = "test_user"
        
        # First validation should pass
        result = secure_service.validate_file_security(str(test_file), user_id)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Check quota was updated
        quota_status = secure_service.get_user_quota_status(user_id)
        assert quota_status['daily_used'] == 1
        assert quota_status['hourly_used'] == 1
    
    def test_upload_quota_limits(self, secure_service, temp_dir):
        """Test upload quota enforcement."""
        # Create valid test file
        test_file = temp_dir / "test.csv"
        test_file.write_text("name,age\nJohn,30")
        
        user_id = "quota_test_user"
        
        # Set low limits for testing
        secure_service.daily_upload_limit = 2
        secure_service.hourly_upload_limit = 1
        
        # First upload should pass
        result1 = secure_service.validate_file_security(str(test_file), user_id)
        assert result1.is_valid is True
        
        # Second upload should fail due to hourly limit
        result2 = secure_service.validate_file_security(str(test_file), user_id)
        assert result2.is_valid is False
        assert "Upload quota exceeded" in result2.errors
    
    def test_load_file_secure(self, secure_service, temp_dir):
        """Test secure file loading."""
        # Create valid CSV file
        test_file = temp_dir / "secure_load.csv"
        test_data = "name,age,city\nJohn,30,NYC\nJane,25,LA"
        test_file.write_text(test_data)
        
        user_id = "load_test_user"
        
        # Load file securely
        dataset = secure_service.load_file_secure(str(test_file), user_id=user_id)
        
        assert dataset is not None
        assert len(dataset.data) == 2
        assert list(dataset.columns) == ['name', 'age', 'city']
        assert dataset.metadata.metadata['security_validated'] is True
        assert dataset.metadata.metadata['user_id'] == user_id
    
    def test_load_file_secure_with_security_failure(self, secure_service, temp_dir):
        """Test secure file loading with security validation failure."""
        # Create file with suspicious content
        test_file = temp_dir / "suspicious.csv"
        test_data = "name,script\nJohn,<script>alert('xss')</script>"
        test_file.write_text(test_data)
        
        user_id = "security_test_user"
        
        # Loading should fail due to security validation
        with pytest.raises(SecurityError) as exc_info:
            secure_service.load_file_secure(str(test_file), user_id=user_id)
        
        assert "security validation failed" in str(exc_info.value).lower()
    
    def test_save_results_secure(self, secure_service, temp_dir):
        """Test secure results saving."""
        # Create test DataFrame
        df = pd.DataFrame({
            'name': ['John', 'Jane'],
            'age': [30, 25],
            'city': ['NYC', 'LA']
        })
        
        output_path = temp_dir / "secure_output.csv"
        user_id = "save_test_user"
        
        # Save results securely
        created_files = secure_service.save_results_secure(
            df, str(output_path), user_id=user_id
        )
        
        assert len(created_files) == 1
        assert Path(created_files[0]).exists()
        
        # Verify file was tracked
        assert str(created_files[0]) in secure_service.processed_files
        record = secure_service.processed_files[str(created_files[0])]
        assert record['user_id'] == user_id
        assert record['file_type'] == 'output'
        assert 'file_hash' in record
    
    def test_save_results_secure_with_encryption(self, secure_service, temp_dir):
        """Test secure results saving with encryption."""
        # Create test DataFrame
        df = pd.DataFrame({'data': ['secret1', 'secret2']})
        
        output_path = temp_dir / "encrypted_output.csv"
        user_id = "encrypt_test_user"
        
        # Save results with encryption
        created_files = secure_service.save_results_secure(
            df, str(output_path), encrypt=True, user_id=user_id
        )
        
        assert len(created_files) == 1
        encrypted_file = Path(created_files[0])
        assert encrypted_file.exists()
        assert encrypted_file.suffix == '.enc'
        
        # Verify file is encrypted (content should be different)
        encrypted_content = encrypted_file.read_bytes()
        assert b'secret1' not in encrypted_content
        assert b'secret2' not in encrypted_content
    
    def test_cleanup_processed_files(self, secure_service, temp_dir):
        """Test cleanup of old processed files."""
        # Create and track some files
        test_files = []
        for i in range(3):
            test_file = temp_dir / f"cleanup_test_{i}.txt"
            test_file.write_text(f"content {i}")
            test_files.append(test_file)
            
            # Add to processed files with old timestamp
            old_time = datetime.now() - timedelta(hours=25)
            secure_service.processed_files[str(test_file)] = {
                'created_at': old_time,
                'user_id': 'cleanup_user'
            }
        
        # Run cleanup
        cleaned_count = secure_service.cleanup_processed_files(max_age_hours=24)
        
        assert cleaned_count >= 3
        assert len(secure_service.processed_files) == 0
    
    def test_get_security_statistics(self, secure_service, temp_dir):
        """Test security statistics generation."""
        # Add some processed file records
        secure_service.processed_files = {
            'file1.csv': {'security_validated': True, 'threats_count': 0},
            'file2.csv': {'security_validated': True, 'threats_count': 1},
            'file3.csv': {'security_validated': False, 'threats_count': 2}
        }
        
        stats = secure_service.get_security_statistics()
        
        assert stats['total_files_processed'] == 3
        assert stats['secure_files'] == 2
        assert stats['files_with_threats'] == 2
        assert stats['security_validation_rate'] == 2/3 * 100
    
    def test_audit_file_access(self, secure_service, temp_dir):
        """Test file access auditing."""
        file_path = str(temp_dir / "audit_test.csv")
        user_id = "audit_user"
        action = "read"
        
        # Mock logger to capture audit log
        with patch.object(secure_service.logger, 'info') as mock_log:
            secure_service.audit_file_access(
                file_path, user_id, action, True, {'details': 'test'}
            )
            
            # Verify audit log was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert 'File access audit' in call_args[0][0]
            
            # Check audit record structure
            audit_record = call_args[1]['extra']
            assert audit_record['file_path'] == file_path
            assert audit_record['user_id'] == user_id
            assert audit_record['action'] == action
            assert audit_record['success'] is True


class TestSecurityIntegration:
    """Integration tests for security features."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_end_to_end_secure_processing(self, temp_dir):
        """Test complete secure file processing workflow."""
        # Create secure service
        security_config = FileSecurityConfig(
            max_file_size=1024 * 1024,
            enable_virus_scan=False,
            temp_dir=str(temp_dir)
        )
        service = SecureFileProcessingService(security_config=security_config)
        
        # Create test input file
        input_file = temp_dir / "input.csv"
        input_data = "name,age,salary\nJohn,30,50000\nJane,25,60000"
        input_file.write_text(input_data)
        
        user_id = "integration_user"
        
        # Step 1: Validate file security
        validation_result = service.validate_file_security(str(input_file), user_id)
        assert validation_result.is_valid is True
        
        # Step 2: Load file securely
        dataset = service.load_file_secure(str(input_file), user_id=user_id)
        assert dataset is not None
        assert len(dataset.data) == 2
        
        # Step 3: Process data (simulate some processing)
        processed_df = dataset.data.copy()
        processed_df['bonus'] = processed_df['salary'] * 0.1
        
        # Step 4: Save results securely
        output_path = temp_dir / "output.csv"
        created_files = service.save_results_secure(
            processed_df, str(output_path), user_id=user_id
        )
        
        assert len(created_files) == 1
        assert Path(created_files[0]).exists()
        
        # Step 5: Verify audit trail
        assert len(service.processed_files) >= 1
        
        # Step 6: Get security statistics
        stats = service.get_security_statistics()
        assert stats['total_files_processed'] >= 1
        assert stats['secure_files'] >= 1
        
        # Step 7: Cleanup
        cleaned_count = service.cleanup_processed_files(max_age_hours=0)
        assert cleaned_count >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])