"""
Security infrastructure for file processing and data protection.
Implements requirements 7.1, 7.4, 2.4: File security, data protection, and error handling.
"""

import os
import hashlib
import mimetypes
import tempfile
import shutil
import subprocess
import magic
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import re
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..domain.exceptions import (
    FileValidationError, SecurityError, FileAccessError, 
    ValidationError
)
from ..infrastructure.logging import get_logger


class SecurityLevel(Enum):
    """Security levels for file processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    MALICIOUS_FILE = "malicious_file"
    OVERSIZED_FILE = "oversized_file"
    INVALID_TYPE = "invalid_type"
    SUSPICIOUS_CONTENT = "suspicious_content"
    VIRUS_DETECTED = "virus_detected"
    ENCODING_ATTACK = "encoding_attack"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    file_path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()


@dataclass
class FileSecurityConfig:
    """Configuration for file security validation."""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: Set[str] = None
    allowed_mime_types: Set[str] = None
    enable_virus_scan: bool = True
    enable_content_scan: bool = True
    enable_encryption: bool = True
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    temp_dir: Optional[str] = None
    encryption_key: Optional[bytes] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'.csv', '.json', '.jsonl', '.xlsx', '.xls', '.txt'}
        if self.allowed_mime_types is None:
            self.allowed_mime_types = {
                'text/csv', 'application/json', 'text/json',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-excel', 'text/plain'
            }
        if self.temp_dir is None:
            self.temp_dir = tempfile.gettempdir()


class FileSecurityValidator:
    """Comprehensive file security validation and threat detection."""
    
    def __init__(self, config: Optional[FileSecurityConfig] = None):
        self.config = config or FileSecurityConfig()
        self.logger = get_logger('file_security')
        self.threats_detected: List[SecurityThreat] = []
        
        # Initialize magic library for MIME type detection
        try:
            self.magic_mime = magic.Magic(mime=True)
        except Exception as e:
            self.logger.warning(f"Failed to initialize python-magic: {e}")
            self.magic_mime = None
        
        # Suspicious patterns for content scanning
        self.suspicious_patterns = [
            # Script injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            
            # Command injection patterns
            r';\s*rm\s+-rf',
            r';\s*cat\s+/etc/passwd',
            r';\s*wget\s+',
            r';\s*curl\s+',
            
            # Path traversal patterns
            r'\.\./',
            r'\.\.\\',
            r'/etc/passwd',
            r'/etc/shadow',
            
            # Executable patterns
            r'#!/bin/',
            r'#!/usr/bin/',
            r'MZ\x90\x00',  # PE header
            r'\x7fELF',     # ELF header
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.suspicious_patterns]
    
    def validate_file_security(self, file_path: Union[str, Path]) -> Tuple[bool, List[SecurityThreat]]:
        """
        Perform comprehensive security validation on a file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_secure, list_of_threats)
        """
        file_path = Path(file_path)
        self.threats_detected = []
        
        try:
            # Basic file existence and access checks
            if not self._validate_file_access(file_path):
                return False, self.threats_detected
            
            # File size validation
            if not self._validate_file_size(file_path):
                return False, self.threats_detected
            
            # File type and extension validation
            if not self._validate_file_type(file_path):
                return False, self.threats_detected
            
            # MIME type validation
            if not self._validate_mime_type(file_path):
                return False, self.threats_detected
            
            # Content scanning for suspicious patterns
            if self.config.enable_content_scan:
                if not self._scan_file_content(file_path):
                    return False, self.threats_detected
            
            # Virus scanning (if enabled and available)
            if self.config.enable_virus_scan:
                if not self._scan_for_viruses(file_path):
                    return False, self.threats_detected
            
            # Additional security checks based on security level
            if self.config.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                if not self._advanced_security_checks(file_path):
                    return False, self.threats_detected
            
            self.logger.info(f"File security validation passed: {file_path}")
            return True, self.threats_detected
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"Security validation failed: {str(e)}",
                file_path=str(file_path),
                details={'error': str(e)}
            )
            self.threats_detected.append(threat)
            self.logger.error(f"Security validation error for {file_path}: {e}")
            return False, self.threats_detected
    
    def _validate_file_access(self, file_path: Path) -> bool:
        """Validate basic file access and properties."""
        try:
            if not file_path.exists():
                threat = SecurityThreat(
                    threat_type=ThreatType.INVALID_TYPE,
                    severity=SecurityLevel.MEDIUM,
                    description="File does not exist",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                return False
            
            if not file_path.is_file():
                threat = SecurityThreat(
                    threat_type=ThreatType.INVALID_TYPE,
                    severity=SecurityLevel.MEDIUM,
                    description="Path is not a regular file",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                return False
            
            if not os.access(file_path, os.R_OK):
                threat = SecurityThreat(
                    threat_type=ThreatType.INVALID_TYPE,
                    severity=SecurityLevel.MEDIUM,
                    description="File is not readable",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                return False
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"File access validation failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _validate_file_size(self, file_path: Path) -> bool:
        """Validate file size against configured limits."""
        try:
            file_size = file_path.stat().st_size
            
            if file_size > self.config.max_file_size:
                threat = SecurityThreat(
                    threat_type=ThreatType.OVERSIZED_FILE,
                    severity=SecurityLevel.MEDIUM,
                    description=f"File size ({file_size} bytes) exceeds maximum allowed ({self.config.max_file_size} bytes)",
                    file_path=str(file_path),
                    details={'file_size': file_size, 'max_size': self.config.max_file_size}
                )
                self.threats_detected.append(threat)
                return False
            
            # Check for suspiciously small files that claim to be data files
            if file_size < 10:  # Less than 10 bytes
                threat = SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_CONTENT,
                    severity=SecurityLevel.LOW,
                    description=f"File is suspiciously small ({file_size} bytes)",
                    file_path=str(file_path),
                    details={'file_size': file_size}
                )
                self.threats_detected.append(threat)
                # Don't fail validation for small files, just warn
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"File size validation failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _validate_file_type(self, file_path: Path) -> bool:
        """Validate file extension against allowed types."""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.config.allowed_extensions:
                threat = SecurityThreat(
                    threat_type=ThreatType.INVALID_TYPE,
                    severity=SecurityLevel.MEDIUM,
                    description=f"File extension '{file_extension}' is not allowed",
                    file_path=str(file_path),
                    details={
                        'file_extension': file_extension,
                        'allowed_extensions': list(self.config.allowed_extensions)
                    }
                )
                self.threats_detected.append(threat)
                return False
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"File type validation failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _validate_mime_type(self, file_path: Path) -> bool:
        """Validate MIME type against allowed types."""
        try:
            # Try python-magic first (more accurate)
            mime_type = None
            if self.magic_mime:
                try:
                    mime_type = self.magic_mime.from_file(str(file_path))
                except Exception as e:
                    self.logger.warning(f"python-magic failed: {e}")
            
            # Fallback to mimetypes module
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if not mime_type:
                threat = SecurityThreat(
                    threat_type=ThreatType.INVALID_TYPE,
                    severity=SecurityLevel.LOW,
                    description="Could not determine MIME type",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                # Don't fail validation if we can't determine MIME type
                return True
            
            if mime_type not in self.config.allowed_mime_types:
                threat = SecurityThreat(
                    threat_type=ThreatType.INVALID_TYPE,
                    severity=SecurityLevel.MEDIUM,
                    description=f"MIME type '{mime_type}' is not allowed",
                    file_path=str(file_path),
                    details={
                        'mime_type': mime_type,
                        'allowed_mime_types': list(self.config.allowed_mime_types)
                    }
                )
                self.threats_detected.append(threat)
                return False
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"MIME type validation failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _scan_file_content(self, file_path: Path) -> bool:
        """Scan file content for suspicious patterns."""
        try:
            # Read file content (limit to first 1MB for performance)
            max_scan_size = 1024 * 1024  # 1MB
            
            with open(file_path, 'rb') as f:
                content = f.read(max_scan_size)
            
            # Try to decode as text for pattern matching
            text_content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                # If we can't decode as text, check for binary threats
                return self._scan_binary_content(content, file_path)
            
            # Scan for suspicious text patterns
            threats_found = []
            for pattern in self.compiled_patterns:
                matches = pattern.findall(text_content)
                if matches:
                    threats_found.extend(matches)
            
            if threats_found:
                threat = SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_CONTENT,
                    severity=SecurityLevel.HIGH,
                    description=f"Suspicious patterns detected in file content",
                    file_path=str(file_path),
                    details={'patterns_found': threats_found[:10]}  # Limit to first 10
                )
                self.threats_detected.append(threat)
                return False
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"Content scanning failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _scan_binary_content(self, content: bytes, file_path: Path) -> bool:
        """Scan binary content for threats."""
        try:
            # Check for executable headers
            if content.startswith(b'MZ'):  # PE executable
                threat = SecurityThreat(
                    threat_type=ThreatType.MALICIOUS_FILE,
                    severity=SecurityLevel.CRITICAL,
                    description="File appears to be a Windows executable",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                return False
            
            if content.startswith(b'\x7fELF'):  # ELF executable
                threat = SecurityThreat(
                    threat_type=ThreatType.MALICIOUS_FILE,
                    severity=SecurityLevel.CRITICAL,
                    description="File appears to be a Linux executable",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                return False
            
            # Check for suspicious byte sequences
            suspicious_bytes = [
                b'\x00' * 100,  # Long null sequences
                b'\xff' * 100,  # Long 0xFF sequences
            ]
            
            for suspicious in suspicious_bytes:
                if suspicious in content:
                    threat = SecurityThreat(
                        threat_type=ThreatType.SUSPICIOUS_CONTENT,
                        severity=SecurityLevel.MEDIUM,
                        description="Suspicious byte patterns detected",
                        file_path=str(file_path)
                    )
                    self.threats_detected.append(threat)
                    # Don't fail validation for suspicious bytes, just warn
                    break
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"Binary content scanning failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _scan_for_viruses(self, file_path: Path) -> bool:
        """Scan file for viruses using available antivirus tools."""
        try:
            # Try ClamAV first (most common on Linux)
            if shutil.which('clamscan'):
                return self._clamscan_virus_check(file_path)
            
            # Try other antivirus tools
            if shutil.which('freshclam'):
                # Update virus definitions first
                try:
                    subprocess.run(['freshclam'], capture_output=True, timeout=30)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Virus definition update timed out")
                
                if shutil.which('clamscan'):
                    return self._clamscan_virus_check(file_path)
            
            # If no antivirus available, log warning and continue
            self.logger.warning("No antivirus scanner available, skipping virus scan")
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"Virus scanning failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _clamscan_virus_check(self, file_path: Path) -> bool:
        """Perform virus scan using ClamAV."""
        try:
            result = subprocess.run(
                ['clamscan', '--no-summary', str(file_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # No virus found
                return True
            elif result.returncode == 1:
                # Virus found
                threat = SecurityThreat(
                    threat_type=ThreatType.VIRUS_DETECTED,
                    severity=SecurityLevel.CRITICAL,
                    description="Virus detected by ClamAV",
                    file_path=str(file_path),
                    details={'clamscan_output': result.stdout}
                )
                self.threats_detected.append(threat)
                return False
            else:
                # Scanner error
                threat = SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_CONTENT,
                    severity=SecurityLevel.HIGH,
                    description=f"ClamAV scanner error (exit code {result.returncode})",
                    file_path=str(file_path),
                    details={'clamscan_error': result.stderr}
                )
                self.threats_detected.append(threat)
                return False
                
        except subprocess.TimeoutExpired:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description="Virus scan timed out",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"ClamAV scan failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def _advanced_security_checks(self, file_path: Path) -> bool:
        """Perform advanced security checks for high security levels."""
        try:
            # Check file permissions
            stat_info = file_path.stat()
            
            # Check for world-writable files
            if stat_info.st_mode & 0o002:
                threat = SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_CONTENT,
                    severity=SecurityLevel.MEDIUM,
                    description="File is world-writable",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
            
            # Check for setuid/setgid files
            if stat_info.st_mode & (0o4000 | 0o2000):
                threat = SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_CONTENT,
                    severity=SecurityLevel.HIGH,
                    description="File has setuid or setgid bit set",
                    file_path=str(file_path)
                )
                self.threats_detected.append(threat)
                return False
            
            # Check file age (very old or very new files might be suspicious)
            file_age = datetime.now().timestamp() - stat_info.st_mtime
            if file_age < 60:  # Less than 1 minute old
                threat = SecurityThreat(
                    threat_type=ThreatType.SUSPICIOUS_CONTENT,
                    severity=SecurityLevel.LOW,
                    description="File was created very recently",
                    file_path=str(file_path),
                    details={'age_seconds': file_age}
                )
                self.threats_detected.append(threat)
            
            return True
            
        except Exception as e:
            threat = SecurityThreat(
                threat_type=ThreatType.SUSPICIOUS_CONTENT,
                severity=SecurityLevel.HIGH,
                description=f"Advanced security checks failed: {str(e)}",
                file_path=str(file_path)
            )
            self.threats_detected.append(threat)
            return False
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        return {
            'total_threats': len(self.threats_detected),
            'threats_by_type': {
                threat_type.value: len([t for t in self.threats_detected if t.threat_type == threat_type])
                for threat_type in ThreatType
            },
            'threats_by_severity': {
                severity.value: len([t for t in self.threats_detected if t.severity == severity])
                for severity in SecurityLevel
            },
            'threats': [
                {
                    'type': threat.threat_type.value,
                    'severity': threat.severity.value,
                    'description': threat.description,
                    'file_path': threat.file_path,
                    'detected_at': threat.detected_at.isoformat(),
                    'details': threat.details
                }
                for threat in self.threats_detected
            ]
        }


class SecureFileManager:
    """Secure file management with encryption and cleanup."""
    
    def __init__(self, config: Optional[FileSecurityConfig] = None):
        self.config = config or FileSecurityConfig()
        self.logger = get_logger('secure_file_manager')
        self.temp_files: Set[Path] = set()
        
        # Initialize encryption
        if self.config.enable_encryption:
            self.cipher_suite = self._initialize_encryption()
        else:
            self.cipher_suite = None
    
    def _initialize_encryption(self) -> Optional[Fernet]:
        """Initialize encryption cipher suite."""
        try:
            if self.config.encryption_key:
                key = self.config.encryption_key
            else:
                # Generate key from password (in production, use proper key management)
                password = os.getenv('FILE_ENCRYPTION_PASSWORD', 'default-password-change-me').encode()
                salt = os.getenv('FILE_ENCRYPTION_SALT', 'default-salt').encode()
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password))
            
            return Fernet(key)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            return None
    
    def create_secure_temp_file(self, suffix: str = '', prefix: str = 'secure_') -> Path:
        """Create a secure temporary file."""
        try:
            # Create temporary file with secure permissions
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=self.config.temp_dir
            )
            os.close(fd)
            
            temp_path = Path(temp_path)
            
            # Set secure permissions (owner read/write only)
            temp_path.chmod(0o600)
            
            # Track temporary file for cleanup
            self.temp_files.add(temp_path)
            
            self.logger.debug(f"Created secure temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to create secure temporary file: {e}")
            raise SecurityError(f"Could not create secure temporary file: {e}")
    
    def encrypt_file(self, source_path: Path, target_path: Optional[Path] = None) -> Path:
        """Encrypt a file and return the encrypted file path."""
        if not self.cipher_suite:
            raise SecurityError("Encryption not available")
        
        try:
            if target_path is None:
                target_path = self.create_secure_temp_file(suffix='.enc')
            
            with open(source_path, 'rb') as source_file:
                data = source_file.read()
            
            encrypted_data = self.cipher_suite.encrypt(data)
            
            with open(target_path, 'wb') as target_file:
                target_file.write(encrypted_data)
            
            # Set secure permissions
            target_path.chmod(0o600)
            
            self.logger.info(f"File encrypted: {source_path} -> {target_path}")
            return target_path
            
        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            raise SecurityError(f"File encryption failed: {e}")
    
    def decrypt_file(self, encrypted_path: Path, target_path: Optional[Path] = None) -> Path:
        """Decrypt a file and return the decrypted file path."""
        if not self.cipher_suite:
            raise SecurityError("Encryption not available")
        
        try:
            if target_path is None:
                target_path = self.create_secure_temp_file(suffix='.dec')
            
            with open(encrypted_path, 'rb') as encrypted_file:
                encrypted_data = encrypted_file.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            with open(target_path, 'wb') as target_file:
                target_file.write(decrypted_data)
            
            # Set secure permissions
            target_path.chmod(0o600)
            
            self.logger.info(f"File decrypted: {encrypted_path} -> {target_path}")
            return target_path
            
        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            raise SecurityError(f"File decryption failed: {e}")
    
    def secure_copy(self, source_path: Path, target_path: Path, encrypt: bool = False) -> Path:
        """Securely copy a file with optional encryption."""
        try:
            if encrypt and self.cipher_suite:
                return self.encrypt_file(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
                # Set secure permissions
                target_path.chmod(0o600)
                self.logger.debug(f"File copied securely: {source_path} -> {target_path}")
                return target_path
                
        except Exception as e:
            self.logger.error(f"Secure file copy failed: {e}")
            raise SecurityError(f"Secure file copy failed: {e}")
    
    def secure_delete(self, file_path: Path, overwrite_passes: int = 3) -> bool:
        """Securely delete a file by overwriting it multiple times."""
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            
            # Overwrite file multiple times with random data
            with open(file_path, 'r+b') as f:
                for _ in range(overwrite_passes):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            file_path.unlink()
            
            # Remove from tracking
            self.temp_files.discard(file_path)
            
            self.logger.debug(f"File securely deleted: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Secure file deletion failed for {file_path}: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up temporary files, optionally filtering by age."""
        cleaned_count = 0
        files_to_remove = set()
        
        for temp_file in self.temp_files:
            try:
                if not temp_file.exists():
                    files_to_remove.add(temp_file)
                    continue
                
                # Check age if specified
                if max_age_hours is not None:
                    file_age_hours = (datetime.now().timestamp() - temp_file.stat().st_mtime) / 3600
                    if file_age_hours < max_age_hours:
                        continue
                
                if self.secure_delete(temp_file):
                    cleaned_count += 1
                    files_to_remove.add(temp_file)
                    
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
        
        # Remove cleaned files from tracking
        self.temp_files -= files_to_remove
        
        self.logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count
    
    def get_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for integrity verification."""
        try:
            hash_func = getattr(hashlib, algorithm)()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            self.logger.error(f"File hash calculation failed: {e}")
            raise SecurityError(f"File hash calculation failed: {e}")
    
    def verify_file_integrity(self, file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity using hash comparison."""
        try:
            actual_hash = self.get_file_hash(file_path, algorithm)
            return actual_hash.lower() == expected_hash.lower()
            
        except Exception as e:
            self.logger.error(f"File integrity verification failed: {e}")
            return False
    
    def __del__(self):
        """Cleanup temporary files on destruction."""
        try:
            self.cleanup_temp_files()
        except Exception:
            pass  # Ignore errors during cleanup