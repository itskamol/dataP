"""
Configuration management system with JSON schema validation and hot-reload capability.
Enhanced with Pydantic models for validation and serialization.
Implements requirements 6.2, 8.2, 4.1: Configuration management with validation.
"""

import json
import os
import threading
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from ...domain.models import (
    FieldMapping, AlgorithmConfig, MatchingConfig, ValidationResult,
    AlgorithmType, MatchingType, FileType
)
from ...domain.exceptions import ConfigurationError, ValidationError
from ...infrastructure.logging import get_logger


class FileConfig(BaseModel):
    """Configuration for file processing."""
    path: str = Field(..., min_length=1, description="Path to the file")
    file_type: FileType = Field(FileType.CSV, description="Type of the file")
    delimiter: Optional[str] = Field(None, description="CSV delimiter character")
    encoding: str = Field("utf-8", description="File encoding")
    sheet_name: Optional[str] = Field(None, description="Excel sheet name")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileConfig':
        """Create instance from dictionary."""
        return cls(**data)


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    format: str = Field("csv", pattern="^(csv|json|excel|both)$", description="Output format")
    path: str = Field("results", description="Output directory path")
    include_unmatched: bool = Field(True, description="Whether to include unmatched records")
    include_confidence_scores: bool = Field(True, description="Whether to include confidence scores")
    file_prefix: str = Field("", description="Prefix for output files")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputConfig':
        """Create instance from dictionary."""
        return cls(**data)


class ApplicationConfig(BaseModel):
    """Main application configuration."""
    file1: FileConfig = Field(..., description="Configuration for first file")
    file2: FileConfig = Field(..., description="Configuration for second file")
    matching: MatchingConfig = Field(..., description="Matching configuration")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    logging_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$", description="Logging level")
    max_workers: int = Field(4, ge=1, le=32, description="Maximum number of worker processes")
    memory_limit_mb: int = Field(1024, ge=128, description="Memory limit in MB")
    timeout_seconds: int = Field(3600, ge=60, description="Operation timeout in seconds")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationConfig':
        """Create instance from dictionary."""
        # Convert nested objects
        if 'file1' in data:
            data['file1'] = FileConfig.from_dict(data['file1']) if isinstance(data['file1'], dict) else data['file1']
        if 'file2' in data:
            data['file2'] = FileConfig.from_dict(data['file2']) if isinstance(data['file2'], dict) else data['file2']
        if 'matching' in data:
            data['matching'] = MatchingConfig.from_dict(data['matching']) if isinstance(data['matching'], dict) else data['matching']
        if 'output' in data:
            data['output'] = OutputConfig.from_dict(data['output']) if isinstance(data['output'], dict) else data['output']
        return cls(**data)
    
    def get_config_hash(self) -> str:
        """Generate hash of configuration for change detection."""
        config_dict = self.model_dump(mode='json')  # Use JSON mode for proper serialization
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


class ConfigurationFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reload."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = get_logger('config_service')
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.config_manager.config_file_path:
            self.logger.info(f"Configuration file changed: {event.src_path}")
            try:
                self.config_manager.reload_config()
            except Exception as e:
                self.logger.error(f"Failed to reload configuration: {str(e)}")


class ConfigurationManager:
    """Centralized configuration management with validation and hot-reload."""
    
    def __init__(self, config_file_path: str = 'config.json'):
        self.config_file_path = Path(config_file_path)
        self.logger = get_logger('config_service')
        self._config: Optional[ApplicationConfig] = None
        self._config_lock = threading.RLock()
        self._observer: Optional[Observer] = None
        self._hot_reload_enabled = False
        self._environment = os.getenv('ENVIRONMENT', 'development')
        self._config_hash: Optional[str] = None
        
        # JSON Schema for configuration validation
        self.config_schema = {
            "type": "object",
            "required": ["file1", "file2", "matching", "output"],
            "properties": {
                "file1": {
                    "type": "object",
                    "required": ["path", "type"],
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string", "enum": ["csv", "json", "excel"]},
                        "delimiter": {"type": ["string", "null"]},
                        "encoding": {"type": "string"},
                        "sheet_name": {"type": ["string", "null"]}
                    }
                },
                "file2": {
                    "type": "object",
                    "required": ["path", "type"],
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string", "enum": ["csv", "json", "excel"]},
                        "delimiter": {"type": ["string", "null"]},
                        "encoding": {"type": "string"},
                        "sheet_name": {"type": ["string", "null"]}
                    }
                },
                "matching": {
                    "type": "object",
                    "required": ["mappings", "algorithms", "thresholds"],
                    "properties": {
                        "mappings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["source_field", "target_field"],
                                "properties": {
                                    "source_field": {"type": "string"},
                                    "target_field": {"type": "string"},
                                    "algorithm": {"type": "string"},
                                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                    "normalization": {"type": "boolean"},
                                    "case_sensitive": {"type": "boolean"}
                                }
                            }
                        },
                        "algorithms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "parameters"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "parameters": {"type": "object"},
                                    "enabled": {"type": "boolean"},
                                    "priority": {"type": "integer", "minimum": 1}
                                }
                            }
                        },
                        "thresholds": {"type": "object"},
                        "matching_type": {"type": "string", "enum": ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]},
                        "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 100}
                    }
                },
                "output": {
                    "type": "object",
                    "required": ["format", "path"],
                    "properties": {
                        "format": {"type": "string", "enum": ["csv", "json", "excel", "both"]},
                        "path": {"type": "string"},
                        "include_unmatched": {"type": "boolean"},
                        "include_confidence_scores": {"type": "boolean"},
                        "file_prefix": {"type": "string"}
                    }
                },
                "logging_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                "max_workers": {"type": "integer", "minimum": 1, "maximum": 32},
                "memory_limit_mb": {"type": "integer", "minimum": 128},
                "timeout_seconds": {"type": "integer", "minimum": 60}
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> ApplicationConfig:
        """
        Load and validate configuration from file with environment-specific merging.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if config_path:
            self.config_file_path = Path(config_path)
        
        try:
            with self._config_lock:
                if not self.config_file_path.exists():
                    # Create default configuration
                    self._create_default_config()
                
                # Load base configuration
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load and merge environment-specific configuration
                config_data = self._merge_environment_config(config_data)
                
                # Validate using Pydantic models
                validation_result = self.validate_config_data(config_data)
                if not validation_result.is_valid:
                    raise ConfigurationError(
                        f"Configuration validation failed: {validation_result.errors}",
                        context={'config_file': str(self.config_file_path)}
                    )
                
                # Log warnings if any
                if validation_result.has_warnings:
                    for warning in validation_result.warnings:
                        self.logger.warning(f"Configuration warning: {warning}")
                
                # Convert to Pydantic model
                self._config = ApplicationConfig.from_dict(config_data)
                
                # Store configuration hash for change detection
                self._config_hash = self._config.get_config_hash()
                
                self.logger.info(f"Configuration loaded successfully", extra={
                    'config_file': str(self.config_file_path),
                    'environment': self._environment,
                    'matching_type': self._config.matching.matching_type,
                    'mappings_count': len(self._config.matching.mappings),
                    'config_hash': self._config_hash[:8]  # First 8 chars for logging
                })
                
                return self._config
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {str(e)}",
                context={'config_file': str(self.config_file_path), 'cause': str(e)}
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                context={'config_file': str(self.config_file_path), 'cause': str(e)}
            )
    
    def validate_config_data(self, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration data using Pydantic models.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Use Pydantic validation
            ApplicationConfig.from_dict(config_data)
            
            # Additional business logic validation
            if 'matching' in config_data:
                matching_config = config_data['matching']
                
                # Check algorithm references (only if algorithms are defined)
                if 'algorithms' in matching_config and matching_config['algorithms']:
                    available_algorithms = {alg['name'] for alg in matching_config['algorithms']}
                    for mapping in matching_config.get('mappings', []):
                        algorithm = mapping.get('algorithm')
                        if algorithm and algorithm not in available_algorithms:
                            validation_result.add_warning(
                                f"Algorithm '{algorithm}' referenced in mapping but not defined"
                            )
            
            # Check file paths exist (warnings only)
            for file_key in ['file1', 'file2']:
                if file_key in config_data:
                    file_path = config_data[file_key].get('path')
                    if file_path and not Path(file_path).exists():
                        validation_result.add_warning(f"File not found: {file_path}")
            
            # Environment-specific validation
            env = os.getenv('ENVIRONMENT', 'development')
            if env == 'production':
                # Stricter validation for production
                if config_data.get('logging_level') == 'DEBUG':
                    validation_result.add_warning("DEBUG logging not recommended for production")
                
                if config_data.get('max_workers', 4) > 16:
                    validation_result.add_warning("High worker count may impact performance in production")
            
        except PydanticValidationError as e:
            validation_result.is_valid = False
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error['loc'])
                validation_result.add_error(f"{field_path}: {error['msg']}")
        except Exception as e:
            validation_result.add_error(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _merge_environment_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge environment-specific configuration with base configuration.
        
        Args:
            base_config: Base configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        # Look for environment-specific config file
        env_config_path = self.config_file_path.parent / f"config.{self._environment}.json"
        
        if env_config_path.exists():
            try:
                with open(env_config_path, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)
                
                # Deep merge environment config into base config
                merged_config = self._deep_merge_dict(base_config, env_config)
                
                self.logger.info(f"Environment-specific configuration merged: {env_config_path}")
                return merged_config
                
            except Exception as e:
                self.logger.warning(f"Failed to load environment config {env_config_path}: {str(e)}")
        
        # Override with environment variables
        merged_config = self._apply_environment_variables(base_config)
        
        return merged_config
    
    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variable overrides
        """
        # Define environment variable mappings
        env_mappings = {
            'FILE_PROCESSING_LOG_LEVEL': ['logging_level'],
            'FILE_PROCESSING_MAX_WORKERS': ['max_workers'],
            'FILE_PROCESSING_MEMORY_LIMIT': ['memory_limit_mb'],
            'FILE_PROCESSING_TIMEOUT': ['timeout_seconds'],
            'FILE_PROCESSING_FILE1_PATH': ['file1', 'path'],
            'FILE_PROCESSING_FILE2_PATH': ['file2', 'path'],
            'FILE_PROCESSING_OUTPUT_PATH': ['output', 'path'],
            'FILE_PROCESSING_OUTPUT_FORMAT': ['output', 'format'],
            'FILE_PROCESSING_CONFIDENCE_THRESHOLD': ['matching', 'confidence_threshold']
        }
        
        result = config.copy()
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested dictionary location
                current = result
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                final_key = config_path[-1]
                if final_key in ['max_workers', 'memory_limit_mb', 'timeout_seconds']:
                    current[final_key] = int(env_value)
                elif final_key == 'confidence_threshold':
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value
                
                self.logger.info(f"Applied environment variable {env_var} = {env_value}")
        
        return result
    
    def has_config_changed(self) -> bool:
        """
        Check if configuration has changed since last load.
        
        Returns:
            True if configuration has changed
        """
        if self._config is None or self._config_hash is None:
            return True
        
        current_hash = self._config.get_config_hash()
        return current_hash != self._config_hash
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "file1": {
                "path": "data/file1.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": "data/file2.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "matching": {
                "mappings": [
                    {
                        "source_field": "name",
                        "target_field": "name",
                        "algorithm": "fuzzy",
                        "weight": 1.0,
                        "normalization": True,
                        "case_sensitive": False
                    }
                ],
                "algorithms": [
                    {
                        "name": "exact",
                        "parameters": {},
                        "enabled": True,
                        "priority": 1
                    },
                    {
                        "name": "fuzzy",
                        "parameters": {
                            "threshold": 80
                        },
                        "enabled": True,
                        "priority": 2
                    }
                ],
                "thresholds": {
                    "minimum_confidence": 75.0
                },
                "matching_type": "one-to-one",
                "confidence_threshold": 75.0
            },
            "output": {
                "format": "csv",
                "path": "results/matched_results",
                "include_unmatched": True,
                "include_confidence_scores": True,
                "file_prefix": ""
            },
            "logging_level": "INFO",
            "max_workers": 4,
            "memory_limit_mb": 1024,
            "timeout_seconds": 3600
        }
        
        # Create directory if it doesn't exist
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Default configuration created: {self.config_file_path}")
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration."""
        with self._config_lock:
            if self._config is None:
                return self.load_config()
            return self._config
    
    def reload_config(self):
        """Reload configuration from file."""
        self.logger.info("Reloading configuration...")
        self.load_config()
    
    def enable_hot_reload(self):
        """Enable hot-reload of configuration file."""
        if self._hot_reload_enabled:
            return
        
        try:
            self._observer = Observer()
            event_handler = ConfigurationFileHandler(self)
            self._observer.schedule(
                event_handler,
                str(self.config_file_path.parent),
                recursive=False
            )
            self._observer.start()
            self._hot_reload_enabled = True
            
            self.logger.info("Configuration hot-reload enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to enable hot-reload: {str(e)}")
    
    def disable_hot_reload(self):
        """Disable hot-reload of configuration file."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._hot_reload_enabled = False
            
            self.logger.info("Configuration hot-reload disabled")
    
    def save_config(self, config: ApplicationConfig, config_path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Optional path to save configuration
        """
        if config_path:
            save_path = Path(config_path)
        else:
            save_path = self.config_file_path
        
        try:
            # Convert Pydantic model to dictionary with JSON serialization
            config_dict = config.model_dump(mode='json')
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved: {save_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {str(e)}",
                context={'config_file': str(save_path), 'cause': str(e)}
            )
    
    def get_matching_config(self) -> MatchingConfig:
        """Get matching-specific configuration."""
        return self.get_config().matching
    
    def get_file_configs(self) -> tuple[FileConfig, FileConfig]:
        """Get file configurations."""
        config = self.get_config()
        return config.file1, config.file2
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        return self.get_config().output