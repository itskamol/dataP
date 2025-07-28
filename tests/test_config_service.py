"""
Unit tests for configuration management system.
Tests validation, environment-specific loading, hot-reload, and edge cases.
"""

import pytest
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.application.services.config_service import (
    ConfigurationManager, FileConfig, OutputConfig, ApplicationConfig
)
from src.domain.models import (
    FieldMapping, AlgorithmConfig, MatchingConfig, ValidationResult,
    AlgorithmType, MatchingType, FileType
)
from src.domain.exceptions import ConfigurationError


class TestFileConfig:
    """Test cases for FileConfig model."""
    
    def test_valid_file_config_creation(self):
        """Test creating a valid file configuration."""
        config = FileConfig(
            path="/path/to/file.csv",
            file_type=FileType.CSV,
            delimiter=",",
            encoding="utf-8"
        )
        
        assert config.path == "/path/to/file.csv"
        assert config.file_type == FileType.CSV  # Enum remains as enum object
        assert config.delimiter == ","
        assert config.encoding == "utf-8"
    
    def test_file_config_validation_errors(self):
        """Test validation errors for file configuration."""
        # Test empty path
        with pytest.raises(ValueError):
            FileConfig(path="")
    
    def test_file_config_serialization(self):
        """Test serialization and deserialization of file config."""
        config = FileConfig(
            path="/test/file.json",
            file_type=FileType.JSON,
            encoding="utf-8"
        )
        
        data = config.to_dict()
        restored = FileConfig.from_dict(data)
        
        assert restored.path == config.path
        assert restored.file_type == config.file_type
        assert restored.encoding == config.encoding


class TestOutputConfig:
    """Test cases for OutputConfig model."""
    
    def test_valid_output_config_creation(self):
        """Test creating a valid output configuration."""
        config = OutputConfig(
            format="json",
            path="/output/results",
            include_unmatched=False,
            file_prefix="test_"
        )
        
        assert config.format == "json"
        assert config.path == "/output/results"
        assert config.include_unmatched is False
        assert config.file_prefix == "test_"
    
    def test_output_config_validation_errors(self):
        """Test validation errors for output configuration."""
        # Test invalid format
        with pytest.raises(ValueError):
            OutputConfig(format="invalid_format")
    
    def test_output_config_serialization(self):
        """Test serialization and deserialization of output config."""
        config = OutputConfig(
            format="excel",
            path="/test/output",
            include_confidence_scores=False
        )
        
        data = config.to_dict()
        restored = OutputConfig.from_dict(data)
        
        assert restored.format == config.format
        assert restored.path == config.path
        assert restored.include_confidence_scores == config.include_confidence_scores


class TestApplicationConfig:
    """Test cases for ApplicationConfig model."""
    
    def test_valid_application_config_creation(self):
        """Test creating a valid application configuration."""
        file1 = FileConfig(path="/file1.csv", file_type=FileType.CSV)
        file2 = FileConfig(path="/file2.csv", file_type=FileType.CSV)
        
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        matching = MatchingConfig(
            mappings=[mapping],
            confidence_threshold=80.0
        )
        
        config = ApplicationConfig(
            file1=file1,
            file2=file2,
            matching=matching,
            max_workers=8,
            memory_limit_mb=2048
        )
        
        assert config.file1.path == "/file1.csv"
        assert config.file2.path == "/file2.csv"
        assert config.max_workers == 8
        assert config.memory_limit_mb == 2048
        assert len(config.matching.mappings) == 1
    
    def test_application_config_validation_errors(self):
        """Test validation errors for application configuration."""
        file1 = FileConfig(path="/file1.csv", file_type=FileType.CSV)
        file2 = FileConfig(path="/file2.csv", file_type=FileType.CSV)
        
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        matching = MatchingConfig(mappings=[mapping])
        
        # Test invalid max_workers (too high)
        with pytest.raises(ValueError):
            ApplicationConfig(
                file1=file1,
                file2=file2,
                matching=matching,
                max_workers=50
            )
        
        # Test invalid memory_limit_mb (too low)
        with pytest.raises(ValueError):
            ApplicationConfig(
                file1=file1,
                file2=file2,
                matching=matching,
                memory_limit_mb=64
            )
        
        # Test invalid logging_level
        with pytest.raises(ValueError):
            ApplicationConfig(
                file1=file1,
                file2=file2,
                matching=matching,
                logging_level="INVALID"
            )
    
    def test_application_config_serialization(self):
        """Test serialization and deserialization of application config."""
        file1 = FileConfig(path="/file1.csv", file_type=FileType.CSV)
        file2 = FileConfig(path="/file2.csv", file_type=FileType.CSV)
        
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        matching = MatchingConfig(mappings=[mapping])
        
        config = ApplicationConfig(
            file1=file1,
            file2=file2,
            matching=matching,
            logging_level="DEBUG"
        )
        
        data = config.to_dict()
        restored = ApplicationConfig.from_dict(data)
        
        assert restored.file1.path == config.file1.path
        assert restored.file2.path == config.file2.path
        assert restored.logging_level == config.logging_level
        assert len(restored.matching.mappings) == len(config.matching.mappings)
    
    def test_config_hash_generation(self):
        """Test configuration hash generation for change detection."""
        file1 = FileConfig(path="/file1.csv", file_type=FileType.CSV)
        file2 = FileConfig(path="/file2.csv", file_type=FileType.CSV)
        
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        matching = MatchingConfig(mappings=[mapping])
        
        config1 = ApplicationConfig(
            file1=file1,
            file2=file2,
            matching=matching
        )
        
        config2 = ApplicationConfig(
            file1=file1,
            file2=file2,
            matching=matching
        )
        
        # Same configurations should have same hash
        assert config1.get_config_hash() == config2.get_config_hash()
        
        # Different configurations should have different hashes
        config2.max_workers = 8
        assert config1.get_config_hash() != config2.get_config_hash()


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
        # Create a valid test configuration
        self.test_config = {
            "file1": {
                "path": "data/file1.csv",
                "file_type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": "data/file2.csv",
                "file_type": "csv",
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
                        "name": "fuzzy",
                        "algorithm_type": "fuzzy",
                        "parameters": {"threshold": 80},
                        "enabled": True,
                        "priority": 1
                    }
                ],
                "thresholds": {"minimum_confidence": 75.0},
                "matching_type": "one-to-one",
                "confidence_threshold": 75.0
            },
            "output": {
                "format": "csv",
                "path": "results",
                "include_unmatched": True,
                "include_confidence_scores": True,
                "file_prefix": ""
            },
            "logging_level": "INFO",
            "max_workers": 4,
            "memory_limit_mb": 1024,
            "timeout_seconds": 3600
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_manager_initialization(self):
        """Test configuration manager initialization."""
        manager = ConfigurationManager(str(self.config_path))
        
        assert manager.config_file_path == self.config_path
        assert manager._config is None
        assert not manager._hot_reload_enabled
        assert manager._environment == os.getenv('ENVIRONMENT', 'development')
    
    def test_load_valid_configuration(self):
        """Test loading a valid configuration."""
        # Write test configuration to file
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        manager = ConfigurationManager(str(self.config_path))
        config = manager.load_config()
        
        assert isinstance(config, ApplicationConfig)
        assert config.file1.path == "data/file1.csv"
        assert config.file2.path == "data/file2.csv"
        assert config.matching.confidence_threshold == 75.0
        assert len(config.matching.mappings) == 1
    
    def test_load_invalid_configuration(self):
        """Test loading an invalid configuration."""
        # Write invalid configuration to file
        invalid_config = {"invalid": "config"}
        with open(self.config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        manager = ConfigurationManager(str(self.config_path))
        
        with pytest.raises(ConfigurationError):
            manager.load_config()
    
    def test_load_malformed_json(self):
        """Test loading malformed JSON configuration."""
        # Write malformed JSON to file
        with open(self.config_path, 'w') as f:
            f.write('{"invalid": json}')
        
        manager = ConfigurationManager(str(self.config_path))
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Invalid JSON" in str(exc_info.value)
    
    def test_create_default_configuration(self):
        """Test creation of default configuration."""
        manager = ConfigurationManager(str(self.config_path))
        
        # Load config when file doesn't exist (should create default)
        config = manager.load_config()
        
        assert self.config_path.exists()
        assert isinstance(config, ApplicationConfig)
        assert config.file1.path == "data/file1.csv"
        assert config.matching.confidence_threshold == 75.0
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        manager = ConfigurationManager(str(self.config_path))
        
        # Test valid configuration
        validation_result = manager.validate_config_data(self.test_config)
        assert validation_result.is_valid
        assert not validation_result.has_errors
        
        # Test invalid configuration
        invalid_config = self.test_config.copy()
        invalid_config['max_workers'] = 100  # Too high
        
        validation_result = manager.validate_config_data(invalid_config)
        assert not validation_result.is_valid
        assert validation_result.has_errors
    
    def test_environment_specific_configuration(self):
        """Test environment-specific configuration loading."""
        # Write base configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Write environment-specific configuration
        env_config_path = self.config_path.parent / "test_config.production.json"
        env_config = {
            "logging_level": "WARNING",
            "max_workers": 8,
            "matching": {
                "confidence_threshold": 85.0
            }
        }
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f)
        
        # Test with production environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            manager = ConfigurationManager(str(self.config_path))
            config = manager.load_config()
            
            # Should have merged environment-specific values
            assert config.logging_level == "WARNING"
            assert config.max_workers == 8
            assert config.matching.confidence_threshold == 85.0
            # Base values should remain
            assert config.file1.path == "data/file1.csv"
    
    def test_environment_variable_overrides(self):
        """Test environment variable configuration overrides."""
        # Write base configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Test environment variable overrides
        env_vars = {
            'FILE_PROCESSING_LOG_LEVEL': 'DEBUG',
            'FILE_PROCESSING_MAX_WORKERS': '6',
            'FILE_PROCESSING_CONFIDENCE_THRESHOLD': '90.0',
            'FILE_PROCESSING_OUTPUT_FORMAT': 'json'
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigurationManager(str(self.config_path))
            config = manager.load_config()
            
            assert config.logging_level == "DEBUG"
            assert config.max_workers == 6
            assert config.matching.confidence_threshold == 90.0
            assert config.output.format == "json"
    
    def test_configuration_hot_reload(self):
        """Test configuration hot-reload functionality."""
        # Write initial configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        manager = ConfigurationManager(str(self.config_path))
        config = manager.load_config()
        
        # Enable hot-reload
        manager.enable_hot_reload()
        assert manager._hot_reload_enabled
        
        # Disable hot-reload
        manager.disable_hot_reload()
        assert not manager._hot_reload_enabled
    
    def test_configuration_change_detection(self):
        """Test configuration change detection."""
        # Write initial configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        manager = ConfigurationManager(str(self.config_path))
        config = manager.load_config()
        
        # Initially no change
        assert not manager.has_config_changed()
        
        # Modify configuration
        config.max_workers = 8
        assert manager.has_config_changed()
    
    def test_save_configuration(self):
        """Test saving configuration to file."""
        manager = ConfigurationManager(str(self.config_path))
        
        # Create a configuration
        file1 = FileConfig(path="/test1.csv", file_type=FileType.CSV)
        file2 = FileConfig(path="/test2.csv", file_type=FileType.CSV)
        
        mapping = FieldMapping(
            source_field="name",
            target_field="full_name",
            algorithm=AlgorithmType.FUZZY
        )
        
        matching = MatchingConfig(mappings=[mapping])
        
        config = ApplicationConfig(
            file1=file1,
            file2=file2,
            matching=matching
        )
        
        # Save configuration
        manager.save_config(config)
        
        # Verify file was created and contains correct data
        assert self.config_path.exists()
        
        with open(self.config_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['file1']['path'] == "/test1.csv"
        assert saved_data['file2']['path'] == "/test2.csv"
    
    def test_get_specific_configurations(self):
        """Test getting specific configuration sections."""
        # Write test configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        manager = ConfigurationManager(str(self.config_path))
        manager.load_config()
        
        # Test getting matching config
        matching_config = manager.get_matching_config()
        assert isinstance(matching_config, MatchingConfig)
        assert matching_config.confidence_threshold == 75.0
        
        # Test getting file configs
        file1, file2 = manager.get_file_configs()
        assert isinstance(file1, FileConfig)
        assert isinstance(file2, FileConfig)
        assert file1.path == "data/file1.csv"
        assert file2.path == "data/file2.csv"
        
        # Test getting output config
        output_config = manager.get_output_config()
        assert isinstance(output_config, OutputConfig)
        assert output_config.format == "csv"
    
    def test_configuration_warnings(self):
        """Test configuration validation warnings."""
        # Create configuration with potential issues
        config_with_warnings = self.test_config.copy()
        
        # Reference non-existent algorithm in mapping
        config_with_warnings['matching']['mappings'][0]['algorithm'] = 'nonexistent'
        
        manager = ConfigurationManager(str(self.config_path))
        validation_result = manager.validate_config_data(config_with_warnings)
        
        assert validation_result.is_valid  # Should still be valid
        assert validation_result.has_warnings
        assert any("nonexistent" in warning for warning in validation_result.warnings)
    
    def test_production_environment_validation(self):
        """Test stricter validation for production environment."""
        config_debug = self.test_config.copy()
        config_debug['logging_level'] = 'DEBUG'
        config_debug['max_workers'] = 20
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            manager = ConfigurationManager(str(self.config_path))
            validation_result = manager.validate_config_data(config_debug)
            
            assert validation_result.is_valid  # Should still be valid
            assert validation_result.has_warnings
            assert any("DEBUG logging not recommended" in warning for warning in validation_result.warnings)
            assert any("High worker count" in warning for warning in validation_result.warnings)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_concurrent_configuration_access(self):
        """Test concurrent access to configuration."""
        import threading
        
        # Create valid configuration
        test_config = {
            "file1": {"path": "file1.csv", "file_type": "csv"},
            "file2": {"path": "file2.csv", "file_type": "csv"},
            "matching": {
                "mappings": [{"source_field": "name", "target_field": "name", "algorithm": "exact"}],
                "algorithms": [{"name": "exact", "algorithm_type": "exact", "parameters": {}}],
                "thresholds": {},
                "confidence_threshold": 75.0
            },
            "output": {"format": "csv", "path": "results"}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        manager = ConfigurationManager(str(self.config_path))
        results = []
        errors = []
        
        def load_config():
            try:
                config = manager.load_config()
                results.append(config)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads to load configuration concurrently
        threads = [threading.Thread(target=load_config) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed
        assert len(errors) == 0
        assert len(results) == 5
        
        # All results should be the same configuration
        for result in results:
            assert result.file1.path == "file1.csv"
    
    def test_configuration_with_unicode_content(self):
        """Test configuration with Unicode characters."""
        unicode_config = {
            "file1": {"path": "файл1.csv", "file_type": "csv"},  # Cyrillic
            "file2": {"path": "文件2.csv", "file_type": "csv"},  # Chinese
            "matching": {
                "mappings": [{"source_field": "имя", "target_field": "姓名", "algorithm": "fuzzy"}],
                "algorithms": [{"name": "fuzzy", "algorithm_type": "fuzzy", "parameters": {}}],
                "thresholds": {},
                "confidence_threshold": 75.0
            },
            "output": {"format": "csv", "path": "результаты"}  # Cyrillic
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(unicode_config, f, ensure_ascii=False)
        
        manager = ConfigurationManager(str(self.config_path))
        config = manager.load_config()
        
        assert config.file1.path == "файл1.csv"
        assert config.file2.path == "文件2.csv"
        assert config.output.path == "результаты"
        assert config.matching.mappings[0].source_field == "имя"
        assert config.matching.mappings[0].target_field == "姓名"
    
    def test_configuration_file_permissions(self):
        """Test handling of configuration file permission issues."""
        # Create configuration file
        test_config = {
            "file1": {"path": "file1.csv", "file_type": "csv"},
            "file2": {"path": "file2.csv", "file_type": "csv"},
            "matching": {
                "mappings": [{"source_field": "name", "target_field": "name", "algorithm": "exact"}],
                "algorithms": [{"name": "exact", "algorithm_type": "exact", "parameters": {}}],
                "thresholds": {},
                "confidence_threshold": 75.0
            },
            "output": {"format": "csv", "path": "results"}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Make file read-only
        self.config_path.chmod(0o444)
        
        manager = ConfigurationManager(str(self.config_path))
        
        # Should still be able to read
        config = manager.load_config()
        assert config.file1.path == "file1.csv"
        
        # Should fail to save due to permissions
        with pytest.raises(ConfigurationError):
            manager.save_config(config)
    
    def test_deep_merge_functionality(self):
        """Test deep dictionary merging functionality."""
        manager = ConfigurationManager(str(self.config_path))
        
        base = {
            "level1": {
                "level2": {
                    "key1": "base_value1",
                    "key2": "base_value2"
                },
                "other_key": "base_other"
            },
            "top_level": "base_top"
        }
        
        override = {
            "level1": {
                "level2": {
                    "key1": "override_value1",
                    "key3": "override_value3"
                },
                "new_key": "override_new"
            },
            "new_top_level": "override_new_top"
        }
        
        result = manager._deep_merge_dict(base, override)
        
        # Check merged values
        assert result["level1"]["level2"]["key1"] == "override_value1"  # Overridden
        assert result["level1"]["level2"]["key2"] == "base_value2"      # Preserved
        assert result["level1"]["level2"]["key3"] == "override_value3"  # Added
        assert result["level1"]["other_key"] == "base_other"            # Preserved
        assert result["level1"]["new_key"] == "override_new"            # Added
        assert result["top_level"] == "base_top"                        # Preserved
        assert result["new_top_level"] == "override_new_top"            # Added


if __name__ == "__main__":
    pytest.main([__file__])