"""
Integration tests for the unified CLI interface.
Tests various CLI usage scenarios and argument combinations.
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from cli import FileMatchingCLI


class TestCLIIntegration:
    """Test CLI integration scenarios"""
    
    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing"""
        return FileMatchingCLI()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_csv_file1(self, temp_dir):
        """Create sample CSV file 1"""
        data = {
            'id': [1, 2, 3],
            'name': ['Toshkent', 'Samarqand', 'Buxoro'],
            'region': ['Toshkent viloyati', 'Samarqand viloyati', 'Buxoro viloyati']
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'file1.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def sample_csv_file2(self, temp_dir):
        """Create sample CSV file 2"""
        data = {
            'district_id': [101, 102, 103],
            'district_name': ['Toshkent tumani', 'Samarqand tumani', 'Buxoro tumani'],
            'parent_region': ['Toshkent', 'Samarqand', 'Buxoro']
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'file2.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create sample JSON file"""
        data = [
            {'id': 1, 'name': 'Toshkent shahri', 'code': 'TSH'},
            {'id': 2, 'name': 'Samarqand shahri', 'code': 'SMQ'},
            {'id': 3, 'name': 'Buxoro shahri', 'code': 'BUX'}
        ]
        file_path = os.path.join(temp_dir, 'file.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path
    
    @pytest.fixture
    def sample_config(self, temp_dir, sample_csv_file1, sample_csv_file2):
        """Create sample configuration file"""
        config = {
            "file1": {
                "path": sample_csv_file1,
                "type": "csv",
                "delimiter": ","
            },
            "file2": {
                "path": sample_csv_file2,
                "type": "csv",
                "delimiter": ","
            },
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "parent_region",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "region"],
                "from_file2": ["district_id", "district_name", "parent_region"]
            },
            "settings": {
                "output_format": "csv",
                "matched_output_path": os.path.join(temp_dir, "matched_results"),
                "file1_output_prefix": "f1_",
                "file2_output_prefix": "f2_",
                "confidence_threshold": 75,
                "matching_type": "one-to-one",
                "unmatched_files": {
                    "generate": True
                }
            }
        }
        
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        return config_path
    
    def test_argument_parser_creation(self, cli):
        """Test that argument parser is created correctly"""
        parser = cli.create_parser()
        assert parser is not None
        
        # Test that all major argument groups exist
        help_text = parser.format_help()
        assert 'Input Options' in help_text
        assert 'Matching Options' in help_text
        assert 'Output Options' in help_text
        assert 'Processing Options' in help_text
        assert 'Interface Options' in help_text
        assert 'Utility Options' in help_text
        assert 'Logging Options' in help_text
    
    def test_validate_args_success(self, cli, sample_config):
        """Test successful argument validation"""
        parser = cli.create_parser()
        args = parser.parse_args(['--config', sample_config])
        assert cli.validate_args(args) is True
    
    def test_validate_args_missing_input(self, cli):
        """Test validation failure with missing input"""
        parser = cli.create_parser()
        args = parser.parse_args([])
        assert cli.validate_args(args) is False
    
    def test_validate_args_invalid_threshold(self, cli, sample_config):
        """Test validation failure with invalid threshold"""
        parser = cli.create_parser()
        args = parser.parse_args(['--config', sample_config, '--threshold', '150'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_conflicting_options(self, cli, sample_config):
        """Test validation failure with conflicting options"""
        parser = cli.create_parser()
        args = parser.parse_args(['--config', sample_config, '--verbose', '--quiet'])
        assert cli.validate_args(args) is False
    
    def test_generate_sample_config(self, cli, temp_dir):
        """Test sample configuration generation"""
        output_path = os.path.join(temp_dir, 'sample.json')
        result = cli.generate_sample_config(output_path)
        
        assert result is True
        assert os.path.exists(output_path)
        
        # Verify the generated config is valid JSON
        with open(output_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        assert 'file1' in config
        assert 'file2' in config
        assert 'mapping_fields' in config
        assert 'output_columns' in config
        assert 'settings' in config
    
    def test_list_available_files(self, cli, temp_dir, sample_csv_file1, sample_csv_file2):
        """Test listing available files"""
        # Change to temp directory to test file discovery
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # This should not raise an exception
            cli.list_available_files()
        finally:
            os.chdir(original_cwd)
    
    def test_show_file_columns_csv(self, cli, sample_csv_file1):
        """Test showing columns for CSV file"""
        # This should not raise an exception
        cli.show_file_columns(sample_csv_file1)
    
    def test_show_file_columns_json(self, cli, sample_json_file):
        """Test showing columns for JSON file"""
        # This should not raise an exception
        cli.show_file_columns(sample_json_file)
    
    def test_detect_file_type(self, cli):
        """Test file type detection"""
        assert cli._detect_file_type('test.csv') == 'csv'
        assert cli._detect_file_type('test.json') == 'json'
        assert cli._detect_file_type('test.txt') == 'csv'  # Default fallback
    
    def test_create_config_from_args(self, cli, sample_csv_file1, sample_csv_file2, temp_dir):
        """Test configuration creation from command line arguments"""
        parser = cli.create_parser()
        args = parser.parse_args([
            '--file1', sample_csv_file1,
            '--file2', sample_csv_file2,
            '--output', os.path.join(temp_dir, 'results'),
            '--threshold', '80',
            '--algorithm', 'fuzzy',
            '--normalize',
            '--matching-type', 'one-to-many',
            '--format', 'json'
        ])
        
        config = cli.create_config_from_args(args)
        
        assert config['file1']['path'] == sample_csv_file1
        assert config['file2']['path'] == sample_csv_file2
        assert config['settings']['confidence_threshold'] == 80
        assert config['settings']['matching_type'] == 'one-to-many'
        assert config['settings']['output_format'] == 'json'
        assert config['mapping_fields'][0]['use_normalization'] is True
    
    @patch('cli.run_processing_optimized')
    def test_run_with_config_file(self, mock_processing, cli, sample_config):
        """Test running CLI with configuration file"""
        mock_processing.return_value = None
        
        result = cli.run(['--config', sample_config])
        
        assert result == 0
        mock_processing.assert_called_once()
    
    @patch('cli.run_processing_optimized')
    def test_run_with_command_line_args(self, mock_processing, cli, sample_csv_file1, sample_csv_file2, temp_dir):
        """Test running CLI with command line arguments"""
        mock_processing.return_value = None
        
        result = cli.run([
            '--file1', sample_csv_file1,
            '--file2', sample_csv_file2,
            '--output', os.path.join(temp_dir, 'results'),
            '--threshold', '80'
        ])
        
        assert result == 0
        mock_processing.assert_called_once()
    
    def test_run_generate_config(self, cli, temp_dir):
        """Test running CLI to generate configuration"""
        output_path = os.path.join(temp_dir, 'generated.json')
        result = cli.run(['--generate-config', '--output', output_path])
        
        assert result == 0
        assert os.path.exists(output_path)
    
    def test_run_list_files(self, cli):
        """Test running CLI to list files"""
        result = cli.run(['--list-files'])
        assert result == 0
    
    def test_run_show_columns(self, cli, sample_csv_file1):
        """Test running CLI to show file columns"""
        result = cli.run(['--show-columns', sample_csv_file1])
        assert result == 0
    
    def test_run_validate_config(self, cli, sample_config):
        """Test running CLI to validate configuration"""
        with patch.object(cli.config_manager, 'load_config') as mock_load:
            mock_load.return_value = {}
            result = cli.run(['--validate-config', '--config', sample_config])
            assert result == 0
            mock_load.assert_called_once_with(sample_config)
    
    def test_run_validate_config_invalid(self, cli, temp_dir):
        """Test running CLI to validate invalid configuration"""
        invalid_config = os.path.join(temp_dir, 'invalid.json')
        with open(invalid_config, 'w') as f:
            f.write('invalid json')
        
        result = cli.run(['--validate-config', '--config', invalid_config])
        assert result == 1
    
    @patch('cli.create_config_interactively')
    @patch('cli.run_processing_optimized')
    def test_run_interactive_mode(self, mock_processing, mock_interactive, cli):
        """Test running CLI in interactive mode"""
        mock_interactive.return_value = {'test': 'config'}
        mock_processing.return_value = None
        
        result = cli.run(['--interactive'])
        
        assert result == 0
        mock_interactive.assert_called_once()
        mock_processing.assert_called_once()
    
    @patch('cli.create_config_interactively')
    def test_run_interactive_mode_cancelled(self, mock_interactive, cli):
        """Test running CLI in interactive mode when cancelled"""
        mock_interactive.return_value = None
        
        result = cli.run(['--interactive'])
        
        assert result == 0  # Should still return 0 for cancelled operation
        mock_interactive.assert_called_once()
    
    def test_run_batch_processing(self, cli, temp_dir, sample_config):
        """Test batch processing mode"""
        # Create config directory with multiple configs
        config_dir = os.path.join(temp_dir, 'configs')
        os.makedirs(config_dir)
        
        # Copy sample config to config directory
        shutil.copy(sample_config, os.path.join(config_dir, 'config1.json'))
        shutil.copy(sample_config, os.path.join(config_dir, 'config2.json'))
        
        output_dir = os.path.join(temp_dir, 'batch_results')
        
        with patch('cli.run_processing_optimized') as mock_processing:
            mock_processing.return_value = None
            result = cli.run(['--batch', '--config-dir', config_dir, '--output-dir', output_dir])
            
            assert result == 0
            assert mock_processing.call_count == 2  # Should process both configs
    
    def test_run_batch_processing_no_configs(self, cli, temp_dir):
        """Test batch processing with no configuration files"""
        config_dir = os.path.join(temp_dir, 'empty_configs')
        os.makedirs(config_dir)
        
        result = cli.run(['--batch', '--config-dir', config_dir])
        assert result == 0  # Should handle gracefully
    
    def test_run_with_keyboard_interrupt(self, cli, sample_config):
        """Test handling of keyboard interrupt"""
        with patch('cli.run_processing_optimized') as mock_processing:
            mock_processing.side_effect = KeyboardInterrupt()
            result = cli.run(['--config', sample_config])
            assert result == 130  # Standard exit code for SIGINT
    
    def test_run_with_exception(self, cli, sample_config):
        """Test handling of general exceptions"""
        with patch('cli.run_processing_optimized') as mock_processing:
            mock_processing.side_effect = Exception("Test error")
            result = cli.run(['--config', sample_config])
            assert result == 1
    
    def test_run_with_verbose_exception(self, cli, sample_config):
        """Test handling of exceptions with verbose output"""
        with patch('cli.run_processing_optimized') as mock_processing:
            mock_processing.side_effect = Exception("Test error")
            result = cli.run(['--config', sample_config, '--verbose'])
            assert result == 1
    
    def test_format_file_size(self, cli):
        """Test file size formatting"""
        assert cli._format_file_size(500) == "500.0B"
        assert cli._format_file_size(1500) == "1.5KB"
        assert cli._format_file_size(1500000) == "1.4MB"
        assert cli._format_file_size(1500000000) == "1.4GB"
    
    def test_get_output_path(self, cli, temp_dir):
        """Test output path generation"""
        parser = cli.create_parser()
        
        # Test without output directory
        args = parser.parse_args(['--output', 'results'])
        assert cli._get_output_path(args) == 'results'
        
        # Test with output directory
        args = parser.parse_args(['--output', 'results', '--output-dir', temp_dir])
        expected = os.path.join(temp_dir, 'results')
        assert cli._get_output_path(args) == expected


class TestCLIProgressCallback:
    """Test CLI progress callback functionality"""
    
    def test_progress_callback_verbose(self, capsys):
        """Test progress callback with verbose output"""
        callback = CLIProgressCallback(verbose=True)
        callback(50, 100, "Processing...")
        
        captured = capsys.readouterr()
        assert "50.00%" in captured.out
        assert "Processing..." in captured.out
        assert "ETA:" in captured.out
    
    def test_progress_callback_simple(self, capsys):
        """Test progress callback with simple output"""
        callback = CLIProgressCallback(verbose=False)
        callback(25, 100, "Loading...")
        
        captured = capsys.readouterr()
        assert "25.00%" in captured.out
        assert "Loading..." in captured.out
        assert "ETA:" not in captured.out
    
    def test_progress_callback_no_total(self, capsys):
        """Test progress callback without total"""
        callback = CLIProgressCallback(verbose=True)
        callback(0, 0, "Initializing...")
        
        captured = capsys.readouterr()
        assert "Initializing..." in captured.out


class TestCLIArgumentValidation:
    """Test CLI argument validation edge cases"""
    
    @pytest.fixture
    def cli(self):
        return FileMatchingCLI()
    
    def test_validate_args_file1_only(self, cli):
        """Test validation with only file1 specified"""
        parser = cli.create_parser()
        args = parser.parse_args(['--file1', 'test.csv'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_file2_only(self, cli):
        """Test validation with only file2 specified"""
        parser = cli.create_parser()
        args = parser.parse_args(['--file2', 'test.csv'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_nonexistent_config(self, cli):
        """Test validation with non-existent config file"""
        parser = cli.create_parser()
        args = parser.parse_args(['--config', 'nonexistent.json'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_nonexistent_files(self, cli):
        """Test validation with non-existent input files"""
        parser = cli.create_parser()
        args = parser.parse_args(['--file1', 'nonexistent1.csv', '--file2', 'nonexistent2.csv'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_threshold_boundaries(self, cli, temp_dir):
        """Test threshold boundary validation"""
        # Create dummy config file
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({}, f)
        
        parser = cli.create_parser()
        
        # Test valid boundaries
        args = parser.parse_args(['--config', config_path, '--threshold', '0'])
        assert cli.validate_args(args) is True
        
        args = parser.parse_args(['--config', config_path, '--threshold', '100'])
        assert cli.validate_args(args) is True
        
        # Test invalid boundaries
        args = parser.parse_args(['--config', config_path, '--threshold', '-1'])
        assert cli.validate_args(args) is False
        
        args = parser.parse_args(['--config', config_path, '--threshold', '101'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_batch_without_config_dir(self, cli):
        """Test batch processing validation without config directory"""
        parser = cli.create_parser()
        args = parser.parse_args(['--batch'])
        assert cli.validate_args(args) is False
    
    def test_validate_args_nonexistent_config_dir(self, cli):
        """Test validation with non-existent config directory"""
        parser = cli.create_parser()
        args = parser.parse_args(['--batch', '--config-dir', 'nonexistent_dir'])
        assert cli.validate_args(args) is False


if __name__ == '__main__':
    pytest.main([__file__])