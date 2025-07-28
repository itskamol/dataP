"""
CLI Configuration Service for handling command-line specific configuration needs.
Extends the base ConfigurationManager with CLI-specific functionality.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError

from .config_service import ConfigurationManager
from ...domain.exceptions import ConfigurationError


class CLIConfigurationManager(ConfigurationManager):
    """Extended configuration manager for CLI operations"""
    
    def __init__(self):
        super().__init__()
        self.cli_schema = self._get_cli_config_schema()
    
    def _get_cli_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for CLI configuration validation"""
        return {
            "type": "object",
            "properties": {
                "file1": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string", "enum": ["csv", "json"]},
                        "delimiter": {"type": ["string", "null"]},
                        "encoding": {"type": "string", "default": "utf-8"}
                    },
                    "required": ["path", "type"]
                },
                "file2": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string", "enum": ["csv", "json"]},
                        "delimiter": {"type": ["string", "null"]},
                        "encoding": {"type": "string", "default": "utf-8"}
                    },
                    "required": ["path", "type"]
                },
                "mapping_fields": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "file1_col": {"type": "string"},
                            "file2_col": {"type": "string"},
                            "match_type": {"type": "string", "enum": ["exact", "fuzzy", "phonetic"]},
                            "use_normalization": {"type": "boolean", "default": False},
                            "case_sensitive": {"type": "boolean", "default": False},
                            "weight": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 1.0}
                        },
                        "required": ["file1_col", "file2_col", "match_type"]
                    }
                },
                "output_columns": {
                    "type": "object",
                    "properties": {
                        "from_file1": {"type": "array", "items": {"type": "string"}},
                        "from_file2": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["from_file1", "from_file2"]
                },
                "settings": {
                    "type": "object",
                    "properties": {
                        "output_format": {"type": "string", "enum": ["csv", "json", "both"], "default": "csv"},
                        "matched_output_path": {"type": "string"},
                        "file1_output_prefix": {"type": "string", "default": "f1_"},
                        "file2_output_prefix": {"type": "string", "default": "f2_"},
                        "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 100, "default": 75},
                        "matching_type": {"type": "string", "enum": ["one-to-one", "one-to-many", "many-to-one", "many-to-many"], "default": "one-to-one"},
                        "unmatched_files": {
                            "type": "object",
                            "properties": {
                                "generate": {"type": "boolean", "default": False}
                            }
                        }
                    },
                    "required": ["matched_output_path"]
                }
            },
            "required": ["file1", "file2", "mapping_fields", "output_columns", "settings"]
        }
    
    def validate_cli_config(self, config: Dict[str, Any]) -> None:
        """Validate CLI configuration against schema"""
        try:
            validate(instance=config, schema=self.cli_schema)
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e.message}")
    
    def load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file"""
        try:
            config = self.load_config(config_path)
            self.validate_cli_config(config)
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def create_config_template(self, output_path: str, 
                             file1_path: str = None, 
                             file2_path: str = None,
                             sample_mappings: List[Dict] = None) -> bool:
        """Create a configuration template file"""
        template = {
            "file1": {
                "path": file1_path or "path/to/first/file.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": file2_path or "path/to/second/file.json",
                "type": "json",
                "delimiter": None,
                "encoding": "utf-8"
            },
            "mapping_fields": sample_mappings or [
                {
                    "file1_col": "name_column",
                    "file2_col": "name_field",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "region"],
                "from_file2": ["identifier", "full_name", "district"]
            },
            "settings": {
                "output_format": "csv",
                "matched_output_path": "results/matched_data",
                "file1_output_prefix": "f1_",
                "file2_output_prefix": "f2_",
                "confidence_threshold": 75,
                "matching_type": "one-to-one",
                "unmatched_files": {
                    "generate": True
                }
            }
        }
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error creating configuration template: {e}", file=sys.stderr)
            return False
    
    def merge_cli_args_with_config(self, config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge CLI arguments with configuration file, CLI args take precedence"""
        merged_config = config.copy()
        
        # Override file paths if provided
        if cli_args.get('file1'):
            merged_config['file1']['path'] = cli_args['file1']
        if cli_args.get('file2'):
            merged_config['file2']['path'] = cli_args['file2']
            
        # Override file types if provided
        if cli_args.get('file1_type') and cli_args['file1_type'] != 'auto':
            merged_config['file1']['type'] = cli_args['file1_type']
        if cli_args.get('file2_type') and cli_args['file2_type'] != 'auto':
            merged_config['file2']['type'] = cli_args['file2_type']
            
        # Override delimiters if provided
        if cli_args.get('delimiter1'):
            merged_config['file1']['delimiter'] = cli_args['delimiter1']
        if cli_args.get('delimiter2'):
            merged_config['file2']['delimiter'] = cli_args['delimiter2']
            
        # Override settings if provided
        settings_overrides = {
            'threshold': 'confidence_threshold',
            'matching_type': 'matching_type',
            'format': 'output_format',
            'output': 'matched_output_path',
            'prefix1': 'file1_output_prefix',
            'prefix2': 'file2_output_prefix'
        }
        
        for cli_key, config_key in settings_overrides.items():
            if cli_args.get(cli_key) is not None:
                merged_config['settings'][config_key] = cli_args[cli_key]
                
        # Override unmatched files generation
        if cli_args.get('include_unmatched') is not None:
            merged_config['settings']['unmatched_files']['generate'] = cli_args['include_unmatched']
            
        # Override matching algorithm settings
        if cli_args.get('algorithm'):
            for mapping in merged_config['mapping_fields']:
                mapping['match_type'] = cli_args['algorithm']
                
        if cli_args.get('normalize') is not None:
            for mapping in merged_config['mapping_fields']:
                mapping['use_normalization'] = cli_args['normalize']
                
        if cli_args.get('case_sensitive') is not None:
            for mapping in merged_config['mapping_fields']:
                mapping['case_sensitive'] = cli_args['case_sensitive']
        
        return merged_config
    
    def auto_detect_file_columns(self, file_path: str, file_type: str = None, delimiter: str = None) -> List[str]:
        """Auto-detect columns in a file"""
        import pandas as pd
        
        if file_type is None:
            file_type = 'csv' if file_path.endswith('.csv') else 'json'
        
        try:
            if file_type == 'csv':
                # Try different delimiters if not specified
                delimiters = [delimiter] if delimiter else [',', ';', '\t', '|']
                
                for delim in delimiters:
                    try:
                        df = pd.read_csv(file_path, delimiter=delim, nrows=0)
                        return df.columns.tolist()
                    except:
                        continue
                        
                # Fallback: try without specifying delimiter
                df = pd.read_csv(file_path, nrows=0)
                return df.columns.tolist()
                
            elif file_type == 'json':
                try:
                    df = pd.read_json(file_path, lines=True, nrows=1)
                except:
                    df = pd.read_json(file_path, nrows=1)
                return df.columns.tolist()
                
        except Exception as e:
            raise ConfigurationError(f"Could not detect columns in {file_path}: {str(e)}")
        
        return []
    
    def suggest_column_mappings(self, file1_columns: List[str], file2_columns: List[str]) -> List[Dict[str, Any]]:
        """Suggest potential column mappings based on column names"""
        from difflib import SequenceMatcher
        
        suggestions = []
        used_file2_cols = set()
        
        for col1 in file1_columns:
            best_match = None
            best_score = 0.0
            
            for col2 in file2_columns:
                if col2 in used_file2_cols:
                    continue
                    
                # Calculate similarity score
                score = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
                
                # Boost score for common patterns
                if any(keyword in col1.lower() and keyword in col2.lower() 
                       for keyword in ['name', 'id', 'code', 'region', 'district', 'city']):
                    score += 0.2
                
                if score > best_score and score > 0.3:  # Minimum similarity threshold
                    best_match = col2
                    best_score = score
            
            if best_match:
                suggestions.append({
                    "file1_col": col1,
                    "file2_col": best_match,
                    "match_type": "fuzzy" if best_score < 0.9 else "exact",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": min(1.0, best_score + 0.1),
                    "confidence": best_score
                })
                used_file2_cols.add(best_match)
        
        return suggestions
    
    def create_interactive_config(self, file1_path: str, file2_path: str) -> Dict[str, Any]:
        """Create configuration interactively by analyzing files"""
        try:
            # Detect file types
            file1_type = 'csv' if file1_path.endswith('.csv') else 'json'
            file2_type = 'csv' if file2_path.endswith('.csv') else 'json'
            
            # Detect columns
            file1_columns = self.auto_detect_file_columns(file1_path, file1_type)
            file2_columns = self.auto_detect_file_columns(file2_path, file2_type)
            
            # Suggest mappings
            suggested_mappings = self.suggest_column_mappings(file1_columns, file2_columns)
            
            # Create configuration
            config = {
                "file1": {
                    "path": file1_path,
                    "type": file1_type,
                    "delimiter": "," if file1_type == "csv" else None,
                    "encoding": "utf-8"
                },
                "file2": {
                    "path": file2_path,
                    "type": file2_type,
                    "delimiter": "," if file2_type == "csv" else None,
                    "encoding": "utf-8"
                },
                "mapping_fields": suggested_mappings[:3] if suggested_mappings else [
                    {
                        "file1_col": file1_columns[0] if file1_columns else "column1",
                        "file2_col": file2_columns[0] if file2_columns else "column1",
                        "match_type": "fuzzy",
                        "use_normalization": True,
                        "case_sensitive": False,
                        "weight": 1.0
                    }
                ],
                "output_columns": {
                    "from_file1": file1_columns,
                    "from_file2": file2_columns
                },
                "settings": {
                    "output_format": "csv",
                    "matched_output_path": "results/matched_data",
                    "file1_output_prefix": "f1_",
                    "file2_output_prefix": "f2_",
                    "confidence_threshold": 75,
                    "matching_type": "one-to-one",
                    "unmatched_files": {
                        "generate": True
                    }
                }
            }
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create interactive configuration: {str(e)}")
    
    def validate_file_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validate that file paths in configuration exist and are accessible"""
        errors = []
        
        for file_key in ['file1', 'file2']:
            file_config = config.get(file_key, {})
            file_path = file_config.get('path')
            
            if not file_path:
                errors.append(f"{file_key}: path is required")
                continue
                
            if not os.path.exists(file_path):
                errors.append(f"{file_key}: file not found: {file_path}")
                continue
                
            if not os.path.isfile(file_path):
                errors.append(f"{file_key}: path is not a file: {file_path}")
                continue
                
            if not os.access(file_path, os.R_OK):
                errors.append(f"{file_key}: file is not readable: {file_path}")
                continue
                
            # Validate file type matches extension
            file_type = file_config.get('type')
            if file_type == 'csv' and not file_path.endswith('.csv'):
                errors.append(f"{file_key}: file type 'csv' but file doesn't end with .csv")
            elif file_type == 'json' and not file_path.endswith('.json'):
                errors.append(f"{file_key}: file type 'json' but file doesn't end with .json")
        
        return errors
    
    def validate_output_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validate output paths and ensure directories are writable"""
        errors = []
        
        output_path = config.get('settings', {}).get('matched_output_path')
        if not output_path:
            errors.append("settings.matched_output_path is required")
            return errors
        
        # Check if output directory exists or can be created
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                if not os.access(output_dir, os.W_OK):
                    errors.append(f"Output directory is not writable: {output_dir}")
            except Exception as e:
                errors.append(f"Cannot create output directory {output_dir}: {str(e)}")
        
        return errors
    
    def comprehensive_config_validation(self, config: Dict[str, Any]) -> List[str]:
        """Perform comprehensive configuration validation"""
        errors = []
        
        # Schema validation
        try:
            self.validate_cli_config(config)
        except ConfigurationError as e:
            errors.append(str(e))
        
        # File path validation
        errors.extend(self.validate_file_paths(config))
        
        # Output path validation
        errors.extend(self.validate_output_paths(config))
        
        # Mapping validation
        mapping_fields = config.get('mapping_fields', [])
        if not mapping_fields:
            errors.append("At least one mapping field is required")
        
        for i, mapping in enumerate(mapping_fields):
            if mapping.get('weight', 1.0) <= 0:
                errors.append(f"mapping_fields[{i}]: weight must be greater than 0")
        
        return errors