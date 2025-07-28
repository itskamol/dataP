"""
Configuration templates and examples for the CLI tool.
Provides pre-built configurations for common use cases.
"""

import json
import os
from typing import Dict, Any, List


class ConfigurationTemplates:
    """Provides configuration templates for common use cases"""
    
    @staticmethod
    def get_basic_csv_matching() -> Dict[str, Any]:
        """Basic CSV to CSV matching template"""
        return {
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
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "region"],
                "from_file2": ["id", "name", "district"]
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
    
    @staticmethod
    def get_csv_to_json_matching() -> Dict[str, Any]:
        """CSV to JSON matching template"""
        return {
            "file1": {
                "path": "data/file1.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": "data/file2.json",
                "type": "json",
                "delimiter": None,
                "encoding": "utf-8"
            },
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 0.8
                },
                {
                    "file1_col": "code",
                    "file2_col": "identifier",
                    "match_type": "exact",
                    "use_normalization": False,
                    "case_sensitive": False,
                    "weight": 0.2
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "code", "region"],
                "from_file2": ["id", "full_name", "identifier", "district"]
            },
            "settings": {
                "output_format": "json",
                "matched_output_path": "results/matched_data",
                "file1_output_prefix": "csv_",
                "file2_output_prefix": "json_",
                "confidence_threshold": 80,
                "matching_type": "one-to-one",
                "unmatched_files": {
                    "generate": True
                }
            }
        }
    
    @staticmethod
    def get_uzbek_text_matching() -> Dict[str, Any]:
        """Uzbek text matching with normalization template"""
        return {
            "file1": {
                "path": "data/uzbek_regions.csv",
                "type": "csv",
                "delimiter": ";",
                "encoding": "utf-8"
            },
            "file2": {
                "path": "data/uzbek_districts.json",
                "type": "json",
                "delimiter": None,
                "encoding": "utf-8"
            },
            "mapping_fields": [
                {
                    "file1_col": "region_name_uz",
                    "file2_col": "parent_region",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "region_name_uz", "region_name_lat", "soato"],
                "from_file2": ["district_id", "district_name", "parent_region", "district_code"]
            },
            "settings": {
                "output_format": "both",
                "matched_output_path": "results/uzbek_matched",
                "file1_output_prefix": "region_",
                "file2_output_prefix": "district_",
                "confidence_threshold": 70,
                "matching_type": "one-to-many",
                "unmatched_files": {
                    "generate": True
                }
            }
        }
    
    @staticmethod
    def get_high_precision_matching() -> Dict[str, Any]:
        """High precision matching template"""
        return {
            "file1": {
                "path": "data/master_data.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": "data/reference_data.json",
                "type": "json",
                "delimiter": None,
                "encoding": "utf-8"
            },
            "mapping_fields": [
                {
                    "file1_col": "primary_name",
                    "file2_col": "reference_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 0.6
                },
                {
                    "file1_col": "code",
                    "file2_col": "ref_code",
                    "match_type": "exact",
                    "use_normalization": False,
                    "case_sensitive": False,
                    "weight": 0.4
                }
            ],
            "output_columns": {
                "from_file1": ["id", "primary_name", "code", "category"],
                "from_file2": ["ref_id", "reference_name", "ref_code", "status"]
            },
            "settings": {
                "output_format": "csv",
                "matched_output_path": "results/high_precision_matches",
                "file1_output_prefix": "master_",
                "file2_output_prefix": "ref_",
                "confidence_threshold": 90,
                "matching_type": "one-to-one",
                "unmatched_files": {
                    "generate": True
                }
            }
        }
    
    @staticmethod
    def get_bulk_processing_template() -> Dict[str, Any]:
        """Template optimized for bulk processing"""
        return {
            "file1": {
                "path": "data/large_dataset1.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "file2": {
                "path": "data/large_dataset2.csv",
                "type": "csv",
                "delimiter": ",",
                "encoding": "utf-8"
            },
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name"],
                "from_file2": ["id", "name"]
            },
            "settings": {
                "output_format": "csv",
                "matched_output_path": "results/bulk_matches",
                "file1_output_prefix": "src_",
                "file2_output_prefix": "tgt_",
                "confidence_threshold": 75,
                "matching_type": "many-to-many",
                "unmatched_files": {
                    "generate": False
                }
            }
        }
    
    @staticmethod
    def get_all_templates() -> Dict[str, Dict[str, Any]]:
        """Get all available templates"""
        return {
            "basic_csv": ConfigurationTemplates.get_basic_csv_matching(),
            "csv_to_json": ConfigurationTemplates.get_csv_to_json_matching(),
            "uzbek_text": ConfigurationTemplates.get_uzbek_text_matching(),
            "high_precision": ConfigurationTemplates.get_high_precision_matching(),
            "bulk_processing": ConfigurationTemplates.get_bulk_processing_template()
        }
    
    @staticmethod
    def save_template(template_name: str, output_path: str) -> bool:
        """Save a specific template to file"""
        templates = ConfigurationTemplates.get_all_templates()
        
        if template_name not in templates:
            return False
        
        try:
            # Create directory if output_path has a directory component
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(templates[template_name], f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving template: {e}")
            return False
    
    @staticmethod
    def list_templates() -> List[str]:
        """List all available template names"""
        return list(ConfigurationTemplates.get_all_templates().keys())
    
    @staticmethod
    def get_template_description(template_name: str) -> str:
        """Get description of a template"""
        descriptions = {
            "basic_csv": "Basic CSV to CSV matching with fuzzy text matching",
            "csv_to_json": "CSV to JSON matching with multiple field mapping",
            "uzbek_text": "Optimized for Uzbek text with normalization and regional data",
            "high_precision": "High precision matching with multiple criteria and strict thresholds",
            "bulk_processing": "Optimized for processing large datasets with minimal output"
        }
        return descriptions.get(template_name, "No description available")


class UsageExamples:
    """Provides usage examples for the CLI tool"""
    
    @staticmethod
    def get_basic_examples() -> List[str]:
        """Get basic usage examples"""
        return [
            "# Basic matching with configuration file",
            "python3 cli.py --config config.json",
            "",
            "# Direct file matching with command line options",
            "python3 cli.py --file1 data1.csv --file2 data2.json --threshold 80",
            "",
            "# Interactive mode for guided configuration",
            "python3 cli.py --interactive",
            "",
            "# Generate sample configuration",
            "python3 cli.py --generate-config --output my_config.json",
            "",
            "# Batch processing multiple configurations",
            "python3 cli.py --batch --config-dir configs/ --output-dir results/",
        ]
    
    @staticmethod
    def get_advanced_examples() -> List[str]:
        """Get advanced usage examples"""
        return [
            "# High precision matching with custom settings",
            "python3 cli.py --file1 master.csv --file2 reference.json \\",
            "  --threshold 90 --algorithm fuzzy --normalize \\",
            "  --matching-type one-to-one --format both",
            "",
            "# Uzbek text processing with normalization",
            "python3 cli.py --file1 regions_uz.csv --file2 districts_uz.json \\",
            "  --normalize --threshold 70 --prefix1 'region_' --prefix2 'district_'",
            "",
            "# Bulk processing with minimal output",
            "python3 cli.py --file1 large1.csv --file2 large2.csv \\",
            "  --matching-type many-to-many --format csv --quiet",
            "",
            "# Debug mode with verbose logging",
            "python3 cli.py --config debug_config.json \\",
            "  --verbose --log-level DEBUG --log-file debug.log",
            "",
            "# Validate configuration without processing",
            "python3 cli.py --validate-config --config production.json",
        ]
    
    @staticmethod
    def get_troubleshooting_examples() -> List[str]:
        """Get troubleshooting examples"""
        return [
            "# Check available files",
            "python3 cli.py --list-files",
            "",
            "# Inspect file columns",
            "python3 cli.py --show-columns data/myfile.csv",
            "",
            "# Test with sample data",
            "python3 cli.py --generate-config --output test.json",
            "# Edit test.json with your file paths",
            "python3 cli.py --config test.json --verbose",
            "",
            "# Debug configuration issues",
            "python3 cli.py --validate-config --config problematic.json",
            "",
            "# Run with maximum verbosity",
            "python3 cli.py --config config.json --verbose --log-level DEBUG",
        ]
    
    @staticmethod
    def save_examples_to_file(output_path: str) -> bool:
        """Save all examples to a file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# File Processing CLI - Usage Examples\n\n")
                
                f.write("## Basic Examples\n\n")
                for example in UsageExamples.get_basic_examples():
                    f.write(f"{example}\n")
                
                f.write("\n## Advanced Examples\n\n")
                for example in UsageExamples.get_advanced_examples():
                    f.write(f"{example}\n")
                
                f.write("\n## Troubleshooting Examples\n\n")
                for example in UsageExamples.get_troubleshooting_examples():
                    f.write(f"{example}\n")
            
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Demo usage
    templates = ConfigurationTemplates()
    
    print("Available templates:")
    for name in templates.list_templates():
        print(f"  {name}: {templates.get_template_description(name)}")
    
    print("\nBasic examples:")
    for example in UsageExamples.get_basic_examples():
        print(example)