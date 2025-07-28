#!/usr/bin/env python3
"""
Unified CLI tool for file processing and data matching system.
Combines functionality from main.py and main.py with modern argument parsing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import shutil

# Import templates and examples
from cli_templates import ConfigurationTemplates, UsageExamples


class CLIProgressCallback:
    """Progress callback for CLI operations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        
    def __call__(self, current: int, total: int, message: str = ""):
        if self.verbose:
            elapsed = time.time() - self.start_time
            if total > 0:
                percent = (current / total) * 100
                eta = (elapsed / current) * (total - current) if current > 0 else 0
                print(f"\r[{percent:6.2f}%] {message} (ETA: {eta:.1f}s)", end="", flush=True)
            else:
                print(f"\r{message} ({elapsed:.1f}s)", end="", flush=True)
        else:
            if total > 0:
                percent = (current / total) * 100
                print(f"\r[{percent:6.2f}%] {message}", end="", flush=True)
            else:
                print(f"\r{message}", end="", flush=True)


class SimpleConfigManager:
    """Simple configuration manager for CLI"""
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")


class FileMatchingCLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.config_manager = SimpleConfigManager()
        self.logger = None
        
    def setup_logging(self, log_level: str, log_file: Optional[str] = None):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file if log_file else None
        )
        self.logger = logging.getLogger(__name__)
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all CLI options"""
        parser = argparse.ArgumentParser(
            description="File Processing and Data Matching System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --config config.json --output results/matched
  %(prog)s --file1 data1.csv --file2 data2.json --interactive
  %(prog)s --batch --config-dir configs/ --output-dir results/
  %(prog)s --generate-config --output sample_config.json
            """
        )
        
        # Input options
        input_group = parser.add_argument_group('Input Options')
        input_group.add_argument(
            '--config', '-c',
            type=str,
            help='Configuration file path (JSON format)'
        )
        input_group.add_argument(
            '--file1',
            type=str,
            help='First input file path'
        )
        input_group.add_argument(
            '--file2', 
            type=str,
            help='Second input file path'
        )
        input_group.add_argument(
            '--file1-type',
            choices=['csv', 'json', 'auto'],
            default='auto',
            help='First file type (default: auto-detect)'
        )
        input_group.add_argument(
            '--file2-type',
            choices=['csv', 'json', 'auto'],
            default='auto',
            help='Second file type (default: auto-detect)'
        )
        input_group.add_argument(
            '--delimiter1',
            type=str,
            help='Delimiter for first CSV file (auto-detect if not specified)'
        )
        input_group.add_argument(
            '--delimiter2',
            type=str,
            help='Delimiter for second CSV file (auto-detect if not specified)'
        )
        
        # Matching options
        matching_group = parser.add_argument_group('Matching Options')
        matching_group.add_argument(
            '--matching-type',
            choices=['one-to-one', 'one-to-many', 'many-to-one', 'many-to-many'],
            default='one-to-one',
            help='Type of matching to perform (default: one-to-one)'
        )
        matching_group.add_argument(
            '--threshold',
            type=float,
            default=75.0,
            help='Confidence threshold for matches (0-100, default: 75)'
        )
        matching_group.add_argument(
            '--algorithm',
            choices=['exact', 'fuzzy', 'phonetic', 'mixed'],
            default='fuzzy',
            help='Matching algorithm to use (default: fuzzy)'
        )
        matching_group.add_argument(
            '--normalize',
            action='store_true',
            help='Enable text normalization for Uzbek text'
        )
        matching_group.add_argument(
            '--case-sensitive',
            action='store_true',
            help='Enable case-sensitive matching'
        )
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '--output', '-o',
            type=str,
            default='matched_results',
            help='Output file base name (default: matched_results)'
        )
        output_group.add_argument(
            '--format',
            choices=['csv', 'json', 'both'],
            default='csv',
            help='Output format (default: csv)'
        )
        output_group.add_argument(
            '--output-dir',
            type=str,
            help='Output directory (default: current directory)'
        )
        output_group.add_argument(
            '--include-unmatched',
            action='store_true',
            help='Generate files for unmatched records'
        )
        output_group.add_argument(
            '--prefix1',
            type=str,
            default='f1_',
            help='Prefix for first file columns (default: f1_)'
        )
        output_group.add_argument(
            '--prefix2',
            type=str,
            default='f2_',
            help='Prefix for second file columns (default: f2_)'
        )
        
        # Processing options
        processing_group = parser.add_argument_group('Processing Options')
        processing_group.add_argument(
            '--batch',
            action='store_true',
            help='Enable batch processing mode'
        )
        processing_group.add_argument(
            '--config-dir',
            type=str,
            help='Directory containing multiple config files for batch processing'
        )
        processing_group.add_argument(
            '--parallel',
            type=int,
            default=0,
            help='Number of parallel processes (0 = auto-detect, default: 0)'
        )
        processing_group.add_argument(
            '--chunk-size',
            type=int,
            default=1000,
            help='Chunk size for processing large files (default: 1000)'
        )
        processing_group.add_argument(
            '--memory-limit',
            type=str,
            default='2GB',
            help='Memory limit for processing (e.g., 1GB, 2GB, default: 2GB)'
        )
        
        # Interface options
        interface_group = parser.add_argument_group('Interface Options')
        interface_group.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Run in interactive mode'
        )
        interface_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        interface_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )
        interface_group.add_argument(
            '--progress',
            choices=['none', 'simple', 'detailed'],
            default='simple',
            help='Progress reporting level (default: simple)'
        )
        
        # Utility options
        utility_group = parser.add_argument_group('Utility Options')
        utility_group.add_argument(
            '--generate-config',
            action='store_true',
            help='Generate a sample configuration file'
        )
        utility_group.add_argument(
            '--template',
            choices=['basic_csv', 'csv_to_json', 'uzbek_text', 'high_precision', 'bulk_processing'],
            help='Use a predefined configuration template'
        )
        utility_group.add_argument(
            '--list-templates',
            action='store_true',
            help='List available configuration templates'
        )
        utility_group.add_argument(
            '--validate-config',
            action='store_true',
            help='Validate configuration file without processing'
        )
        utility_group.add_argument(
            '--list-files',
            action='store_true',
            help='List available data files'
        )
        utility_group.add_argument(
            '--show-columns',
            type=str,
            help='Show columns in specified file'
        )
        utility_group.add_argument(
            '--examples',
            action='store_true',
            help='Show usage examples'
        )
        utility_group.add_argument(
            '--generate-completion',
            choices=['bash', 'zsh', 'fish'],
            help='Generate shell completion script'
        )
        
        # Logging options
        logging_group = parser.add_argument_group('Logging Options')
        logging_group.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        logging_group.add_argument(
            '--log-file',
            type=str,
            help='Log file path (default: console only)'
        )
        
        # Version
        parser.add_argument(
            '--version',
            action='version',
            version='File Processing System v2.0.0'
        )
        
        return parser
        
    def validate_args(self, args: argparse.Namespace) -> bool:
        """Validate command line arguments"""
        errors = []
        
        # Check for required input files or config (skip for utility commands)
        utility_commands = [
            args.generate_config, args.list_files, args.show_columns, 
            args.list_templates, args.examples, args.generate_completion,
            args.template, args.validate_config
        ]
        
        if not any(utility_commands) and not args.config and not (args.file1 and args.file2) and not args.interactive:
            errors.append("Either --config or both --file1 and --file2 must be specified (or use --interactive)")
            
        # Check file existence
        if args.config and not os.path.exists(args.config):
            errors.append(f"Configuration file not found: {args.config}")
            
        if args.file1 and not os.path.exists(args.file1):
            errors.append(f"First input file not found: {args.file1}")
            
        if args.file2 and not os.path.exists(args.file2):
            errors.append(f"Second input file not found: {args.file2}")
            
        # Check threshold range
        if not 0 <= args.threshold <= 100:
            errors.append("Threshold must be between 0 and 100")
            
        # Check batch processing requirements
        if args.batch and not args.config_dir:
            errors.append("Batch processing requires --config-dir")
            
        if args.config_dir and not os.path.isdir(args.config_dir):
            errors.append(f"Config directory not found: {args.config_dir}")
            
        # Check conflicting options
        if args.verbose and args.quiet:
            errors.append("Cannot use both --verbose and --quiet")
            
        if errors:
            for error in errors:
                print(f"Error: {error}", file=sys.stderr)
            return False
            
        return True
        
    def generate_sample_config(self, output_path: str) -> bool:
        """Generate a sample configuration file"""
        sample_config = {
            "file1": {
                "path": "data/file1.csv",
                "type": "csv",
                "delimiter": ","
            },
            "file2": {
                "path": "data/file2.json",
                "type": "json",
                "delimiter": None
            },
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "region"],
                "from_file2": ["id", "full_name", "district"]
            },
            "settings": {
                "output_format": "csv",
                "matched_output_path": "results/matched",
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
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, indent=2, ensure_ascii=False)
            print(f"Sample configuration generated: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating config: {e}", file=sys.stderr)
            return False
            
    def list_available_files(self) -> None:
        """List available data files"""
        print("\n=== AVAILABLE FILES ===")
        file_list = []
        
        # Check data directory
        if os.path.exists('data'):
            for root, dirs, files in os.walk('data'):
                for file in files:
                    if file.endswith(('.csv', '.json')):
                        file_list.append(os.path.join(root, file))
        
        # Check current directory
        current_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.json'))]
        file_list.extend(current_files)
        
        if not file_list:
            print("No CSV or JSON files found!")
            return
            
        for idx, file_path in enumerate(file_list, 1):
            file_size = os.path.getsize(file_path)
            file_size_str = self._format_file_size(file_size)
            print(f"{idx:3d}. {file_path:<40} ({file_size_str})")
            
    def show_file_columns(self, file_path: str) -> None:
        """Show columns in specified file"""
        try:
            if file_path.endswith('.csv'):
                import pandas as pd
                # Try different delimiters
                for delimiter in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(file_path, delimiter=delimiter, nrows=0)
                        print(f"\nColumns in {file_path} (delimiter: '{delimiter}'):")
                        for idx, col in enumerate(df.columns, 1):
                            print(f"{idx:3d}. {col}")
                        return
                    except:
                        continue
                print(f"Could not read CSV file: {file_path}")
                
            elif file_path.endswith('.json'):
                import pandas as pd
                try:
                    df = pd.read_json(file_path, lines=True, nrows=1)
                except:
                    df = pd.read_json(file_path, nrows=1)
                    
                print(f"\nColumns in {file_path}:")
                for idx, col in enumerate(df.columns, 1):
                    print(f"{idx:3d}. {col}")
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}", file=sys.stderr)
            
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
        
    def create_config_from_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Create configuration from command line arguments"""
        config = {
            "file1": {
                "path": args.file1,
                "type": args.file1_type if args.file1_type != 'auto' else self._detect_file_type(args.file1),
                "delimiter": args.delimiter1
            },
            "file2": {
                "path": args.file2,
                "type": args.file2_type if args.file2_type != 'auto' else self._detect_file_type(args.file2),
                "delimiter": args.delimiter2
            },
            "mapping_fields": [
                {
                    "file1_col": "auto",  # Will be determined interactively or from file analysis
                    "file2_col": "auto",
                    "match_type": args.algorithm,
                    "use_normalization": args.normalize,
                    "case_sensitive": args.case_sensitive,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": [],  # Will be determined from file analysis
                "from_file2": []
            },
            "settings": {
                "output_format": args.format,
                "matched_output_path": self._get_output_path(args),
                "file1_output_prefix": args.prefix1,
                "file2_output_prefix": args.prefix2,
                "confidence_threshold": args.threshold,
                "matching_type": args.matching_type,
                "unmatched_files": {
                    "generate": args.include_unmatched
                }
            }
        }
        
        return config
        
    def _detect_file_type(self, file_path: str) -> str:
        """Auto-detect file type from extension"""
        if file_path.endswith('.csv'):
            return 'csv'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            return 'csv'  # Default fallback
            
    def _get_output_path(self, args: argparse.Namespace) -> str:
        """Get full output path from arguments"""
        if args.output_dir:
            return os.path.join(args.output_dir, args.output)
        return args.output
        
    def run_interactive_mode(self) -> bool:
        """Run the CLI in interactive mode"""
        print("="*60)
        print("           INTERACTIVE FILE MATCHING SYSTEM")
        print("="*60)
        
        # Import interactive functions from existing code
        try:
            from main import create_config_interactively, run_processing_optimized
            config = create_config_interactively()
            if config:
                print("\n🚀 Starting processing...")
                progress_callback = CLIProgressCallback(verbose=True)
                run_processing_optimized(config, progress_callback)
                return True
            else:
                print("Configuration cancelled.")
                return False
        except ImportError as e:
            print(f"Error importing interactive functions: {e}")
            return False
            
    def run_batch_processing(self, config_dir: str, output_dir: str, parallel: bool = True) -> bool:
        """Run batch processing on multiple configuration files"""
        try:
            # Find all JSON config files
            config_files = []
            for file in os.listdir(config_dir):
                if file.endswith('.json'):
                    config_files.append(os.path.join(config_dir, file))
                    
            if not config_files:
                print(f"No configuration files found in {config_dir}")
                return False
                
            print(f"Found {len(config_files)} configuration files")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            success_count = 0
            failed_configs = []
            
            for i, config_file in enumerate(config_files, 1):
                print(f"\n{'='*60}")
                print(f"Processing {i}/{len(config_files)}: {os.path.basename(config_file)}")
                print(f"{'='*60}")
                
                try:
                    # Load configuration
                    config = self.config_manager.load_config(config_file)
                    
                    # Update output path for batch processing
                    base_name = os.path.splitext(os.path.basename(config_file))[0]
                    config['settings']['matched_output_path'] = os.path.join(output_dir, base_name)
                    
                    # Run processing
                    try:
                        from main import run_processing_optimized
                        progress_callback = CLIProgressCallback(verbose=False)
                        run_processing_optimized(config, progress_callback)
                        success_count += 1
                        print(f"\n✅ Successfully processed {os.path.basename(config_file)}")
                        
                    except ImportError:
                        # Fallback to main.py
                        from main import run_processing_optimized
                        progress_callback = CLIProgressCallback(verbose=False)
                        run_processing_optimized(config, progress_callback)
                        success_count += 1
                        print(f"\n✅ Successfully processed {os.path.basename(config_file)}")
                        
                except Exception as e:
                    failed_configs.append((config_file, str(e)))
                    print(f"\n❌ Failed to process {os.path.basename(config_file)}: {e}")
                    
            # Print summary
            print(f"\n{'='*60}")
            print(f"BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total configurations: {len(config_files)}")
            print(f"Successfully processed: {success_count}")
            print(f"Failed: {len(failed_configs)}")
            print(f"Success rate: {(success_count/len(config_files)*100):.1f}%")
            
            if failed_configs:
                print(f"\nFailed configurations:")
                for config_file, error in failed_configs:
                    print(f"  - {os.path.basename(config_file)}: {error}")
            
            print(f"{'='*60}")
            
            # Save batch report
            report = {
                "total_configs": len(config_files),
                "successful": success_count,
                "failed": len(failed_configs),
                "success_rate": (success_count/len(config_files)*100),
                "failed_configs": [{"file": f, "error": e} for f, e in failed_configs],
                "timestamp": time.time()
            }
            
            report_path = os.path.join(output_dir, 'batch_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Batch report saved to: {report_path}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"Batch processing failed: {e}", file=sys.stderr)
            return False
    
    def generate_shell_completion(self, shell: str) -> bool:
        """Generate shell completion script"""
        try:
            if shell == 'bash':
                completion_script = self._generate_bash_completion()
            elif shell == 'zsh':
                completion_script = self._generate_zsh_completion()
            elif shell == 'fish':
                completion_script = self._generate_fish_completion()
            else:
                print(f"Unsupported shell: {shell}", file=sys.stderr)
                return False
            
            output_file = f"cli_completion.{shell}"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(completion_script)
            
            print(f"Shell completion script generated: {output_file}")
            print(f"To enable completion, add the following to your shell configuration:")
            
            if shell == 'bash':
                print(f"  source {os.path.abspath(output_file)}")
                print("  # Or add to ~/.bashrc")
            elif shell == 'zsh':
                print(f"  source {os.path.abspath(output_file)}")
                print("  # Or add to ~/.zshrc")
            elif shell == 'fish':
                print(f"  source {os.path.abspath(output_file)}")
                print("  # Or copy to ~/.config/fish/completions/")
            
            return True
            
        except Exception as e:
            print(f"Error generating completion script: {e}", file=sys.stderr)
            return False
    
    def _generate_bash_completion(self) -> str:
        """Generate bash completion script"""
        return r'''#!/bin/bash
# Bash completion for file processing CLI

_cli_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main options
    opts="--config --file1 --file2 --file1-type --file2-type --delimiter1 --delimiter2
          --matching-type --threshold --algorithm --normalize --case-sensitive
          --output --format --output-dir --include-unmatched --prefix1 --prefix2
          --batch --config-dir --parallel --chunk-size --memory-limit
          --interactive --verbose --quiet --progress
          --generate-config --template --list-templates --validate-config
          --list-files --show-columns --examples --generate-completion
          --log-level --log-file --help --version"

    case "${prev}" in
        --config|--file1|--file2|--show-columns)
            COMPREPLY=( $(compgen -f "${cur}") )
            return 0
            ;;
        --config-dir|--output-dir)
            COMPREPLY=( $(compgen -d "${cur}") )
            return 0
            ;;
        --file1-type|--file2-type)
            COMPREPLY=( $(compgen -W "csv json auto" -- "${cur}") )
            return 0
            ;;
        --matching-type)
            COMPREPLY=( $(compgen -W "one-to-one one-to-many many-to-one many-to-many" -- "${cur}") )
            return 0
            ;;
        --algorithm)
            COMPREPLY=( $(compgen -W "exact fuzzy phonetic mixed" -- "${cur}") )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "csv json both" -- "${cur}") )
            return 0
            ;;
        --template)
            COMPREPLY=( $(compgen -W "basic_csv csv_to_json uzbek_text high_precision bulk_processing" -- "${cur}") )
            return 0
            ;;
        --progress)
            COMPREPLY=( $(compgen -W "none simple detailed" -- "${cur}") )
            return 0
            ;;
        --log-level)
            COMPREPLY=( $(compgen -W "DEBUG INFO WARNING ERROR CRITICAL" -- "${cur}") )
            return 0
            ;;
        --generate-completion)
            COMPREPLY=( $(compgen -W "bash zsh fish" -- "${cur}") )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
}

complete -F _cli_completion cli.py
complete -F _cli_completion python3\ cli.py
'''
    
    def _generate_zsh_completion(self) -> str:
        """Generate zsh completion script"""
        return '''#compdef cli.py

# Zsh completion for file processing CLI

_cli_completion() {
    local context state line
    typeset -A opt_args

    _arguments -C \\
        '(-c --config)'{-c,--config}'[Configuration file path]:config file:_files -g "*.json"' \\
        '--file1[First input file path]:input file:_files' \\
        '--file2[Second input file path]:input file:_files' \\
        '--file1-type[First file type]:file type:(csv json auto)' \\
        '--file2-type[Second file type]:file type:(csv json auto)' \\
        '--delimiter1[Delimiter for first CSV file]:delimiter:' \\
        '--delimiter2[Delimiter for second CSV file]:delimiter:' \\
        '--matching-type[Type of matching]:matching type:(one-to-one one-to-many many-to-one many-to-many)' \\
        '--threshold[Confidence threshold]:threshold:' \\
        '--algorithm[Matching algorithm]:algorithm:(exact fuzzy phonetic mixed)' \\
        '--normalize[Enable text normalization]' \\
        '--case-sensitive[Enable case-sensitive matching]' \\
        '(-o --output)'{-o,--output}'[Output file base name]:output name:' \\
        '--format[Output format]:format:(csv json both)' \\
        '--output-dir[Output directory]:output directory:_directories' \\
        '--include-unmatched[Generate files for unmatched records]' \\
        '--prefix1[Prefix for first file columns]:prefix:' \\
        '--prefix2[Prefix for second file columns]:prefix:' \\
        '--batch[Enable batch processing mode]' \\
        '--config-dir[Directory with config files]:config directory:_directories' \\
        '--parallel[Number of parallel processes]:parallel:' \\
        '--chunk-size[Chunk size for large files]:chunk size:' \\
        '--memory-limit[Memory limit]:memory limit:' \\
        '(-i --interactive)'{-i,--interactive}'[Run in interactive mode]' \\
        '(-v --verbose)'{-v,--verbose}'[Enable verbose output]' \\
        '(-q --quiet)'{-q,--quiet}'[Suppress non-essential output]' \\
        '--progress[Progress reporting level]:progress:(none simple detailed)' \\
        '--generate-config[Generate sample configuration file]' \\
        '--template[Use predefined template]:template:(basic_csv csv_to_json uzbek_text high_precision bulk_processing)' \\
        '--list-templates[List available templates]' \\
        '--validate-config[Validate configuration file]' \\
        '--list-files[List available data files]' \\
        '--show-columns[Show columns in file]:file:_files' \\
        '--examples[Show usage examples]' \\
        '--generate-completion[Generate shell completion]:shell:(bash zsh fish)' \\
        '--log-level[Logging level]:log level:(DEBUG INFO WARNING ERROR CRITICAL)' \\
        '--log-file[Log file path]:log file:_files' \\
        '--help[Show help message]' \\
        '--version[Show version information]'
}

_cli_completion "$@"
'''
    
    def _generate_fish_completion(self) -> str:
        """Generate fish completion script"""
        return '''# Fish completion for file processing CLI

# File completions
complete -c cli.py -l config -d "Configuration file path" -F
complete -c cli.py -l file1 -d "First input file path" -F
complete -c cli.py -l file2 -d "Second input file path" -F
complete -c cli.py -l show-columns -d "Show columns in file" -F
complete -c cli.py -l config-dir -d "Config directory" -x -a "(__fish_complete_directories)"
complete -c cli.py -l output-dir -d "Output directory" -x -a "(__fish_complete_directories)"
complete -c cli.py -l log-file -d "Log file path" -F

# Choice completions
complete -c cli.py -l file1-type -d "First file type" -x -a "csv json auto"
complete -c cli.py -l file2-type -d "Second file type" -x -a "csv json auto"
complete -c cli.py -l matching-type -d "Matching type" -x -a "one-to-one one-to-many many-to-one many-to-many"
complete -c cli.py -l algorithm -d "Matching algorithm" -x -a "exact fuzzy phonetic mixed"
complete -c cli.py -l format -d "Output format" -x -a "csv json both"
complete -c cli.py -l template -d "Configuration template" -x -a "basic_csv csv_to_json uzbek_text high_precision bulk_processing"
complete -c cli.py -l progress -d "Progress reporting level" -x -a "none simple detailed"
complete -c cli.py -l log-level -d "Logging level" -x -a "DEBUG INFO WARNING ERROR CRITICAL"
complete -c cli.py -l generate-completion -d "Generate shell completion" -x -a "bash zsh fish"

# Flag completions
complete -c cli.py -l normalize -d "Enable text normalization"
complete -c cli.py -l case-sensitive -d "Enable case-sensitive matching"
complete -c cli.py -l include-unmatched -d "Generate unmatched files"
complete -c cli.py -l batch -d "Enable batch processing"
complete -c cli.py -s i -l interactive -d "Run in interactive mode"
complete -c cli.py -s v -l verbose -d "Enable verbose output"
complete -c cli.py -s q -l quiet -d "Suppress non-essential output"
complete -c cli.py -l generate-config -d "Generate sample config"
complete -c cli.py -l list-templates -d "List available templates"
complete -c cli.py -l validate-config -d "Validate configuration"
complete -c cli.py -l list-files -d "List available files"
complete -c cli.py -l examples -d "Show usage examples"
complete -c cli.py -l help -d "Show help message"
complete -c cli.py -l version -d "Show version information"

# Value completions
complete -c cli.py -l threshold -d "Confidence threshold (0-100)" -x
complete -c cli.py -l parallel -d "Number of parallel processes" -x
complete -c cli.py -l chunk-size -d "Chunk size for large files" -x
complete -c cli.py -l memory-limit -d "Memory limit (e.g., 2GB)" -x
complete -c cli.py -s o -l output -d "Output file base name" -x
complete -c cli.py -l prefix1 -d "Prefix for first file columns" -x
complete -c cli.py -l prefix2 -d "Prefix for second file columns" -x
complete -c cli.py -l delimiter1 -d "Delimiter for first CSV file" -x
complete -c cli.py -l delimiter2 -d "Delimiter for second CSV file" -x
'''
        
    def run(self, args: List[str] = None) -> int:
        """Main entry point for the CLI"""
        parser = self.create_parser()
        args = parser.parse_args(args)
        
        # Setup logging
        self.setup_logging(args.log_level, args.log_file)
        
        # Validate arguments
        if not self.validate_args(args):
            return 1
            
        try:
            # Handle utility commands
            if args.generate_config:
                output_path = args.output if args.output != 'matched_results' else 'sample_config.json'
                return 0 if self.generate_sample_config(output_path) else 1
                
            if args.list_files:
                self.list_available_files()
                return 0
                
            if args.show_columns:
                self.show_file_columns(args.show_columns)
                return 0
                
            if args.template:
                output_path = args.output if args.output != 'matched_results' else f'{args.template}_config.json'
                success = ConfigurationTemplates.save_template(args.template, output_path)
                if success:
                    print(f"Template '{args.template}' saved to: {output_path}")
                    print(f"Description: {ConfigurationTemplates.get_template_description(args.template)}")
                    return 0
                else:
                    print(f"Error: Failed to save template '{args.template}'", file=sys.stderr)
                    return 1
                    
            if args.list_templates:
                print("\n=== AVAILABLE CONFIGURATION TEMPLATES ===")
                templates = ConfigurationTemplates.get_all_templates()
                for name, _ in templates.items():
                    description = ConfigurationTemplates.get_template_description(name)
                    print(f"{name:15s} - {description}")
                print(f"\nUsage: {sys.argv[0]} --template <template_name> --output <config_file>")
                return 0
                
            if args.examples:
                print("\n=== USAGE EXAMPLES ===\n")
                
                print("## Basic Examples")
                for example in UsageExamples.get_basic_examples():
                    print(example)
                
                print("\n## Advanced Examples")
                for example in UsageExamples.get_advanced_examples():
                    print(example)
                
                print("\n## Troubleshooting Examples")
                for example in UsageExamples.get_troubleshooting_examples():
                    print(example)
                
                return 0
                
            if args.generate_completion:
                return 0 if self.generate_shell_completion(args.generate_completion) else 1
                
            if args.validate_config:
                if not args.config:
                    print("Error: --validate-config requires --config", file=sys.stderr)
                    return 1
                try:
                    config = self.config_manager.load_config(args.config)
                    print(f"Configuration file {args.config} is valid")
                    return 0
                except Exception as e:
                    print(f"Configuration validation failed: {e}", file=sys.stderr)
                    return 1
                    
            # Handle processing modes
            if args.interactive:
                return 0 if self.run_interactive_mode() else 1
                
            if args.batch:
                return 0 if self.run_batch_processing(args.config_dir, args.output_dir or 'results') else 1
                
            # Handle regular processing
            if args.config:
                # Load configuration from file
                config = self.config_manager.load_config(args.config)
            else:
                # Create configuration from command line arguments
                config = self.create_config_from_args(args)
                
            # Import and run processing
            from main import run_processing_optimized
            progress_callback = CLIProgressCallback(verbose=args.verbose)
            run_processing_optimized(config, progress_callback)
            
            return 0
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1


def main():
    """Entry point for the CLI application"""
    cli = FileMatchingCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())