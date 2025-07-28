#!/usr/bin/env python3
"""
Main entry point for the File Processing and Data Matching System.
This file uses the refactored code structure from the src folder.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Add project root to Python path for proper package imports
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import refactored services using absolute imports
    from src.application.services.cli_config_service import CLIConfigurationManager
    from src.application.services.batch_processor import BatchProcessor, BatchProgressReporter
    from src.application.services.file_service import FileProcessingService
    from src.application.services.result_manager import ResultManager
    from src.infrastructure.progress_tracker import ProgressTracker
    from src.domain.exceptions import ConfigurationError, FileProcessingError
    
    # Import domain matching components
    from src.domain.matching.engine import MatchingEngine
    from src.domain.matching.fuzzy_matcher import FuzzyMatcher
    from src.domain.matching.exact_matcher import ExactMatcher
    from src.domain.matching.phonetic_matcher import PhoneticMatcher
    from src.domain.matching.blocking import OptimizedBlockingIndex
    from src.domain.matching.uzbek_normalizer import UzbekTextNormalizer
    
except ImportError as e:
    print(f"Error importing refactored modules: {e}")
    print("Falling back to legacy main.py functionality...")
    
    # Fallback to original implementation if refactored modules are not available
    import pandas as pd
    from tqdm import tqdm
    import multiprocessing as mp
    import numpy as np
    import re
    from rapidfuzz import fuzz
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
    from collections import defaultdict
    from functools import lru_cache
    import pickle
    import warnings
    warnings.filterwarnings('ignore')
    
    # Include the original classes and functions here as fallback
    # (This would be the original OptimizedMatcher, etc. classes)
    REFACTORED_AVAILABLE = False
else:
    REFACTORED_AVAILABLE = True


class MainApplication:
    """Main application class that orchestrates the file matching process"""
    
    def __init__(self):
        self.config_manager = CLIConfigurationManager() if REFACTORED_AVAILABLE else None
        self.batch_processor = BatchProcessor() if REFACTORED_AVAILABLE else None
        self.file_service = FileProcessingService() if REFACTORED_AVAILABLE else None
        self.result_manager = ResultManager() if REFACTORED_AVAILABLE else None
        self.progress_tracker = ProgressTracker() if REFACTORED_AVAILABLE else None
    
    def run_processing_optimized(self, config: Dict[str, Any], progress_callback: Optional[Callable] = None):
        """
        Main processing function using refactored architecture
        """
        if not REFACTORED_AVAILABLE:
            return self._run_legacy_processing(config, progress_callback)
        
        try:
            def update_progress(progress, message):
                if progress_callback:
                    progress_callback('processing', progress, message)
                print(f"📊 {progress}% - {message}")

            print("\n" + "="*60)
            print("           🚀 REFACTORED PROCESSING ENGINE")
            print("="*60)

            update_progress(5, "Loading data...")
            start_time = time.time()
            
            # Load data using refactored file service
            if self.file_service:
                dataset1 = self.file_service.load_file(config['file1']['path'], config['file1'])
                dataset2 = self.file_service.load_file(config['file2']['path'], config['file2'])
                df1 = dataset1.data
                df2 = dataset2.data
            else:
                # Fallback to pandas loading
                import pandas as pd
                df1 = pd.read_csv(config['file1']['path']) if config['file1']['type'] == 'csv' else pd.read_json(config['file1']['path'])
                df2 = pd.read_csv(config['file2']['path']) if config['file2']['type'] == 'csv' else pd.read_json(config['file2']['path'])
                
                # Create simple dataset objects
                from src.domain.models import Dataset, DatasetMetadata, FileType
                dataset1 = Dataset(
                    name="file1",
                    data=df1,
                    metadata=DatasetMetadata(
                        name="file1",
                        file_path=config['file1']['path'],
                        file_type=FileType.CSV if config['file1']['type'] == 'csv' else FileType.JSON,
                        row_count=len(df1),
                        column_count=len(df1.columns)
                    )
                )
                dataset2 = Dataset(
                    name="file2", 
                    data=df2,
                    metadata=DatasetMetadata(
                        name="file2",
                        file_path=config['file2']['path'],
                        file_type=FileType.CSV if config['file2']['type'] == 'csv' else FileType.JSON,
                        row_count=len(df2),
                        column_count=len(df2.columns)
                    )
                )
            
            load_time = time.time() - start_time
            print(f"⏱️  Load time: {load_time:.2f}s")
            print(f"📊 File 1: {len(df1):,} rows")
            print(f"📊 File 2: {len(df2):,} rows")
            
            total_comparisons = len(df1) * len(df2)
            print(f"� Totval comparisons: {total_comparisons:,}")
            
            update_progress(10, "Initializing matching engine...")
            print("\n🔍 Initializing matching engine...")
            
            # Create matching configuration from config dict
            from src.domain.models import MatchingConfig, FieldMapping, AlgorithmConfig, MatchingType, AlgorithmType
            
            # Convert config to MatchingConfig object
            matching_config = MatchingConfig(
                confidence_threshold=config['settings'].get('confidence_threshold', 75),
                parallel_processing=True,
                max_workers=4,
                use_blocking=True
            )
            
            # Add field mappings
            for mapping in config['mapping_fields']:
                # Algorithm turini aniqlash
                algorithm_type = mapping.get('match_type', 'fuzzy')
                print(f"🔍 Debug: Field {mapping['file1_col']} -> {mapping['file2_col']}, match_type: {algorithm_type}")
                
                if algorithm_type == 'exact':
                    algorithm = AlgorithmType.EXACT
                    print(f"✅ Using EXACT matcher for {mapping['file1_col']}")
                elif algorithm_type == 'phonetic':
                    algorithm = AlgorithmType.PHONETIC
                    print(f"✅ Using PHONETIC matcher for {mapping['file1_col']}")
                else:
                    algorithm = AlgorithmType.FUZZY  # default
                    print(f"✅ Using FUZZY matcher for {mapping['file1_col']}")
                
                field_mapping = FieldMapping(
                    source_field=mapping['file1_col'],
                    target_field=mapping['file2_col'],
                    algorithm=algorithm,
                    weight=mapping.get('weight', 1.0),
                    normalization=mapping.get('use_normalization', False),
                    case_sensitive=mapping.get('case_sensitive', False)
                )
                matching_config.mappings.append(field_mapping)
            
            # Add algorithm configurations based on field mappings
            used_algorithms = set()
            for mapping in matching_config.mappings:
                used_algorithms.add(mapping.algorithm)
            
            # Create algorithm configs for each used algorithm type
            print(f"🔍 Debug: Used algorithms: {list(used_algorithms)}")
            
            for algorithm_type in used_algorithms:
                print(f"🔍 Debug: Processing algorithm_type: {algorithm_type}, type: {type(algorithm_type)}")
                
                # String-ni enum bilan taqqoslash uchun value ishlatamiz
                if algorithm_type == AlgorithmType.EXACT or algorithm_type == AlgorithmType.EXACT.value:
                    algo_config = AlgorithmConfig(
                        name="exact_matcher",
                        algorithm_type="exact",
                        enabled=True,
                        parameters={}
                    )
                    print(f"✅ Created EXACT algorithm config")
                elif algorithm_type == AlgorithmType.PHONETIC or algorithm_type == AlgorithmType.PHONETIC.value:
                    algo_config = AlgorithmConfig(
                        name="phonetic_matcher",
                        algorithm_type="phonetic",
                        enabled=True,
                        parameters={}
                    )
                    print(f"✅ Created PHONETIC algorithm config")
                else:  # FUZZY
                    algo_config = AlgorithmConfig(
                        name="fuzzy_matcher",
                        algorithm_type="fuzzy",
                        enabled=True,
                        parameters={}
                    )
                    print(f"✅ Created FUZZY algorithm config (fallback)")
                matching_config.algorithms.append(algo_config)
            
            # Initialize matching engine
            matching_engine = MatchingEngine(matching_config)
            
            update_progress(20, "Finding matches...")
            print("\n🔍 Finding matches...")
            matching_start = time.time()
            
            # Define progress callback for matching engine
            def matching_progress_callback(status):
                progress = 20 + int((status.progress / 100) * 60)
                update_progress(progress, status.message)
            
            # Run matching
            result = matching_engine.find_matches(dataset1, dataset2, matching_progress_callback)
            
            matching_time = time.time() - matching_start
            print(f"⏱️  Matching time: {matching_time:.2f}s")
            print(f"📊 {len(result.matched_records):,} matches found")
            
            if not result.matched_records:
                update_progress(100, "No matches found!")
                print("❌ No matches found!")
                return

            update_progress(80, "Saving results...")
            print("\n💾 Saving results...")
            save_start = time.time()
            
            # Convert matched records to DataFrame format
            high_conf_results = []
            low_conf_results = []
            threshold = config['settings'].get('confidence_threshold', 75)
            
            for match in result.matched_records:
                match_dict = {
                    'match_score': match.confidence_score,
                    **match.record1,
                    **match.record2
                }
                
                if match.confidence_score >= threshold:
                    high_conf_results.append(match_dict)
                else:
                    low_conf_results.append(match_dict)
            
            # Save results using refactored result manager or fallback
            if self.result_manager:
                # Store results first
                result_id = self.result_manager.store_results(result)
                
                # Export high confidence results
                if high_conf_results:
                    from src.application.services.result_manager import ExportConfig, ExportFormat
                    
                    # Determine export format
                    output_format = config['settings']['output_format']
                    if output_format == 'csv':
                        export_format = ExportFormat.CSV
                    elif output_format == 'json':
                        export_format = ExportFormat.JSON
                    elif output_format == 'excel':
                        export_format = ExportFormat.EXCEL
                    else:
                        export_format = ExportFormat.JSON  # default
                    
                    # Confidence threshold uchun filter yaratish
                    confidence_threshold = config['settings'].get('confidence_threshold', 75)
                    filters = {'min_confidence': confidence_threshold} if confidence_threshold > 0 else None
                    
                    export_config = ExportConfig(
                        format=export_format,
                        include_metadata=True,
                        include_unmatched=False,  # Faqat high confidence results
                        include_statistics=True,
                        filters=filters
                    )
                    
                    exported_file = self.result_manager.export_results(result_id, export_config)
                    
                    # Export qilingan faylni kerakli joyga ko'chirish
                    import shutil
                    desired_output_path = config['settings']['matched_output_path']
                    
                    # File extension aniqlash
                    if export_format == ExportFormat.CSV:
                        final_path = f"{desired_output_path}.csv"
                    elif export_format == ExportFormat.EXCEL:
                        final_path = f"{desired_output_path}.xlsx"
                    else:
                        final_path = f"{desired_output_path}.json"
                    
                    # Faylni ko'chirish
                    shutil.copy2(exported_file, final_path)
                    print(f"✅ Results exported to: {final_path}")
                    
                    # Temp faylni o'chirish
                    try:
                        import os
                        os.remove(exported_file)
                    except:
                        pass
                else:
                    print("⚠️  No high confidence results to export")
            else:
                # Fallback to pandas saving
                import pandas as pd
                if high_conf_results:
                    matched_df = pd.DataFrame(high_conf_results)
                    output_path = config['settings']['matched_output_path']
                    if config['settings']['output_format'] in ['csv', 'both']:
                        matched_df.to_csv(f"{output_path}.csv", index=False)
                    if config['settings']['output_format'] in ['json', 'both']:
                        matched_df.to_json(f"{output_path}.json", orient='records', indent=2)
                    print(f"✅ High confidence results saved")
                
                if low_conf_results:
                    low_conf_df = pd.DataFrame(low_conf_results)
                    low_conf_path = f"{config['settings']['matched_output_path']}_low_confidence"
                    if config['settings']['output_format'] in ['csv', 'both']:
                        low_conf_df.to_csv(f"{low_conf_path}.csv", index=False)
                    if config['settings']['output_format'] in ['json', 'both']:
                        low_conf_df.to_json(f"{low_conf_path}.json", orient='records', indent=2)
                    print(f"⚠️  Low confidence results saved")

            # Handle unmatched records
            if config['settings'].get('unmatched_files', {}).get('generate', False):
                update_progress(90, "Saving unmatched records...")
                print("\n📝 Saving unmatched records...")
                
                if result.unmatched_records:
                    import pandas as pd
                    
                    # # Debug: unmatched_records formatini ko'rish
                    # print(f"🔍 Debug: unmatched_records type: {type(result.unmatched_records)}")
                    # print(f"🔍 Debug: unmatched_records content: {result.unmatched_records}")
                    
                    # Agar list bo'lsa
                    if isinstance(result.unmatched_records, list) and result.unmatched_records:
                        print(f"🔍 Debug: First unmatched record: {result.unmatched_records[0]}")
                        print(f"🔍 Debug: First record type: {type(result.unmatched_records[0])}")
                    # Agar dict bo'lsa
                    elif isinstance(result.unmatched_records, dict):
                        print(f"🔍 Debug: Dict keys: {list(result.unmatched_records.keys())}")
                        if result.unmatched_records:
                            first_key = list(result.unmatched_records.keys())[0]
                            print(f"🔍 Debug: First key: {first_key}, value: {result.unmatched_records[first_key]}")
                    else:
                        print(f"🔍 Debug: Unexpected unmatched_records format")
                    
                    # Unmatched records-ni to'g'ri formatda olish
                    unmatched_records_1 = []
                    unmatched_records_2 = []
                    
                    for record in result.unmatched_records:
                        if isinstance(record, dict):
                            if record.get('source') == 'file1':
                                unmatched_records_1.append(record)
                            elif record.get('source') == 'file2':
                                unmatched_records_2.append(record)
                        else:
                            # Agar record string yoki boshqa format bo'lsa
                            print(f"⚠️  Unexpected record format: {type(record)}: {record}")
                    
                    unmatched_df1 = pd.DataFrame(unmatched_records_1)
                    unmatched_df2 = pd.DataFrame(unmatched_records_2)
                    
                    if not unmatched_df1.empty:
                        unmatched_path_1 = f"{config['settings']['matched_output_path']}_unmatched_1"
                        if config['settings']['output_format'] in ['csv', 'both']:
                            unmatched_df1.to_csv(f"{unmatched_path_1}.csv", index=False)
                        if config['settings']['output_format'] in ['json', 'both']:
                            unmatched_df1.to_json(f"{unmatched_path_1}.json", orient='records', indent=2)
                        print(f"📄 File 1 unmatched: {len(unmatched_df1):,} records")
                    else:
                        print(f"🎉 All records from File 1 matched!")

                    if not unmatched_df2.empty:
                        unmatched_path_2 = f"{config['settings']['matched_output_path']}_unmatched_2"
                        if config['settings']['output_format'] in ['csv', 'both']:
                            unmatched_df2.to_csv(f"{unmatched_path_2}.csv", index=False)
                        if config['settings']['output_format'] in ['json', 'both']:
                            unmatched_df2.to_json(f"{unmatched_path_2}.json", orient='records', indent=2)
                        print(f"📄 File 2 unmatched: {len(unmatched_df2):,} records")
                    else:
                        print(f"🎉 All records from File 2 matched!")
            
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            
            update_progress(100, f"Process completed successfully! ({total_time:.2f}s)")
            print(f"\n🎉 Process completed successfully!")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"   📁 Loading: {load_time:.2f}s")
            print(f"   🔍 Matching: {matching_time:.2f}s") 
            print(f"   💾 Saving: {save_time:.2f}s")
            
            if matching_time > 0:
                comparisons_per_second = len(df1) * len(df2) / matching_time
                print(f"⚡ Speed: {comparisons_per_second:,.0f} comparisons/second")

            print("\nAll processes completed successfully!")

        except Exception as e:
            if progress_callback:
                progress_callback('error', 0, f"Error occurred: {str(e)}")
            import traceback
            print(f"ERROR: Unexpected problem during processing: {e}")
            traceback.print_exc()
    
    def _run_legacy_processing(self, config: Dict[str, Any], progress_callback: Optional[Callable] = None):
        """Fallback to legacy processing if refactored modules are not available"""
        print("⚠️  Running in legacy mode - refactored modules not available")
        # Here you would include the original processing logic
        # For now, we'll just show an error message
        print("❌ Legacy processing not implemented in this version")
        print("Please ensure the refactored modules in src/ are available")
    
    def create_config_interactively(self) -> Optional[Dict[str, Any]]:
        """Create configuration interactively"""
        if not REFACTORED_AVAILABLE:
            print("❌ Interactive configuration requires refactored modules")
            return None
        
        try:
            print("\n" + "="*60)
            print("           📝 INTERACTIVE CONFIGURATION")
            print("="*60)
            
            # Get file paths
            print("\n1. Select input files:")
            file1_path = input("Enter path to first file: ").strip()
            if not os.path.exists(file1_path):
                print(f"❌ File not found: {file1_path}")
                return None
            
            file2_path = input("Enter path to second file: ").strip()
            if not os.path.exists(file2_path):
                print(f"❌ File not found: {file2_path}")
                return None
            
            # Create interactive configuration
            config = self.config_manager.create_interactive_config(file1_path, file2_path)
            
            print("\n✅ Configuration created successfully!")
            print(f"📊 File 1 columns: {len(config['output_columns']['from_file1'])}")
            print(f"📊 File 2 columns: {len(config['output_columns']['from_file2'])}")
            print(f"🔗 Suggested mappings: {len(config['mapping_fields'])}")
            
            # Ask for output path
            output_path = input("\nEnter output path (default: results/matched): ").strip()
            if not output_path:
                output_path = "results/matched"
            config['settings']['matched_output_path'] = output_path
            
            return config
            
        except Exception as e:
            print(f"❌ Error creating interactive configuration: {e}")
            return None
    
    def show_files(self):
        """Show available files"""
        print("\n=== AVAILABLE FILES ===")
        file_list = []
        
        if os.path.exists('data'):
            for root, dirs, files in os.walk('data'):
                for file in files:
                    if file.endswith(('.csv', '.json')):
                        file_list.append(os.path.join(root, file))
        
        current_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.json'))]
        for file in current_files:
            file_list.append(file)
        
        if not file_list:
            print("No CSV or JSON files found!")
            return None
        
        for idx, file_path in enumerate(file_list, 1):
            file_size = os.path.getsize(file_path)
            file_size_str = self._format_file_size(file_size)
            print(f"{idx:3d}. {file_path:<40} ({file_size_str})")
        
        print("0. Go back")
        
        while True:
            try:
                choice = input("\nSelect file number: ").strip()
                if choice == '0':
                    return None
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(file_list):
                    return file_list[choice_idx]
                else:
                    print("Invalid number! Try again.")
            except ValueError:
                print("Enter numbers only!")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
    
    def main_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "="*60)
            print("           🚀 FILE PROCESSING SYSTEM")
            print("="*60)
            print("1. 📊 Process files with configuration")
            print("2. 📝 Create configuration interactively")
            print("3. 📁 Show available files")
            print("4. 🔄 Batch processing")
            print("5. ❓ Help")
            print("0. 🚪 Exit")
            print("="*60)
            
            choice = input("Select option (0-5): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                self._handle_process_files()
            elif choice == '2':
                self._handle_interactive_config()
            elif choice == '3':
                self.show_files()
            elif choice == '4':
                self._handle_batch_processing()
            elif choice == '5':
                self._show_help()
            else:
                print("❌ Invalid choice! Please try again.")
    
    def _handle_process_files(self):
        """Handle file processing option"""
        config_path = input("Enter configuration file path: ").strip()
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        try:
            if REFACTORED_AVAILABLE:
                config = self.config_manager.load_and_validate_config(config_path)
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            print("\n🚀 Starting processing...")
            self.run_processing_optimized(config)
            
        except Exception as e:
            print(f"❌ Error processing files: {e}")
    
    def _handle_interactive_config(self):
        """Handle interactive configuration creation"""
        config = self.create_config_interactively()
        if config:
            save_config = input("\nSave configuration to file? (y/n): ").strip().lower()
            if save_config == 'y':
                config_path = input("Enter configuration file path: ").strip()
                if not config_path.endswith('.json'):
                    config_path += '.json'
                
                try:
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    print(f"✅ Configuration saved to: {config_path}")
                except Exception as e:
                    print(f"❌ Error saving configuration: {e}")
            
            process_now = input("Process files now? (y/n): ").strip().lower()
            if process_now == 'y':
                print("\n🚀 Starting processing...")
                self.run_processing_optimized(config)
    
    def _handle_batch_processing(self):
        """Handle batch processing option"""
        if not REFACTORED_AVAILABLE or not self.batch_processor:
            print("❌ Batch processing requires refactored modules")
            return
        
        config_dir = input("Enter configuration directory path: ").strip()
        if not os.path.isdir(config_dir):
            print(f"❌ Directory not found: {config_dir}")
            return
        
        output_dir = input("Enter output directory path: ").strip()
        if not output_dir:
            output_dir = "batch_results"
        
        try:
            print(f"\n🔄 Starting batch processing...")
            print(f"📁 Config directory: {config_dir}")
            print(f"📁 Output directory: {output_dir}")
            
            progress_reporter = BatchProgressReporter(verbose=True)
            result = self.batch_processor.run_batch(
                config_dir=config_dir,
                output_dir=output_dir,
                parallel=True,
                progress_callback=progress_reporter
            )
            
            progress_reporter.print_summary(result)
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("                    📖 HELP")
        print("="*60)
        print("This system helps you match records between two data files.")
        print("\nSupported file formats:")
        print("  • CSV files (.csv)")
        print("  • JSON files (.json)")
        print("\nMatching algorithms:")
        print("  • Exact matching")
        print("  • Fuzzy matching (recommended)")
        print("  • Phonetic matching")
        print("\nFor more information, check the documentation in docs/")
        print("="*60)


def main():
    """Main entry point"""
    app = MainApplication()
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Handle command line arguments (could integrate with scripts/cli.py)
        parser = argparse.ArgumentParser(description="File Processing and Data Matching System")
        parser.add_argument('--config', '-c', help='Configuration file path')
        parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
        parser.add_argument('--batch', action='store_true', help='Run batch processing')
        parser.add_argument('--config-dir', help='Configuration directory for batch processing')
        parser.add_argument('--output-dir', help='Output directory for batch processing')
        
        args = parser.parse_args()
        
        if args.config:
            try:
                if REFACTORED_AVAILABLE:
                    config = app.config_manager.load_and_validate_config(args.config)
                else:
                    with open(args.config, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                
                print("🚀 Starting processing from command line...")
                app.run_processing_optimized(config)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                sys.exit(1)
        
        elif args.batch:
            if not args.config_dir:
                print("❌ Batch processing requires --config-dir")
                sys.exit(1)
            
            output_dir = args.output_dir or "batch_results"
            
            try:
                if REFACTORED_AVAILABLE and app.batch_processor:
                    progress_reporter = BatchProgressReporter(verbose=True)
                    result = app.batch_processor.run_batch(
                        config_dir=args.config_dir,
                        output_dir=output_dir,
                        parallel=True,
                        progress_callback=progress_reporter
                    )
                    progress_reporter.print_summary(result)
                else:
                    print("❌ Batch processing requires refactored modules")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"❌ Batch processing failed: {e}")
                sys.exit(1)
        
        elif args.interactive:
            app.main_menu()
        
        else:
            parser.print_help()
    
    else:
        # Run interactive menu
        app.main_menu()


if __name__ == "__main__":
    main()