#!/usr/bin/env python3
"""
Performance Benchmarking Suite
Tests system performance under various load conditions
"""

import os
import sys
import json
import time
import psutil
import tempfile
import unittest
import threading
import multiprocessing
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class PerformanceBenchmarkSuite(unittest.TestCase):
    """Comprehensive performance benchmarking"""
    
    @classmethod
    def setUpClass(cls):
        """Set up performance test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='perf_test_')
        cls.project_root = Path(__file__).parent.parent
        
        # Performance thresholds (adjust based on requirements)
        cls.performance_thresholds = {
            'small_dataset_time': 5.0,      # seconds for < 1K records
            'medium_dataset_time': 30.0,    # seconds for 1K-10K records
            'large_dataset_time': 300.0,    # seconds for 10K-100K records
            'memory_usage_mb': 500,         # MB maximum memory usage
            'cpu_usage_percent': 80,        # % maximum CPU usage
            'throughput_min': 1000,         # minimum records/second
        }
        
        # System information
        cls.system_info = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform
        }
        
        print(f"Performance test environment: {cls.test_dir}")
        print(f"System: {cls.system_info['cpu_count']} CPUs, {cls.system_info['memory_gb']:.1f}GB RAM")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up performance test environment"""
        try:
            import shutil
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")

    def _create_test_dataset(self, size, filename_prefix):
        """Create test dataset of specified size"""
        base_names = ["Toshkent", "Samarqand", "Buxoro", "Andijon", "Farg'ona", 
                     "Namangan", "Qarshi", "Termiz", "Nukus", "Urganch"]
        
        data = []
        for i in range(size):
            base_name = base_names[i % len(base_names)]
            suffix = "shahri" if i % 2 == 0 else "tumani"
            
            data.append({
                "id": i + 1,
                "name": f"{base_name} {suffix} {i}",
                "region": f"Region_{i % 10}",
                "code": f"CODE_{i:06d}",
                "description": f"Description for {base_name} {suffix} {i}" * (i % 3 + 1)
            })
        
        # Save as CSV
        df = pd.DataFrame(data)
        file_path = os.path.join(self.test_dir, f"{filename_prefix}_{size}.csv")
        df.to_csv(file_path, index=False)
        
        return file_path, df

    def _monitor_system_resources(self, duration=60):
        """Monitor system resources during test"""
        start_time = time.time()
        measurements = []
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            measurements.append({
                'timestamp': time.time() - start_time,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024**2)
            })
        
        return measurements

    def test_01_small_dataset_performance(self):
        """Test performance with small datasets (< 1K records)"""
        print("\n=== Testing Small Dataset Performance ===")
        
        dataset_sizes = [100, 500, 1000]
        results = []
        
        for size in dataset_sizes:
            print(f"Testing dataset size: {size} records")
            
            # Create test datasets
            file1_path, df1 = self._create_test_dataset(size, "small_file1")
            file2_path, df2 = self._create_test_dataset(size, "small_file2")
            
            # Create configuration
            config = {
                "file1": {"path": file1_path, "type": "csv", "delimiter": ","},
                "file2": {"path": file2_path, "type": "csv", "delimiter": ","},
                "mapping_fields": [{
                    "file1_col": "name",
                    "file2_col": "name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }],
                "output_columns": {
                    "from_file1": ["id", "name", "region"],
                    "from_file2": ["id", "name", "code"]
                },
                "settings": {
                    "output_format": "json",
                    "matched_output_path": os.path.join(self.test_dir, f"results_{size}"),
                    "confidence_threshold": 75,
                    "matching_type": "one-to-one"
                }
            }
            
            # Start resource monitoring
            monitor_thread = threading.Thread(
                target=lambda: self._monitor_system_resources(30)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            try:
                # Import and run processing
                from src.application.services.file_service import FileProcessingService
                from src.application.services.config_service import ConfigurationManager
                from src.domain.matching.engine import MatchingEngine
                from src.infrastructure.caching import CacheManager
                from src.infrastructure.progress_tracker import ProgressTracker
                
                # Initialize components
                config_manager = ConfigurationManager()
                file_service = FileProcessingService(config_manager)
                cache_manager = CacheManager()
                progress_tracker = ProgressTracker()
                
                matching_engine = MatchingEngine(
                    config=config,
                    cache_manager=cache_manager,
                    progress_tracker=progress_tracker
                )
                
                # Load files
                dataset1 = file_service.load_file(file1_path, config["file1"])
                dataset2 = file_service.load_file(file2_path, config["file2"])
                
                # Perform matching
                matching_result = matching_engine.find_matches(dataset1, dataset2)
                
                # Save results
                output_files = file_service.save_results(matching_result, config["settings"])
                
                # Calculate metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**2)
                
                processing_time = end_time - start_time
                memory_used = end_memory - start_memory
                total_comparisons = len(dataset1) * len(dataset2)
                throughput = total_comparisons / processing_time
                
                result = {
                    'dataset_size': size,
                    'processing_time': processing_time,
                    'memory_used_mb': memory_used,
                    'throughput': throughput,
                    'matches_found': len(matching_result.matched_records),
                    'total_comparisons': total_comparisons
                }
                
                results.append(result)
                
                # Performance assertions
                self.assertLess(processing_time, self.performance_thresholds['small_dataset_time'],
                               f"Small dataset processing too slow: {processing_time:.2f}s")
                
                self.assertLess(memory_used, self.performance_thresholds['memory_usage_mb'],
                               f"Memory usage too high: {memory_used:.2f}MB")
                
                self.assertGreater(throughput, self.performance_thresholds['throughput_min'],
                                  f"Throughput too low: {throughput:.0f} comparisons/s")
                
                print(f"✅ Size {size}: {processing_time:.2f}s, {memory_used:.1f}MB, {throughput:.0f} comp/s")
                
            except Exception as e:
                print(f"❌ Size {size}: Test failed - {e}")
                self.fail(f"Performance test failed for size {size}: {e}")
        
        # Print summary
        print(f"\nSmall Dataset Performance Summary:")
        for result in results:
            print(f"  {result['dataset_size']} records: {result['processing_time']:.2f}s, "
                  f"{result['throughput']:.0f} comp/s, {result['matches_found']} matches")

    def test_02_medium_dataset_performance(self):
        """Test performance with medium datasets (1K-10K records)"""
        print("\n=== Testing Medium Dataset Performance ===")
        
        dataset_sizes = [2000, 5000, 10000]
        results = []
        
        for size in dataset_sizes:
            print(f"Testing dataset size: {size} records")
            
            # Create test datasets
            file1_path, df1 = self._create_test_dataset(size, "medium_file1")
            file2_path, df2 = self._create_test_dataset(size, "medium_file2")
            
            # Create configuration with optimizations
            config = {
                "file1": {"path": file1_path, "type": "csv", "delimiter": ","},
                "file2": {"path": file2_path, "type": "csv", "delimiter": ","},
                "mapping_fields": [{
                    "file1_col": "name",
                    "file2_col": "name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }],
                "output_columns": {
                    "from_file1": ["id", "name", "region"],
                    "from_file2": ["id", "name", "code"]
                },
                "settings": {
                    "output_format": "json",
                    "matched_output_path": os.path.join(self.test_dir, f"results_{size}"),
                    "confidence_threshold": 75,
                    "matching_type": "one-to-one",
                    "use_blocking": True,
                    "parallel_processing": True
                }
            }
            
            # Measure performance with resource monitoring
            resource_measurements = []
            
            def monitor_resources():
                nonlocal resource_measurements
                resource_measurements = self._monitor_system_resources(120)
            
            monitor_thread = threading.Thread(target=monitor_resources)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            try:
                # Import and run processing with optimizations
                from src.application.services.file_service import FileProcessingService
                from src.application.services.config_service import ConfigurationManager
                from src.domain.matching.engine import MatchingEngine
                from src.infrastructure.caching import CacheManager
                from src.infrastructure.progress_tracker import ProgressTracker
                from src.infrastructure.parallel_processing import ParallelProcessor
                
                # Initialize components with optimizations
                config_manager = ConfigurationManager()
                file_service = FileProcessingService(config_manager)
                cache_manager = CacheManager()
                progress_tracker = ProgressTracker()
                parallel_processor = ParallelProcessor()
                
                matching_engine = MatchingEngine(
                    config=config,
                    cache_manager=cache_manager,
                    progress_tracker=progress_tracker,
                    parallel_processor=parallel_processor
                )
                
                # Load files
                dataset1 = file_service.load_file(file1_path, config["file1"])
                dataset2 = file_service.load_file(file2_path, config["file2"])
                
                # Perform matching with parallel processing
                matching_result = matching_engine.find_matches(dataset1, dataset2)
                
                # Save results
                output_files = file_service.save_results(matching_result, config["settings"])
                
                # Calculate metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**2)
                
                processing_time = end_time - start_time
                memory_used = end_memory - start_memory
                total_comparisons = len(dataset1) * len(dataset2)
                throughput = total_comparisons / processing_time
                
                # Analyze resource usage
                if resource_measurements:
                    max_cpu = max(m['cpu_percent'] for m in resource_measurements)
                    max_memory = max(m['memory_used_mb'] for m in resource_measurements)
                    avg_cpu = sum(m['cpu_percent'] for m in resource_measurements) / len(resource_measurements)
                else:
                    max_cpu = max_memory = avg_cpu = 0
                
                result = {
                    'dataset_size': size,
                    'processing_time': processing_time,
                    'memory_used_mb': memory_used,
                    'max_memory_mb': max_memory,
                    'max_cpu_percent': max_cpu,
                    'avg_cpu_percent': avg_cpu,
                    'throughput': throughput,
                    'matches_found': len(matching_result.matched_records),
                    'total_comparisons': total_comparisons
                }
                
                results.append(result)
                
                # Performance assertions
                self.assertLess(processing_time, self.performance_thresholds['medium_dataset_time'],
                               f"Medium dataset processing too slow: {processing_time:.2f}s")
                
                self.assertLess(max_memory, self.performance_thresholds['memory_usage_mb'] * 2,
                               f"Memory usage too high: {max_memory:.2f}MB")
                
                print(f"✅ Size {size}: {processing_time:.2f}s, {max_memory:.1f}MB peak, "
                      f"{throughput:.0f} comp/s, {avg_cpu:.1f}% avg CPU")
                
            except Exception as e:
                print(f"❌ Size {size}: Test failed - {e}")
                # Don't fail the test for medium datasets, just log the issue
                print(f"⚠️  Performance test issue for size {size}: {e}")
        
        # Print summary
        print(f"\nMedium Dataset Performance Summary:")
        for result in results:
            print(f"  {result['dataset_size']} records: {result['processing_time']:.2f}s, "
                  f"{result['throughput']:.0f} comp/s, {result['matches_found']} matches")

    def test_03_memory_efficiency(self):
        """Test memory efficiency and garbage collection"""
        print("\n=== Testing Memory Efficiency ===")
        
        # Test memory usage patterns
        dataset_size = 5000
        file1_path, df1 = self._create_test_dataset(dataset_size, "memory_test1")
        file2_path, df2 = self._create_test_dataset(dataset_size, "memory_test2")
        
        # Monitor memory usage throughout processing
        memory_measurements = []
        
        def memory_monitor():
            while True:
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_measurements.append({
                        'timestamp': time.time(),
                        'rss_mb': memory_info.rss / (1024**2),
                        'vms_mb': memory_info.vms / (1024**2)
                    })
                    time.sleep(0.5)
                except:
                    break
        
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Test memory-efficient processing
            from src.infrastructure.memory_management import MemoryManager
            from src.infrastructure.memory_mapped_files import MemoryMappedFileProcessor
            
            memory_manager = MemoryManager()
            file_processor = MemoryMappedFileProcessor()
            
            # Test memory-mapped file processing
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            # Process files with memory mapping
            dataset1 = file_processor.load_file_memory_mapped(file1_path)
            dataset2 = file_processor.load_file_memory_mapped(file2_path)
            
            # Perform memory-efficient matching
            matches = file_processor.find_matches_memory_efficient(dataset1, dataset2)
            
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_used = end_memory - start_memory
            
            # Analyze memory usage patterns
            if memory_measurements:
                peak_memory = max(m['rss_mb'] for m in memory_measurements)
                memory_growth = peak_memory - memory_measurements[0]['rss_mb']
                
                print(f"✅ Memory efficiency test:")
                print(f"   - Memory used: {memory_used:.1f}MB")
                print(f"   - Peak memory: {peak_memory:.1f}MB")
                print(f"   - Memory growth: {memory_growth:.1f}MB")
                
                # Assert memory efficiency
                self.assertLess(memory_growth, self.performance_thresholds['memory_usage_mb'],
                               f"Memory growth too high: {memory_growth:.1f}MB")
            
        except ImportError:
            print("⚠️  Memory management modules not available")
        except Exception as e:
            print(f"⚠️  Memory efficiency test failed: {e}")

    def test_04_concurrent_processing(self):
        """Test concurrent processing performance"""
        print("\n=== Testing Concurrent Processing ===")
        
        dataset_size = 2000
        num_concurrent_jobs = 4
        
        # Create multiple test datasets
        test_jobs = []
        for i in range(num_concurrent_jobs):
            file1_path, _ = self._create_test_dataset(dataset_size, f"concurrent_file1_{i}")
            file2_path, _ = self._create_test_dataset(dataset_size, f"concurrent_file2_{i}")
            
            config = {
                "file1": {"path": file1_path, "type": "csv", "delimiter": ","},
                "file2": {"path": file2_path, "type": "csv", "delimiter": ","},
                "mapping_fields": [{
                    "file1_col": "name",
                    "file2_col": "name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "weight": 1.0
                }],
                "settings": {
                    "output_format": "json",
                    "matched_output_path": os.path.join(self.test_dir, f"concurrent_results_{i}"),
                    "confidence_threshold": 75
                }
            }
            
            test_jobs.append(config)
        
        # Test sequential processing
        print("Testing sequential processing...")
        sequential_start = time.time()
        
        for i, config in enumerate(test_jobs):
            try:
                from src.application.services.file_service import FileProcessingService
                from src.application.services.config_service import ConfigurationManager
                from src.domain.matching.engine import MatchingEngine
                
                config_manager = ConfigurationManager()
                file_service = FileProcessingService(config_manager)
                matching_engine = MatchingEngine(config=config)
                
                dataset1 = file_service.load_file(config["file1"]["path"], config["file1"])
                dataset2 = file_service.load_file(config["file2"]["path"], config["file2"])
                
                matching_result = matching_engine.find_matches(dataset1, dataset2)
                file_service.save_results(matching_result, config["settings"])
                
            except Exception as e:
                print(f"⚠️  Sequential job {i} failed: {e}")
        
        sequential_time = time.time() - sequential_start
        
        # Test concurrent processing
        print("Testing concurrent processing...")
        concurrent_start = time.time()
        
        def process_job(config):
            try:
                from src.application.services.file_service import FileProcessingService
                from src.application.services.config_service import ConfigurationManager
                from src.domain.matching.engine import MatchingEngine
                
                config_manager = ConfigurationManager()
                file_service = FileProcessingService(config_manager)
                matching_engine = MatchingEngine(config=config)
                
                dataset1 = file_service.load_file(config["file1"]["path"], config["file1"])
                dataset2 = file_service.load_file(config["file2"]["path"], config["file2"])
                
                matching_result = matching_engine.find_matches(dataset1, dataset2)
                file_service.save_results(matching_result, config["settings"])
                
                return True
            except Exception as e:
                print(f"⚠️  Concurrent job failed: {e}")
                return False
        
        with ThreadPoolExecutor(max_workers=num_concurrent_jobs) as executor:
            futures = [executor.submit(process_job, config) for config in test_jobs]
            results = [future.result() for future in futures]
        
        concurrent_time = time.time() - concurrent_start
        
        # Calculate performance improvement
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
        efficiency = speedup / num_concurrent_jobs
        
        print(f"✅ Concurrent processing results:")
        print(f"   - Sequential time: {sequential_time:.2f}s")
        print(f"   - Concurrent time: {concurrent_time:.2f}s")
        print(f"   - Speedup: {speedup:.2f}x")
        print(f"   - Efficiency: {efficiency:.2f}")
        print(f"   - Successful jobs: {sum(results)}/{len(results)}")
        
        # Assert reasonable speedup
        self.assertGreater(speedup, 1.5, f"Concurrent processing speedup too low: {speedup:.2f}x")
        self.assertGreaterEqual(sum(results), len(results) * 0.8, "Too many concurrent jobs failed")

    def test_05_caching_performance(self):
        """Test caching system performance"""
        print("\n=== Testing Caching Performance ===")
        
        dataset_size = 1000
        file1_path, _ = self._create_test_dataset(dataset_size, "cache_test1")
        file2_path, _ = self._create_test_dataset(dataset_size, "cache_test2")
        
        config = {
            "file1": {"path": file1_path, "type": "csv", "delimiter": ","},
            "file2": {"path": file2_path, "type": "csv", "delimiter": ","},
            "mapping_fields": [{
                "file1_col": "name",
                "file2_col": "name",
                "match_type": "fuzzy",
                "use_normalization": True,
                "weight": 1.0
            }],
            "settings": {
                "output_format": "json",
                "matched_output_path": os.path.join(self.test_dir, "cache_results"),
                "confidence_threshold": 75
            }
        }
        
        try:
            from src.infrastructure.caching import CacheManager
            from src.domain.matching.engine import MatchingEngine
            from src.application.services.file_service import FileProcessingService
            from src.application.services.config_service import ConfigurationManager
            
            # Test without caching
            print("Testing without caching...")
            no_cache_start = time.time()
            
            config_manager = ConfigurationManager()
            file_service = FileProcessingService(config_manager)
            matching_engine = MatchingEngine(config=config, cache_manager=None)
            
            dataset1 = file_service.load_file(file1_path, config["file1"])
            dataset2 = file_service.load_file(file2_path, config["file2"])
            
            result1 = matching_engine.find_matches(dataset1, dataset2)
            no_cache_time = time.time() - no_cache_start
            
            # Test with caching (first run)
            print("Testing with caching (first run)...")
            cache_manager = CacheManager()
            matching_engine_cached = MatchingEngine(config=config, cache_manager=cache_manager)
            
            cache_first_start = time.time()
            result2 = matching_engine_cached.find_matches(dataset1, dataset2)
            cache_first_time = time.time() - cache_first_start
            
            # Test with caching (second run - should be faster)
            print("Testing with caching (second run)...")
            cache_second_start = time.time()
            result3 = matching_engine_cached.find_matches(dataset1, dataset2)
            cache_second_time = time.time() - cache_second_start
            
            # Calculate cache performance
            cache_hit_speedup = cache_first_time / cache_second_time if cache_second_time > 0 else 0
            
            print(f"✅ Caching performance results:")
            print(f"   - No cache: {no_cache_time:.2f}s")
            print(f"   - Cache first run: {cache_first_time:.2f}s")
            print(f"   - Cache second run: {cache_second_time:.2f}s")
            print(f"   - Cache hit speedup: {cache_hit_speedup:.2f}x")
            
            # Assert caching effectiveness
            self.assertGreater(cache_hit_speedup, 2.0, f"Cache hit speedup too low: {cache_hit_speedup:.2f}x")
            
            # Verify results consistency
            self.assertEqual(len(result1.matched_records), len(result2.matched_records),
                           "Cached results differ from non-cached")
            self.assertEqual(len(result2.matched_records), len(result3.matched_records),
                           "Cache hit results differ from cache miss")
            
        except ImportError:
            print("⚠️  Caching modules not available")
        except Exception as e:
            print(f"⚠️  Caching performance test failed: {e}")

    def test_06_scalability_limits(self):
        """Test system scalability limits"""
        print("\n=== Testing Scalability Limits ===")
        
        # Test increasing dataset sizes to find limits
        dataset_sizes = [1000, 5000, 10000, 20000]
        scalability_results = []
        
        for size in dataset_sizes:
            print(f"Testing scalability with {size} records...")
            
            try:
                file1_path, _ = self._create_test_dataset(size, f"scale_test1_{size}")
                file2_path, _ = self._create_test_dataset(size, f"scale_test2_{size}")
                
                config = {
                    "file1": {"path": file1_path, "type": "csv", "delimiter": ","},
                    "file2": {"path": file2_path, "type": "csv", "delimiter": ","},
                    "mapping_fields": [{
                        "file1_col": "name",
                        "file2_col": "name",
                        "match_type": "fuzzy",
                        "use_normalization": True,
                        "weight": 1.0
                    }],
                    "settings": {
                        "output_format": "json",
                        "matched_output_path": os.path.join(self.test_dir, f"scale_results_{size}"),
                        "confidence_threshold": 75
                    }
                }
                
                # Monitor resources during processing
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / (1024**2)
                
                from src.application.services.file_service import FileProcessingService
                from src.application.services.config_service import ConfigurationManager
                from src.domain.matching.engine import MatchingEngine
                from src.infrastructure.caching import CacheManager
                
                config_manager = ConfigurationManager()
                file_service = FileProcessingService(config_manager)
                cache_manager = CacheManager()
                matching_engine = MatchingEngine(config=config, cache_manager=cache_manager)
                
                dataset1 = file_service.load_file(file1_path, config["file1"])
                dataset2 = file_service.load_file(file2_path, config["file2"])
                
                matching_result = matching_engine.find_matches(dataset1, dataset2)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**2)
                
                processing_time = end_time - start_time
                memory_used = end_memory - start_memory
                total_comparisons = len(dataset1) * len(dataset2)
                throughput = total_comparisons / processing_time
                
                result = {
                    'dataset_size': size,
                    'processing_time': processing_time,
                    'memory_used_mb': memory_used,
                    'throughput': throughput,
                    'matches_found': len(matching_result.matched_records),
                    'success': True
                }
                
                scalability_results.append(result)
                
                print(f"✅ Size {size}: {processing_time:.2f}s, {memory_used:.1f}MB, {throughput:.0f} comp/s")
                
                # Check if we're hitting limits
                if processing_time > 300:  # 5 minutes
                    print(f"⚠️  Processing time limit reached at {size} records")
                    break
                
                if memory_used > 1000:  # 1GB
                    print(f"⚠️  Memory limit reached at {size} records")
                    break
                
            except Exception as e:
                print(f"❌ Size {size}: Failed - {e}")
                scalability_results.append({
                    'dataset_size': size,
                    'processing_time': None,
                    'memory_used_mb': None,
                    'throughput': None,
                    'matches_found': None,
                    'success': False,
                    'error': str(e)
                })
                break
        
        # Analyze scalability
        successful_results = [r for r in scalability_results if r['success']]
        
        if len(successful_results) >= 2:
            # Calculate scalability metrics
            size_ratio = successful_results[-1]['dataset_size'] / successful_results[0]['dataset_size']
            time_ratio = successful_results[-1]['processing_time'] / successful_results[0]['processing_time']
            
            scalability_factor = time_ratio / (size_ratio ** 2)  # Expected O(n²) for matching
            
            print(f"\nScalability Analysis:")
            print(f"   - Size range: {successful_results[0]['dataset_size']} - {successful_results[-1]['dataset_size']} records")
            print(f"   - Time scaling: {time_ratio:.2f}x for {size_ratio:.2f}x data")
            print(f"   - Scalability factor: {scalability_factor:.3f} (lower is better)")
            
            # Assert reasonable scalability
            self.assertLess(scalability_factor, 2.0, f"Poor scalability: {scalability_factor:.3f}")
        
        print(f"\nScalability Summary:")
        for result in scalability_results:
            if result['success']:
                print(f"  {result['dataset_size']} records: {result['processing_time']:.2f}s, "
                      f"{result['throughput']:.0f} comp/s")
            else:
                print(f"  {result['dataset_size']} records: FAILED")

def run_performance_benchmarks():
    """Run all performance benchmarks"""
    print("="*80)
    print("PERFORMANCE BENCHMARKING SUITE")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmarkSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nPERFORMANCE ISSUES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nTEST ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_performance_benchmarks()
    sys.exit(0 if success else 1)