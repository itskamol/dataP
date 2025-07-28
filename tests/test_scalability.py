"""
Scalability tests for various deployment configurations.
Tests for task 11.2: Parallel processing and GPU acceleration.
"""

import time
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.infrastructure.parallel_processing import (
    ProcessingConfig, ProcessingMode, WorkerPool, ChunkProcessor, 
    AdaptiveResourceManager, parallel_processing_context
)
from src.infrastructure.gpu_acceleration import (
    GPUConfig, GPUBackend, GPUAcceleratedMatchingService, get_gpu_matching_service
)
from src.infrastructure.message_queue import (
    MessageQueueManager, Message, MessagePriority, MessageWorker
)


class ScalabilityTestSuite:
    """Comprehensive scalability test suite."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.temp_dir = None
    
    def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataframe(self, rows: int, columns: int = 5) -> pd.DataFrame:
        """Create test DataFrame for scalability tests."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        for i in range(columns):
            if i == 0:
                data[f'col_{i}'] = [f'text_{j}_{np.random.randint(0, 1000)}' for j in range(rows)]
            elif i == 1:
                data[f'col_{i}'] = np.random.randn(rows)
            else:
                data[f'col_{i}'] = np.random.randint(0, 100, rows)
        
        return pd.DataFrame(data)
    
    def test_parallel_processing_scalability(self):
        """Test parallel processing with different data sizes and worker counts."""
        print("\n=== Parallel Processing Scalability Tests ===")
        
        data_sizes = [1000, 10000, 50000, 100000]
        worker_counts = [1, 2, 4, 8]
        processing_modes = [ProcessingMode.THREADING, ProcessingMode.MULTIPROCESSING]
        
        def simple_processor(chunk_df: pd.DataFrame) -> int:
            """Simple processing function for testing."""
            # Simulate some computation
            result = chunk_df.iloc[:, 0].str.len().sum()
            time.sleep(0.001)  # Small delay to simulate work
            return result
        
        for data_size in data_sizes:
            df = self.create_test_dataframe(data_size)
            
            for mode in processing_modes:
                for worker_count in worker_counts:
                    config = ProcessingConfig(
                        mode=mode,
                        max_workers=worker_count,
                        chunk_size=max(100, data_size // (worker_count * 4))
                    )
                    
                    processor = ChunkProcessor(config)
                    
                    start_time = time.time()
                    try:
                        results = processor.process_dataframe_chunks(df, simple_processor)
                        execution_time = time.time() - start_time
                        
                        test_result = {
                            'test_type': 'parallel_processing',
                            'data_size': data_size,
                            'processing_mode': mode.value,
                            'worker_count': worker_count,
                            'execution_time': execution_time,
                            'throughput': data_size / execution_time,
                            'success': True,
                            'results_count': len(results)
                        }
                        
                        self.results.append(test_result)
                        print(f"✓ {mode.value} - {data_size} rows, {worker_count} workers: "
                              f"{execution_time:.3f}s ({data_size/execution_time:.0f} rows/s)")
                    
                    except Exception as e:
                        test_result = {
                            'test_type': 'parallel_processing',
                            'data_size': data_size,
                            'processing_mode': mode.value,
                            'worker_count': worker_count,
                            'execution_time': 0,
                            'throughput': 0,
                            'success': False,
                            'error': str(e)
                        }
                        
                        self.results.append(test_result)
                        print(f"✗ {mode.value} - {data_size} rows, {worker_count} workers: {str(e)}")
    
    def test_adaptive_resource_management(self):
        """Test adaptive resource management with different workloads."""
        print("\n=== Adaptive Resource Management Tests ===")
        
        resource_manager = AdaptiveResourceManager()
        
        test_scenarios = [
            {'data_size': 1000, 'complexity': 1.0, 'expected_mode': ProcessingMode.SEQUENTIAL},
            {'data_size': 50000, 'complexity': 1.0, 'expected_mode': ProcessingMode.THREADING},
            {'data_size': 200000, 'complexity': 1.0, 'expected_mode': ProcessingMode.MULTIPROCESSING},
            {'data_size': 100000, 'complexity': 3.0, 'expected_mode': ProcessingMode.MULTIPROCESSING},
        ]
        
        for scenario in test_scenarios:
            config = resource_manager.analyze_workload(
                scenario['data_size'], 
                scenario['complexity']
            )
            
            test_result = {
                'test_type': 'adaptive_resource_management',
                'data_size': scenario['data_size'],
                'complexity_score': scenario['complexity'],
                'recommended_mode': config.mode.value,
                'recommended_workers': config.max_workers,
                'recommended_chunk_size': config.chunk_size,
                'memory_limit_mb': config.memory_limit_mb,
                'expected_mode': scenario['expected_mode'].value,
                'success': True
            }
            
            self.results.append(test_result)
            print(f"✓ Data size: {scenario['data_size']}, Complexity: {scenario['complexity']:.1f}")
            print(f"  Recommended: {config.mode.value}, {config.max_workers} workers, "
                  f"chunk size: {config.chunk_size}")
    
    def test_gpu_acceleration_scalability(self):
        """Test GPU acceleration with different data sizes."""
        print("\n=== GPU Acceleration Scalability Tests ===")
        
        # Test data sizes
        data_sizes = [100, 1000, 5000, 10000]
        
        # Create GPU service (will fallback to CPU if GPU not available)
        gpu_config = GPUConfig(
            backend=GPUBackend.CUPY,
            fallback_to_cpu=True,
            batch_size=1000
        )
        
        gpu_service = get_gpu_matching_service(gpu_config)
        
        if gpu_service is None:
            print("GPU service not available, skipping GPU tests")
            return
        
        for data_size in data_sizes:
            # Create test string data
            strings1 = [f"test_string_{i}_{np.random.randint(0, 1000)}" for i in range(data_size)]
            strings2 = [f"match_string_{i}_{np.random.randint(0, 1000)}" for i in range(data_size)]
            
            start_time = time.time()
            try:
                results = gpu_service.match_datasets(
                    strings1, strings2,
                    algorithm='levenshtein',
                    threshold=0.7,
                    top_k=3
                )
                
                execution_time = time.time() - start_time
                
                test_result = {
                    'test_type': 'gpu_acceleration',
                    'data_size': data_size,
                    'execution_time': execution_time,
                    'throughput': data_size / execution_time,
                    'gpu_backend': results['gpu_stats']['backend'],
                    'gpu_available': results['gpu_stats']['gpu_available'],
                    'total_matches': results['statistics']['total_matches'],
                    'match_rate': results['statistics']['match_rate'],
                    'success': True
                }
                
                self.results.append(test_result)
                print(f"✓ GPU matching - {data_size} strings: {execution_time:.3f}s "
                      f"({data_size/execution_time:.0f} strings/s)")
                print(f"  Backend: {results['gpu_stats']['backend']}, "
                      f"Matches: {results['statistics']['total_matches']}")
            
            except Exception as e:
                test_result = {
                    'test_type': 'gpu_acceleration',
                    'data_size': data_size,
                    'execution_time': 0,
                    'throughput': 0,
                    'success': False,
                    'error': str(e)
                }
                
                self.results.append(test_result)
                print(f"✗ GPU matching - {data_size} strings: {str(e)}")
        
        # Cleanup GPU resources
        gpu_service.cleanup()
    
    def test_message_queue_scalability(self):
        """Test message queue scalability with different loads."""
        print("\n=== Message Queue Scalability Tests ===")
        
        message_counts = [100, 1000, 5000, 10000]
        worker_counts = [1, 2, 4, 8]
        
        def simple_handler(payload: Any) -> str:
            """Simple message handler for testing."""
            time.sleep(0.001)  # Simulate processing time
            return f"processed_{payload}"
        
        for message_count in message_counts:
            for worker_count in worker_counts:
                # Test in-memory queue
                queue_manager = MessageQueueManager(use_redis=False)
                queue_name = f"test_queue_{message_count}_{worker_count}"
                
                # Create workers
                workers = []
                for i in range(worker_count):
                    worker = queue_manager.create_worker(
                        queue_name, 
                        simple_handler,
                        f"worker_{i}"
                    )
                    workers.append(worker)
                
                # Start workers
                for worker in workers:
                    worker.start()
                
                # Send messages
                start_time = time.time()
                message_ids = []
                
                for i in range(message_count):
                    message_id = queue_manager.send_message(
                        queue_name,
                        f"message_{i}",
                        MessagePriority.NORMAL
                    )
                    message_ids.append(message_id)
                
                # Wait for processing to complete
                queue = queue_manager.get_queue(queue_name)
                while queue.size() > 0 or any(worker.get_stats()['messages_processed'] < message_count // worker_count for worker in workers):
                    time.sleep(0.1)
                    if time.time() - start_time > 30:  # Timeout after 30 seconds
                        break
                
                execution_time = time.time() - start_time
                
                # Stop workers
                for worker in workers:
                    worker.stop()
                
                # Collect statistics
                total_processed = sum(worker.get_stats()['messages_processed'] for worker in workers)
                total_failed = sum(worker.get_stats()['messages_failed'] for worker in workers)
                
                test_result = {
                    'test_type': 'message_queue',
                    'message_count': message_count,
                    'worker_count': worker_count,
                    'execution_time': execution_time,
                    'throughput': total_processed / execution_time if execution_time > 0 else 0,
                    'messages_processed': total_processed,
                    'messages_failed': total_failed,
                    'success_rate': total_processed / message_count if message_count > 0 else 0,
                    'success': True
                }
                
                self.results.append(test_result)
                print(f"✓ Queue - {message_count} messages, {worker_count} workers: "
                      f"{execution_time:.3f}s ({total_processed/execution_time:.0f} msg/s)")
                
                # Cleanup
                queue_manager.cleanup()
    
    def test_concurrent_load(self):
        """Test system behavior under concurrent load."""
        print("\n=== Concurrent Load Tests ===")
        
        def concurrent_processing_task(task_id: int, data_size: int) -> Dict[str, Any]:
            """Task for concurrent processing test."""
            df = self.create_test_dataframe(data_size)
            
            def processor(chunk_df: pd.DataFrame) -> int:
                return len(chunk_df)
            
            with parallel_processing_context(data_size, complexity_score=1.0) as processor_instance:
                start_time = time.time()
                results = processor_instance.process_dataframe_chunks(df, processor)
                execution_time = time.time() - start_time
                
                return {
                    'task_id': task_id,
                    'data_size': data_size,
                    'execution_time': execution_time,
                    'results_count': len(results),
                    'success': True
                }
        
        # Test concurrent processing with different loads
        concurrent_scenarios = [
            {'num_tasks': 2, 'data_size': 5000},
            {'num_tasks': 4, 'data_size': 2500},
            {'num_tasks': 8, 'data_size': 1250},
        ]
        
        for scenario in concurrent_scenarios:
            num_tasks = scenario['num_tasks']
            data_size = scenario['data_size']
            
            start_time = time.time()
            
            # Run tasks concurrently
            with ThreadPoolExecutor(max_workers=num_tasks) as executor:
                futures = [
                    executor.submit(concurrent_processing_task, i, data_size)
                    for i in range(num_tasks)
                ]
                
                # Wait for completion
                task_results = [future.result() for future in futures]
            
            total_execution_time = time.time() - start_time
            
            # Analyze results
            successful_tasks = [r for r in task_results if r['success']]
            avg_task_time = np.mean([r['execution_time'] for r in successful_tasks])
            total_data_processed = sum(r['data_size'] for r in successful_tasks)
            
            test_result = {
                'test_type': 'concurrent_load',
                'num_concurrent_tasks': num_tasks,
                'data_size_per_task': data_size,
                'total_execution_time': total_execution_time,
                'avg_task_execution_time': avg_task_time,
                'successful_tasks': len(successful_tasks),
                'total_data_processed': total_data_processed,
                'overall_throughput': total_data_processed / total_execution_time,
                'success': len(successful_tasks) == num_tasks
            }
            
            self.results.append(test_result)
            print(f"✓ Concurrent load - {num_tasks} tasks, {data_size} rows each: "
                  f"{total_execution_time:.3f}s total, {avg_task_time:.3f}s avg per task")
    
    def test_memory_scaling(self):
        """Test memory usage scaling with different data sizes."""
        print("\n=== Memory Scaling Tests ===")
        
        from src.infrastructure.memory_management import get_memory_manager
        
        memory_manager = get_memory_manager()
        data_sizes = [1000, 10000, 50000, 100000]
        
        for data_size in data_sizes:
            # Get initial memory stats
            initial_stats = memory_manager.get_comprehensive_stats()
            initial_memory = initial_stats['memory_stats']['process_memory_mb']
            
            # Create and process data
            df = self.create_test_dataframe(data_size)
            
            def memory_intensive_processor(chunk_df: pd.DataFrame) -> pd.DataFrame:
                # Create some additional data structures
                result = chunk_df.copy()
                result['new_col'] = result.iloc[:, 0].str.upper()
                return result
            
            with parallel_processing_context(data_size, complexity_score=1.0) as processor:
                start_time = time.time()
                results = processor.process_dataframe_chunks(df, memory_intensive_processor)
                execution_time = time.time() - start_time
            
            # Get final memory stats
            final_stats = memory_manager.get_comprehensive_stats()
            final_memory = final_stats['memory_stats']['process_memory_mb']
            memory_increase = final_memory - initial_memory
            
            # Force garbage collection
            memory_manager.emergency_cleanup()
            
            test_result = {
                'test_type': 'memory_scaling',
                'data_size': data_size,
                'execution_time': execution_time,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'memory_per_row_kb': (memory_increase * 1024) / data_size if data_size > 0 else 0,
                'success': True
            }
            
            self.results.append(test_result)
            print(f"✓ Memory scaling - {data_size} rows: {execution_time:.3f}s, "
                  f"memory increase: {memory_increase:.1f}MB ({memory_increase*1024/data_size:.2f}KB/row)")
    
    def run_all_tests(self):
        """Run all scalability tests."""
        print("Starting Scalability Test Suite...")
        print("=" * 60)
        
        self.setup()
        
        try:
            self.test_parallel_processing_scalability()
            self.test_adaptive_resource_management()
            self.test_gpu_acceleration_scalability()
            self.test_message_queue_scalability()
            self.test_concurrent_load()
            self.test_memory_scaling()
            
            # Generate summary report
            self._generate_summary_report()
        
        finally:
            self.teardown()
    
    def _generate_summary_report(self):
        """Generate summary report of all test results."""
        print("\n" + "=" * 60)
        print("SCALABILITY TEST SUMMARY")
        print("=" * 60)
        
        # Group results by test type
        test_types = {}
        for result in self.results:
            test_type = result['test_type']
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        for test_type, results in test_types.items():
            print(f"\n{test_type.upper()} RESULTS:")
            print("-" * 40)
            
            successful_results = [r for r in results if r.get('success', False)]
            failed_results = [r for r in results if not r.get('success', True)]
            
            print(f"Total tests: {len(results)}")
            print(f"Successful: {len(successful_results)}")
            print(f"Failed: {len(failed_results)}")
            
            if successful_results:
                if 'throughput' in successful_results[0]:
                    throughputs = [r['throughput'] for r in successful_results if r['throughput'] > 0]
                    if throughputs:
                        print(f"Throughput range: {min(throughputs):.0f} - {max(throughputs):.0f} items/s")
                        print(f"Average throughput: {np.mean(throughputs):.0f} items/s")
                
                if 'execution_time' in successful_results[0]:
                    execution_times = [r['execution_time'] for r in successful_results if r['execution_time'] > 0]
                    if execution_times:
                        print(f"Execution time range: {min(execution_times):.3f} - {max(execution_times):.3f}s")
            
            if failed_results:
                print("Failed test errors:")
                for result in failed_results[:3]:  # Show first 3 errors
                    print(f"  - {result.get('error', 'Unknown error')}")
        
        # Overall statistics
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.get('success', False)])
        
        print(f"\nOVERALL RESULTS:")
        print(f"Total tests run: {total_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        # Performance insights
        print(f"\nPERFORMANCE INSIGHTS:")
        
        # Find best performing configurations
        parallel_results = [r for r in self.results if r['test_type'] == 'parallel_processing' and r.get('success')]
        if parallel_results:
            best_parallel = max(parallel_results, key=lambda x: x.get('throughput', 0))
            print(f"Best parallel config: {best_parallel['processing_mode']} with {best_parallel['worker_count']} workers")
            print(f"  Throughput: {best_parallel['throughput']:.0f} rows/s on {best_parallel['data_size']} rows")
        
        gpu_results = [r for r in self.results if r['test_type'] == 'gpu_acceleration' and r.get('success')]
        if gpu_results:
            best_gpu = max(gpu_results, key=lambda x: x.get('throughput', 0))
            print(f"Best GPU performance: {best_gpu['throughput']:.0f} strings/s on {best_gpu['data_size']} strings")


# Test functions for individual components
def test_parallel_processing_basic():
    """Basic test for parallel processing functionality."""
    config = ProcessingConfig(
        mode=ProcessingMode.THREADING,
        max_workers=2,
        chunk_size=100
    )
    
    processor = ChunkProcessor(config)
    
    # Create test data
    df = pd.DataFrame({
        'col1': [f'text_{i}' for i in range(500)],
        'col2': range(500)
    })
    
    def simple_processor(chunk_df: pd.DataFrame) -> int:
        return len(chunk_df)
    
    results = processor.process_dataframe_chunks(df, simple_processor)
    
    # Verify results
    assert len(results) > 0
    assert sum(results) == len(df)
    print("✅ Basic parallel processing test passed!")


def test_adaptive_resource_management_basic():
    """Basic test for adaptive resource management."""
    resource_manager = AdaptiveResourceManager()
    
    # Test different workload scenarios
    small_config = resource_manager.analyze_workload(1000, 1.0)
    assert small_config.mode == ProcessingMode.SEQUENTIAL
    
    medium_config = resource_manager.analyze_workload(50000, 1.0)
    assert medium_config.mode in [ProcessingMode.THREADING, ProcessingMode.MULTIPROCESSING]
    
    large_config = resource_manager.analyze_workload(200000, 1.0)
    assert large_config.mode == ProcessingMode.MULTIPROCESSING
    
    print("✅ Basic adaptive resource management test passed!")


def test_message_queue_basic():
    """Basic test for message queue functionality."""
    queue_manager = MessageQueueManager(use_redis=False)
    
    # Test message sending and receiving
    queue_name = "test_queue"
    
    # Send messages
    message_ids = []
    for i in range(10):
        message_id = queue_manager.send_message(queue_name, f"test_message_{i}")
        message_ids.append(message_id)
    
    # Create and start worker
    processed_messages = []
    
    def test_handler(payload: Any) -> str:
        processed_messages.append(payload)
        return f"processed_{payload}"
    
    worker = queue_manager.create_worker(queue_name, test_handler)
    worker.start()
    
    # Wait for processing
    import time
    time.sleep(2)
    
    worker.stop()
    
    # Verify results
    assert len(processed_messages) == 10
    assert all(f"test_message_{i}" in processed_messages for i in range(10))
    
    queue_manager.cleanup()
    print("✅ Basic message queue test passed!")


if __name__ == "__main__":
    # Run individual tests first
    print("Running basic functionality tests...")
    test_parallel_processing_basic()
    test_adaptive_resource_management_basic()
    test_message_queue_basic()
    
    print("\nRunning comprehensive scalability test suite...")
    # Run comprehensive scalability tests
    test_suite = ScalabilityTestSuite()
    test_suite.run_all_tests()