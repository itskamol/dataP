"""
Parallel processing infrastructure for large-scale operations.
Implements requirement 3.4, 8.3, 3.1: Parallel processing and horizontal scaling.
"""

import multiprocessing as mp
import threading
import concurrent.futures
import queue
import time
import os
from typing import Dict, List, Optional, Any, Callable, Iterator, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
import hashlib
from contextlib import contextmanager

import pandas as pd
import numpy as np

from src.infrastructure.logging import get_logger
from src.infrastructure.memory_management import get_memory_manager
from src.domain.exceptions import ProcessingError


class ProcessingMode(Enum):
    """Processing modes for different workload types."""
    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    HYBRID = "hybrid"


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    mode: ProcessingMode = ProcessingMode.MULTIPROCESSING
    max_workers: Optional[int] = None
    chunk_size: int = 1000
    memory_limit_mb: int = 1024
    timeout_seconds: Optional[int] = None
    enable_progress_tracking: bool = True
    enable_error_recovery: bool = True
    batch_size: int = 10


@dataclass
class ProcessingResult:
    """Result from parallel processing operation."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None
    chunk_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkerPool:
    """Managed worker pool for parallel processing."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = get_logger('worker_pool')
        
        # Determine optimal worker count
        if config.max_workers is None:
            if config.mode == ProcessingMode.THREADING:
                self.max_workers = min(32, (os.cpu_count() or 1) * 2)
            else:
                self.max_workers = os.cpu_count() or 1
        else:
            self.max_workers = config.max_workers
        
        self._executor: Optional[concurrent.futures.Executor] = None
        self._active_futures: List[concurrent.futures.Future] = []
        self._results_queue = queue.Queue()
        self._shutdown = False
        
        self.logger.info(f"Worker pool initialized: {config.mode.value} mode, {self.max_workers} workers")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def start(self):
        """Start the worker pool."""
        if self._executor is not None:
            return
        
        if self.config.mode == ProcessingMode.THREADING:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="worker"
            )
        elif self.config.mode == ProcessingMode.MULTIPROCESSING:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context('spawn')
            )
        else:
            raise ValueError(f"Unsupported processing mode: {self.config.mode}")
        
        self.logger.info(f"Worker pool started with {self.max_workers} workers")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the worker pool."""
        if self._executor is None:
            raise RuntimeError("Worker pool not started")
        
        future = self._executor.submit(func, *args, **kwargs)
        self._active_futures.append(future)
        return future
    
    def submit_batch(self, func: Callable, tasks: List[Tuple[Any, ...]], 
                    callback: Optional[Callable] = None) -> List[concurrent.futures.Future]:
        """Submit a batch of tasks to the worker pool."""
        futures = []
        
        for task_args in tasks:
            if isinstance(task_args, tuple):
                future = self.submit_task(func, *task_args)
            else:
                future = self.submit_task(func, task_args)
            
            if callback:
                future.add_done_callback(callback)
            
            futures.append(future)
        
        return futures
    
    def wait_for_completion(self, futures: List[concurrent.futures.Future], 
                          timeout: Optional[float] = None) -> List[ProcessingResult]:
        """Wait for futures to complete and return results."""
        results = []
        
        try:
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(ProcessingResult(
                        success=True,
                        result=result,
                        processing_time=0.0  # TODO: Track actual processing time
                    ))
                except Exception as e:
                    results.append(ProcessingResult(
                        success=False,
                        error=str(e),
                        processing_time=0.0
                    ))
        
        except concurrent.futures.TimeoutError:
            self.logger.warning(f"Timeout waiting for {len(futures)} tasks")
            # Cancel remaining futures
            for future in futures:
                if not future.done():
                    future.cancel()
        
        return results
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        if self._executor is None:
            return
        
        self._shutdown = True
        
        # Cancel active futures
        for future in self._active_futures:
            if not future.done():
                future.cancel()
        
        # Shutdown executor
        self._executor.shutdown(wait=wait)
        self._executor = None
        
        self.logger.info("Worker pool shutdown completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        active_count = sum(1 for f in self._active_futures if not f.done())
        completed_count = sum(1 for f in self._active_futures if f.done())
        
        return {
            'max_workers': self.max_workers,
            'processing_mode': self.config.mode.value,
            'active_tasks': active_count,
            'completed_tasks': completed_count,
            'total_submitted': len(self._active_futures),
            'is_shutdown': self._shutdown
        }


class ChunkProcessor:
    """Process data in chunks using parallel workers."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = get_logger('chunk_processor')
        self.memory_manager = get_memory_manager()
    
    def process_dataframe_chunks(self, df: pd.DataFrame, 
                                processor_func: Callable[[pd.DataFrame], Any],
                                chunk_size: Optional[int] = None) -> List[Any]:
        """
        Process DataFrame in chunks using parallel workers.
        
        Args:
            df: DataFrame to process
            processor_func: Function to process each chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of processing results
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        # Split DataFrame into chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunks.append((chunk, i // chunk_size))
        
        self.logger.info(f"Processing DataFrame in {len(chunks)} chunks of size {chunk_size}")
        
        # Process chunks in parallel
        with WorkerPool(self.config) as pool:
            # Create wrapper function that includes chunk metadata
            def chunk_wrapper(chunk_data):
                chunk_df, chunk_id = chunk_data
                try:
                    result = processor_func(chunk_df)
                    return ProcessingResult(
                        success=True,
                        result=result,
                        chunk_id=chunk_id,
                        metadata={'chunk_size': len(chunk_df)}
                    )
                except Exception as e:
                    return ProcessingResult(
                        success=False,
                        error=str(e),
                        chunk_id=chunk_id
                    )
            
            # Submit all chunks
            futures = pool.submit_batch(chunk_wrapper, chunks)
            
            # Wait for completion
            results = pool.wait_for_completion(futures, timeout=self.config.timeout_seconds)
        
        # Sort results by chunk_id to maintain order
        results.sort(key=lambda x: x.chunk_id or 0)
        
        # Extract successful results
        successful_results = []
        failed_chunks = []
        
        for result in results:
            if result.success:
                successful_results.append(result.result)
            else:
                failed_chunks.append(result)
        
        if failed_chunks:
            self.logger.warning(f"{len(failed_chunks)} chunks failed processing")
            if not self.config.enable_error_recovery:
                raise ProcessingError(f"Chunk processing failed: {failed_chunks[0].error}")
        
        return successful_results
    
    def process_file_chunks(self, file_paths: List[str],
                           processor_func: Callable[[str], Any]) -> List[Any]:
        """
        Process multiple files in parallel.
        
        Args:
            file_paths: List of file paths to process
            processor_func: Function to process each file
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Processing {len(file_paths)} files in parallel")
        
        with WorkerPool(self.config) as pool:
            # Create wrapper function for file processing
            def file_wrapper(file_path):
                try:
                    result = processor_func(file_path)
                    return ProcessingResult(
                        success=True,
                        result=result,
                        metadata={'file_path': file_path}
                    )
                except Exception as e:
                    return ProcessingResult(
                        success=False,
                        error=str(e),
                        metadata={'file_path': file_path}
                    )
            
            # Submit all files
            futures = pool.submit_batch(file_wrapper, file_paths)
            
            # Wait for completion
            results = pool.wait_for_completion(futures, timeout=self.config.timeout_seconds)
        
        # Extract successful results
        successful_results = []
        failed_files = []
        
        for result in results:
            if result.success:
                successful_results.append(result.result)
            else:
                failed_files.append(result)
        
        if failed_files:
            self.logger.warning(f"{len(failed_files)} files failed processing")
            if not self.config.enable_error_recovery:
                raise ProcessingError(f"File processing failed: {failed_files[0].error}")
        
        return successful_results


class DistributedTaskQueue:
    """Distributed task queue for horizontal scaling."""
    
    def __init__(self, queue_name: str = "default"):
        self.queue_name = queue_name
        self.logger = get_logger('distributed_queue')
        
        # In-memory queue for now (could be replaced with Redis, RabbitMQ, etc.)
        self._task_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._shutdown = False
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Submit a task to the distributed queue."""
        task = {
            'task_id': task_id,
            'func': pickle.dumps(func),
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        self._task_queue.put(task)
        self.logger.debug(f"Task submitted: {task_id}")
    
    def start_workers(self, num_workers: int = None):
        """Start worker threads to process tasks."""
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(f"worker_{i}",),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        self.logger.info(f"Started {num_workers} distributed workers")
    
    def _worker_loop(self, worker_id: str):
        """Worker loop to process tasks from the queue."""
        while not self._shutdown:
            try:
                # Get task with timeout
                task = self._task_queue.get(timeout=1.0)
                
                # Process task
                start_time = time.time()
                try:
                    func = pickle.loads(task['func'])
                    result = func(*task['args'], **task['kwargs'])
                    
                    # Put result
                    self._result_queue.put({
                        'task_id': task['task_id'],
                        'success': True,
                        'result': result,
                        'worker_id': worker_id,
                        'processing_time': time.time() - start_time
                    })
                    
                except Exception as e:
                    self._result_queue.put({
                        'task_id': task['task_id'],
                        'success': False,
                        'error': str(e),
                        'worker_id': worker_id,
                        'processing_time': time.time() - start_time
                    })
                
                finally:
                    self._task_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a result from the result queue."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def shutdown(self):
        """Shutdown the distributed task queue."""
        self._shutdown = True
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self.logger.info("Distributed task queue shutdown completed")


class AdaptiveResourceManager:
    """Adaptive resource allocation based on workload characteristics."""
    
    def __init__(self):
        self.logger = get_logger('adaptive_resource_manager')
        self.memory_manager = get_memory_manager()
        
        # Resource usage history
        self._cpu_usage_history: List[float] = []
        self._memory_usage_history: List[float] = []
        self._task_completion_times: List[float] = []
        
        # Current resource allocation
        self._current_workers = os.cpu_count() or 1
        self._current_chunk_size = 1000
        
    def analyze_workload(self, data_size: int, complexity_score: float = 1.0) -> ProcessingConfig:
        """
        Analyze workload and recommend optimal processing configuration.
        
        Args:
            data_size: Size of data to process
            complexity_score: Complexity score (1.0 = normal, >1.0 = complex)
            
        Returns:
            Recommended processing configuration
        """
        # Get current system resources
        memory_stats = self.memory_manager.get_comprehensive_stats()
        available_memory_mb = memory_stats['memory_stats']['available_memory_mb']
        
        # Determine processing mode based on data size and complexity
        if data_size < 10000:
            mode = ProcessingMode.SEQUENTIAL
            workers = 1
            chunk_size = data_size
        elif data_size < 100000:
            mode = ProcessingMode.THREADING
            workers = min(4, os.cpu_count() or 1)
            chunk_size = max(1000, data_size // workers)
        else:
            mode = ProcessingMode.MULTIPROCESSING
            workers = os.cpu_count() or 1
            chunk_size = max(5000, data_size // (workers * 2))
        
        # Adjust for complexity
        if complexity_score > 2.0:
            workers = min(workers, max(1, workers // 2))  # Reduce workers for complex tasks
            chunk_size = max(chunk_size // 2, 100)  # Smaller chunks for complex tasks
        
        # Adjust for available memory
        memory_per_worker = available_memory_mb / workers
        if memory_per_worker < 100:  # Less than 100MB per worker
            workers = max(1, int(available_memory_mb / 100))
            chunk_size = min(chunk_size, 1000)  # Smaller chunks to reduce memory usage
        
        config = ProcessingConfig(
            mode=mode,
            max_workers=workers,
            chunk_size=chunk_size,
            memory_limit_mb=int(memory_per_worker * 0.8),  # 80% of available memory per worker
            timeout_seconds=max(60, data_size // 1000),  # Scale timeout with data size
            enable_progress_tracking=data_size > 10000,
            enable_error_recovery=True
        )
        
        self.logger.info(f"Workload analysis: {data_size} items, complexity {complexity_score:.1f}")
        self.logger.info(f"Recommended config: {mode.value}, {workers} workers, chunk size {chunk_size}")
        
        return config
    
    def update_performance_metrics(self, processing_time: float, memory_usage: float, 
                                 cpu_usage: float):
        """Update performance metrics for adaptive optimization."""
        self._task_completion_times.append(processing_time)
        self._memory_usage_history.append(memory_usage)
        self._cpu_usage_history.append(cpu_usage)
        
        # Keep only recent history (last 100 measurements)
        if len(self._task_completion_times) > 100:
            self._task_completion_times = self._task_completion_times[-100:]
            self._memory_usage_history = self._memory_usage_history[-100:]
            self._cpu_usage_history = self._cpu_usage_history[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._task_completion_times:
            return {'status': 'no_data'}
        
        return {
            'avg_completion_time': np.mean(self._task_completion_times),
            'avg_memory_usage': np.mean(self._memory_usage_history),
            'avg_cpu_usage': np.mean(self._cpu_usage_history),
            'current_workers': self._current_workers,
            'current_chunk_size': self._current_chunk_size,
            'total_tasks_processed': len(self._task_completion_times)
        }


# Global instances
_adaptive_resource_manager: Optional[AdaptiveResourceManager] = None
_resource_manager_lock = threading.Lock()


def get_adaptive_resource_manager() -> AdaptiveResourceManager:
    """Get global adaptive resource manager instance."""
    global _adaptive_resource_manager
    
    if _adaptive_resource_manager is None:
        with _resource_manager_lock:
            if _adaptive_resource_manager is None:
                _adaptive_resource_manager = AdaptiveResourceManager()
    
    return _adaptive_resource_manager


@contextmanager
def parallel_processing_context(data_size: int, complexity_score: float = 1.0):
    """Context manager for optimized parallel processing."""
    resource_manager = get_adaptive_resource_manager()
    config = resource_manager.analyze_workload(data_size, complexity_score)
    
    processor = ChunkProcessor(config)
    
    start_time = time.time()
    try:
        yield processor
    finally:
        # Update performance metrics
        processing_time = time.time() - start_time
        memory_stats = resource_manager.memory_manager.get_comprehensive_stats()
        
        resource_manager.update_performance_metrics(
            processing_time=processing_time,
            memory_usage=memory_stats['memory_stats']['process_memory_mb'],
            cpu_usage=0.0  # TODO: Add CPU usage tracking
        )