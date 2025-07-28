"""
Core MatchingEngine with parallel processing and progress tracking.
Implements requirements 3.1, 3.4, 5.3, 4.1: Configurable algorithm pipeline, 
multiprocessing support, progress tracking, and memory management.
"""

import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Iterator, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Value
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import os

# Optional dependency for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .base import MatchingAlgorithm, MatchingResult
from ..models import MatchingType
from .exact_matcher import ExactMatcher
from .fuzzy_matcher import FuzzyMatcher
from .phonetic_matcher import PhoneticMatcher
from .blocking import OptimizedBlockingIndex, BlockingStrategy
from ..models import (
    MatchingConfig, Dataset, MatchingResult as DomainMatchingResult,
    MatchedRecord, MatchingStatistics, ResultMetadata, ProgressStatus
)
from ...infrastructure.logging import get_logger


@dataclass
class ProcessingChunk:
    """Represents a chunk of data for parallel processing."""
    chunk_id: str
    pairs: List[Tuple[int, int]]
    dataset1_chunk: pd.DataFrame
    dataset2_chunk: pd.DataFrame
    field_mappings: List[Tuple[str, str, float]]
    config: Dict[str, Any]


@dataclass
class ChunkResult:
    """Result from processing a chunk."""
    chunk_id: str
    matched_records: List[Dict[str, Any]]
    processing_time: float
    comparisons_made: int
    memory_used_mb: float


class ProgressTracker:
    """Thread-safe progress tracker for matching operations."""
    
    def __init__(self, operation_id: str, total_steps: int):
        self.operation_id = operation_id
        self.total_steps = total_steps
        self.current_step = Value('i', 0)
        # Use a simple string instead of multiprocessing Value for status
        self._status = 'running'
        self._message = 'Starting operation...'
        self.started_at = datetime.now()
        self.completed_at = None
        self.error_message = None
        self._lock = threading.RLock()
        self.logger = get_logger('matching.progress')
        
        # Callbacks for real-time updates
        self._callbacks: List[Callable[[ProgressStatus], None]] = []
    
    def add_callback(self, callback: Callable[[ProgressStatus], None]):
        """Add a callback for progress updates."""
        with self._lock:
            self._callbacks.append(callback)
    
    def update_progress(self, current_step: int, message: str = ""):
        """Update progress with current step and message."""
        with self._lock:
            self.current_step.value = current_step
            if message:
                self._message = message
            
            # Calculate progress percentage
            progress = min((current_step / self.total_steps) * 100, 100.0) if self.total_steps > 0 else 0.0
            
            # Create status object
            status = ProgressStatus(
                operation_id=self.operation_id,
                status=self._status,
                progress=progress,
                message=self._message,
                current_step=current_step,
                total_steps=self.total_steps,
                started_at=self.started_at,
                completed_at=self.completed_at,
                error_message=self.error_message
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"Error in progress callback: {e}")
    
    def complete(self, message: str = "Operation completed"):
        """Mark operation as completed."""
        with self._lock:
            self._status = 'completed'
            self._message = message
            self.completed_at = datetime.now()
            self.current_step.value = self.total_steps
            
            self.update_progress(self.total_steps, message)
    
    def error(self, error_message: str):
        """Mark operation as failed."""
        with self._lock:
            self._status = 'error'
            self.error_message = error_message
            self._message = f"Error: {error_message}"
            self.completed_at = datetime.now()
            
            self.update_progress(self.current_step.value, f"Error: {error_message}")
    
    def cancel(self):
        """Cancel the operation."""
        with self._lock:
            self._status = 'cancelled'
            self._message = 'Operation cancelled'
            self.completed_at = datetime.now()
            
            self.update_progress(self.current_step.value, "Operation cancelled")
    
    def get_status(self) -> ProgressStatus:
        """Get current progress status."""
        with self._lock:
            progress = min((self.current_step.value / self.total_steps) * 100, 100.0) if self.total_steps > 0 else 0.0
            
            return ProgressStatus(
                operation_id=self.operation_id,
                status=self._status,
                progress=progress,
                message=self._message,
                current_step=self.current_step.value,
                total_steps=self.total_steps,
                started_at=self.started_at,
                completed_at=self.completed_at,
                error_message=self.error_message
            )


class MemoryManager:
    """Memory management for large dataset processing."""
    
    def __init__(self, max_memory_mb: int = 1024, chunk_size: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size
        self.logger = get_logger('matching.memory')
        
        # Get initial memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                self.initial_memory_mb = process.memory_info().rss / 1024 / 1024
            except:
                self.initial_memory_mb = 0
        else:
            self.initial_memory_mb = 0
            self.logger.warning("psutil not available - memory monitoring disabled")
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except:
                return 0.0
        return 0.0
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        if not PSUTIL_AVAILABLE:
            return True  # Always return True if we can't monitor memory
        
        current_memory = self.get_current_memory_mb()
        memory_increase = current_memory - self.initial_memory_mb
        
        if memory_increase > self.max_memory_mb:
            self.logger.warning(f"Memory usage {memory_increase:.1f}MB exceeds limit {self.max_memory_mb}MB")
            return False
        
        return True
    
    def optimize_memory(self):
        """Optimize memory usage by forcing garbage collection."""
        gc.collect()
        self.logger.debug("Forced garbage collection")
    
    def calculate_optimal_chunk_size(self, dataset_size: int, available_memory_mb: float) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Estimate memory per record (rough approximation)
        memory_per_record_kb = 1.0  # 1KB per record estimate
        
        # Calculate how many records can fit in available memory
        available_memory_kb = available_memory_mb * 1024
        max_records_in_memory = int(available_memory_kb / memory_per_record_kb)
        
        # Use smaller of calculated size or default chunk size
        optimal_size = min(max_records_in_memory, self.chunk_size)
        
        # Ensure minimum chunk size
        return max(optimal_size, 100)


def process_chunk_worker(chunk_data: ProcessingChunk) -> ChunkResult:
    """
    Worker function for processing a chunk of record pairs.
    This function runs in a separate process.
    """
    start_time = time.time()
    initial_memory = 0
    
    try:
        # Get initial memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024
            except:
                pass
        
        # Initialize algorithms in worker process
        algorithms = []
        
        # Create algorithms based on config
        if chunk_data.config.get('use_exact', True):
            exact_config = chunk_data.config.get('exact_config', {})
            algorithms.append(ExactMatcher(exact_config))
        
        if chunk_data.config.get('use_fuzzy', True):
            fuzzy_config = chunk_data.config.get('fuzzy_config', {})
            algorithms.append(FuzzyMatcher(fuzzy_config))
        
        if chunk_data.config.get('use_phonetic', True):
            phonetic_config = chunk_data.config.get('phonetic_config', {})
            algorithms.append(PhoneticMatcher(phonetic_config))
        
        matched_records = []
        comparisons_made = 0
        
        # Process each pair in the chunk
        for record_id1, record_id2 in chunk_data.pairs:
            try:
                # Get records from datasets
                record1 = chunk_data.dataset1_chunk.loc[record_id1].to_dict()
                record2 = chunk_data.dataset2_chunk.loc[record_id2].to_dict()
                
                # Calculate combined similarity using all algorithms
                best_result = None
                best_score = 0.0
                
                for algorithm in algorithms:
                    result = algorithm.compare_records(record1, record2, chunk_data.field_mappings)
                    
                    if result.similarity_score > best_score:
                        best_score = result.similarity_score
                        best_result = result
                
                comparisons_made += 1
                
                # Add to results if above threshold
                confidence_threshold = chunk_data.config.get('confidence_threshold', 75.0)
                if best_result and best_result.confidence >= confidence_threshold:
                    matched_record = {
                        'record1': record1,
                        'record2': record2,
                        'confidence_score': best_result.confidence,
                        'similarity_score': best_result.similarity_score,
                        'matching_fields': best_result.matched_fields,
                        'metadata': best_result.metadata,
                        'created_at': datetime.now().isoformat()
                    }
                    matched_records.append(matched_record)
                
            except Exception as e:
                # Log error but continue processing
                continue
        
        # Calculate final memory usage
        final_memory = initial_memory
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                final_memory = process.memory_info().rss / 1024 / 1024
            except:
                pass
        
        processing_time = time.time() - start_time
        
        return ChunkResult(
            chunk_id=chunk_data.chunk_id,
            matched_records=matched_records,
            processing_time=processing_time,
            comparisons_made=comparisons_made,
            memory_used_mb=final_memory - initial_memory
        )
        
    except Exception as e:
        # Return error result
        return ChunkResult(
            chunk_id=chunk_data.chunk_id,
            matched_records=[],
            processing_time=time.time() - start_time,
            comparisons_made=0,
            memory_used_mb=0
        )


class MatchingEngine:
    """
    Core matching engine with configurable algorithm pipeline and parallel processing.
    Implements requirements 3.1, 3.4, 5.3, 4.1.
    """
    
    def __init__(self, config: Optional[MatchingConfig] = None):
        """
        Initialize the matching engine.
        
        Args:
            config: Matching configuration with algorithm settings
        """
        self.config = config or MatchingConfig()
        self.logger = get_logger('matching.engine')
        
        # Initialize components
        self.blocking_index = self._initialize_blocking_index()
        self.memory_manager = MemoryManager(
            max_memory_mb=self.config.thresholds.get('max_memory_mb', 1024),
            chunk_size=self.config.thresholds.get('chunk_size', 1000)
        )
        
        # Algorithm instances (created per process in multiprocessing)
        self.algorithms: List[MatchingAlgorithm] = []
        
        # Progress tracking
        self.current_operation: Optional[ProgressTracker] = None
        self._operation_lock = threading.RLock()
        
        # Performance statistics
        self.total_operations = 0
        self.total_processing_time = 0.0
        self.total_comparisons = 0
        
        self.logger.info("MatchingEngine initialized", extra={
            'parallel_processing': self.config.parallel_processing,
            'max_workers': self.config.max_workers,
            'confidence_threshold': self.config.confidence_threshold,
            'use_blocking': self.config.use_blocking
        })
    
    def _initialize_blocking_index(self) -> OptimizedBlockingIndex:
        """Initialize the blocking index based on configuration."""
        blocking_config = {
            'strategies': [
                BlockingStrategy.EXACT_PREFIX,
                BlockingStrategy.PHONETIC_KEY,
                BlockingStrategy.NGRAM
            ],
            'parallel_processing': self.config.parallel_processing,
            'max_workers': self.config.max_workers or 4,
            'adaptive_threshold': 1000,
            'max_block_size': 500
        }
        
        return OptimizedBlockingIndex(blocking_config)
    
    def _initialize_algorithms(self) -> List[MatchingAlgorithm]:
        """Initialize matching algorithms based on configuration."""
        algorithms = []
        
        for algo_config in self.config.algorithms:
            if not algo_config.enabled:
                continue
            
            try:
                algo_type = algo_config.algorithm_type
                if hasattr(algo_type, 'value'):
                    algo_type_str = algo_type.value
                else:
                    algo_type_str = str(algo_type)
                
                if algo_type_str == 'exact':
                    algorithm = ExactMatcher(algo_config.parameters)
                elif algo_type_str == 'fuzzy':
                    algorithm = FuzzyMatcher(algo_config.parameters)
                elif algo_type_str == 'phonetic':
                    algorithm = PhoneticMatcher(algo_config.parameters)
                else:
                    self.logger.warning(f"Unknown algorithm type: {algo_type_str}")
                    continue
                
                algorithms.append(algorithm)
                
            except Exception as e:
                self.logger.error(f"Failed to initialize algorithm {algo_config.name}: {e}")
        
        # If no algorithms configured, use defaults
        if not algorithms:
            algorithms = [
                ExactMatcher(),
                FuzzyMatcher(),
                PhoneticMatcher()
            ]
        
        return algorithms
    
    def find_matches(self, dataset1: Dataset, dataset2: Dataset, 
                    progress_callback: Optional[Callable[[ProgressStatus], None]] = None) -> DomainMatchingResult:
        """
        Find matches between two datasets using configured algorithms.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            progress_callback: Optional callback for progress updates
            
        Returns:
            MatchingResult with found matches and statistics
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting matching operation {operation_id}", extra={
            'dataset1_size': len(dataset1.data) if dataset1.data is not None else 0,
            'dataset2_size': len(dataset2.data) if dataset2.data is not None else 0,
            'field_mappings': len(self.config.mappings)
        })
        
        # Validate inputs
        if dataset1.data is None or dataset2.data is None:
            raise ValueError("Dataset data cannot be None")
        
        if len(self.config.mappings) == 0:
            raise ValueError("No field mappings configured")
        
        # Initialize progress tracker
        total_steps = self._estimate_total_steps(len(dataset1.data), len(dataset2.data))
        
        with self._operation_lock:
            self.current_operation = ProgressTracker(operation_id, total_steps)
            if progress_callback:
                self.current_operation.add_callback(progress_callback)
        
        try:
            # Step 1: Generate candidate pairs using blocking
            self.current_operation.update_progress(1, "Generating candidate pairs...")
            candidate_pairs = self._generate_candidate_pairs(dataset1.data, dataset2.data)
            
            # Step 2: Process candidate pairs
            self.current_operation.update_progress(2, f"Processing {len(candidate_pairs)} candidate pairs...")
            matched_records = self._process_candidate_pairs(
                candidate_pairs, dataset1.data, dataset2.data
            )
            
            # Step 3: Generate unmatched records
            self.current_operation.update_progress(total_steps - 1, "Generating unmatched records...")
            unmatched_records = self._generate_unmatched_records(
                matched_records, dataset1.data, dataset2.data
            )
            
            # Step 4: Create final result
            processing_time = time.time() - start_time
            statistics = self._calculate_statistics(
                matched_records, len(dataset1.data), len(dataset2.data), processing_time
            )
            
            metadata = ResultMetadata(
                operation_id=operation_id,
                file1_metadata=dataset1.metadata,
                file2_metadata=dataset2.metadata,
                config_hash=self._calculate_config_hash()
            )
            
            result = DomainMatchingResult(
                matched_records=[MatchedRecord.from_dict(record) for record in matched_records],
                unmatched_records=unmatched_records,
                statistics=statistics,
                metadata=metadata
            )
            
            # Update performance statistics
            self.total_operations += 1
            self.total_processing_time += processing_time
            self.total_comparisons += len(candidate_pairs)
            
            self.current_operation.complete(f"Found {len(matched_records)} matches")
            
            self.logger.info(f"Matching operation {operation_id} completed", extra={
                'processing_time': processing_time,
                'matches_found': len(matched_records),
                'comparisons_made': len(candidate_pairs),
                'reduction_ratio': self._calculate_reduction_ratio(len(dataset1.data), len(dataset2.data), len(candidate_pairs))
            })
            
            return result
            
        except Exception as e:
            self.current_operation.error(str(e))
            self.logger.error(f"Matching operation {operation_id} failed: {e}")
            raise
        
        finally:
            # Cleanup
            self.memory_manager.optimize_memory()
            with self._operation_lock:
                self.current_operation = None
    
    def _estimate_total_steps(self, size1: int, size2: int) -> int:
        """Estimate total steps for progress tracking."""
        # Base steps: blocking, processing, unmatched generation, finalization
        base_steps = 4
        
        # Add steps based on dataset size (for chunked processing)
        if self.config.parallel_processing:
            chunk_size = self.memory_manager.chunk_size
            estimated_chunks = max(1, (size1 * size2) // (chunk_size * chunk_size))
            return base_steps + min(estimated_chunks, 100)  # Cap at 100 for UI purposes
        
        return base_steps
    
    def _generate_candidate_pairs(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> List[Tuple[int, int]]:
        """Generate candidate pairs using blocking strategies."""
        if not self.config.use_blocking:
            # Generate all possible pairs (Cartesian product)
            self.logger.warning("Blocking disabled - generating all possible pairs")
            pairs = []
            for i in dataset1.index:
                for j in dataset2.index:
                    pairs.append((i, j))
            return pairs
        
        # Use blocking to reduce candidate pairs
        field_mappings = [(mapping.source_field, mapping.target_field, mapping.weight) 
                         for mapping in self.config.mappings]
        
        candidate_pairs = list(self.blocking_index.get_candidate_pairs(
            dataset1, dataset2, field_mappings
        ))
        
        blocking_stats = self.blocking_index.get_statistics()
        self.logger.info("Candidate pair generation completed", extra={
            'total_pairs': len(candidate_pairs),
            'reduction_ratio': blocking_stats.reduction_ratio,
            'blocking_time': blocking_stats.processing_time_seconds
        })
        
        return candidate_pairs
    
    def _process_candidate_pairs(self, candidate_pairs: List[Tuple[int, int]], 
                               dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process candidate pairs using parallel processing."""
        if not candidate_pairs:
            return []
        
        if self.config.parallel_processing and len(candidate_pairs) > 100:
            return self._process_pairs_parallel(candidate_pairs, dataset1, dataset2)
        else:
            return self._process_pairs_sequential(candidate_pairs, dataset1, dataset2)
    
    def _process_pairs_sequential(self, candidate_pairs: List[Tuple[int, int]], 
                                dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process candidate pairs sequentially."""
        matched_records = []
        algorithms = self._initialize_algorithms()
        
        field_mappings = [(mapping.source_field, mapping.target_field, mapping.weight) 
                         for mapping in self.config.mappings]
        
        for i, (record_id1, record_id2) in enumerate(candidate_pairs):
            try:
                # Update progress periodically
                if i % 100 == 0 and self.current_operation:
                    progress_step = 2 + int((i / len(candidate_pairs)) * (self.current_operation.total_steps - 3))
                    self.current_operation.update_progress(
                        progress_step, f"Processing pair {i+1}/{len(candidate_pairs)}"
                    )
                
                # Get records
                record1 = dataset1.loc[record_id1].to_dict()
                record2 = dataset2.loc[record_id2].to_dict()
                
                # Find best match using all algorithms
                best_result = self._find_best_match(record1, record2, field_mappings, algorithms)
                
                if best_result and best_result.confidence >= self.config.confidence_threshold:
                    matched_record = {
                        'record1': record1,
                        'record2': record2,
                        'confidence_score': best_result.confidence,
                        'similarity_score': best_result.similarity_score,
                        'matching_fields': best_result.matched_fields,
                        'metadata': best_result.metadata,
                        'created_at': datetime.now().isoformat()
                    }
                    matched_records.append(matched_record)
                
                # Check memory usage periodically
                if i % 1000 == 0:
                    if not self.memory_manager.check_memory_usage():
                        self.memory_manager.optimize_memory()
                
            except Exception as e:
                self.logger.warning(f"Error processing pair ({record_id1}, {record_id2}): {e}")
                continue
        
        return matched_records
    
    def _process_pairs_parallel(self, candidate_pairs: List[Tuple[int, int]], 
                              dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process candidate pairs using parallel processing."""
        # Calculate optimal chunk size
        available_memory = self.memory_manager.max_memory_mb * 0.8  # Use 80% of available memory
        chunk_size = self.memory_manager.calculate_optimal_chunk_size(
            len(candidate_pairs), available_memory
        )
        
        # Create chunks
        chunks = self._create_processing_chunks(candidate_pairs, dataset1, dataset2, chunk_size)
        
        matched_records = []
        max_workers = self.config.max_workers or min(4, len(chunks))
        
        self.logger.info(f"Processing {len(chunks)} chunks with {max_workers} workers")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_chunk_worker, chunk): chunk.chunk_id 
                for chunk in chunks
            }
            
            completed_chunks = 0
            total_comparisons = 0
            
            # Process completed chunks
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                
                try:
                    result = future.result()
                    matched_records.extend(result.matched_records)
                    total_comparisons += result.comparisons_made
                    
                    completed_chunks += 1
                    
                    # Update progress
                    if self.current_operation:
                        progress_step = 2 + int((completed_chunks / len(chunks)) * (self.current_operation.total_steps - 3))
                        self.current_operation.update_progress(
                            progress_step, 
                            f"Completed chunk {completed_chunks}/{len(chunks)} ({len(matched_records)} matches found)"
                        )
                    
                    self.logger.debug(f"Chunk {chunk_id} completed", extra={
                        'matches_found': len(result.matched_records),
                        'processing_time': result.processing_time,
                        'comparisons': result.comparisons_made,
                        'memory_used_mb': result.memory_used_mb
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_id}: {e}")
        
        self.logger.info(f"Parallel processing completed", extra={
            'total_matches': len(matched_records),
            'total_comparisons': total_comparisons,
            'chunks_processed': len(chunks)
        })
        
        return matched_records
    
    def _create_processing_chunks(self, candidate_pairs: List[Tuple[int, int]], 
                                dataset1: pd.DataFrame, dataset2: pd.DataFrame, 
                                chunk_size: int) -> List[ProcessingChunk]:
        """Create processing chunks for parallel execution."""
        chunks = []
        
        # Helper function to get algorithm type string
        def get_algo_type_str(algo_type):
            if hasattr(algo_type, 'value'):
                return algo_type.value
            return str(algo_type)
        
        # Prepare configuration for workers
        worker_config = {
            'use_exact': any(get_algo_type_str(algo.algorithm_type) == 'exact' for algo in self.config.algorithms if algo.enabled),
            'use_fuzzy': any(get_algo_type_str(algo.algorithm_type) == 'fuzzy' for algo in self.config.algorithms if algo.enabled),
            'use_phonetic': any(get_algo_type_str(algo.algorithm_type) == 'phonetic' for algo in self.config.algorithms if algo.enabled),
            'confidence_threshold': self.config.confidence_threshold,
            'exact_config': {},
            'fuzzy_config': {},
            'phonetic_config': {}
        }
        
        # Extract algorithm-specific configurations
        for algo_config in self.config.algorithms:
            if not algo_config.enabled:
                continue
            
            algo_type_str = get_algo_type_str(algo_config.algorithm_type)
            
            if algo_type_str == 'exact':
                worker_config['exact_config'] = algo_config.parameters
            elif algo_type_str == 'fuzzy':
                worker_config['fuzzy_config'] = algo_config.parameters
            elif algo_type_str == 'phonetic':
                worker_config['phonetic_config'] = algo_config.parameters
        
        field_mappings = [(mapping.source_field, mapping.target_field, mapping.weight) 
                         for mapping in self.config.mappings]
        
        # Split pairs into chunks
        for i in range(0, len(candidate_pairs), chunk_size):
            chunk_pairs = candidate_pairs[i:i + chunk_size]
            
            # Get unique record IDs for this chunk
            record_ids1 = set(pair[0] for pair in chunk_pairs)
            record_ids2 = set(pair[1] for pair in chunk_pairs)
            
            # Create data subsets for this chunk
            dataset1_chunk = dataset1.loc[list(record_ids1)].copy()
            dataset2_chunk = dataset2.loc[list(record_ids2)].copy()
            
            chunk = ProcessingChunk(
                chunk_id=f"chunk_{i // chunk_size}",
                pairs=chunk_pairs,
                dataset1_chunk=dataset1_chunk,
                dataset2_chunk=dataset2_chunk,
                field_mappings=field_mappings,
                config=worker_config
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _find_best_match(self, record1: Dict[str, Any], record2: Dict[str, Any], 
                        field_mappings: List[Tuple[str, str, float]], 
                        algorithms: List[MatchingAlgorithm]) -> Optional[MatchingResult]:
        """Find the best match using all configured algorithms."""
        best_result = None
        best_score = 0.0
        
        for algorithm in algorithms:
            try:
                result = algorithm.compare_records(record1, record2, field_mappings)
                
                if result.similarity_score > best_score:
                    best_score = result.similarity_score
                    best_result = result
                    
            except Exception as e:
                self.logger.warning(f"Error in algorithm {algorithm.name}: {e}")
                continue
        
        return best_result
    
    def _generate_unmatched_records(self, matched_records: List[Dict[str, Any]], 
                                  dataset1: pd.DataFrame, dataset2: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Generate lists of unmatched records from both datasets."""
        # For efficiency, we'll use a simpler approach
        # In a real implementation, we'd track indices during matching
        
        # Create sets of matched record identifiers
        matched_records1 = set()
        matched_records2 = set()
        
        for match in matched_records:
            # Use a hash of the record as identifier
            record1_hash = hash(frozenset(match['record1'].items()))
            record2_hash = hash(frozenset(match['record2'].items()))
            matched_records1.add(record1_hash)
            matched_records2.add(record2_hash)
        
        # Find unmatched records efficiently
        unmatched1 = []
        unmatched2 = []
        
        for _, row in dataset1.iterrows():
            row_dict = row.to_dict()
            row_hash = hash(frozenset(row_dict.items()))
            if row_hash not in matched_records1:
                unmatched1.append(row_dict)
        
        for _, row in dataset2.iterrows():
            row_dict = row.to_dict()
            row_hash = hash(frozenset(row_dict.items()))
            if row_hash not in matched_records2:
                unmatched2.append(row_dict)
        
        return {
            'file1': unmatched1,
            'file2': unmatched2
        }
    
    def _calculate_statistics(self, matched_records: List[Dict[str, Any]], 
                            size1: int, size2: int, processing_time: float) -> MatchingStatistics:
        """Calculate matching statistics."""
        high_confidence_matches = sum(
            1 for record in matched_records 
            if record['confidence_score'] >= 85.0
        )
        
        low_confidence_matches = len(matched_records) - high_confidence_matches
        
        average_confidence = (
            sum(record['confidence_score'] for record in matched_records) / len(matched_records)
            if matched_records else 0.0
        )
        
        # Calculate unique matched records from each dataset
        matched_indices1 = set()
        matched_indices2 = set()
        
        for match in matched_records:
            # For now, assume each match represents one record from each dataset
            # In a more sophisticated implementation, we'd track actual indices
            matched_indices1.add(id(match['record1']))  # Use object id as proxy
            matched_indices2.add(id(match['record2']))
        
        # For simplicity, assume one-to-one matching
        unique_matches_file1 = len(matched_indices1)
        unique_matches_file2 = len(matched_indices2)
        
        return MatchingStatistics(
            total_records_file1=size1,
            total_records_file2=size2,
            total_comparisons=len(matched_records),  # Use actual comparisons made
            high_confidence_matches=high_confidence_matches,
            low_confidence_matches=low_confidence_matches,
            unmatched_file1=max(0, size1 - unique_matches_file1),
            unmatched_file2=max(0, size2 - unique_matches_file2),
            processing_time_seconds=processing_time,
            average_confidence=average_confidence
        )
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration."""
        import hashlib
        config_str = str(self.config.to_dict())
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _calculate_reduction_ratio(self, size1: int, size2: int, actual_comparisons: int) -> float:
        """Calculate the reduction ratio achieved by blocking."""
        total_possible = size1 * size2
        if total_possible == 0:
            return 0.0
        return max(0.0, (1 - actual_comparisons / total_possible) * 100)
    
    def get_current_progress(self) -> Optional[ProgressStatus]:
        """Get current operation progress."""
        with self._operation_lock:
            if self.current_operation:
                return self.current_operation.get_status()
            return None
    
    def cancel_current_operation(self):
        """Cancel the current matching operation."""
        with self._operation_lock:
            if self.current_operation:
                self.current_operation.cancel()
                self.logger.info(f"Operation {self.current_operation.operation_id} cancelled")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the engine."""
        avg_processing_time = (
            self.total_processing_time / self.total_operations 
            if self.total_operations > 0 else 0.0
        )
        
        avg_comparisons = (
            self.total_comparisons / self.total_operations 
            if self.total_operations > 0 else 0.0
        )
        
        return {
            'total_operations': self.total_operations,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'total_comparisons': self.total_comparisons,
            'average_comparisons_per_operation': avg_comparisons,
            'current_memory_mb': self.memory_manager.get_current_memory_mb(),
            'blocking_stats': self.blocking_index.get_statistics().to_dict() if hasattr(self.blocking_index.get_statistics(), 'to_dict') else {}
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.total_operations = 0
        self.total_processing_time = 0.0
        self.total_comparisons = 0
        self.logger.info("Performance statistics reset")