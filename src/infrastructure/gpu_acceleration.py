"""
GPU acceleration for large-scale fuzzy matching operations.
Implements requirement 3.4, 3.1: GPU-accelerated fuzzy matching.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from enum import Enum

# GPU libraries (optional dependencies)
try:
    import cupy as cp
    import cupyx.scipy.spatial.distance as gpu_distance
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import numba
    from numba import cuda, jit
    NUMBA_CUDA_AVAILABLE = cuda.is_available() if hasattr(cuda, 'is_available') else False
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    numba = None
    cuda = None

from src.infrastructure.logging import get_logger
from src.domain.exceptions import ProcessingError


class GPUBackend(Enum):
    """Available GPU backends."""
    CUPY = "cupy"
    NUMBA_CUDA = "numba_cuda"
    CPU_FALLBACK = "cpu_fallback"


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration."""
    backend: GPUBackend = GPUBackend.CUPY
    device_id: int = 0
    memory_pool_size_mb: int = 1024
    batch_size: int = 10000
    enable_memory_pool: bool = True
    fallback_to_cpu: bool = True


class GPUMemoryManager:
    """Manage GPU memory allocation and cleanup."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = get_logger('gpu_memory_manager')
        self._memory_pool = None
        self._allocated_arrays = []
        
        if config.backend == GPUBackend.CUPY and CUPY_AVAILABLE:
            self._init_cupy_memory()
    
    def _init_cupy_memory(self):
        """Initialize CuPy memory pool."""
        try:
            cp.cuda.Device(self.config.device_id).use()
            
            if self.config.enable_memory_pool:
                # Set memory pool size
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.config.memory_pool_size_mb * 1024 * 1024)
                self._memory_pool = mempool
                
                self.logger.info(f"CuPy memory pool initialized: {self.config.memory_pool_size_mb}MB")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize CuPy memory: {str(e)}")
            raise
    
    def allocate_array(self, shape: Tuple[int, ...], dtype=np.float32) -> Any:
        """Allocate GPU array."""
        if self.config.backend == GPUBackend.CUPY and CUPY_AVAILABLE:
            array = cp.zeros(shape, dtype=dtype)
            self._allocated_arrays.append(array)
            return array
        else:
            # CPU fallback
            return np.zeros(shape, dtype=dtype)
    
    def free_array(self, array: Any):
        """Free GPU array."""
        if array in self._allocated_arrays:
            self._allocated_arrays.remove(array)
        
        if self.config.backend == GPUBackend.CUPY and CUPY_AVAILABLE:
            if hasattr(array, 'data') and hasattr(array.data, 'mem'):
                array.data.mem.free()
    
    def cleanup(self):
        """Clean up all allocated GPU memory."""
        for array in self._allocated_arrays:
            try:
                self.free_array(array)
            except Exception as e:
                self.logger.warning(f"Failed to free GPU array: {str(e)}")
        
        self._allocated_arrays.clear()
        
        if self._memory_pool:
            self._memory_pool.free_all_blocks()
            self.logger.info("GPU memory pool cleaned up")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self.config.backend == GPUBackend.CUPY and CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                return {
                    'backend': 'cupy',
                    'device_id': self.config.device_id,
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes(),
                    'allocated_arrays': len(self._allocated_arrays)
                }
            except Exception as e:
                return {'error': str(e)}
        else:
            return {'backend': 'cpu_fallback'}


class GPUFuzzyMatcher:
    """GPU-accelerated fuzzy string matching."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.logger = get_logger('gpu_fuzzy_matcher')
        self.memory_manager = GPUMemoryManager(config)
        
        # Check GPU availability
        self.gpu_available = self._check_gpu_availability()
        
        if not self.gpu_available and not config.fallback_to_cpu:
            raise ProcessingError("GPU not available and CPU fallback disabled")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        if self.config.backend == GPUBackend.CUPY:
            return CUPY_AVAILABLE
        elif self.config.backend == GPUBackend.NUMBA_CUDA:
            return NUMBA_CUDA_AVAILABLE
        else:
            return False
    
    def _prepare_strings_for_gpu(self, strings: List[str], max_length: int = 100) -> np.ndarray:
        """
        Prepare strings for GPU processing by converting to fixed-length arrays.
        
        Args:
            strings: List of strings to process
            max_length: Maximum string length
            
        Returns:
            NumPy array of character codes
        """
        # Convert strings to fixed-length character arrays
        char_arrays = np.zeros((len(strings), max_length), dtype=np.int32)
        
        for i, string in enumerate(strings):
            # Truncate or pad string to max_length
            string = string[:max_length]
            char_codes = [ord(c) for c in string]
            char_arrays[i, :len(char_codes)] = char_codes
        
        return char_arrays
    
    def _levenshtein_distance_gpu_cupy(self, strings1: np.ndarray, strings2: np.ndarray) -> np.ndarray:
        """
        Compute Levenshtein distance using CuPy.
        
        Args:
            strings1: First set of strings as character arrays
            strings2: Second set of strings as character arrays
            
        Returns:
            Distance matrix
        """
        if not CUPY_AVAILABLE:
            raise ProcessingError("CuPy not available")
        
        # Transfer data to GPU
        gpu_strings1 = cp.asarray(strings1)
        gpu_strings2 = cp.asarray(strings2)
        
        n1, max_len1 = gpu_strings1.shape
        n2, max_len2 = gpu_strings2.shape
        
        # Allocate result matrix
        distances = cp.zeros((n1, n2), dtype=cp.int32)
        
        # Custom CUDA kernel for Levenshtein distance
        levenshtein_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void levenshtein_distance(const int* strings1, const int* strings2,
                                int* distances, int n1, int n2, int max_len1, int max_len2) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i >= n1 || j >= n2) return;
            
            // Dynamic programming matrix (shared memory would be better for larger strings)
            int dp[101][101];  // Assuming max_length <= 100
            
            const int* s1 = &strings1[i * max_len1];
            const int* s2 = &strings2[j * max_len2];
            
            // Find actual string lengths
            int len1 = 0, len2 = 0;
            for (int k = 0; k < max_len1 && s1[k] != 0; k++) len1++;
            for (int k = 0; k < max_len2 && s2[k] != 0; k++) len2++;
            
            // Initialize DP matrix
            for (int k = 0; k <= len1; k++) dp[k][0] = k;
            for (int k = 0; k <= len2; k++) dp[0][k] = k;
            
            // Fill DP matrix
            for (int k1 = 1; k1 <= len1; k1++) {
                for (int k2 = 1; k2 <= len2; k2++) {
                    int cost = (s1[k1-1] == s2[k2-1]) ? 0 : 1;
                    dp[k1][k2] = min(min(dp[k1-1][k2] + 1, dp[k1][k2-1] + 1), dp[k1-1][k2-1] + cost);
                }
            }
            
            distances[i * n2 + j] = dp[len1][len2];
        }
        ''', 'levenshtein_distance')
        
        # Launch kernel
        block_size = (16, 16)
        grid_size = ((n1 + block_size[0] - 1) // block_size[0],
                    (n2 + block_size[1] - 1) // block_size[1])
        
        levenshtein_kernel(
            grid_size, block_size,
            (gpu_strings1, gpu_strings2, distances, n1, n2, max_len1, max_len2)
        )
        
        # Transfer result back to CPU
        return cp.asnumpy(distances)
    
    def _levenshtein_distance_cpu_fallback(self, strings1: List[str], strings2: List[str]) -> np.ndarray:
        """
        CPU fallback for Levenshtein distance computation.
        
        Args:
            strings1: First set of strings
            strings2: Second set of strings
            
        Returns:
            Distance matrix
        """
        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distances = np.zeros((len(strings1), len(strings2)), dtype=np.int32)
        
        for i, s1 in enumerate(strings1):
            for j, s2 in enumerate(strings2):
                distances[i, j] = levenshtein(s1, s2)
        
        return distances
    
    def compute_similarity_matrix(self, strings1: List[str], strings2: List[str],
                                algorithm: str = 'levenshtein') -> np.ndarray:
        """
        Compute similarity matrix between two sets of strings.
        
        Args:
            strings1: First set of strings
            strings2: Second set of strings
            algorithm: Similarity algorithm ('levenshtein', 'jaccard')
            
        Returns:
            Similarity matrix (higher values = more similar)
        """
        start_time = time.time()
        
        try:
            if algorithm == 'levenshtein':
                if self.gpu_available and self.config.backend == GPUBackend.CUPY:
                    # GPU acceleration with CuPy
                    max_length = max(
                        max(len(s) for s in strings1) if strings1 else 0,
                        max(len(s) for s in strings2) if strings2 else 0,
                        1
                    )
                    max_length = min(max_length, 100)  # Limit for GPU kernel
                    
                    char_arrays1 = self._prepare_strings_for_gpu(strings1, max_length)
                    char_arrays2 = self._prepare_strings_for_gpu(strings2, max_length)
                    
                    distances = self._levenshtein_distance_gpu_cupy(char_arrays1, char_arrays2)
                    
                    # Convert distances to similarities (0-1 scale)
                    max_distances = np.maximum(
                        np.array([len(s) for s in strings1])[:, np.newaxis],
                        np.array([len(s) for s in strings2])[np.newaxis, :]
                    )
                    similarities = 1.0 - (distances / np.maximum(max_distances, 1))
                    
                    self.logger.info(f"GPU Levenshtein computation completed in {time.time() - start_time:.3f}s")
                    
                else:
                    # CPU fallback
                    distances = self._levenshtein_distance_cpu_fallback(strings1, strings2)
                    
                    # Convert distances to similarities
                    max_distances = np.maximum(
                        np.array([len(s) for s in strings1])[:, np.newaxis],
                        np.array([len(s) for s in strings2])[np.newaxis, :]
                    )
                    similarities = 1.0 - (distances / np.maximum(max_distances, 1))
                    
                    self.logger.info(f"CPU Levenshtein computation completed in {time.time() - start_time:.3f}s")
            
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            return similarities.astype(np.float32)
        
        except Exception as e:
            if self.config.fallback_to_cpu and self.gpu_available:
                self.logger.warning(f"GPU computation failed, falling back to CPU: {str(e)}")
                return self.compute_similarity_matrix(strings1, strings2, algorithm)
            else:
                raise ProcessingError(f"Similarity computation failed: {str(e)}")
    
    def batch_similarity_computation(self, strings1: List[str], strings2: List[str],
                                   algorithm: str = 'levenshtein') -> np.ndarray:
        """
        Compute similarities in batches to manage memory usage.
        
        Args:
            strings1: First set of strings
            strings2: Second set of strings
            algorithm: Similarity algorithm
            
        Returns:
            Similarity matrix
        """
        batch_size = self.config.batch_size
        n1, n2 = len(strings1), len(strings2)
        
        # Allocate result matrix
        similarities = np.zeros((n1, n2), dtype=np.float32)
        
        # Process in batches
        for i in range(0, n1, batch_size):
            for j in range(0, n2, batch_size):
                i_end = min(i + batch_size, n1)
                j_end = min(j + batch_size, n2)
                
                batch_strings1 = strings1[i:i_end]
                batch_strings2 = strings2[j:j_end]
                
                batch_similarities = self.compute_similarity_matrix(
                    batch_strings1, batch_strings2, algorithm
                )
                
                similarities[i:i_end, j:j_end] = batch_similarities
                
                self.logger.debug(f"Processed batch ({i}:{i_end}, {j}:{j_end})")
        
        return similarities
    
    def find_best_matches(self, query_strings: List[str], candidate_strings: List[str],
                         threshold: float = 0.8, top_k: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Find best matches for query strings against candidates.
        
        Args:
            query_strings: Strings to find matches for
            candidate_strings: Candidate strings to match against
            threshold: Minimum similarity threshold
            top_k: Maximum number of matches per query
            
        Returns:
            List of matches for each query string (index, similarity)
        """
        similarities = self.batch_similarity_computation(query_strings, candidate_strings)
        
        matches = []
        for i, query in enumerate(query_strings):
            # Get similarities for this query
            query_similarities = similarities[i]
            
            # Find indices above threshold
            above_threshold = np.where(query_similarities >= threshold)[0]
            
            if len(above_threshold) > 0:
                # Sort by similarity (descending)
                sorted_indices = above_threshold[np.argsort(query_similarities[above_threshold])[::-1]]
                
                # Take top_k matches
                top_matches = sorted_indices[:top_k]
                query_matches = [(int(idx), float(query_similarities[idx])) for idx in top_matches]
            else:
                query_matches = []
            
            matches.append(query_matches)
        
        return matches
    
    def cleanup(self):
        """Clean up GPU resources."""
        self.memory_manager.cleanup()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GPU acceleration statistics."""
        return {
            'gpu_available': self.gpu_available,
            'backend': self.config.backend.value,
            'memory_info': self.memory_manager.get_memory_info(),
            'batch_size': self.config.batch_size
        }


class GPUAcceleratedMatchingService:
    """High-level service for GPU-accelerated matching operations."""
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self.logger = get_logger('gpu_matching_service')
        
        # Initialize GPU matcher
        try:
            self.gpu_matcher = GPUFuzzyMatcher(self.config)
            self.logger.info("GPU matching service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU matching service: {str(e)}")
            if not self.config.fallback_to_cpu:
                raise
            self.gpu_matcher = None
    
    def match_datasets(self, dataset1_strings: List[str], dataset2_strings: List[str],
                      algorithm: str = 'levenshtein', threshold: float = 0.8,
                      top_k: int = 5) -> Dict[str, Any]:
        """
        Match strings between two datasets using GPU acceleration.
        
        Args:
            dataset1_strings: Strings from first dataset
            dataset2_strings: Strings from second dataset
            algorithm: Matching algorithm
            threshold: Similarity threshold
            top_k: Maximum matches per string
            
        Returns:
            Matching results with statistics
        """
        if not self.gpu_matcher:
            raise ProcessingError("GPU matcher not available")
        
        start_time = time.time()
        
        try:
            # Find matches
            matches = self.gpu_matcher.find_best_matches(
                dataset1_strings, dataset2_strings, threshold, top_k
            )
            
            # Calculate statistics
            total_matches = sum(len(match_list) for match_list in matches)
            matched_queries = sum(1 for match_list in matches if match_list)
            
            processing_time = time.time() - start_time
            
            results = {
                'matches': matches,
                'statistics': {
                    'total_queries': len(dataset1_strings),
                    'total_candidates': len(dataset2_strings),
                    'matched_queries': matched_queries,
                    'total_matches': total_matches,
                    'match_rate': matched_queries / len(dataset1_strings) if dataset1_strings else 0,
                    'avg_matches_per_query': total_matches / len(dataset1_strings) if dataset1_strings else 0,
                    'processing_time_seconds': processing_time,
                    'throughput_queries_per_second': len(dataset1_strings) / processing_time if processing_time > 0 else 0
                },
                'gpu_stats': self.gpu_matcher.get_stats()
            }
            
            self.logger.info(f"GPU matching completed: {matched_queries}/{len(dataset1_strings)} queries matched "
                           f"in {processing_time:.3f}s")
            
            return results
        
        except Exception as e:
            self.logger.error(f"GPU matching failed: {str(e)}")
            raise ProcessingError(f"GPU matching failed: {str(e)}")
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.gpu_matcher:
            self.gpu_matcher.cleanup()


# Global GPU service instance
_gpu_matching_service: Optional[GPUAcceleratedMatchingService] = None
_gpu_service_lock = threading.Lock()


def get_gpu_matching_service(config: Optional[GPUConfig] = None) -> Optional[GPUAcceleratedMatchingService]:
    """Get global GPU matching service instance."""
    global _gpu_matching_service
    
    if _gpu_matching_service is None:
        with _gpu_service_lock:
            if _gpu_matching_service is None:
                try:
                    _gpu_matching_service = GPUAcceleratedMatchingService(config)
                except Exception as e:
                    logger = get_logger('gpu_service')
                    logger.warning(f"GPU matching service not available: {str(e)}")
                    return None
    
    return _gpu_matching_service


def cleanup_gpu_resources():
    """Clean up all GPU resources."""
    global _gpu_matching_service
    if _gpu_matching_service is not None:
        _gpu_matching_service.cleanup()