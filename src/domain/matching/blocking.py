"""
Optimized blocking and indexing strategies for efficient record matching.
Implements requirements 3.1, 3.5, 4.4: Multi-level blocking, adaptive strategies, and parallel processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import threading
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import re

from .uzbek_normalizer import UzbekTextNormalizer
from ...infrastructure.logging import get_logger


class BlockingStrategy(Enum):
    """Types of blocking strategies."""
    EXACT_PREFIX = "exact_prefix"
    PHONETIC_KEY = "phonetic_key"
    NGRAM = "ngram"
    SORTED_NEIGHBORHOOD = "sorted_neighborhood"
    CANOPY = "canopy"
    ADAPTIVE = "adaptive"


@dataclass
class BlockingKey:
    """Represents a blocking key with metadata."""
    key: str
    strategy: BlockingStrategy
    field_name: str
    record_ids: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_record(self, record_id: int):
        """Add a record ID to this blocking key."""
        self.record_ids.add(record_id)
    
    def __hash__(self):
        return hash((self.key, self.strategy.value, self.field_name))
    
    def __eq__(self, other):
        if not isinstance(other, BlockingKey):
            return False
        return (self.key == other.key and 
                self.strategy == other.strategy and 
                self.field_name == other.field_name)


@dataclass
class BlockingStatistics:
    """Statistics for blocking operations."""
    total_records: int = 0
    total_blocks: int = 0
    average_block_size: float = 0.0
    max_block_size: int = 0
    min_block_size: int = 0
    empty_blocks: int = 0
    reduction_ratio: float = 0.0  # Percentage of comparisons avoided
    processing_time_seconds: float = 0.0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    
    def calculate_metrics(self, block_sizes: List[int], total_possible_comparisons: int):
        """Calculate derived metrics from block sizes."""
        if block_sizes:
            self.total_blocks = len(block_sizes)
            self.average_block_size = sum(block_sizes) / len(block_sizes)
            self.max_block_size = max(block_sizes)
            self.min_block_size = min(block_sizes)
            self.empty_blocks = sum(1 for size in block_sizes if size == 0)
            
            # Calculate actual comparisons needed within blocks
            actual_comparisons = sum(size * (size - 1) // 2 for size in block_sizes if size > 1)
            if total_possible_comparisons > 0:
                self.reduction_ratio = max(0.0, (1 - actual_comparisons / total_possible_comparisons) * 100)
            else:
                self.reduction_ratio = 0.0


class BlockingKeyGenerator(ABC):
    """Abstract base class for blocking key generators."""
    
    def __init__(self, strategy: BlockingStrategy, config: Optional[Dict[str, Any]] = None):
        self.strategy = strategy
        self.config = config or {}
        self.logger = get_logger(f'blocking.{strategy.value}')
    
    @abstractmethod
    def generate_keys(self, text: str, field_name: str) -> List[BlockingKey]:
        """Generate blocking keys for a text value."""
        pass
    
    @abstractmethod
    def estimate_selectivity(self, values: List[str]) -> float:
        """Estimate the selectivity of this blocking strategy for given values."""
        pass


class ExactPrefixGenerator(BlockingKeyGenerator):
    """Generates blocking keys based on exact prefixes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(BlockingStrategy.EXACT_PREFIX, config)
        self.prefix_lengths = self.config.get('prefix_lengths', [2, 3, 4])
        self.case_sensitive = self.config.get('case_sensitive', False)
    
    def generate_keys(self, text: str, field_name: str) -> List[BlockingKey]:
        """Generate exact prefix blocking keys."""
        if not text:
            return []
        
        normalized_text = text if self.case_sensitive else text.lower()
        normalized_text = normalized_text.strip()
        
        keys = []
        for length in self.prefix_lengths:
            if len(normalized_text) >= length:
                prefix = normalized_text[:length]
                key = BlockingKey(
                    key=prefix,
                    strategy=self.strategy,
                    field_name=field_name,
                    metadata={'prefix_length': length, 'original_text': text}
                )
                keys.append(key)
        
        return keys
    
    def estimate_selectivity(self, values: List[str]) -> float:
        """Estimate selectivity based on prefix distribution."""
        if not values:
            return 0.0
        
        prefix_counts = defaultdict(int)
        for value in values:
            if value:
                normalized = value if self.case_sensitive else value.lower()
                for length in self.prefix_lengths:
                    if len(normalized) >= length:
                        prefix_counts[normalized[:length]] += 1
        
        if not prefix_counts:
            return 0.0
        
        # Calculate average block size
        total_prefixes = len(prefix_counts)
        total_records = len(values)
        average_block_size = total_records / total_prefixes if total_prefixes > 0 else total_records
        
        # Selectivity is inverse of average block size (normalized)
        return min(1.0, 1.0 / (average_block_size / 10))  # Normalize to reasonable range


class PhoneticKeyGenerator(BlockingKeyGenerator):
    """Generates blocking keys based on phonetic similarity."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(BlockingStrategy.PHONETIC_KEY, config)
        self.uzbek_normalizer = UzbekTextNormalizer()
        self.use_soundex = self.config.get('use_soundex', True)
        self.use_uzbek_phonetic = self.config.get('use_uzbek_phonetic', True)
    
    def generate_keys(self, text: str, field_name: str) -> List[BlockingKey]:
        """Generate phonetic blocking keys."""
        if not text:
            return []
        
        keys = []
        
        # Generate Uzbek phonetic key if text appears to be Uzbek
        if self.use_uzbek_phonetic and self.uzbek_normalizer.is_uzbek_text(text):
            phonetic_key = self.uzbek_normalizer.generate_phonetic_key(text)
            if phonetic_key:
                key = BlockingKey(
                    key=phonetic_key,
                    strategy=self.strategy,
                    field_name=field_name,
                    metadata={'method': 'uzbek_phonetic', 'original_text': text}
                )
                keys.append(key)
        
        # Generate Soundex key for general phonetic matching
        if self.use_soundex:
            soundex_key = self._generate_soundex(text)
            if soundex_key:
                key = BlockingKey(
                    key=soundex_key,
                    strategy=self.strategy,
                    field_name=field_name,
                    metadata={'method': 'soundex', 'original_text': text}
                )
                keys.append(key)
        
        return keys
    
    def _generate_soundex(self, text: str) -> str:
        """Generate Soundex code for text."""
        if not text:
            return ""
        
        # Simplified Soundex implementation
        text = text.upper().strip()
        if not text:
            return ""
        
        # Soundex mapping
        soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        soundex = text[0]  # Keep first letter
        
        # Convert remaining letters
        for char in text[1:]:
            if char in soundex_map:
                code = soundex_map[char]
                if not soundex or soundex[-1] != code:
                    soundex += code
        
        # Remove vowels and specific consonants (except first letter)
        soundex = soundex[0] + ''.join(c for c in soundex[1:] if c.isdigit())
        
        # Pad or truncate to 4 characters
        soundex = (soundex + "000")[:4]
        return soundex
    
    def estimate_selectivity(self, values: List[str]) -> float:
        """Estimate selectivity based on phonetic key distribution."""
        if not values:
            return 0.0
        
        phonetic_counts = defaultdict(int)
        for value in values:
            keys = self.generate_keys(value, "temp")
            for key in keys:
                phonetic_counts[key.key] += 1
        
        if not phonetic_counts:
            return 0.0
        
        total_keys = len(phonetic_counts)
        total_records = len(values)
        average_block_size = total_records / total_keys if total_keys > 0 else total_records
        
        return min(1.0, 1.0 / (average_block_size / 15))  # Phonetic keys tend to be less selective


class NGramGenerator(BlockingKeyGenerator):
    """Generates blocking keys based on n-grams."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(BlockingStrategy.NGRAM, config)
        self.n = self.config.get('n', 3)
        self.skip_grams = self.config.get('skip_grams', False)
        self.case_sensitive = self.config.get('case_sensitive', False)
    
    def generate_keys(self, text: str, field_name: str) -> List[BlockingKey]:
        """Generate n-gram blocking keys."""
        if not text:
            return []
        
        normalized_text = text if self.case_sensitive else text.lower()
        normalized_text = re.sub(r'[^\w\s]', '', normalized_text)  # Remove punctuation
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        if len(normalized_text) < self.n:
            # For short text, use the whole text as key
            return [BlockingKey(
                key=normalized_text,
                strategy=self.strategy,
                field_name=field_name,
                metadata={'n': self.n, 'original_text': text}
            )]
        
        keys = []
        ngrams = set()
        
        # Generate regular n-grams
        for i in range(len(normalized_text) - self.n + 1):
            ngram = normalized_text[i:i + self.n]
            if ngram not in ngrams:
                ngrams.add(ngram)
                key = BlockingKey(
                    key=ngram,
                    strategy=self.strategy,
                    field_name=field_name,
                    metadata={'n': self.n, 'type': 'regular', 'original_text': text}
                )
                keys.append(key)
        
        # Generate skip-grams if enabled
        if self.skip_grams and len(normalized_text) >= self.n + 1:
            for i in range(len(normalized_text) - self.n):
                for skip in range(1, min(3, len(normalized_text) - i - self.n + 1)):
                    skip_gram = normalized_text[i:i + self.n - 1] + normalized_text[i + self.n - 1 + skip]
                    if skip_gram not in ngrams:
                        ngrams.add(skip_gram)
                        key = BlockingKey(
                            key=skip_gram,
                            strategy=self.strategy,
                            field_name=field_name,
                            metadata={'n': self.n, 'type': 'skip', 'skip': skip, 'original_text': text}
                        )
                        keys.append(key)
        
        return keys
    
    def estimate_selectivity(self, values: List[str]) -> float:
        """Estimate selectivity based on n-gram distribution."""
        if not values:
            return 0.0
        
        ngram_counts = defaultdict(int)
        for value in values:
            keys = self.generate_keys(value, "temp")
            for key in keys:
                ngram_counts[key.key] += 1
        
        if not ngram_counts:
            return 0.0
        
        total_ngrams = len(ngram_counts)
        total_records = len(values)
        average_block_size = total_records / total_ngrams if total_ngrams > 0 else total_records
        
        return min(1.0, 1.0 / (average_block_size / 8))  # N-grams are usually quite selective


class OptimizedBlockingIndex:
    """
    Multi-level blocking index with adaptive strategies and parallel processing.
    Implements requirements 3.1, 3.5, 4.4.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the optimized blocking index.
        
        Args:
            config: Configuration parameters including:
                - strategies: List of blocking strategies to use
                - adaptive_threshold: Threshold for switching strategies
                - max_block_size: Maximum allowed block size
                - parallel_processing: Whether to enable parallel processing
                - max_workers: Maximum number of worker threads
        """
        self.config = config or {}
        self.logger = get_logger('blocking.index')
        
        # Configuration parameters
        self.adaptive_threshold = self.config.get('adaptive_threshold', 1000)
        self.max_block_size = self.config.get('max_block_size', 500)
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Initialize blocking key generators
        self.generators = self._initialize_generators()
        
        # Thread-safe data structures
        self._lock = threading.RLock()
        self._blocks = defaultdict(lambda: defaultdict(set))  # strategy -> key -> record_ids
        self._record_metadata = {}  # record_id -> metadata
        self._statistics = BlockingStatistics()
        
        self.logger.info("OptimizedBlockingIndex initialized", extra={
            'strategies': [s.value for s in self.generators.keys()],
            'adaptive_threshold': self.adaptive_threshold,
            'max_block_size': self.max_block_size,
            'parallel_processing': self.parallel_processing
        })
    
    def _initialize_generators(self) -> Dict[BlockingStrategy, BlockingKeyGenerator]:
        """Initialize blocking key generators based on configuration."""
        generators = {}
        
        # Default strategies if none specified
        strategies = self.config.get('strategies', [
            BlockingStrategy.EXACT_PREFIX,
            BlockingStrategy.PHONETIC_KEY,
            BlockingStrategy.NGRAM
        ])
        
        for strategy in strategies:
            strategy_config = self.config.get(f'{strategy.value}_config', {})
            
            if strategy == BlockingStrategy.EXACT_PREFIX:
                generators[strategy] = ExactPrefixGenerator(strategy_config)
            elif strategy == BlockingStrategy.PHONETIC_KEY:
                generators[strategy] = PhoneticKeyGenerator(strategy_config)
            elif strategy == BlockingStrategy.NGRAM:
                generators[strategy] = NGramGenerator(strategy_config)
            # Add more generators as needed
        
        return generators
    
    def build_index(self, dataset: pd.DataFrame, field_mappings: List[Tuple[str, float]]) -> BlockingStatistics:
        """
        Build blocking index for a dataset.
        
        Args:
            dataset: DataFrame containing the records
            field_mappings: List of (field_name, weight) tuples
            
        Returns:
            BlockingStatistics with index building metrics
        """
        start_time = time.time()
        
        with self._lock:
            # Clear existing index
            self._blocks.clear()
            self._record_metadata.clear()
            
            # Analyze dataset characteristics for adaptive strategy selection
            dataset_characteristics = self._analyze_dataset(dataset, field_mappings)
            
            # Select optimal strategies based on dataset characteristics
            selected_strategies = self._select_strategies(dataset_characteristics)
            
            if self.parallel_processing and len(dataset) > 100:
                self._build_index_parallel(dataset, field_mappings, selected_strategies)
            else:
                self._build_index_sequential(dataset, field_mappings, selected_strategies)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            self._calculate_statistics(len(dataset), processing_time)
            
            self.logger.info("Blocking index built", extra={
                'total_records': len(dataset),
                'total_blocks': self._statistics.total_blocks,
                'processing_time': processing_time,
                'reduction_ratio': self._statistics.reduction_ratio
            })
            
            return self._statistics
    
    def _analyze_dataset(self, dataset: pd.DataFrame, field_mappings: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Analyze dataset characteristics to inform strategy selection."""
        characteristics = {
            'total_records': len(dataset),
            'field_characteristics': {}
        }
        
        for field_name, weight in field_mappings:
            if field_name not in dataset.columns:
                continue
            
            field_values = dataset[field_name].dropna().astype(str)
            if len(field_values) == 0:
                continue
            
            # Calculate field characteristics
            unique_values = field_values.nunique()
            avg_length = field_values.str.len().mean()
            
            # Check for Uzbek text if phonetic generator is available
            has_uzbek_text = False
            if BlockingStrategy.PHONETIC_KEY in self.generators:
                has_uzbek_text = any(
                    self.generators[BlockingStrategy.PHONETIC_KEY].uzbek_normalizer.is_uzbek_text(val)
                    for val in field_values.head(100)
                    if isinstance(val, str)
                )
            
            characteristics['field_characteristics'][field_name] = {
                'unique_values': unique_values,
                'total_values': len(field_values),
                'uniqueness_ratio': unique_values / len(field_values),
                'average_length': avg_length,
                'has_uzbek_text': has_uzbek_text,
                'weight': weight
            }
        
        return characteristics
    
    def _select_strategies(self, characteristics: Dict[str, Any]) -> List[BlockingStrategy]:
        """Select optimal blocking strategies based on dataset characteristics."""
        selected_strategies = []
        total_records = characteristics['total_records']
        
        # Always include exact prefix for high-selectivity blocking
        selected_strategies.append(BlockingStrategy.EXACT_PREFIX)
        
        # Add phonetic key if Uzbek text is detected
        for field_char in characteristics['field_characteristics'].values():
            if field_char.get('has_uzbek_text', False):
                selected_strategies.append(BlockingStrategy.PHONETIC_KEY)
                break
        
        # Add n-gram for fuzzy matching on larger datasets
        if total_records > 1000:
            selected_strategies.append(BlockingStrategy.NGRAM)
        
        # Limit strategies for very large datasets to avoid overhead
        if total_records > 10000:
            selected_strategies = selected_strategies[:2]
        
        self.logger.info("Selected blocking strategies", extra={
            'strategies': [s.value for s in selected_strategies],
            'total_records': total_records
        })
        
        return selected_strategies
    
    def _build_index_sequential(self, dataset: pd.DataFrame, field_mappings: List[Tuple[str, float]], 
                               strategies: List[BlockingStrategy]):
        """Build index sequentially."""
        for idx, row in dataset.iterrows():
            record_id = int(idx)
            self._record_metadata[record_id] = dict(row)
            
            for field_name, weight in field_mappings:
                if field_name not in row or pd.isna(row[field_name]):
                    continue
                
                field_value = str(row[field_name])
                
                for strategy in strategies:
                    if strategy in self.generators:
                        keys = self.generators[strategy].generate_keys(field_value, field_name)
                        for key in keys:
                            self._blocks[strategy][key.key].add(record_id)
    
    def _build_index_parallel(self, dataset: pd.DataFrame, field_mappings: List[Tuple[str, float]], 
                             strategies: List[BlockingStrategy]):
        """Build index using parallel processing."""
        # Split dataset into chunks for parallel processing
        chunk_size = max(100, len(dataset) // self.max_workers)
        chunks = [dataset.iloc[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
        
        # Thread-safe collections for results
        thread_blocks = defaultdict(lambda: defaultdict(set))
        thread_metadata = {}
        thread_lock = threading.Lock()
        
        def process_chunk(chunk):
            """Process a chunk of records."""
            local_blocks = defaultdict(lambda: defaultdict(set))
            local_metadata = {}
            
            for idx, row in chunk.iterrows():
                record_id = int(idx)
                local_metadata[record_id] = dict(row)
                
                for field_name, weight in field_mappings:
                    if field_name not in row or pd.isna(row[field_name]):
                        continue
                    
                    field_value = str(row[field_name])
                    
                    for strategy in strategies:
                        if strategy in self.generators:
                            keys = self.generators[strategy].generate_keys(field_value, field_name)
                            for key in keys:
                                local_blocks[strategy][key.key].add(record_id)
            
            # Merge results thread-safely
            with thread_lock:
                for strategy, strategy_blocks in local_blocks.items():
                    for key, record_ids in strategy_blocks.items():
                        thread_blocks[strategy][key].update(record_ids)
                thread_metadata.update(local_metadata)
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing chunk: {e}")
        
        # Update main data structures
        self._blocks.update(thread_blocks)
        self._record_metadata.update(thread_metadata)
    
    def get_candidate_pairs(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, 
                           field_mappings: List[Tuple[str, str, float]]) -> Iterator[Tuple[int, int]]:
        """
        Get candidate record pairs for matching using blocking.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            field_mappings: List of (field1, field2, weight) tuples
            
        Yields:
            Tuples of (record_id1, record_id2) for candidate pairs
        """
        # Build index for dataset1 if not already built
        if not self._blocks:
            field_mappings_1 = [(field1, weight) for field1, field2, weight in field_mappings]
            self.build_index(dataset1, field_mappings_1)
        
        # Generate candidate pairs
        candidate_pairs = set()
        
        for idx2, row2 in dataset2.iterrows():
            record_id2 = int(idx2)
            
            # Find matching blocks for this record
            matching_record_ids = set()
            
            for field1, field2, weight in field_mappings:
                if field2 not in row2 or pd.isna(row2[field2]):
                    continue
                
                field_value = str(row2[field2])
                
                # Check all strategies for matching blocks
                for strategy in self.generators:
                    keys = self.generators[strategy].generate_keys(field_value, field2)
                    for key in keys:
                        if key.key in self._blocks[strategy]:
                            matching_record_ids.update(self._blocks[strategy][key.key])
            
            # Add candidate pairs
            for record_id1 in matching_record_ids:
                pair = (record_id1, record_id2)
                if pair not in candidate_pairs:
                    candidate_pairs.add(pair)
                    yield pair
    
    def _calculate_statistics(self, total_records: int, processing_time: float):
        """Calculate blocking statistics."""
        block_sizes = []
        strategy_distribution = defaultdict(int)
        
        for strategy, strategy_blocks in self._blocks.items():
            strategy_distribution[strategy.value] = len(strategy_blocks)
            for key, record_ids in strategy_blocks.items():
                block_size = len(record_ids)
                block_sizes.append(block_size)
        
        # Calculate total possible comparisons
        total_possible_comparisons = total_records * (total_records - 1) // 2
        
        self._statistics = BlockingStatistics(
            total_records=total_records,
            processing_time_seconds=processing_time,
            strategy_distribution=dict(strategy_distribution)
        )
        
        if block_sizes:
            self._statistics.calculate_metrics(block_sizes, total_possible_comparisons)
        else:
            # No blocks created - this means no blocking keys were generated
            self._statistics.total_blocks = 0
            self._statistics.average_block_size = 0.0
            self._statistics.max_block_size = 0
            self._statistics.min_block_size = 0
            self._statistics.empty_blocks = 0
            self._statistics.reduction_ratio = 0.0
    
    def get_statistics(self) -> BlockingStatistics:
        """Get current blocking statistics."""
        with self._lock:
            return self._statistics
    
    def clear_index(self):
        """Clear the blocking index."""
        with self._lock:
            self._blocks.clear()
            self._record_metadata.clear()
            self._statistics = BlockingStatistics()
            self.logger.info("Blocking index cleared")
    
    def optimize_block_sizes(self):
        """Optimize block sizes by splitting large blocks."""
        with self._lock:
            optimized_blocks = defaultdict(lambda: defaultdict(set))
            
            for strategy, strategy_blocks in self._blocks.items():
                for key, record_ids in strategy_blocks.items():
                    if len(record_ids) > self.max_block_size:
                        # Split large block into smaller sub-blocks
                        record_list = list(record_ids)
                        for i in range(0, len(record_list), self.max_block_size):
                            sub_block = set(record_list[i:i + self.max_block_size])
                            sub_key = f"{key}_{i // self.max_block_size}"
                            optimized_blocks[strategy][sub_key] = sub_block
                    else:
                        optimized_blocks[strategy][key] = record_ids
            
            self._blocks = optimized_blocks
            self.logger.info("Block sizes optimized", extra={
                'max_block_size': self.max_block_size
            })
    
    def get_block_info(self, strategy: Optional[BlockingStrategy] = None) -> Dict[str, Any]:
        """Get information about blocks."""
        with self._lock:
            if strategy:
                if strategy in self._blocks:
                    strategy_blocks = self._blocks[strategy]
                    return {
                        'strategy': strategy.value,
                        'total_blocks': len(strategy_blocks),
                        'block_sizes': [len(record_ids) for record_ids in strategy_blocks.values()],
                        'total_records': sum(len(record_ids) for record_ids in strategy_blocks.values())
                    }
                else:
                    return {'strategy': strategy.value, 'total_blocks': 0}
            else:
                # Return info for all strategies
                info = {}
                for strat, strategy_blocks in self._blocks.items():
                    info[strat.value] = {
                        'total_blocks': len(strategy_blocks),
                        'block_sizes': [len(record_ids) for record_ids in strategy_blocks.values()],
                        'total_records': sum(len(record_ids) for record_ids in strategy_blocks.values())
                    }
                return info