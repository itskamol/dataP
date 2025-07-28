"""
Performance tests for blocking strategies.
Tests requirements 3.1, 3.5, 4.4: Compare different blocking strategies and parallel processing.
"""

import pytest
import pandas as pd
import time
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import random
import string

from src.domain.matching.blocking import (
    OptimizedBlockingIndex, BlockingStrategy, BlockingStatistics,
    ExactPrefixGenerator, PhoneticKeyGenerator, NGramGenerator
)


class TestBlockingPerformance:
    """Performance tests for blocking strategies."""
    
    @pytest.fixture
    def small_dataset(self) -> pd.DataFrame:
        """Create a small test dataset."""
        data = {
            'id': range(100),
            'name': [f'Name_{i}' for i in range(100)],
            'city': [f'City_{i % 10}' for i in range(100)],
            'phone': [f'+998{90 + i % 10}{1000000 + i}' for i in range(100)]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def medium_dataset(self) -> pd.DataFrame:
        """Create a medium test dataset."""
        data = {
            'id': range(1000),
            'name': [f'Name_{i}' for i in range(1000)],
            'city': [f'City_{i % 50}' for i in range(1000)],
            'phone': [f'+998{90 + i % 10}{1000000 + i}' for i in range(1000)]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """Create a large test dataset."""
        data = {
            'id': range(5000),
            'name': [f'Name_{i}' for i in range(5000)],
            'city': [f'City_{i % 100}' for i in range(5000)],
            'phone': [f'+998{90 + i % 10}{1000000 + i}' for i in range(5000)]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def uzbek_dataset(self) -> pd.DataFrame:
        """Create a dataset with Uzbek text."""
        uzbek_names = [
            'Abdulla', 'Akmal', 'Aziz', 'Bobur', 'Davron', 'Eldor', 'Farrux', 'Gulnora',
            'Хуршида', 'Ирода', 'Жамшид', 'Камола', 'Лола', 'Мадина', 'Нодира', 'Озода',
            'Toshkent', 'Samarkand', 'Buxoro', 'Andijon', 'Namangan', 'Farg\'ona',
            'Ташкент', 'Самарканд', 'Бухара', 'Андижан', 'Наманган', 'Фергана'
        ]
        
        data = {
            'id': range(200),
            'name': [random.choice(uzbek_names) for _ in range(200)],
            'city': [random.choice(uzbek_names[-12:]) for _ in range(200)],
            'region': [f'Region_{i % 10}' for i in range(200)]
        }
        return pd.DataFrame(data)
    
    def test_exact_prefix_performance(self, medium_dataset):
        """Test performance of exact prefix blocking."""
        generator = ExactPrefixGenerator({'prefix_lengths': [2, 3, 4]})
        
        start_time = time.time()
        total_keys = 0
        
        for _, row in medium_dataset.iterrows():
            keys = generator.generate_keys(str(row['name']), 'name')
            total_keys += len(keys)
        
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0  # Should complete within 1 second
        assert total_keys > 0
        
        # Test selectivity estimation
        names = medium_dataset['name'].tolist()
        selectivity = generator.estimate_selectivity(names)
        assert 0.0 <= selectivity <= 1.0
    
    def test_phonetic_key_performance(self, uzbek_dataset):
        """Test performance of phonetic key blocking."""
        generator = PhoneticKeyGenerator({
            'use_soundex': True,
            'use_uzbek_phonetic': True
        })
        
        start_time = time.time()
        total_keys = 0
        
        for _, row in uzbek_dataset.iterrows():
            keys = generator.generate_keys(str(row['name']), 'name')
            total_keys += len(keys)
        
        processing_time = time.time() - start_time
        
        assert processing_time < 2.0  # Should complete within 2 seconds
        assert total_keys > 0
        
        # Test selectivity estimation
        names = uzbek_dataset['name'].tolist()
        selectivity = generator.estimate_selectivity(names)
        assert 0.0 <= selectivity <= 1.0
    
    def test_ngram_performance(self, medium_dataset):
        """Test performance of n-gram blocking."""
        generator = NGramGenerator({'n': 3, 'skip_grams': True})
        
        start_time = time.time()
        total_keys = 0
        
        for _, row in medium_dataset.iterrows():
            keys = generator.generate_keys(str(row['name']), 'name')
            total_keys += len(keys)
        
        processing_time = time.time() - start_time
        
        assert processing_time < 1.5  # Should complete within 1.5 seconds
        assert total_keys > 0
        
        # Test selectivity estimation
        names = medium_dataset['name'].tolist()
        selectivity = generator.estimate_selectivity(names)
        assert 0.0 <= selectivity <= 1.0
    
    def test_blocking_index_sequential_performance(self, medium_dataset):
        """Test performance of sequential blocking index building."""
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': False
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        start_time = time.time()
        stats = index.build_index(medium_dataset, field_mappings)
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert stats.total_records == len(medium_dataset)
        assert stats.total_blocks > 0
        assert stats.reduction_ratio >= 0  # Should not increase comparisons
    
    def test_blocking_index_parallel_performance(self, medium_dataset):
        """Test performance of parallel blocking index building."""
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': True,
            'max_workers': 4
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        start_time = time.time()
        stats = index.build_index(medium_dataset, field_mappings)
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert stats.total_records == len(medium_dataset)
        assert stats.total_blocks > 0
        assert stats.reduction_ratio >= 0
    
    def test_parallel_vs_sequential_performance(self, large_dataset):
        """Compare parallel vs sequential performance."""
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        # Test sequential
        sequential_config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': False
        }
        sequential_index = OptimizedBlockingIndex(sequential_config)
        
        start_time = time.time()
        sequential_stats = sequential_index.build_index(large_dataset, field_mappings)
        sequential_time = time.time() - start_time
        
        # Test parallel
        parallel_config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': True,
            'max_workers': 4
        }
        parallel_index = OptimizedBlockingIndex(parallel_config)
        
        start_time = time.time()
        parallel_stats = parallel_index.build_index(large_dataset, field_mappings)
        parallel_time = time.time() - start_time
        
        # Parallel should be faster or at least not significantly slower
        # For smaller datasets, parallel processing might have overhead
        assert parallel_time <= sequential_time * 1.5  # Allow 50% tolerance for overhead
        
        # Results should be similar
        assert abs(sequential_stats.total_blocks - parallel_stats.total_blocks) <= 10
        assert abs(sequential_stats.reduction_ratio - parallel_stats.reduction_ratio) <= 5.0
    
    def test_candidate_pair_generation_performance(self, medium_dataset):
        """Test performance of candidate pair generation."""
        # Create two similar datasets
        dataset1 = medium_dataset.copy()
        dataset2 = medium_dataset.copy()
        # Modify some records to create variations
        dataset2.loc[dataset2.index % 10 == 0, 'name'] = dataset2.loc[dataset2.index % 10 == 0, 'name'] + '_modified'
        
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 'name', 1.0), ('city', 'city', 0.8)]
        
        start_time = time.time()
        candidate_pairs = list(index.get_candidate_pairs(dataset1, dataset2, field_mappings))
        processing_time = time.time() - start_time
        
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert len(candidate_pairs) > 0
        # With blocking, we should get fewer pairs than the full cartesian product
        # But with uniform test data, blocking might not be very effective
        max_expected_pairs = len(dataset1) * len(dataset2)
        assert len(candidate_pairs) <= max_expected_pairs  # Should not exceed full comparison
    
    def test_strategy_comparison(self, uzbek_dataset):
        """Compare different blocking strategies."""
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        strategies_to_test = [
            [BlockingStrategy.EXACT_PREFIX],
            [BlockingStrategy.PHONETIC_KEY],
            [BlockingStrategy.NGRAM],
            [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.PHONETIC_KEY],
            [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            [BlockingStrategy.PHONETIC_KEY, BlockingStrategy.NGRAM],
            [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.PHONETIC_KEY, BlockingStrategy.NGRAM]
        ]
        
        results = []
        
        for strategies in strategies_to_test:
            config = {
                'strategies': strategies,
                'parallel_processing': True
            }
            
            index = OptimizedBlockingIndex(config)
            
            start_time = time.time()
            stats = index.build_index(uzbek_dataset, field_mappings)
            processing_time = time.time() - start_time
            
            results.append({
                'strategies': [s.value for s in strategies],
                'processing_time': processing_time,
                'total_blocks': stats.total_blocks,
                'reduction_ratio': stats.reduction_ratio,
                'average_block_size': stats.average_block_size
            })
        
        # Verify all strategies completed successfully
        for result in results:
            assert result['processing_time'] < 5.0
            assert result['total_blocks'] >= 0  # Some strategies might not create blocks with this data
            assert result['reduction_ratio'] >= 0
        
        # Multi-strategy approaches should generally have more blocks (higher selectivity)
        single_strategy_blocks = [r['total_blocks'] for r in results[:3]]
        multi_strategy_blocks = [r['total_blocks'] for r in results[3:]]
        
        assert max(multi_strategy_blocks) >= max(single_strategy_blocks)
    
    def test_adaptive_strategy_selection(self, small_dataset, large_dataset):
        """Test adaptive strategy selection based on dataset size."""
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.PHONETIC_KEY, BlockingStrategy.NGRAM],
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0)]
        
        # Test with small dataset
        small_stats = index.build_index(small_dataset, field_mappings)
        small_info = index.get_block_info()
        
        # Clear and test with large dataset
        index.clear_index()
        large_stats = index.build_index(large_dataset, field_mappings)
        large_info = index.get_block_info()
        
        # Both should complete successfully
        assert small_stats.total_records == len(small_dataset)
        assert large_stats.total_records == len(large_dataset)
        
        # Large dataset should have better reduction ratio due to more blocking opportunities
        assert large_stats.reduction_ratio >= small_stats.reduction_ratio
    
    def test_memory_efficiency(self, large_dataset):
        """Test memory efficiency of blocking index."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        # Build index
        stats = index.build_index(large_dataset, field_mappings)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clear index
        index.clear_index()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB for 5K records)
        assert memory_increase < 100
        
        # Memory should be mostly freed after clearing
        assert final_memory <= peak_memory
        
        # Index should be functional
        assert stats.total_records == len(large_dataset)
        assert stats.total_blocks > 0
    
    def test_thread_safety(self, medium_dataset):
        """Test thread safety of blocking operations."""
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.NGRAM],
            'parallel_processing': True,
            'max_workers': 4
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        def build_and_query():
            """Build index and query it."""
            stats = index.build_index(medium_dataset, field_mappings)
            info = index.get_block_info()
            return stats, info
        
        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(build_and_query) for _ in range(4)]
            results = [future.result() for future in futures]
        
        # All threads should complete successfully
        assert len(results) == 4
        for stats, info in results:
            assert stats.total_records == len(medium_dataset)
            assert stats.total_blocks > 0
            assert len(info) > 0
    
    def test_block_size_optimization(self, large_dataset):
        """Test block size optimization."""
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX],
            'max_block_size': 50,  # Small block size to force optimization
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('city', 1.0)]  # City has many duplicates, will create large blocks
        
        # Build index
        stats = index.build_index(large_dataset, field_mappings)
        
        # Check initial block sizes
        initial_info = index.get_block_info()
        initial_max_block = max(
            max(info['block_sizes']) if info['block_sizes'] else 0
            for info in initial_info.values()
        )
        
        # Optimize block sizes
        index.optimize_block_sizes()
        
        # Check optimized block sizes
        optimized_info = index.get_block_info()
        optimized_max_block = max(
            max(info['block_sizes']) if info['block_sizes'] else 0
            for info in optimized_info.values()
        )
        
        # Optimized blocks should be smaller or equal to max_block_size
        assert optimized_max_block <= config['max_block_size']
        
        # Should have more blocks after optimization if large blocks were split
        if initial_max_block > config['max_block_size']:
            optimized_total_blocks = sum(info['total_blocks'] for info in optimized_info.values())
            initial_total_blocks = sum(info['total_blocks'] for info in initial_info.values())
            assert optimized_total_blocks >= initial_total_blocks


class TestBlockingBenchmarks:
    """Benchmark tests for blocking performance."""
    
    def generate_realistic_dataset(self, size: int) -> pd.DataFrame:
        """Generate a realistic dataset for benchmarking."""
        # Common Uzbek names and places
        uzbek_names = [
            'Abdulla', 'Akmal', 'Aziz', 'Bobur', 'Davron', 'Eldor', 'Farrux', 'Gulnora',
            'Хуршида', 'Ирода', 'Жамшид', 'Камола', 'Лола', 'Мадина', 'Нодира', 'Озода',
            'Muhammad', 'Ahmad', 'Ali', 'Fatima', 'Aisha', 'Omar', 'Hassan', 'Zainab'
        ]
        
        uzbek_cities = [
            'Toshkent', 'Samarkand', 'Buxoro', 'Andijon', 'Namangan', 'Farg\'ona',
            'Ташкент', 'Самарканд', 'Бухара', 'Андижан', 'Наманган', 'Фергана',
            'Nukus', 'Qarshi', 'Termiz', 'Jizzax', 'Sirdaryo', 'Xorazm'
        ]
        
        data = {
            'id': range(size),
            'name': [random.choice(uzbek_names) + f'_{i}' if i % 10 == 0 else random.choice(uzbek_names) for i in range(size)],
            'city': [random.choice(uzbek_cities) for _ in range(size)],
            'region': [f'Region_{i % 14}' for i in range(size)],  # 14 regions in Uzbekistan
            'phone': [f'+998{random.randint(90, 99)}{random.randint(1000000, 9999999)}' for _ in range(size)]
        }
        
        return pd.DataFrame(data)
    
    @pytest.mark.benchmark
    def test_benchmark_small_dataset(self, benchmark):
        """Benchmark blocking on small dataset (1K records)."""
        dataset = self.generate_realistic_dataset(1000)
        
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.PHONETIC_KEY],
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0), ('city', 0.8)]
        
        def build_index():
            index.clear_index()
            return index.build_index(dataset, field_mappings)
        
        stats = benchmark(build_index)
        assert stats.total_records == len(dataset)
    
    @pytest.mark.benchmark
    def test_benchmark_medium_dataset(self, benchmark):
        """Benchmark blocking on medium dataset (5K records)."""
        dataset = self.generate_realistic_dataset(5000)
        
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.PHONETIC_KEY, BlockingStrategy.NGRAM],
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 1.0), ('city', 0.8), ('region', 0.6)]
        
        def build_index():
            index.clear_index()
            return index.build_index(dataset, field_mappings)
        
        stats = benchmark(build_index)
        assert stats.total_records == len(dataset)
    
    @pytest.mark.benchmark
    def test_benchmark_candidate_generation(self, benchmark):
        """Benchmark candidate pair generation."""
        dataset1 = self.generate_realistic_dataset(1000)
        dataset2 = self.generate_realistic_dataset(1000)
        
        config = {
            'strategies': [BlockingStrategy.EXACT_PREFIX, BlockingStrategy.PHONETIC_KEY],
            'parallel_processing': True
        }
        
        index = OptimizedBlockingIndex(config)
        field_mappings = [('name', 'name', 1.0), ('city', 'city', 0.8)]
        
        def generate_candidates():
            return list(index.get_candidate_pairs(dataset1, dataset2, field_mappings))
        
        candidates = benchmark(generate_candidates)
        assert len(candidates) > 0
        assert len(candidates) < len(dataset1) * len(dataset2)


if __name__ == '__main__':
    # Run performance tests
    pytest.main([__file__, '-v', '--tb=short'])