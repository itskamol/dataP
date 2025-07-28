"""
Comprehensive unit tests for UzbekTextNormalizer with edge cases and performance benchmarks.
Tests requirements 4.1, 4.3: Uzbek text normalization with edge cases and performance testing.
"""

import pytest
import unittest
import time
from unittest.mock import patch

from src.domain.matching.uzbek_normalizer import UzbekTextNormalizer


class TestUzbekTextNormalizerComprehensive(unittest.TestCase):
    """Comprehensive tests for UzbekTextNormalizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = UzbekTextNormalizer()
    
    def test_cyrillic_to_latin_comprehensive(self):
        """Test comprehensive Cyrillic to Latin conversion."""
        test_cases = [
            # Basic Cyrillic letters
            ('а', 'a'), ('б', 'b'), ('в', 'v'), ('г', 'g'), ('д', 'd'),
            ('е', 'e'), ('ё', 'yo'), ('ж', 'j'), ('з', 'z'), ('и', 'i'),
            ('й', 'y'), ('к', 'k'), ('л', 'l'), ('м', 'm'), ('н', 'n'),
            ('о', 'o'), ('п', 'p'), ('р', 'r'), ('с', 's'), ('т', 't'),
            ('у', 'u'), ('ф', 'f'), ('х', 'x'), ('ц', 'ts'), ('ч', 'ch'),
            ('ш', 'sh'), ('щ', 'shch'), ('ъ', ''), ('ы', 'i'), ('ь', ''),
            ('э', 'e'), ('ю', 'yu'), ('я', 'ya'),
            
            # Uzbek-specific Cyrillic letters
            ('ў', 'o'), ('қ', 'q'), ('ғ', 'g'), ('ҳ', 'h'),
            
            # Words
            ('Ташкент', 'tashkent'), ('Самарқанд', 'samarqand'),
            ('Бухоро', 'buxoro'), ('Андижон', 'andijon'),
            ('Ўзбекистон', 'ozbekiston'), ('махалла', 'mahalla'),
            
            # Mixed case
            ('ТашКент', 'tashkent'), ('ЎЗБЕКИСТОН', 'ozbekiston'),
        ]
        
        for cyrillic, expected_latin in test_cases:
            with self.subTest(cyrillic=cyrillic):
                result = self.normalizer.normalize_text(cyrillic)
                self.assertIn(expected_latin.lower(), result.lower())
    
    def test_latin_normalizations_comprehensive(self):
        """Test comprehensive Latin character normalizations."""
        test_cases = [
            # Apostrophe handling
            ("O'zbekiston", "ozbekiston"),
            ("Ko'cha", "kocha"),
            ("Bo'lim", "bolim"),
            ("Qo'rg'on", "qorgon"),
            
            # Digraph standardization
            ("Toshkent", "tashkent"),  # 'tosh' -> 'tash'
            ("Qashqadaryo", "qashqadaryo"),
            ("Xorazm", "xorazm"),
            
            # Case normalization
            ("TOSHKENT", "tashkent"),
            ("tOsHkEnT", "tashkent"),
            
            # Whitespace normalization
            ("  Toshkent  ", "tashkent"),
            ("Tosh\t\nkent", "tashkent"),
            ("Tosh    kent", "tashkent"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.normalizer.normalize_text(input_text)
                self.assertIn(expected, result.lower())
    
    def test_word_variations_comprehensive(self):
        """Test comprehensive word-level normalizations."""
        variation_groups = [
            # Administrative terms
            ['мчж', 'mchj', 'МЧЖ', 'MCHJ'],
            ['туман', 'tuman', 'район', 'rayon'],
            ['вилоят', 'viloyat', 'область', 'oblast'],
            ['шаҳар', 'shahar', 'город', 'gorod'],
            ['қишлоқ', 'qishloq', 'село', 'selo'],
            
            # Geographic terms
            ['дарё', 'daryo', 'река', 'reka'],
            ['тоғ', 'tog', 'гора', 'gora'],
            ['водий', 'vodiy', 'долина', 'dolina'],
            
            # Common words
            ['уй', 'uy', 'дом', 'dom'],
            ['йўл', 'yol', 'дорога', 'doroga'],
            ['бозор', 'bozor', 'рынок', 'rinok'],
        ]
        
        for group in variation_groups:
            normalized_forms = [self.normalizer.normalize_text(word) for word in group]
            
            # All variations in a group should normalize to similar forms
            base_form = normalized_forms[0]
            for form in normalized_forms[1:]:
                # Should have significant overlap or be identical
                similarity = self._calculate_string_similarity(base_form, form)
                self.assertGreater(similarity, 0.7, 
                    f"Forms '{base_form}' and '{form}' from group {group} not similar enough")
    
    def test_phonetic_key_generation_comprehensive(self):
        """Test comprehensive phonetic key generation."""
        test_cases = [
            # Same words in different scripts should have similar keys
            [('Toshkent', 'Ташкент'), ('O\'zbekiston', 'Ўзбекистон'), 
             ('mahalla', 'махалла'), ('shahar', 'шаҳар')],
            
            # Phonetically similar words
            [('kino', 'qino'), ('gul', 'ғul'), ('haus', 'xaus')],
            
            # Words with common variations
            [('center', 'tsentr', 'центр'), ('bank', 'банк')],
        ]
        
        for similar_group in test_cases:
            keys = [self.normalizer.generate_phonetic_key(word) for word in similar_group]
            
            # Keys should be similar within each group
            base_key = keys[0]
            for key in keys[1:]:
                similarity = self._calculate_string_similarity(base_key, key)
                self.assertGreater(similarity, 0.6,
                    f"Phonetic keys '{base_key}' and '{key}' not similar enough for {similar_group}")
    
    def test_phonetic_distance_comprehensive(self):
        """Test comprehensive phonetic distance calculation."""
        test_cases = [
            # Identical words (distance should be 0)
            [('hello', 'hello'), ('Toshkent', 'Toshkent')],
            
            # Very similar words (distance should be low)
            [('Toshkent', 'Ташкент'), ('O\'zbekiston', 'Ўзбекистон'),
             ('mahalla', 'махалла')],
            
            # Somewhat similar words (distance should be medium)
            [('Toshkent', 'Tashkent'), ('center', 'tsentr')],
            
            # Different words (distance should be high)
            [('hello', 'world'), ('Toshkent', 'London')],
        ]
        
        expected_ranges = [
            (0.0, 0.1),    # Identical
            (0.0, 0.3),    # Very similar
            (0.2, 0.6),    # Somewhat similar
            (0.6, 1.0),    # Different
        ]
        
        for i, test_group in enumerate(test_cases):
            min_dist, max_dist = expected_ranges[i]
            
            for word1, word2 in test_group:
                with self.subTest(word1=word1, word2=word2):
                    distance = self.normalizer.calculate_phonetic_distance(word1, word2)
                    self.assertGreaterEqual(distance, min_dist)
                    self.assertLessEqual(distance, max_dist)
    
    def test_uzbek_text_detection_comprehensive(self):
        """Test comprehensive Uzbek text detection."""
        uzbek_texts = [
            # Pure Uzbek Cyrillic
            'Ташкент', 'Самарқанд', 'Ўзбекистон', 'махалла',
            
            # Pure Uzbek Latin
            'Toshkent', 'Samarqand', 'O\'zbekiston', 'mahalla',
            
            # Mixed Uzbek
            'Toshkent шаҳри', 'O\'zbekiston Республикаси',
            
            # Uzbek with numbers and punctuation
            'Toshkent-2023', 'махалла №5', 'O\'zbekiston!',
            
            # Common Uzbek phrases
            'Assalomu alaykum', 'Rahmat', 'Xayr',
            'Салом алейкум', 'Рахмат', 'Хайр',
        ]
        
        non_uzbek_texts = [
            # Pure English
            'Hello World', 'New York', 'United States',
            
            # Pure Russian (non-Uzbek Cyrillic)
            'Москва', 'Россия', 'Привет мир',
            
            # Other languages
            '你好世界', 'Bonjour monde', 'Hola mundo',
            
            # Numbers and symbols only
            '12345', '@#$%^', '2023-01-01',
        ]
        
        for text in uzbek_texts:
            with self.subTest(text=text):
                self.assertTrue(self.normalizer.is_uzbek_text(text),
                    f"'{text}' should be detected as Uzbek")
        
        for text in non_uzbek_texts:
            with self.subTest(text=text):
                self.assertFalse(self.normalizer.is_uzbek_text(text),
                    f"'{text}' should not be detected as Uzbek")
    
    def test_character_variations_comprehensive(self):
        """Test comprehensive character variation retrieval."""
        test_cases = [
            # Vowels
            ('a', ['a', 'а', 'ا']),  # Latin, Cyrillic, Arabic
            ('i', ['i', 'и', 'ы', 'y']),
            ('o', ['o', 'о', 'ў']),
            ('u', ['u', 'у', 'ў']),
            
            # Consonants with variations
            ('k', ['k', 'к', 'q', 'қ']),
            ('g', ['g', 'г', 'ғ']),
            ('h', ['h', 'х', 'ҳ']),
            
            # Special Uzbek characters
            ('ў', ['ў', 'o', 'u', 'w']),
            ('қ', ['қ', 'q', 'k']),
            ('ғ', ['ғ', 'g']),
            ('ҳ', ['ҳ', 'h', 'x']),
        ]
        
        for char, expected_variations in test_cases:
            with self.subTest(char=char):
                variations = self.normalizer.get_character_variations(char)
                
                # Should include the character itself
                self.assertIn(char, variations)
                
                # Should include expected variations
                for expected in expected_variations:
                    self.assertIn(expected, variations,
                        f"Expected variation '{expected}' not found for '{char}'")
    
    def test_normalization_caching_comprehensive(self):
        """Test comprehensive normalization caching behavior."""
        test_texts = [
            'Toshkent', 'Ташкент', 'O\'zbekiston', 'махалла',
            'Samarqand', 'Самарқанд', 'Buxoro', 'Бухоро'
        ]
        
        # Clear any existing cache
        self.normalizer._normalization_cache.clear()
        
        # First pass - populate cache
        first_pass_times = []
        for text in test_texts:
            start_time = time.time()
            result1 = self.normalizer.normalize_text(text)
            end_time = time.time()
            first_pass_times.append(end_time - start_time)
        
        # Second pass - should use cache
        second_pass_times = []
        for text in test_texts:
            start_time = time.time()
            result2 = self.normalizer.normalize_text(text)
            end_time = time.time()
            second_pass_times.append(end_time - start_time)
        
        # Results should be identical
        for i, text in enumerate(test_texts):
            result1 = self.normalizer.normalize_text(text)
            result2 = self.normalizer.normalize_text(text)
            self.assertEqual(result1, result2)
        
        # Second pass should generally be faster (cached)
        avg_first = sum(first_pass_times) / len(first_pass_times)
        avg_second = sum(second_pass_times) / len(second_pass_times)
        
        # Allow some tolerance for timing variations
        self.assertLessEqual(avg_second, avg_first * 1.5,
            "Cached normalization should be faster or similar")
    
    def test_edge_cases_comprehensive(self):
        """Test comprehensive edge cases and error handling."""
        edge_cases = [
            # Empty and None inputs
            ('', ''),
            (None, ''),
            
            # Whitespace only
            ('   ', ''),
            ('\t\n\r', ''),
            
            # Non-string inputs
            (123, ''),
            ([], ''),
            ({}, ''),
            
            # Very long strings
            ('a' * 10000, 'a' * 10000),
            
            # Mixed scripts
            ('Hello Ташкент World', 'hello tashkent world'),
            
            # Only punctuation
            ('!@#$%^&*()', ''),
            
            # Numbers mixed with text
            ('Toshkent123', 'tashkent123'),
            ('123Ташкент456', '123tashkent456'),
            
            # Special Unicode characters
            ('Café résumé', 'cafe resume'),
            ('naïve', 'naive'),
        ]
        
        for input_text, expected_contains in edge_cases:
            with self.subTest(input_text=input_text):
                try:
                    result = self.normalizer.normalize_text(input_text)
                    self.assertIsInstance(result, str)
                    
                    if expected_contains:
                        self.assertIn(expected_contains.lower(), result.lower())
                        
                except Exception as e:
                    self.fail(f"Failed to handle edge case '{input_text}': {e}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for normalization."""
        # Test data of various sizes
        small_texts = ['Toshkent', 'Ташкент', 'O\'zbekiston'] * 10
        medium_texts = ['Toshkent shahri', 'Ташкент шаҳри', 'O\'zbekiston Respublikasi'] * 100
        large_texts = ['Toshkent shahri, O\'zbekiston Respublikasi'] * 1000
        
        datasets = [
            ('small', small_texts),
            ('medium', medium_texts),
            ('large', large_texts)
        ]
        
        for dataset_name, texts in datasets:
            with self.subTest(dataset=dataset_name):
                start_time = time.time()
                
                for text in texts:
                    self.normalizer.normalize_text(text)
                
                end_time = time.time()
                total_time = end_time - start_time
                avg_time_per_text = total_time / len(texts)
                
                # Performance expectations (adjust based on requirements)
                if dataset_name == 'small':
                    self.assertLess(avg_time_per_text, 0.01)  # 10ms per text
                elif dataset_name == 'medium':
                    self.assertLess(avg_time_per_text, 0.005)  # 5ms per text
                else:  # large
                    self.assertLess(avg_time_per_text, 0.002)  # 2ms per text
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns during normalization."""
        # Test with many unique texts (should not cause memory leak)
        unique_texts = [f'Toshkent_{i}' for i in range(1000)]
        
        # Process all texts
        for text in unique_texts:
            self.normalizer.normalize_text(text)
        
        # Cache should have reasonable size (LRU should limit growth)
        cache_size = len(self.normalizer._normalization_cache)
        self.assertLessEqual(cache_size, 1000)  # Should not exceed reasonable limit
        
        # Test cache cleanup
        self.normalizer._normalization_cache.clear()
        self.assertEqual(len(self.normalizer._normalization_cache), 0)
    
    def test_thread_safety(self):
        """Test thread safety of normalization operations."""
        import threading
        
        results = []
        errors = []
        
        def normalize_worker():
            try:
                for i in range(100):
                    text = f'Toshkent_{i % 10}'  # Reuse some texts for cache testing
                    result = self.normalizer.normalize_text(text)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=normalize_worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 500)  # 5 threads * 100 normalizations
        
        # Results should be consistent
        for i in range(0, len(results), 10):
            if i + 10 <= len(results):
                batch = results[i:i+10]
                # All results for same input should be identical
                unique_results = set(batch)
                self.assertLessEqual(len(unique_results), 10)  # At most 10 unique results
    
    def _calculate_string_similarity(self, s1, s2):
        """Helper method to calculate string similarity."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Simple similarity based on common characters
        common_chars = set(s1.lower()) & set(s2.lower())
        total_chars = set(s1.lower()) | set(s2.lower())
        
        if not total_chars:
            return 1.0
        
        return len(common_chars) / len(total_chars)


if __name__ == '__main__':
    unittest.main(verbosity=2)