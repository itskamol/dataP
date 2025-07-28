"""
Phonetic matching algorithm implementation with Uzbek language support.
Implements requirements 1.1, 3.3, 4.1: Phonetic matching with caching and Uzbek phonetic rules.
"""

from typing import Dict, Any, Optional, List
import time
import re

from .base import MatchingAlgorithm, MatchingResult, MatchingType
from .cache import get_global_cache
from .uzbek_normalizer import UzbekTextNormalizer


class PhoneticMatcher(MatchingAlgorithm):
    """
    Phonetic matching algorithm that compares sounds rather than exact spellings.
    Specialized for Uzbek language with support for Cyrillic/Latin variations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the phonetic matcher.
        
        Args:
            config: Configuration parameters including:
                - phonetic_method: Method to use ('soundex', 'metaphone', 'uzbek_phonetic')
                - min_similarity: Minimum similarity threshold (default: 70.0)
                - normalize_uzbek: Whether to apply Uzbek normalization (default: True)
                - use_cache: Whether to use caching (default: True)
                - strict_phonetic: Whether to use strict phonetic matching (default: False)
        """
        super().__init__("PhoneticMatcher", MatchingType.PHONETIC, config)
        
        # Configuration parameters
        self.phonetic_method = self.config.get('phonetic_method', 'uzbek_phonetic')
        self.min_similarity = self.config.get('min_similarity', 70.0)
        self.normalize_uzbek = self.config.get('normalize_uzbek', True)
        self.use_cache = self.config.get('use_cache', True)
        self.strict_phonetic = self.config.get('strict_phonetic', False)
        
        # Initialize Uzbek normalizer
        self.uzbek_normalizer = UzbekTextNormalizer() if self.normalize_uzbek else None
        
        # Get global cache instance
        self.cache = get_global_cache() if self.use_cache else None
        
        # Validate phonetic method
        valid_methods = ['soundex', 'metaphone', 'uzbek_phonetic']
        if self.phonetic_method not in valid_methods:
            self.logger.warning(f"Invalid phonetic method '{self.phonetic_method}', using 'uzbek_phonetic'")
            self.phonetic_method = 'uzbek_phonetic'
        
        # Initialize phonetic mapping tables
        self._init_phonetic_mappings()
        
        self.logger.info(f"PhoneticMatcher initialized", extra={
            'phonetic_method': self.phonetic_method,
            'min_similarity': self.min_similarity,
            'normalize_uzbek': self.normalize_uzbek,
            'use_cache': self.use_cache,
            'strict_phonetic': self.strict_phonetic
        })
    
    def _init_phonetic_mappings(self):
        """Initialize phonetic mapping tables for different methods."""
        
        # Soundex mapping for English/Latin characters
        self.soundex_mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        # Uzbek-specific phonetic groups (sounds that are similar)
        self.uzbek_phonetic_groups = {
            # Vowels
            'A': ['A', 'Ə', 'E'],
            'E': ['E', 'A', 'Ə'],
            'I': ['I', 'Y', 'Ы'],
            'O': ['O', 'U', 'Ў'],
            'U': ['U', 'O', 'Ў'],
            
            # Consonants with similar sounds
            'B': ['B', 'P'],
            'P': ['P', 'B'],
            'D': ['D', 'T'],
            'T': ['T', 'D'],
            'G': ['G', 'K', 'Ғ'],
            'K': ['K', 'G', 'Q', 'Қ'],
            'Q': ['Q', 'K', 'Қ'],
            'S': ['S', 'Z', 'Ш', 'Ж'],
            'Z': ['Z', 'S', 'Ж', 'Ш'],
            'F': ['F', 'V'],
            'V': ['V', 'F'],
            'H': ['H', 'Ҳ', 'X'],
            'X': ['X', 'H', 'Ҳ'],
            
            # Uzbek-specific characters
            'Ў': ['Ў', 'O', 'U'],
            'Ғ': ['Ғ', 'G'],
            'Қ': ['Қ', 'K', 'Q'],
            'Ҳ': ['Ҳ', 'H', 'X'],
            'Ш': ['Ш', 'SH', 'S'],
            'Ч': ['Ч', 'CH', 'C'],
            'Ж': ['Ж', 'J', 'Z'],
            'Ё': ['Ё', 'YO', 'O'],
            'Ю': ['Ю', 'YU', 'U'],
            'Я': ['Я', 'YA', 'A']
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for phonetic comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get_normalized_text(
                text, self.name, self.config
            )
            if cached_result is not None:
                self._track_cache_hit()
                return cached_result
            else:
                self._track_cache_miss()
        
        # Start normalization
        normalized = text.strip().upper()
        
        # Apply Uzbek normalization if enabled
        if self.normalize_uzbek and self.uzbek_normalizer:
            if self.uzbek_normalizer.is_uzbek_text(text):
                # Use Uzbek normalizer but keep uppercase for phonetic processing
                temp_normalized = self.uzbek_normalizer.normalize_text(text, aggressive=True)
                normalized = temp_normalized.upper()
            else:
                # Apply basic normalization for non-Uzbek text
                import unicodedata
                normalized = unicodedata.normalize('NFC', normalized)
        
        # Remove non-alphabetic characters for phonetic matching
        normalized = re.sub(r'[^A-ZА-ЯЎҒҚҲ]', '', normalized)
        
        # Cache the result
        if self.cache:
            self.cache.set_normalized_text(text, self.name, normalized, self.config)
        
        return normalized
    
    def calculate_similarity(self, text1: str, text2: str, 
                           field_name: Optional[str] = None) -> MatchingResult:
        """
        Calculate phonetic similarity between two text values.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            field_name: Optional field name for context
            
        Returns:
            MatchingResult with similarity score and metadata
        """
        start_time = time.time()
        
        # Handle empty inputs
        if not text1 and not text2:
            return MatchingResult(
                similarity_score=100.0,
                confidence=100.0,
                matched_fields=[field_name] if field_name else [],
                metadata={
                    'algorithm': self.name,
                    'match_type': 'both_empty',
                    'phonetic_method': self.phonetic_method
                },
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        if not text1 or not text2:
            return MatchingResult(
                similarity_score=0.0,
                confidence=0.0,
                matched_fields=[],
                metadata={
                    'algorithm': self.name,
                    'match_type': 'one_empty',
                    'phonetic_method': self.phonetic_method
                },
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check cache first
        cache_key_config = self.config.copy() if self.config else {}
        cache_key_config['field_name'] = field_name
        
        if self.cache:
            cached_result = self.cache.get_similarity(
                text1, text2, self.name, cache_key_config
            )
            if cached_result is not None:
                self._track_cache_hit()
                processing_time = (time.time() - start_time) * 1000
                is_match = cached_result >= self.min_similarity
                return MatchingResult(
                    similarity_score=cached_result,
                    confidence=cached_result,
                    matched_fields=[field_name] if field_name and is_match else [],
                    metadata={
                        'algorithm': self.name,
                        'phonetic_method': self.phonetic_method,
                        'cached': True
                    },
                    processing_time_ms=processing_time
                )
            else:
                self._track_cache_miss()
        
        # Normalize both texts
        normalized_text1 = self.normalize_text(text1)
        normalized_text2 = self.normalize_text(text2)
        
        # Calculate phonetic similarity based on method
        if self.phonetic_method == 'soundex':
            similarity_score = self._soundex_similarity(normalized_text1, normalized_text2)
        elif self.phonetic_method == 'metaphone':
            similarity_score = self._metaphone_similarity(normalized_text1, normalized_text2)
        else:  # uzbek_phonetic
            similarity_score = self._uzbek_phonetic_similarity(normalized_text1, normalized_text2)
        
        # Calculate confidence (phonetic matching is less certain than exact)
        confidence = max(similarity_score - 10.0, 0.0) if similarity_score < 100.0 else 100.0
        
        # Cache the result
        if self.cache:
            self.cache.set_similarity(text1, text2, self.name, similarity_score, cache_key_config)
        
        processing_time = (time.time() - start_time) * 1000
        is_match = similarity_score >= self.min_similarity
        
        # Update performance tracking
        self._total_comparisons += 1
        self._total_processing_time += processing_time
        
        return MatchingResult(
            similarity_score=similarity_score,
            confidence=confidence,
            matched_fields=[field_name] if field_name and is_match else [],
            metadata={
                'algorithm': self.name,
                'phonetic_method': self.phonetic_method,
                'normalized_text1': normalized_text1,
                'normalized_text2': normalized_text2,
                'original_text1': text1,
                'original_text2': text2,
                'min_similarity_threshold': self.min_similarity,
                'strict_phonetic': self.strict_phonetic,
                'cached': False
            },
            processing_time_ms=processing_time
        )
    
    def _soundex_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using Soundex algorithm."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        soundex1 = self._generate_soundex(text1)
        soundex2 = self._generate_soundex(text2)
        
        if soundex1 == soundex2:
            return 100.0
        
        # Calculate partial similarity based on matching characters
        matches = sum(1 for a, b in zip(soundex1, soundex2) if a == b)
        max_len = max(len(soundex1), len(soundex2))
        return (matches / max_len) * 100 if max_len > 0 else 0.0
    
    def _generate_soundex(self, text: str) -> str:
        """Generate Soundex code for text."""
        if not text:
            return "0000"
        
        text = text.upper()
        soundex = text[0]  # Keep first letter
        
        # Convert remaining letters using mapping
        for char in text[1:]:
            if char in self.soundex_mapping:
                code = self.soundex_mapping[char]
                # Don't add consecutive duplicates
                if not soundex or soundex[-1] != code:
                    soundex += code
        
        # Remove vowels and H, W, Y (except first letter)
        soundex = soundex[0] + ''.join(c for c in soundex[1:] if c.isdigit())
        
        # Pad or truncate to 4 characters
        soundex = (soundex + "000")[:4]
        return soundex
    
    def _metaphone_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using simplified Metaphone algorithm."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        metaphone1 = self._generate_metaphone(text1)
        metaphone2 = self._generate_metaphone(text2)
        
        if metaphone1 == metaphone2:
            return 100.0
        
        # Calculate similarity using Levenshtein distance on metaphone codes
        max_len = max(len(metaphone1), len(metaphone2))
        if max_len == 0:
            return 100.0
        
        distance = self._levenshtein_distance(metaphone1, metaphone2)
        similarity = (1.0 - distance / max_len) * 100
        return max(similarity, 0.0)
    
    def _generate_metaphone(self, text: str) -> str:
        """Generate simplified Metaphone code for text."""
        if not text:
            return ""
        
        text = text.upper()
        metaphone = ""
        
        # Simplified Metaphone rules
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in 'AEIOU':
                if i == 0:  # Keep initial vowels
                    metaphone += char
            elif char == 'B':
                metaphone += 'B'
            elif char == 'C':
                if i + 1 < len(text) and text[i + 1] in 'HI':
                    metaphone += 'X'
                else:
                    metaphone += 'K'
            elif char in 'DT':
                metaphone += 'T'
            elif char in 'FV':
                metaphone += 'F'
            elif char in 'GJ':
                metaphone += 'J'
            elif char in 'KQ':
                metaphone += 'K'
            elif char == 'L':
                metaphone += 'L'
            elif char in 'MN':
                metaphone += 'M'
            elif char in 'PB':
                metaphone += 'P'
            elif char == 'R':
                metaphone += 'R'
            elif char in 'SZ':
                metaphone += 'S'
            elif char == 'X':
                metaphone += 'KS'
            elif char == 'Y':
                if i == 0:
                    metaphone += 'Y'
            
            i += 1
        
        return metaphone
    
    def _uzbek_phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using Uzbek-specific phonetic rules."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        # Use Uzbek normalizer's phonetic key if available
        if self.uzbek_normalizer:
            key1 = self.uzbek_normalizer.generate_phonetic_key(text1)
            key2 = self.uzbek_normalizer.generate_phonetic_key(text2)
            
            if key1 == key2:
                return 100.0
            
            # Calculate similarity based on phonetic distance
            phonetic_distance = self.uzbek_normalizer.calculate_phonetic_distance(text1, text2)
            similarity = (1.0 - phonetic_distance) * 100
            
            # Apply additional Uzbek-specific rules
            similarity = self._apply_uzbek_phonetic_rules(text1, text2, similarity)
            
            return max(similarity, 0.0)
        else:
            # Fallback to character-based phonetic matching
            return self._character_phonetic_similarity(text1, text2)
    
    def _apply_uzbek_phonetic_rules(self, text1: str, text2: str, base_similarity: float) -> float:
        """Apply Uzbek-specific phonetic rules to boost similarity."""
        # Check for common Uzbek word patterns and variations
        words1 = text1.split()
        words2 = text2.split()
        
        if len(words1) != len(words2):
            return base_similarity
        
        word_similarities = []
        for w1, w2 in zip(words1, words2):
            # Check for common Uzbek suffixes and prefixes
            if self._are_uzbek_variants(w1, w2):
                word_similarities.append(90.0)  # High similarity for variants
            else:
                # Use character-based similarity
                word_sim = self._character_phonetic_similarity(w1, w2)
                word_similarities.append(word_sim)
        
        # Average word similarities
        avg_similarity = sum(word_similarities) / len(word_similarities)
        
        # Return the higher of base similarity and word-based similarity
        return max(base_similarity, avg_similarity)
    
    def _are_uzbek_variants(self, word1: str, word2: str) -> bool:
        """Check if two words are Uzbek variants of each other."""
        # Common Uzbek suffixes that might vary
        uzbek_suffixes = ['LAR', 'LAR', 'DA', 'TA', 'NI', 'GA', 'DAN', 'TAN']
        
        # Remove common suffixes and compare roots
        root1 = word1
        root2 = word2
        
        for suffix in uzbek_suffixes:
            if root1.endswith(suffix):
                root1 = root1[:-len(suffix)]
            if root2.endswith(suffix):
                root2 = root2[:-len(suffix)]
        
        # If roots are similar, consider them variants
        if root1 and root2:
            similarity = self._character_phonetic_similarity(root1, root2)
            return similarity >= 80.0
        
        return False
    
    def _character_phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity based on character groups."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        # Convert to phonetic representations
        phonetic1 = self._to_phonetic_representation(text1)
        phonetic2 = self._to_phonetic_representation(text2)
        
        # Calculate similarity using Levenshtein distance
        max_len = max(len(phonetic1), len(phonetic2))
        if max_len == 0:
            return 100.0
        
        distance = self._levenshtein_distance(phonetic1, phonetic2)
        similarity = (1.0 - distance / max_len) * 100
        return max(similarity, 0.0)
    
    def _to_phonetic_representation(self, text: str) -> str:
        """Convert text to phonetic representation using Uzbek phonetic groups."""
        phonetic = ""
        for char in text:
            if char in self.uzbek_phonetic_groups:
                # Use the first (canonical) character from the phonetic group
                phonetic += self.uzbek_phonetic_groups[char][0]
            else:
                phonetic += char
        return phonetic
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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
    
    def validate_config(self) -> List[str]:
        """
        Validate algorithm configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = super().validate_config()
        
        # Validate phonetic_method
        valid_methods = ['soundex', 'metaphone', 'uzbek_phonetic']
        if 'phonetic_method' in self.config:
            if self.config['phonetic_method'] not in valid_methods:
                errors.append(f"phonetic_method must be one of {valid_methods}")
        
        # Validate min_similarity
        if 'min_similarity' in self.config:
            min_sim = self.config['min_similarity']
            if not isinstance(min_sim, (int, float)) or min_sim < 0 or min_sim > 100:
                errors.append("min_similarity must be a number between 0 and 100")
        
        # Validate boolean parameters
        bool_params = ['normalize_uzbek', 'use_cache', 'strict_phonetic']
        for param in bool_params:
            if param in self.config and not isinstance(self.config[param], bool):
                errors.append(f"{param} must be a boolean value")
        
        return errors
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about this algorithm."""
        info = {
            'name': self.name,
            'type': self.algorithm_type.value,
            'description': 'Phonetic matching with Uzbek language support',
            'config': self.config,
            'features': [
                'Multiple phonetic methods (Soundex, Metaphone, Uzbek-specific)',
                'Uzbek phonetic rules and character groups',
                'Cyrillic/Latin script handling',
                'Word variant recognition',
                'LRU caching for performance'
            ],
            'performance': self.get_performance_stats()
        }
        
        # Add Uzbek normalizer stats if available
        if self.uzbek_normalizer:
            info['uzbek_normalizer_stats'] = self.uzbek_normalizer.get_normalization_stats()
        
        return info