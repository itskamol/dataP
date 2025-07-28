"""
Fuzzy matching algorithm implementation with optimized similarity calculations.
Implements requirements 1.1, 3.3, 4.1: Fuzzy string matching with caching and Uzbek support.
"""

from typing import Dict, Any, Optional, List
import time
import difflib

from .base import MatchingAlgorithm, MatchingResult, MatchingType
from .cache import get_global_cache
from .uzbek_normalizer import UzbekTextNormalizer


class FuzzyMatcher(MatchingAlgorithm):
    """
    Fuzzy matching algorithm using various similarity metrics.
    Supports Levenshtein distance, Jaro-Winkler, and sequence matching.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fuzzy matcher.
        
        Args:
            config: Configuration parameters including:
                - similarity_method: Method to use ('levenshtein', 'jaro_winkler', 'sequence', 'combined')
                - min_similarity: Minimum similarity threshold (default: 60.0)
                - normalize_uzbek: Whether to apply Uzbek normalization (default: True)
                - use_cache: Whether to use caching (default: True)
                - case_sensitive: Whether to perform case-sensitive matching (default: False)
                - phonetic_boost: Whether to boost phonetically similar matches (default: True)
        """
        super().__init__("FuzzyMatcher", MatchingType.FUZZY, config)
        
        # Configuration parameters
        self.similarity_method = self.config.get('similarity_method', 'combined')
        self.min_similarity = self.config.get('min_similarity', 60.0)
        self.normalize_uzbek = self.config.get('normalize_uzbek', True)
        self.use_cache = self.config.get('use_cache', True)
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.phonetic_boost = self.config.get('phonetic_boost', True)
        
        # Initialize Uzbek normalizer if needed
        self.uzbek_normalizer = UzbekTextNormalizer() if self.normalize_uzbek else None
        
        # Get global cache instance
        self.cache = get_global_cache() if self.use_cache else None
        
        # Validate similarity method
        valid_methods = ['levenshtein', 'jaro_winkler', 'sequence', 'combined']
        if self.similarity_method not in valid_methods:
            self.logger.warning(f"Invalid similarity method '{self.similarity_method}', using 'combined'")
            self.similarity_method = 'combined'
        
        self.logger.info(f"FuzzyMatcher initialized", extra={
            'similarity_method': self.similarity_method,
            'min_similarity': self.min_similarity,
            'normalize_uzbek': self.normalize_uzbek,
            'use_cache': self.use_cache,
            'case_sensitive': self.case_sensitive,
            'phonetic_boost': self.phonetic_boost
        })
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for fuzzy comparison.
        
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
        normalized = text.strip()
        
        # Apply case normalization if not case sensitive
        if not self.case_sensitive:
            normalized = normalized.lower()
        
        # Apply Uzbek normalization if enabled
        if self.normalize_uzbek and self.uzbek_normalizer:
            if self.uzbek_normalizer.is_uzbek_text(normalized):
                normalized = self.uzbek_normalizer.normalize_text(normalized, aggressive=True)
            else:
                # Apply basic normalization for non-Uzbek text
                import unicodedata
                import re
                normalized = unicodedata.normalize('NFC', normalized)
                # Remove extra whitespace and punctuation for fuzzy matching
                normalized = re.sub(r'[^\w\s]', ' ', normalized)
                normalized = re.sub(r'\s+', ' ', normalized)
        
        # Cache the result
        if self.cache:
            self.cache.set_normalized_text(text, self.name, normalized, self.config)
        
        return normalized.strip()
    
    def calculate_similarity(self, text1: str, text2: str, 
                           field_name: Optional[str] = None) -> MatchingResult:
        """
        Calculate fuzzy similarity between two text values.
        
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
                    'similarity_method': self.similarity_method
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
                    'similarity_method': self.similarity_method
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
                        'similarity_method': self.similarity_method,
                        'cached': True
                    },
                    processing_time_ms=processing_time
                )
            else:
                self._track_cache_miss()
        
        # Normalize both texts
        normalized_text1 = self.normalize_text(text1)
        normalized_text2 = self.normalize_text(text2)
        
        # Calculate similarity based on method
        if self.similarity_method == 'levenshtein':
            similarity_score = self._levenshtein_similarity(normalized_text1, normalized_text2)
        elif self.similarity_method == 'jaro_winkler':
            similarity_score = self._jaro_winkler_similarity(normalized_text1, normalized_text2)
        elif self.similarity_method == 'sequence':
            similarity_score = self._sequence_similarity(normalized_text1, normalized_text2)
        else:  # combined
            similarity_score = self._combined_similarity(normalized_text1, normalized_text2)
        
        # Apply phonetic boost if enabled and texts are Uzbek
        phonetic_boost_applied = False
        if (self.phonetic_boost and self.uzbek_normalizer and 
            self.uzbek_normalizer.is_uzbek_text(text1) and 
            self.uzbek_normalizer.is_uzbek_text(text2)):
            
            phonetic_distance = self.uzbek_normalizer.calculate_phonetic_distance(text1, text2)
            phonetic_similarity = (1.0 - phonetic_distance) * 100
            
            # Boost similarity if phonetic similarity is high
            if phonetic_similarity > similarity_score:
                boost_amount = min((phonetic_similarity - similarity_score) * 0.3, 15.0)
                similarity_score = min(similarity_score + boost_amount, 100.0)
                phonetic_boost_applied = True
        
        # Calculate confidence (slightly lower than similarity for fuzzy matches)
        confidence = max(similarity_score - 5.0, 0.0) if similarity_score < 100.0 else 100.0
        
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
                'similarity_method': self.similarity_method,
                'normalized_text1': normalized_text1,
                'normalized_text2': normalized_text2,
                'original_text1': text1,
                'original_text2': text2,
                'min_similarity_threshold': self.min_similarity,
                'phonetic_boost_applied': phonetic_boost_applied,
                'cached': False
            },
            processing_time_ms=processing_time
        )
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using Levenshtein distance."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 100.0
        
        distance = self._levenshtein_distance(text1, text2)
        similarity = (1.0 - distance / max_len) * 100
        return max(similarity, 0.0)
    
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
    
    def _jaro_winkler_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using Jaro-Winkler algorithm."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        # Simplified Jaro-Winkler implementation
        jaro_sim = self._jaro_similarity(text1, text2)
        
        # Calculate common prefix length (up to 4 characters)
        prefix_len = 0
        for i in range(min(len(text1), len(text2), 4)):
            if text1[i] == text2[i]:
                prefix_len += 1
            else:
                break
        
        # Winkler modification
        jaro_winkler_sim = jaro_sim + (0.1 * prefix_len * (1 - jaro_sim))
        return jaro_winkler_sim * 100
    
    def _jaro_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaro similarity."""
        if text1 == text2:
            return 1.0
        
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Calculate the match window
        match_window = max(len1, len2) // 2 - 1
        if match_window < 0:
            match_window = 0
        
        # Initialize match arrays
        text1_matches = [False] * len1
        text2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)
            
            for j in range(start, end):
                if text2_matches[j] or text1[i] != text2[j]:
                    continue
                text1_matches[i] = text2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not text1_matches[i]:
                continue
            while not text2_matches[k]:
                k += 1
            if text1[i] != text2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
        return jaro
    
    def _sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sequence matching."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        # Use difflib's SequenceMatcher
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio() * 100
    
    def _combined_similarity(self, text1: str, text2: str) -> float:
        """Calculate combined similarity using multiple methods."""
        if not text1 and not text2:
            return 100.0
        if not text1 or not text2:
            return 0.0
        
        # Calculate similarities using different methods
        levenshtein_sim = self._levenshtein_similarity(text1, text2)
        jaro_winkler_sim = self._jaro_winkler_similarity(text1, text2)
        sequence_sim = self._sequence_similarity(text1, text2)
        
        # Weighted combination (can be adjusted based on performance)
        combined_sim = (
            levenshtein_sim * 0.4 +
            jaro_winkler_sim * 0.4 +
            sequence_sim * 0.2
        )
        
        return combined_sim
    
    def validate_config(self) -> List[str]:
        """
        Validate algorithm configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = super().validate_config()
        
        # Validate similarity_method
        valid_methods = ['levenshtein', 'jaro_winkler', 'sequence', 'combined']
        if 'similarity_method' in self.config:
            if self.config['similarity_method'] not in valid_methods:
                errors.append(f"similarity_method must be one of {valid_methods}")
        
        # Validate min_similarity
        if 'min_similarity' in self.config:
            min_sim = self.config['min_similarity']
            if not isinstance(min_sim, (int, float)) or min_sim < 0 or min_sim > 100:
                errors.append("min_similarity must be a number between 0 and 100")
        
        # Validate boolean parameters
        bool_params = ['normalize_uzbek', 'use_cache', 'case_sensitive', 'phonetic_boost']
        for param in bool_params:
            if param in self.config and not isinstance(self.config[param], bool):
                errors.append(f"{param} must be a boolean value")
        
        return errors
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about this algorithm."""
        info = {
            'name': self.name,
            'type': self.algorithm_type.value,
            'description': 'Fuzzy string matching with multiple similarity methods',
            'config': self.config,
            'features': [
                'Multiple similarity methods (Levenshtein, Jaro-Winkler, Sequence)',
                'Combined similarity scoring',
                'Uzbek text normalization and phonetic boosting',
                'Configurable similarity thresholds',
                'LRU caching for performance'
            ],
            'performance': self.get_performance_stats()
        }
        
        # Add Uzbek normalizer stats if available
        if self.uzbek_normalizer:
            info['uzbek_normalizer_stats'] = self.uzbek_normalizer.get_normalization_stats()
        
        return info