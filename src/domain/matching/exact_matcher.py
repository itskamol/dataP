"""
Exact matching algorithm implementation.
Implements requirements 1.1, 3.3, 4.1: Exact string matching with normalization and caching.
"""

from typing import Dict, Any, Optional
import time

from .base import MatchingAlgorithm, MatchingResult, MatchingType
from .cache import get_global_cache
from .uzbek_normalizer import UzbekTextNormalizer


class ExactMatcher(MatchingAlgorithm):
    """
    Exact matching algorithm that compares normalized text for exact matches.
    Supports case-insensitive matching and Uzbek text normalization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the exact matcher.
        
        Args:
            config: Configuration parameters including:
                - case_sensitive: Whether to perform case-sensitive matching (default: False)
                - normalize_uzbek: Whether to apply Uzbek normalization (default: True)
                - use_cache: Whether to use caching (default: True)
        """
        super().__init__("ExactMatcher", MatchingType.EXACT, config)
        
        # Configuration parameters
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.normalize_uzbek = self.config.get('normalize_uzbek', True)
        self.use_cache = self.config.get('use_cache', True)
        
        # Initialize Uzbek normalizer if needed
        self.uzbek_normalizer = UzbekTextNormalizer() if self.normalize_uzbek else None
        
        # Get global cache instance
        self.cache = get_global_cache() if self.use_cache else None
        
        self.logger.info(f"ExactMatcher initialized", extra={
            'case_sensitive': self.case_sensitive,
            'normalize_uzbek': self.normalize_uzbek,
            'use_cache': self.use_cache
        })
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for exact comparison.
        
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
        
        # Apply Uzbek normalization if enabled (before case normalization)
        if self.normalize_uzbek and self.uzbek_normalizer:
            # Check if text appears to be Uzbek
            if self.uzbek_normalizer.is_uzbek_text(text):  # Use original text for detection
                normalized = self.uzbek_normalizer.normalize_text(normalized, aggressive=False)
            else:
                # Apply basic normalization for non-Uzbek text
                import unicodedata
                import re
                normalized = unicodedata.normalize('NFC', normalized)
                # Normalize whitespace for non-Uzbek text too
                normalized = re.sub(r'\s+', ' ', normalized)
        
        # Apply case normalization if not case sensitive
        if not self.case_sensitive:
            normalized = normalized.lower()
        
        # Cache the result
        if self.cache:
            self.cache.set_normalized_text(text, self.name, normalized, self.config)
        
        return normalized
    
    def calculate_similarity(self, text1: str, text2: str, 
                           field_name: Optional[str] = None) -> MatchingResult:
        """
        Calculate similarity between two text values using exact matching.
        
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
                    'normalized_text1': '',
                    'normalized_text2': ''
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
                    'normalized_text1': str(text1),
                    'normalized_text2': str(text2)
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
                return MatchingResult(
                    similarity_score=cached_result,
                    confidence=cached_result,
                    matched_fields=[field_name] if field_name and cached_result == 100.0 else [],
                    metadata={
                        'algorithm': self.name,
                        'match_type': 'exact' if cached_result == 100.0 else 'no_match',
                        'cached': True
                    },
                    processing_time_ms=processing_time
                )
            else:
                self._track_cache_miss()
        
        # Normalize both texts
        normalized_text1 = self.normalize_text(text1)
        normalized_text2 = self.normalize_text(text2)
        
        # Perform exact comparison
        is_exact_match = normalized_text1 == normalized_text2
        similarity_score = 100.0 if is_exact_match else 0.0
        confidence = similarity_score
        
        # Cache the result
        if self.cache:
            self.cache.set_similarity(text1, text2, self.name, similarity_score, cache_key_config)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update performance tracking
        self._total_comparisons += 1
        self._total_processing_time += processing_time
        
        return MatchingResult(
            similarity_score=similarity_score,
            confidence=confidence,
            matched_fields=[field_name] if field_name and is_exact_match else [],
            metadata={
                'algorithm': self.name,
                'match_type': 'exact' if is_exact_match else 'no_match',
                'normalized_text1': normalized_text1,
                'normalized_text2': normalized_text2,
                'original_text1': text1,
                'original_text2': text2,
                'case_sensitive': self.case_sensitive,
                'uzbek_normalized': self.normalize_uzbek,
                'cached': False
            },
            processing_time_ms=processing_time
        )
    
    def validate_config(self) -> list[str]:
        """
        Validate algorithm configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = super().validate_config()
        
        # Validate case_sensitive parameter
        if 'case_sensitive' in self.config:
            if not isinstance(self.config['case_sensitive'], bool):
                errors.append("case_sensitive must be a boolean value")
        
        # Validate normalize_uzbek parameter
        if 'normalize_uzbek' in self.config:
            if not isinstance(self.config['normalize_uzbek'], bool):
                errors.append("normalize_uzbek must be a boolean value")
        
        # Validate use_cache parameter
        if 'use_cache' in self.config:
            if not isinstance(self.config['use_cache'], bool):
                errors.append("use_cache must be a boolean value")
        
        return errors
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about this algorithm."""
        info = {
            'name': self.name,
            'type': self.algorithm_type.value,
            'description': 'Exact string matching with normalization support',
            'config': self.config,
            'features': [
                'Case-sensitive/insensitive matching',
                'Uzbek text normalization',
                'Unicode normalization',
                'LRU caching for performance'
            ],
            'performance': self.get_performance_stats()
        }
        
        # Add Uzbek normalizer stats if available
        if self.uzbek_normalizer:
            info['uzbek_normalizer_stats'] = self.uzbek_normalizer.get_normalization_stats()
        
        return info