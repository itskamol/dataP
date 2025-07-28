"""
Base classes for matching algorithms with pluggable architecture.
Implements requirements 1.1, 3.3, 4.1: Modular architecture with clear interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from ...infrastructure.logging import get_logger


class MatchingType(Enum):
    """Types of matching algorithms."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    PHONETIC = "phonetic"


@dataclass
class MatchingResult:
    """Result of a single matching operation."""
    similarity_score: float
    confidence: float
    matched_fields: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float
    
    @property
    def is_match(self) -> bool:
        """Check if this is considered a match based on confidence."""
        return self.confidence >= 50.0  # Default threshold


class MatchingAlgorithm(ABC):
    """
    Abstract base class for all matching algorithms.
    Provides pluggable architecture for different matching strategies.
    """
    
    def __init__(self, name: str, algorithm_type: MatchingType, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the matching algorithm.
        
        Args:
            name: Human-readable name for the algorithm
            algorithm_type: Type of matching algorithm
            config: Algorithm-specific configuration parameters
        """
        self.name = name
        self.algorithm_type = algorithm_type
        self.config = config or {}
        self.logger = get_logger(f'matching.{name.lower()}')
        
        # Performance tracking
        self._total_comparisons = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str, 
                           field_name: Optional[str] = None) -> MatchingResult:
        """
        Calculate similarity between two text values.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            field_name: Optional field name for context
            
        Returns:
            MatchingResult with similarity score and metadata
        """
        pass
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        pass
    
    def compare_records(self, record1: Dict[str, Any], record2: Dict[str, Any],
                       field_mappings: List[Tuple[str, str, float]]) -> MatchingResult:
        """
        Compare two records using field mappings.
        
        Args:
            record1: First record to compare
            record2: Second record to compare
            field_mappings: List of (field1, field2, weight) tuples
            
        Returns:
            Combined MatchingResult for all fields
        """
        start_time = time.time()
        
        total_weighted_score = 0.0
        total_weight = 0.0
        matched_fields = []
        field_results = {}
        
        for field1, field2, weight in field_mappings:
            # Get field values with safe access
            value1 = str(record1.get(field1, "")).strip()
            value2 = str(record2.get(field2, "")).strip()
            
            # Skip empty values
            if not value1 or not value2:
                continue
            
            # Calculate similarity for this field
            field_result = self.calculate_similarity(value1, value2, field1)
            
            # Apply weight
            weighted_score = field_result.similarity_score * weight
            total_weighted_score += weighted_score
            total_weight += weight
            
            # Track matched fields
            if field_result.is_match:
                matched_fields.append(field1)
            
            # Store field-specific result
            field_results[field1] = {
                'similarity': field_result.similarity_score,
                'confidence': field_result.confidence,
                'weight': weight,
                'weighted_score': weighted_score
            }
        
        # Calculate overall similarity and confidence
        if total_weight > 0:
            overall_similarity = total_weighted_score / total_weight
            # Confidence is based on similarity and number of matched fields
            confidence_boost = min(len(matched_fields) * 10, 30)  # Up to 30% boost
            overall_confidence = min(overall_similarity + confidence_boost, 100.0)
        else:
            overall_similarity = 0.0
            overall_confidence = 0.0
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update performance tracking
        self._total_comparisons += 1
        self._total_processing_time += processing_time
        
        return MatchingResult(
            similarity_score=overall_similarity,
            confidence=overall_confidence,
            matched_fields=matched_fields,
            metadata={
                'algorithm': self.name,
                'algorithm_type': self.algorithm_type.value,
                'field_results': field_results,
                'total_fields_compared': len(field_mappings),
                'matched_fields_count': len(matched_fields)
            },
            processing_time_ms=processing_time
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this algorithm."""
        avg_processing_time = (
            self._total_processing_time / self._total_comparisons 
            if self._total_comparisons > 0 else 0.0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses) * 100
            if (self._cache_hits + self._cache_misses) > 0 else 0.0
        )
        
        return {
            'algorithm_name': self.name,
            'algorithm_type': self.algorithm_type.value,
            'total_comparisons': self._total_comparisons,
            'total_processing_time_ms': self._total_processing_time,
            'average_processing_time_ms': avg_processing_time,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate_percent': cache_hit_rate
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self._total_comparisons = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _track_cache_hit(self):
        """Track a cache hit for performance monitoring."""
        self._cache_hits += 1
    
    def _track_cache_miss(self):
        """Track a cache miss for performance monitoring."""
        self._cache_misses += 1
    
    def validate_config(self) -> List[str]:
        """
        Validate algorithm configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation - subclasses can override
        if not self.name:
            errors.append("Algorithm name is required")
        
        return errors
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.algorithm_type.value}')"
    
    def __repr__(self) -> str:
        return self.__str__()