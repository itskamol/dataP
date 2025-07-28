"""
Matching algorithms package for the file processing system.
Contains base classes and implementations for various matching strategies.
"""

from .base import MatchingAlgorithm, MatchingResult as AlgorithmResult
from .exact_matcher import ExactMatcher
from .fuzzy_matcher import FuzzyMatcher
from .phonetic_matcher import PhoneticMatcher
from .cache import MatchingCache
from .uzbek_normalizer import UzbekTextNormalizer

__all__ = [
    'MatchingAlgorithm',
    'AlgorithmResult',
    'ExactMatcher',
    'FuzzyMatcher',
    'PhoneticMatcher',
    'MatchingCache',
    'UzbekTextNormalizer'
]