"""
Uzbek text normalization with character mapping and phonetic rules.
Implements requirements 4.3: Optimize Uzbek text normalization with character mapping and phonetic rules.
"""

import re
import unicodedata
from typing import Dict, Set, Optional
from functools import lru_cache

from ...infrastructure.logging import get_logger


class UzbekTextNormalizer:
    """
    Specialized text normalizer for Uzbek language with character mapping and phonetic rules.
    Handles Cyrillic to Latin conversion, common character variations, and phonetic similarities.
    """
    
    def __init__(self):
        self.logger = get_logger('matching.uzbek_normalizer')
        
        # Cyrillic to Latin character mapping for Uzbek
        self.cyrillic_to_latin = {
            'А': 'A', 'а': 'a',
            'Б': 'B', 'б': 'b',
            'В': 'V', 'в': 'v',
            'Г': 'G', 'г': 'g',
            'Д': 'D', 'д': 'd',
            'Е': 'E', 'е': 'e',
            'Ё': 'Yo', 'ё': 'yo',
            'Ж': 'J', 'ж': 'j',
            'З': 'Z', 'з': 'z',
            'И': 'I', 'и': 'i',
            'Й': 'Y', 'й': 'y',
            'К': 'K', 'к': 'k',
            'Л': 'L', 'л': 'l',
            'М': 'M', 'м': 'm',
            'Н': 'N', 'н': 'n',
            'О': 'O', 'о': 'o',
            'П': 'P', 'п': 'p',
            'Р': 'R', 'р': 'r',
            'С': 'S', 'с': 's',
            'Т': 'T', 'т': 't',
            'У': 'U', 'у': 'u',
            'Ф': 'F', 'ф': 'f',
            'Х': 'X', 'х': 'x',
            'Ц': 'Ts', 'ц': 'ts',
            'Ч': 'Ch', 'ч': 'ch',
            'Ш': 'SH', 'ш': 'sh',
            'Щ': 'Shch', 'щ': 'shch',
            'Ъ': '', 'ъ': '',  # Hard sign - usually omitted
            'Ы': 'I', 'ы': 'i',
            'Ь': '', 'ь': '',  # Soft sign - usually omitted
            'Э': 'E', 'э': 'e',
            'Ю': 'Yu', 'ю': 'yu',
            'Я': 'Ya', 'я': 'ya',
            # Uzbek-specific Cyrillic letters
            'Ў': 'O\'', 'ў': 'o\'',  # O with breve
            'Қ': 'Q', 'қ': 'q',      # Ka with descender
            'Ғ': 'G\'', 'ғ': 'g\'',  # Ghayn
            'Ҳ': 'H', 'ҳ': 'h',      # Ha with tail
        }
        
        # Latin character variations and normalizations
        self.latin_normalizations = {
            'O\'': 'O', 'o\'': 'o',  # O with apostrophe
            'G\'': 'G', 'g\'': 'g',  # G with apostrophe
            'Sh': 'sh', 'SH': 'sh',  # Standardize sh to lowercase
            'Ch': 'ch', 'CH': 'ch',  # Standardize ch to lowercase
            'Ng': 'ng', 'NG': 'ng',  # Standardize ng to lowercase
            'Yo': 'yo', 'YO': 'yo',  # Standardize yo to lowercase
            'Yu': 'yu', 'YU': 'yu',  # Standardize yu to lowercase
            'Ya': 'ya', 'YA': 'ya',  # Standardize ya to lowercase
        }
        
        # Phonetically similar characters for fuzzy matching
        self.phonetic_similarities = {
            'i': ['i', 'y', 'ы'],
            'y': ['y', 'i', 'ы'],
            'ы': ['ы', 'i', 'y'],
            'o': ['o', 'u', 'ў'],
            'u': ['u', 'o', 'ў'],
            'ў': ['ў', 'o', 'u'],
            'k': ['k', 'q', 'қ'],
            'q': ['q', 'k', 'қ'],
            'қ': ['қ', 'k', 'q'],
            'g': ['g', 'ғ'],
            'ғ': ['ғ', 'g'],
            'h': ['h', 'ҳ', 'x'],
            'ҳ': ['ҳ', 'h', 'x'],
            'x': ['x', 'h', 'ҳ'],
            's': ['s', 'z'],
            'z': ['z', 's'],
            'b': ['b', 'p'],
            'p': ['p', 'b'],
            'd': ['d', 't'],
            't': ['t', 'd'],
            'v': ['v', 'w'],
            'w': ['w', 'v'],
        }
        
        # Common word variations and abbreviations
        self.word_variations = {
            'мчж': ['мчж', 'mchj', 'махалла'],
            'mchj': ['mchj', 'мчж', 'mahalla'],
            'махалла': ['махалла', 'mahalla', 'мчж', 'mchj'],
            'mahalla': ['mahalla', 'махалла', 'mchj', 'мчж'],
            'туман': ['туман', 'tuman', 'район'],
            'tuman': ['tuman', 'туман', 'rayon'],
            'район': ['район', 'rayon', 'туман'],
            'rayon': ['rayon', 'район', 'tuman'],
            'вилоят': ['вилоят', 'viloyat', 'область'],
            'viloyat': ['viloyat', 'вилоят', 'oblast'],
            'область': ['область', 'oblast', 'вилоят'],
            'oblast': ['oblast', 'область', 'viloyat'],
        }
        
        # Precompile regex patterns for efficiency
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\d+')
    
    @lru_cache(maxsize=5000)
    def normalize_text(self, text: str, aggressive: bool = False) -> str:
        """
        Normalize Uzbek text for comparison.
        
        Args:
            text: Text to normalize
            aggressive: Whether to apply aggressive normalization
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Start with the original text
        normalized = text.strip()
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Unicode normalization (NFC form)
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Convert Cyrillic to Latin
        normalized = self._convert_cyrillic_to_latin(normalized)
        
        # Apply Latin normalizations
        normalized = self._apply_latin_normalizations(normalized)
        
        # Remove or normalize punctuation
        if aggressive:
            normalized = self.punctuation_pattern.sub('', normalized)
        else:
            normalized = self.punctuation_pattern.sub(' ', normalized)
        
        # Normalize whitespace
        normalized = self.whitespace_pattern.sub(' ', normalized)
        
        # Apply word-level normalizations
        normalized = self._normalize_words(normalized)
        
        return normalized.strip()
    
    def _convert_cyrillic_to_latin(self, text: str) -> str:
        """Convert Cyrillic characters to Latin equivalents."""
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            if char in self.cyrillic_to_latin:
                result.append(self.cyrillic_to_latin[char])
            else:
                result.append(char)
            i += 1
        return ''.join(result)
    
    def _apply_latin_normalizations(self, text: str) -> str:
        """Apply Latin character normalizations."""
        # First handle digraphs and special sequences
        for original, normalized in self.latin_normalizations.items():
            text = text.replace(original, normalized)
        
        # Handle common transliteration variations
        # These help match different ways of writing the same Uzbek sounds
        transliteration_map = {
            'tosh': 'tash',  # Common variation for Tashkent
            'kh': 'x',       # kh -> x
            'ts': 'c',       # ts -> c
            'iy': 'i',       # iy -> i
            'ey': 'e',       # ey -> e
        }
        
        for original, normalized in transliteration_map.items():
            text = text.replace(original, normalized)
        
        return text
    
    def _normalize_words(self, text: str) -> str:
        """Apply word-level normalizations and handle common variations."""
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Check for exact word variations
            if word in self.word_variations:
                # Use the first (canonical) form
                normalized_words.append(self.word_variations[word][0])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    @lru_cache(maxsize=2000)
    def generate_phonetic_key(self, text: str) -> str:
        """
        Generate a phonetic key for Uzbek text to improve blocking accuracy.
        
        Args:
            text: Text to generate phonetic key for
            
        Returns:
            Phonetic key string
        """
        if not text:
            return ""
        
        # Start with normalized text
        normalized = self.normalize_text(text, aggressive=True)
        
        # Generate phonetic key by replacing similar sounds
        phonetic_key = []
        for char in normalized:
            if char in self.phonetic_similarities:
                # Use the first (canonical) character from similar group
                phonetic_key.append(self.phonetic_similarities[char][0])
            else:
                phonetic_key.append(char)
        
        # Remove consecutive duplicates
        result = []
        prev_char = None
        for char in phonetic_key:
            if char != prev_char:
                result.append(char)
                prev_char = char
        
        return ''.join(result)
    
    def get_character_variations(self, char: str) -> Set[str]:
        """
        Get all possible variations of a character.
        
        Args:
            char: Character to get variations for
            
        Returns:
            Set of character variations
        """
        variations = {char}  # Include the original character
        
        # Add phonetic similarities
        if char in self.phonetic_similarities:
            variations.update(self.phonetic_similarities[char])
        
        # Add reverse mappings from Cyrillic
        for cyrillic, latin in self.cyrillic_to_latin.items():
            if latin == char:
                variations.add(cyrillic)
        
        return variations
    
    def calculate_phonetic_distance(self, text1: str, text2: str) -> float:
        """
        Calculate phonetic distance between two Uzbek texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Phonetic distance (0.0 = identical, 1.0 = completely different)
        """
        key1 = self.generate_phonetic_key(text1)
        key2 = self.generate_phonetic_key(text2)
        
        if not key1 and not key2:
            return 0.0
        if not key1 or not key2:
            return 1.0
        
        # Simple Levenshtein distance on phonetic keys
        return self._levenshtein_distance(key1, key2) / max(len(key1), len(key2))
    
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
    
    def is_uzbek_text(self, text: str) -> bool:
        """
        Determine if text contains Uzbek characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be Uzbek
        """
        if not text:
            return False
        
        # Uzbek-specific characters (not common in other languages)
        uzbek_specific_chars = {'ў', 'қ', 'ғ', 'ҳ', 'Ў', 'Қ', 'Ғ', 'Ҳ', "'"}
        
        # Common Cyrillic characters used in Uzbek
        uzbek_cyrillic_chars = set(self.cyrillic_to_latin.keys())
        
        # Uzbek words and patterns
        uzbek_words = set(self.word_variations.keys())
        
        # Check for Uzbek-specific characters
        for char in text:
            if char in uzbek_specific_chars:
                return True
        
        # Check for Uzbek-specific words
        words = text.lower().split()
        for word in words:
            if word in uzbek_words:
                return True
        
        # Check for common Uzbek patterns (apostrophe usage)
        if "o'" in text.lower() or "g'" in text.lower():
            return True
        
        # Check for Cyrillic characters (could be Uzbek Cyrillic)
        cyrillic_count = sum(1 for char in text if char in uzbek_cyrillic_chars)
        if cyrillic_count > 0:
            return True
        
        # Check for common Uzbek city/place names
        uzbek_places = {'tashkent', 'toshkent', 'ташкент', 'samarkand', 'самарканд', 
                       'bukhara', 'бухара', 'andijan', 'андижан', 'namangan', 'наманган'}
        text_lower = text.lower()
        for place in uzbek_places:
            if place in text_lower:
                return True
        
        return False
    
    def get_normalization_stats(self) -> Dict[str, int]:
        """Get statistics about normalization cache usage."""
        return {
            'normalize_text_cache_size': self.normalize_text.cache_info().currsize,
            'normalize_text_cache_hits': self.normalize_text.cache_info().hits,
            'normalize_text_cache_misses': self.normalize_text.cache_info().misses,
            'phonetic_key_cache_size': self.generate_phonetic_key.cache_info().currsize,
            'phonetic_key_cache_hits': self.generate_phonetic_key.cache_info().hits,
            'phonetic_key_cache_misses': self.generate_phonetic_key.cache_info().misses,
        }
    
    def clear_cache(self):
        """Clear normalization caches."""
        self.normalize_text.cache_clear()
        self.generate_phonetic_key.cache_clear()
        self.logger.info("Uzbek normalizer caches cleared")