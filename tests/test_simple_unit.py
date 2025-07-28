"""
Simple unit tests to verify testing framework works.
"""

import unittest
import sys
import os

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from domain.models import FieldMapping, AlgorithmType
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


class TestSimpleUnit(unittest.TestCase):
    """Simple unit tests to verify framework."""
    
    def test_basic_functionality(self):
        """Test basic Python functionality."""
        self.assertEqual(1 + 1, 2)
        self.assertTrue(True)
        self.assertFalse(False)
    
    def test_string_operations(self):
        """Test string operations."""
        text = "Hello World"
        self.assertEqual(text.lower(), "hello world")
        self.assertEqual(text.upper(), "HELLO WORLD")
        self.assertTrue(text.startswith("Hello"))
    
    def test_list_operations(self):
        """Test list operations."""
        test_list = [1, 2, 3, 4, 5]
        self.assertEqual(len(test_list), 5)
        self.assertIn(3, test_list)
        self.assertEqual(test_list[0], 1)
        self.assertEqual(test_list[-1], 5)
    
    def test_dict_operations(self):
        """Test dictionary operations."""
        test_dict = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(len(test_dict), 3)
        self.assertIn('a', test_dict)
        self.assertEqual(test_dict['a'], 1)
        self.assertEqual(list(test_dict.keys()), ['a', 'b', 'c'])
    
    @unittest.skipUnless(MODELS_AVAILABLE, "Models not available")
    def test_model_import(self):
        """Test that we can import models."""
        mapping = FieldMapping(
            source_field="test",
            target_field="test",
            algorithm=AlgorithmType.EXACT
        )
        self.assertEqual(mapping.source_field, "test")
        self.assertEqual(mapping.target_field, "test")
    
    def test_exception_handling(self):
        """Test exception handling."""
        with self.assertRaises(ValueError):
            raise ValueError("Test error")
        
        with self.assertRaises(KeyError):
            test_dict = {'a': 1}
            _ = test_dict['b']
    
    def test_file_operations(self):
        """Test basic file operations."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                content = f.read()
            self.assertEqual(content, "test content")
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)