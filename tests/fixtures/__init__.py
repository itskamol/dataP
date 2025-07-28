"""
Test fixtures for reproducible testing environments.
Provides common test data, configurations, and utilities.
"""

import os
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import json

# Test data directory
FIXTURES_DIR = Path(__file__).parent
TEST_DATA_DIR = FIXTURES_DIR / "data"
CONFIG_DIR = FIXTURES_DIR / "configs"

# Ensure directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)


class TestFixtures:
    """Central class for managing test fixtures."""
    
    @staticmethod
    def get_sample_person_data() -> pd.DataFrame:
        """Get sample person data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'age': [25, 30, 35, 28, 42],
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 
                     'alice@example.com', 'charlie@example.com'],
            'phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', '+1-555-0104', '+1-555-0105']
        })
    
    @staticmethod
    def get_sample_organization_data() -> pd.DataFrame:
        """Get sample organization data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Tech Corp', 'Data Solutions LLC', 'AI Innovations Inc', 'Smart Systems', 'Future Tech'],
            'industry': ['Technology', 'Consulting', 'AI/ML', 'Software', 'Research'],
            'employees': [100, 50, 200, 75, 150],
            'city': ['San Francisco', 'New York', 'Boston', 'Seattle', 'Austin'],
            'revenue': [1000000, 500000, 2000000, 750000, 1500000]
        })
    
    @staticmethod
    def get_uzbek_test_data() -> pd.DataFrame:
        """Get Uzbek text data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name_latin': ['Abdulla Karimov', 'Gulnora Rahimova', 'Bobur Toshmatov', 
                          'Malika Yusupova', 'Davron Ismoilov'],
            'name_cyrillic': ['Абдулла Каримов', 'Гулнора Рахимова', 'Бобур Тошматов',
                             'Малика Юсупова', 'Даврон Исмоилов'],
            'city_latin': ['Toshkent', 'Samarkand', 'Buxoro', 'Andijon', 'Namangan'],
            'city_cyrillic': ['Ташкент', 'Самарканд', 'Бухара', 'Андижан', 'Наманган'],
            'region': ['Toshkent shahri', 'Samarqand viloyati', 'Buxoro viloyati',
                      'Andijon viloyati', 'Namangan viloyati']
        })
    
    @staticmethod
    def get_matching_config() -> Dict[str, Any]:
        """Get standard matching configuration for testing."""
        return {
            'mappings': [
                {
                    'source_field': 'name',
                    'target_field': 'name',
                    'algorithm': 'fuzzy',
                    'weight': 1.0,
                    'normalization': True,
                    'case_sensitive': False
                }
            ],
            'algorithms': [
                {
                    'name': 'fuzzy',
                    'algorithm_type': 'fuzzy',
                    'parameters': {'similarity_method': 'combined', 'min_similarity': 75.0},
                    'enabled': True,
                    'priority': 1
                },
                {
                    'name': 'exact',
                    'algorithm_type': 'exact',
                    'parameters': {'case_sensitive': False},
                    'enabled': True,
                    'priority': 2
                }
            ],
            'thresholds': {'minimum_confidence': 75.0},
            'matching_type': 'one-to-one',
            'confidence_threshold': 75.0,
            'use_blocking': True,
            'parallel_processing': False
        }
    
    @staticmethod
    def get_application_config() -> Dict[str, Any]:
        """Get standard application configuration for testing."""
        return {
            'file1': {
                'path': 'test_file1.csv',
                'file_type': 'csv',
                'delimiter': ',',
                'encoding': 'utf-8'
            },
            'file2': {
                'path': 'test_file2.csv',
                'file_type': 'csv',
                'delimiter': ',',
                'encoding': 'utf-8'
            },
            'matching': TestFixtures.get_matching_config(),
            'output': {
                'format': 'csv',
                'path': 'test_results',
                'include_unmatched': True,
                'include_confidence_scores': True,
                'file_prefix': 'test_'
            },
            'logging_level': 'INFO',
            'max_workers': 2,
            'memory_limit_mb': 512,
            'timeout_seconds': 300
        }
    
    @staticmethod
    def create_temp_csv_file(data: pd.DataFrame, filename: str = None) -> Path:
        """Create a temporary CSV file with test data."""
        if filename is None:
            fd, temp_path = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
            temp_path = Path(temp_path)
        else:
            temp_path = Path(tempfile.gettempdir()) / filename
        
        data.to_csv(temp_path, index=False)
        return temp_path
    
    @staticmethod
    def create_temp_json_file(data: pd.DataFrame, filename: str = None) -> Path:
        """Create a temporary JSON file with test data."""
        if filename is None:
            fd, temp_path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            temp_path = Path(temp_path)
        else:
            temp_path = Path(tempfile.gettempdir()) / filename
        
        data.to_json(temp_path, orient='records', indent=2)
        return temp_path
    
    @staticmethod
    def create_temp_config_file(config: Dict[str, Any], filename: str = None) -> Path:
        """Create a temporary configuration file."""
        if filename is None:
            fd, temp_path = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            temp_path = Path(temp_path)
        else:
            temp_path = Path(tempfile.gettempdir()) / filename
        
        with open(temp_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return temp_path
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[Path]):
        """Clean up temporary files."""
        for path in file_paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {path}: {e}")


# Pre-created fixture data files
def create_fixture_files():
    """Create standard fixture files."""
    fixtures = TestFixtures()
    
    # Create sample data files
    person_data = fixtures.get_sample_person_data()
    person_data.to_csv(TEST_DATA_DIR / "sample_persons.csv", index=False)
    person_data.to_json(TEST_DATA_DIR / "sample_persons.json", orient='records', indent=2)
    
    org_data = fixtures.get_sample_organization_data()
    org_data.to_csv(TEST_DATA_DIR / "sample_organizations.csv", index=False)
    
    uzbek_data = fixtures.get_uzbek_test_data()
    uzbek_data.to_csv(TEST_DATA_DIR / "uzbek_test_data.csv", index=Fal