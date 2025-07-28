"""
Synthetic test data generators for various testing scenarios.
Tests requirements 4.1, 4.3: Test data generators for reproducible testing environments.
"""

import random
import string
import pandas as pd
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid


class TestDataGenerator:
    """Base class for generating synthetic test data."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def generate_random_string(self, length: int = 10, charset: str = None) -> str:
        """Generate a random string of specified length."""
        if charset is None:
            charset = string.ascii_letters + string.digits
        return ''.join(random.choices(charset, k=length))
    
    def generate_random_name(self) -> str:
        """Generate a random person name."""
        first_names = [
            'John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Emily',
            'James', 'Jessica', 'William', 'Ashley', 'Richard', 'Amanda', 'Thomas', 'Jennifer'
        ]
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Taylor'
        ]
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def generate_uzbek_name(self) -> str:
        """Generate a random Uzbek name."""
        uzbek_first_names = [
            'Abdulla', 'Akmal', 'Aziz', 'Bobur', 'Davron', 'Eldor', 'Farrux', 'Gulnora',
            'Хуршида', 'Ирода', 'Жамшид', 'Камола', 'Лола', 'Мадина', 'Нодира', 'Озода',
            'Muhammad', 'Ahmad', 'Ali', 'Fatima', 'Aisha', 'Omar', 'Hassan', 'Zainab'
        ]
        uzbek_last_names = [
            'Karimov', 'Rahimov', 'Toshmatov', 'Abdullayev', 'Ismoilov', 'Yusupov',
            'Каримов', 'Рахимов', 'Тошматов', 'Абдуллаев', 'Исмоилов', 'Юсупов'
        ]
        
        return f"{random.choice(uzbek_first_names)} {random.choice(uzbek_last_names)}"
    
    def generate_city_name(self) -> str:
        """Generate a random city name."""
        cities = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
            'Toshkent', 'Samarkand', 'Buxoro', 'Andijon', 'Namangan', 'Fargona',
            'Ташкент', 'Самарканд', 'Бухара', 'Андижан', 'Наманган', 'Фергана'
        ]
        return random.choice(cities)
    
    def generate_phone_number(self) -> str:
        """Generate a random phone number."""
        formats = [
            '+1-{}-{}-{}',
            '+998-{}-{}-{}',
            '({}) {}-{}',
            '{}.{}.{}'
        ]
        
        format_str = random.choice(formats)
        
        if '+1' in format_str:
            return format_str.format(
                random.randint(200, 999),
                random.randint(200, 999),
                random.randint(1000, 9999)
            )
        elif '+998' in format_str:
            return format_str.format(
                random.randint(90, 99),
                random.randint(100, 999),
                random.randint(1000, 9999)
            )
        else:
            return format_str.format(
                random.randint(200, 999),
                random.randint(200, 999),
                random.randint(1000, 9999)
            )
    
    def generate_email(self, name: str = None) -> str:
        """Generate a random email address."""
        if name:
            # Create email from name
            clean_name = name.lower().replace(' ', '.').replace("'", "")
            domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'mail.ru']
            return f"{clean_name}@{random.choice(domains)}"
        else:
            username = self.generate_random_string(8, string.ascii_lowercase + string.digits)
            domains = ['example.com', 'test.org', 'sample.net']
            return f"{username}@{random.choice(domains)}"
    
    def generate_date(self, start_date: datetime = None, end_date: datetime = None) -> datetime:
        """Generate a random date within specified range."""
        if start_date is None:
            start_date = datetime(1980, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        return start_date + timedelta(days=random_days)


class PersonDataGenerator(TestDataGenerator):
    """Generator for person/customer data."""
    
    def generate_person_record(self, include_uzbek: bool = False) -> Dict[str, Any]:
        """Generate a single person record."""
        if include_uzbek and random.random() < 0.3:  # 30% chance of Uzbek name
            name = self.generate_uzbek_name()
        else:
            name = self.generate_random_name()
        
        return {
            'id': str(uuid.uuid4()),
            'name': name,
            'first_name': name.split()[0],
            'last_name': name.split()[-1],
            'age': random.randint(18, 80),
            'email': self.generate_email(name),
            'phone': self.generate_phone_number(),
            'city': self.generate_city_name(),
            'country': random.choice(['USA', 'Uzbekistan', 'UK', 'Canada']),
            'birth_date': self.generate_date().strftime('%Y-%m-%d'),
            'created_at': datetime.now().isoformat()
        }
    
    def generate_person_dataset(self, size: int, include_uzbek: bool = False) -> pd.DataFrame:
        """Generate a dataset of person records."""
        records = [self.generate_person_record(include_uzbek) for _ in range(size)]
        return pd.DataFrame(records)
    
    def generate_similar_person_datasets(self, size: int, similarity_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate two datasets with controlled similarity."""
        # Generate base dataset
        base_records = [self.generate_person_record() for _ in range(size)]
        
        # Create first dataset
        dataset1 = pd.DataFrame(base_records)
        
        # Create second dataset with some similar records
        similar_count = int(size * similarity_ratio)
        different_count = size - similar_count
        
        # Take similar records and modify them slightly
        similar_records = []
        for i in range(similar_count):
            record = base_records[i].copy()
            
            # Introduce small variations
            if random.random() < 0.3:  # 30% chance to modify name
                record['name'] = self.introduce_name_variation(record['name'])
            
            if random.random() < 0.2:  # 20% chance to modify age
                record['age'] += random.randint(-2, 2)
                record['age'] = max(18, min(80, record['age']))
            
            if random.random() < 0.1:  # 10% chance to modify phone
                record['phone'] = self.introduce_phone_variation(record['phone'])
            
            similar_records.append(record)
        
        # Add completely different records
        different_records = [self.generate_person_record() for _ in range(different_count)]
        
        dataset2_records = similar_records + different_records
        random.shuffle(dataset2_records)
        
        dataset2 = pd.DataFrame(dataset2_records)
        
        return dataset1, dataset2
    
    def introduce_name_variation(self, name: str) -> str:
        """Introduce small variations in a name."""
        variations = [
            lambda n: n.replace('John', 'Jon'),
            lambda n: n.replace('Michael', 'Mike'),
            lambda n: n.replace('Robert', 'Bob'),
            lambda n: n.replace('William', 'Bill'),
            lambda n: n.replace('Richard', 'Rick'),
            lambda n: n + ' Jr.',
            lambda n: n.replace(' ', '-'),
            lambda n: n.upper(),
            lambda n: n.lower()
        ]
        
        variation = random.choice(variations)
        try:
            return variation(name)
        except:
            return name
    
    def introduce_phone_variation(self, phone: str) -> str:
        """Introduce small variations in a phone number."""
        # Remove/add formatting
        if random.random() < 0.5:
            return ''.join(c for c in phone if c.isdigit())
        else:
            digits = ''.join(c for c in phone if c.isdigit())
            if len(digits) >= 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}"
        
        return phone


class OrganizationDataGenerator(TestDataGenerator):
    """Generator for organization/company data."""
    
    def generate_organization_record(self) -> Dict[str, Any]:
        """Generate a single organization record."""
        company_types = ['LLC', 'Inc', 'Corp', 'Ltd', 'Co']
        industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail']
        
        base_name = random.choice([
            'Tech Solutions', 'Global Systems', 'Advanced Analytics', 'Smart Innovations',
            'Digital Dynamics', 'Future Technologies', 'Prime Services', 'Elite Consulting'
        ])
        
        company_name = f"{base_name} {random.choice(company_types)}"
        
        return {
            'id': str(uuid.uuid4()),
            'name': company_name,
            'short_name': base_name,
            'industry': random.choice(industries),
            'employees': random.randint(10, 10000),
            'revenue': random.randint(100000, 100000000),
            'city': self.generate_city_name(),
            'country': random.choice(['USA', 'Uzbekistan', 'UK', 'Canada']),
            'phone': self.generate_phone_number(),
            'email': self.generate_email(),
            'website': f"www.{base_name.lower().replace(' ', '')}.com",
            'founded_year': random.randint(1950, 2023),
            'created_at': datetime.now().isoformat()
        }
    
    def generate_organization_dataset(self, size: int) -> pd.DataFrame:
        """Generate a dataset of organization records."""
        records = [self.generate_organization_record() for _ in range(size)]
        return pd.DataFrame(records)


class MatchingTestDataGenerator(TestDataGenerator):
    """Generator specifically for matching algorithm testing."""
    
    def generate_exact_match_pairs(self, count: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate pairs that should match exactly."""
        dataset1 = []
        dataset2 = []
        
        for i in range(count):
            record = {
                'id': i,
                'name': self.generate_random_name(),
                'value': self.generate_random_string(10)
            }
            
            # Create identical record for dataset2
            dataset1.append(record)
            dataset2.append(record.copy())
        
        return dataset1, dataset2
    
    def generate_fuzzy_match_pairs(self, count: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate pairs that should match with fuzzy matching."""
        dataset1 = []
        dataset2 = []
        
        for i in range(count):
            name = self.generate_random_name()
            value = self.generate_random_string(10)
            
            record1 = {
                'id': i,
                'name': name,
                'value': value
            }
            
            # Create similar record with small differences
            record2 = {
                'id': i,
                'name': self.introduce_typos(name),
                'value': self.introduce_typos(value)
            }
            
            dataset1.append(record1)
            dataset2.append(record2)
        
        return dataset1, dataset2
    
    def generate_no_match_pairs(self, count: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate pairs that should not match."""
        dataset1 = []
        dataset2 = []
        
        for i in range(count):
            dataset1.append({
                'id': i,
                'name': self.generate_random_name(),
                'value': self.generate_random_string(10)
            })
            
            dataset2.append({
                'id': i + 1000,  # Different ID
                'name': self.generate_random_name(),
                'value': self.generate_random_string(10)
            })
        
        return dataset1, dataset2
    
    def introduce_typos(self, text: str, typo_rate: float = 0.1) -> str:
        """Introduce random typos in text."""
        if not text:
            return text
        
        chars = list(text)
        num_typos = max(1, int(len(chars) * typo_rate))
        
        for _ in range(num_typos):
            if len(chars) > 1:
                pos = random.randint(0, len(chars) - 1)
                
                # Different types of typos
                typo_type = random.choice(['substitute', 'delete', 'insert', 'transpose'])
                
                if typo_type == 'substitute' and chars[pos].isalpha():
                    chars[pos] = random.choice(string.ascii_letters)
                elif typo_type == 'delete':
                    chars.pop(pos)
                elif typo_type == 'insert':
                    chars.insert(pos, random.choice(string.ascii_letters))
                elif typo_type == 'transpose' and pos < len(chars) - 1:
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        return ''.join(chars)
    
    def generate_performance_test_data(self, size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate large datasets for performance testing."""
        # Create datasets with controlled overlap
        overlap_ratio = 0.3  # 30% overlap
        overlap_count = int(size * overlap_ratio)
        
        # Generate base records
        base_records = []
        for i in range(overlap_count):
            record = {
                'id': i,
                'name': self.generate_random_name(),
                'category': f'Category_{i % 20}',
                'value': random.randint(1, 1000),
                'description': self.generate_random_string(50)
            }
            base_records.append(record)
        
        # Create dataset1 with base records + unique records
        dataset1_records = base_records.copy()
        for i in range(overlap_count, size):
            record = {
                'id': i,
                'name': self.generate_random_name(),
                'category': f'Category_{i % 20}',
                'value': random.randint(1, 1000),
                'description': self.generate_random_string(50)
            }
            dataset1_records.append(record)
        
        # Create dataset2 with modified base records + unique records
        dataset2_records = []
        for record in base_records:
            modified_record = record.copy()
            # Introduce small variations
            if random.random() < 0.5:
                modified_record['name'] = self.introduce_typos(record['name'], 0.05)
            if random.random() < 0.3:
                modified_record['value'] += random.randint(-10, 10)
            dataset2_records.append(modified_record)
        
        # Add unique records to dataset2
        for i in range(overlap_count, size):
            record = {
                'id': i + size,  # Different ID range
                'name': self.generate_random_name(),
                'category': f'Category_{i % 20}',
                'value': random.randint(1, 1000),
                'description': self.generate_random_string(50)
            }
            dataset2_records.append(record)
        
        # Shuffle to randomize order
        random.shuffle(dataset1_records)
        random.shuffle(dataset2_records)
        
        return pd.DataFrame(dataset1_records), pd.DataFrame(dataset2_records)


class TestDataManager:
    """Manager for test data lifecycle and cleanup."""
    
    def __init__(self, base_dir: str = "test_data"):
        """Initialize test data manager."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.created_files = []
        self.generators = {
            'person': PersonDataGenerator(),
            'organization': OrganizationDataGenerator(),
            'matching': MatchingTestDataGenerator()
        }
    
    def create_test_file(self, data: pd.DataFrame, filename: str, 
                        format_type: str = 'csv') -> Path:
        """Create a test data file."""
        file_path = self.base_dir / filename
        
        if format_type.lower() == 'csv':
            data.to_csv(file_path, index=False)
        elif format_type.lower() == 'json':
            data.to_json(file_path, orient='records', indent=2)
        elif format_type.lower() == 'excel':
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.created_files.append(file_path)
        return file_path
    
    def create_test_scenario(self, scenario_name: str, **kwargs) -> Dict[str, Path]:
        """Create a complete test scenario with multiple files."""
        scenario_dir = self.base_dir / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        files = {}
        
        if scenario_name == 'person_matching':
            size = kwargs.get('size', 100)
            generator = self.generators['person']
            
            # Create two similar datasets
            dataset1, dataset2 = generator.generate_similar_person_datasets(size)
            
            files['dataset1'] = self.create_test_file(
                dataset1, f"{scenario_name}/dataset1.csv"
            )
            files['dataset2'] = self.create_test_file(
                dataset2, f"{scenario_name}/dataset2.csv"
            )
            
            # Create configuration file
            config = {
                'mappings': [
                    {
                        'source_field': 'name',
                        'target_field': 'name',
                        'algorithm': 'fuzzy',
                        'weight': 1.0
                    }
                ],
                'confidence_threshold': 75.0
            }
            
            config_path = scenario_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            files['config'] = config_path
            self.created_files.append(config_path)
        
        elif scenario_name == 'performance_test':
            size = kwargs.get('size', 1000)
            generator = self.generators['matching']
            
            dataset1, dataset2 = generator.generate_performance_test_data(size)
            
            files['large_dataset1'] = self.create_test_file(
                dataset1, f"{scenario_name}/large_dataset1.csv"
            )
            files['large_dataset2'] = self.create_test_file(
                dataset2, f"{scenario_name}/large_dataset2.csv"
            )
        
        elif scenario_name == 'uzbek_text_matching':
            size = kwargs.get('size', 200)
            generator = self.generators['person']
            
            dataset = generator.generate_person_dataset(size, include_uzbek=True)
            
            files['uzbek_dataset'] = self.create_test_file(
                dataset, f"{scenario_name}/uzbek_dataset.csv"
            )
        
        return files
    
    def cleanup_test_data(self):
        """Clean up all created test files."""
        for file_path in self.created_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Error cleaning up {file_path}: {e}")
        
        # Remove empty directories
        try:
            if self.base_dir.exists():
                for item in self.base_dir.iterdir():
                    if item.is_dir() and not any(item.iterdir()):
                        item.rmdir()
                
                # Remove base directory if empty
                if not any(self.base_dir.iterdir()):
                    self.base_dir.rmdir()
        except Exception as e:
            print(f"Error cleaning up directories: {e}")
        
        self.created_files.clear()
    
    def get_test_data_info(self) -> Dict[str, Any]:
        """Get information about created test data."""
        info = {
            'base_directory': str(self.base_dir),
            'total_files': len(self.created_files),
            'files': []
        }
        
        for file_path in self.created_files:
            if file_path.exists():
                stat = file_path.stat()
                info['files'].append({
                    'path': str(file_path),
                    'size_bytes': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
        
        return info


# Example usage and test functions
def create_sample_test_data():
    """Create sample test data for demonstration."""
    manager = TestDataManager()
    
    try:
        # Create person matching scenario
        person_files = manager.create_test_scenario('person_matching', size=50)
        print(f"Created person matching scenario: {person_files}")
        
        # Create performance test scenario
        perf_files = manager.create_test_scenario('performance_test', size=500)
        print(f"Created performance test scenario: {perf_files}")
        
        # Create Uzbek text scenario
        uzbek_files = manager.create_test_scenario('uzbek_text_matching', size=100)
        print(f"Created Uzbek text scenario: {uzbek_files}")
        
        # Print info
        info = manager.get_test_data_info()
        print(f"Test data info: {info}")
        
        return manager
        
    except Exception as e:
        print(f"Error creating test data: {e}")
        manager.cleanup_test_data()
        raise


if __name__ == '__main__':
    # Create sample test data
    manager = create_sample_test_data()
    
    # Clean up after demonstration
    input("Press Enter to clean up test data...")
    manager.cleanup_test_data()
    print("Test data cleaned up.")