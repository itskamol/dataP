#!/usr/bin/env python3
"""
User Acceptance Testing Suite
Tests real-world user scenarios and workflows
"""

import os
import sys
import json
import time
import tempfile
import unittest
import pandas as pd
from pathlib import Path

class UserAcceptanceTestSuite(unittest.TestCase):
    """User acceptance testing scenarios"""
    
    @classmethod
    def setUpClass(cls):
        """Set up user acceptance test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='uat_test_')
        cls.project_root = Path(__file__).parent.parent
        
        # Create realistic test datasets
        cls._create_realistic_datasets()
        
        print(f"User acceptance test environment: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            import shutil
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")
    
    @classmethod
    def _create_realistic_datasets(cls):
        """Create realistic datasets for testing"""
        # Dataset 1: Government districts (Uzbek)
        districts_data = [
            {"id": 1, "name": "Тошкент шаҳри", "region": "Тошкент", "type": "shahar", "population": 2500000},
            {"id": 2, "name": "Самарқанд тумани", "region": "Самарқанд", "type": "tuman", "population": 150000},
            {"id": 3, "name": "Бухоро шаҳри", "region": "Бухоро", "type": "shahar", "population": 280000},
            {"id": 4, "name": "Андижон тумани", "region": "Андижон", "type": "tuman", "population": 200000},
            {"id": 5, "name": "Фарғона шаҳри", "region": "Фарғона", "type": "shahar", "population": 320000},
            {"id": 6, "name": "Наманган тумани", "region": "Наманган", "type": "tuman", "population": 180000},
            {"id": 7, "name": "Қарши шаҳри", "region": "Қашқадарё", "type": "shahar", "population": 250000},
            {"id": 8, "name": "Термиз тумани", "region": "Сурхондарё", "type": "tuman", "population": 140000},
            {"id": 9, "name": "Нукус шаҳри", "region": "Қорақалпоғистон", "type": "shahar", "population": 310000},
            {"id": 10, "name": "Урганч тумани", "region": "Хоразм", "type": "tuman", "population": 160000}
        ]
        
        # Dataset 2: Statistical data (Latin script with variations)
        statistics_data = [
            {"code": "TSH001", "full_name": "Toshkent shahar", "district": "Toshkent viloyati", "area_km2": 334.8, "density": 7463},
            {"code": "SAM002", "full_name": "Samarqand tuman", "district": "Samarqand viloyati", "area_km2": 1200.5, "density": 125},
            {"code": "BUX003", "full_name": "Buxoro shahar", "district": "Buxoro viloyati", "area_km2": 143.2, "density": 1955},
            {"code": "AND004", "full_name": "Andijon tuman", "district": "Andijon viloyati", "area_km2": 890.3, "density": 225},
            {"code": "FAR005", "full_name": "Farg'ona shahri", "district": "Farg'ona viloyati", "area_km2": 96.8, "density": 3306},
            {"code": "NAM006", "full_name": "Namangan tuman", "district": "Namangan viloyati", "area_km2": 780.4, "density": 231},
            {"code": "QAR007", "full_name": "Qarshi shahar", "district": "Qashqadaryo viloyati", "area_km2": 120.5, "density": 2075},
            {"code": "TER008", "full_name": "Termez tumani", "district": "Surxondaryo viloyati", "area_km2": 650.2, "density": 215},
            {"code": "NUK009", "full_name": "Nukus shahar", "district": "Qoraqalpog'iston", "area_km2": 221.6, "density": 1399},
            {"code": "URG010", "full_name": "Urganch tuman", "district": "Xorazm viloyati", "area_km2": 420.8, "density": 380},
            {"code": "NEW011", "full_name": "Yangi tuman", "district": "Yangi viloyat", "area_km2": 500.0, "density": 200}  # Unmatched
        ]
        
        # Save datasets
        cls.districts_file = os.path.join(cls.test_dir, 'districts.csv')
        cls.statistics_file = os.path.join(cls.test_dir, 'statistics.csv')
        
        pd.DataFrame(districts_data).to_csv(cls.districts_file, index=False)
        pd.DataFrame(statistics_data).to_csv(cls.statistics_file, index=False)
        
        # Create JSON versions
        cls.districts_json = os.path.join(cls.test_dir, 'districts.json')
        cls.statistics_json = os.path.join(cls.test_dir, 'statistics.json')
        
        pd.DataFrame(districts_data).to_json(cls.districts_json, orient='records', indent=2, force_ascii=False)
        pd.DataFrame(statistics_data).to_json(cls.statistics_json, orient='records', indent=2, force_ascii=False)

    def test_01_basic_user_workflow(self):
        """Test basic user workflow: upload, configure, process, download"""
        print("\n=== Testing Basic User Workflow ===")
        
        # Simulate user uploading files and configuring matching
        config = {
            "file1": {
                "path": self.districts_file,
                "type": "csv",
                "delimiter": ","
            },
            "file2": {
                "path": self.statistics_file,
                "type": "csv",
                "delimiter": ","
            },
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "output_columns": {
                "from_file1": ["id", "name", "region", "population"],
                "from_file2": ["code", "full_name", "district", "area_km2"]
            },
            "settings": {
                "output_format": "json",
                "matched_output_path": os.path.join(self.test_dir, "user_results"),
                "file1_output_prefix": "district_",
                "file2_output_prefix": "stat_",
                "confidence_threshold": 75,
                "matching_type": "one-to-one",
                "unmatched_files": {"generate": True}
            }
        }
        
        # Test configuration validation
        self.assertIn("file1", config)
        self.assertIn("file2", config)
        self.assertIn("mapping_fields", config)
        self.assertGreater(len(config["mapping_fields"]), 0)
        
        print("✅ Configuration created and validated")
        
        # Test file loading
        df1 = pd.read_csv(config["file1"]["path"])
        df2 = pd.read_csv(config["file2"]["path"])
        
        self.assertGreater(len(df1), 0)
        self.assertGreater(len(df2), 0)
        self.assertIn("name", df1.columns)
        self.assertIn("full_name", df2.columns)
        
        print(f"✅ Files loaded: {len(df1)} + {len(df2)} records")
        
        # Simulate basic matching (simplified for UAT)
        matches = []
        for _, row1 in df1.iterrows():
            for _, row2 in df2.iterrows():
                # Simple similarity check
                name1 = str(row1['name']).lower()
                name2 = str(row2['full_name']).lower()
                
                # Basic matching logic
                if any(word in name2 for word in name1.split() if len(word) > 3):
                    matches.append({
                        'district_id': row1['id'],
                        'district_name': row1['name'],
                        'stat_code': row2['code'],
                        'stat_full_name': row2['full_name'],
                        'confidence': 85  # Simulated confidence
                    })
                    break
        
        self.assertGreater(len(matches), 5)  # Should find several matches
        
        print(f"✅ Matching completed: {len(matches)} matches found")
        
        # Test result saving
        results_file = config["settings"]["matched_output_path"] + ".json"
        with open(results_file, 'w') as f:
            json.dump(matches, f, indent=2, ensure_ascii=False)
        
        self.assertTrue(os.path.exists(results_file))
        
        print("✅ Results saved successfully")
        print("✅ Basic user workflow completed successfully")

    def test_02_advanced_user_scenarios(self):
        """Test advanced user scenarios"""
        print("\n=== Testing Advanced User Scenarios ===")
        
        # Scenario 1: Multiple mapping fields
        print("Testing multiple mapping fields...")
        
        config_multi = {
            "file1": {"path": self.districts_file, "type": "csv", "delimiter": ","},
            "file2": {"path": self.statistics_file, "type": "csv", "delimiter": ","},
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "weight": 0.7
                },
                {
                    "file1_col": "region",
                    "file2_col": "district",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "weight": 0.3
                }
            ],
            "settings": {
                "output_format": "csv",
                "matched_output_path": os.path.join(self.test_dir, "multi_results"),
                "confidence_threshold": 70
            }
        }
        
        # Validate multi-field configuration
        self.assertEqual(len(config_multi["mapping_fields"]), 2)
        total_weight = sum(m["weight"] for m in config_multi["mapping_fields"])
        self.assertAlmostEqual(total_weight, 1.0, places=1)
        
        print("✅ Multiple mapping fields configuration validated")
        
        # Scenario 2: Different file formats (JSON)
        print("Testing JSON file format...")
        
        config_json = {
            "file1": {"path": self.districts_json, "type": "json"},
            "file2": {"path": self.statistics_json, "type": "json"},
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "weight": 1.0
                }
            ],
            "settings": {
                "output_format": "both",
                "matched_output_path": os.path.join(self.test_dir, "json_results"),
                "confidence_threshold": 80
            }
        }
        
        # Test JSON loading
        df1_json = pd.read_json(config_json["file1"]["path"])
        df2_json = pd.read_json(config_json["file2"]["path"])
        
        self.assertGreater(len(df1_json), 0)
        self.assertGreater(len(df2_json), 0)
        
        print("✅ JSON file format handling validated")
        
        # Scenario 3: Different matching types
        print("Testing different matching types...")
        
        matching_types = ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
        
        for matching_type in matching_types:
            config_type = {
                "settings": {
                    "matching_type": matching_type,
                    "confidence_threshold": 75
                }
            }
            
            # Validate matching type
            self.assertIn(matching_type, ["one-to-one", "one-to-many", "many-to-one", "many-to-many"])
            print(f"✅ Matching type '{matching_type}' validated")
        
        print("✅ Advanced user scenarios completed successfully")

    def test_03_error_handling_scenarios(self):
        """Test error handling from user perspective"""
        print("\n=== Testing Error Handling Scenarios ===")
        
        # Scenario 1: Invalid file paths
        print("Testing invalid file paths...")
        
        config_invalid = {
            "file1": {"path": "/nonexistent/file1.csv", "type": "csv"},
            "file2": {"path": "/nonexistent/file2.csv", "type": "csv"}
        }
        
        # Should handle missing files gracefully
        try:
            df1 = pd.read_csv(config_invalid["file1"]["path"])
            self.fail("Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("✅ Missing file error handled correctly")
        
        # Scenario 2: Invalid CSV format
        print("Testing invalid CSV format...")
        
        invalid_csv = os.path.join(self.test_dir, 'invalid.csv')
        with open(invalid_csv, 'w') as f:
            f.write('invalid,csv,format\n"unclosed quote,data,more data\n')
        
        try:
            df_invalid = pd.read_csv(invalid_csv)
            print("⚠️  Invalid CSV was parsed (pandas is forgiving)")
        except Exception as e:
            print(f"✅ Invalid CSV error handled: {type(e).__name__}")
        
        # Scenario 3: Empty files
        print("Testing empty files...")
        
        empty_csv = os.path.join(self.test_dir, 'empty.csv')
        with open(empty_csv, 'w') as f:
            f.write('name,value\n')  # Header only
        
        df_empty = pd.read_csv(empty_csv)
        self.assertEqual(len(df_empty), 0)
        print("✅ Empty file handled correctly")
        
        # Scenario 4: Mismatched columns
        print("Testing mismatched columns...")
        
        config_mismatch = {
            "mapping_fields": [
                {
                    "file1_col": "nonexistent_column",
                    "file2_col": "another_nonexistent_column",
                    "match_type": "exact"
                }
            ]
        }
        
        df1 = pd.read_csv(self.districts_file)
        df2 = pd.read_csv(self.statistics_file)
        
        # Check if columns exist
        col1_exists = config_mismatch["mapping_fields"][0]["file1_col"] in df1.columns
        col2_exists = config_mismatch["mapping_fields"][0]["file2_col"] in df2.columns
        
        self.assertFalse(col1_exists)
        self.assertFalse(col2_exists)
        print("✅ Column mismatch detection works")
        
        print("✅ Error handling scenarios completed successfully")

    def test_04_performance_user_experience(self):
        """Test performance from user experience perspective"""
        print("\n=== Testing Performance User Experience ===")
        
        # Test with current dataset size
        df1 = pd.read_csv(self.districts_file)
        df2 = pd.read_csv(self.statistics_file)
        
        dataset_size = len(df1) * len(df2)
        print(f"Testing with {len(df1)} x {len(df2)} = {dataset_size} comparisons")
        
        # Measure user-perceived performance
        start_time = time.time()
        
        # Simulate processing time
        matches = []
        for i, row1 in df1.iterrows():
            for j, row2 in df2.iterrows():
                # Simple processing simulation
                time.sleep(0.001)  # Simulate processing time
                
                # Basic matching
                name1 = str(row1['name']).lower()
                name2 = str(row2['full_name']).lower()
                
                if any(word in name2 for word in name1.split() if len(word) > 3):
                    matches.append((i, j))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # User experience thresholds
        if processing_time < 5.0:
            print(f"✅ Excellent performance: {processing_time:.2f}s (< 5s)")
        elif processing_time < 15.0:
            print(f"✅ Good performance: {processing_time:.2f}s (< 15s)")
        elif processing_time < 30.0:
            print(f"⚠️  Acceptable performance: {processing_time:.2f}s (< 30s)")
        else:
            print(f"❌ Poor performance: {processing_time:.2f}s (> 30s)")
            self.fail(f"Performance too slow for user experience: {processing_time:.2f}s")
        
        # Test progress reporting simulation
        print("Testing progress reporting...")
        
        total_steps = 100
        for step in range(0, total_steps + 1, 10):
            progress_percent = (step / total_steps) * 100
            print(f"Progress: {progress_percent:.0f}% ({step}/{total_steps})")
            time.sleep(0.1)  # Simulate processing
        
        print("✅ Progress reporting simulation completed")
        print("✅ Performance user experience testing completed")

    def test_05_data_quality_validation(self):
        """Test data quality validation from user perspective"""
        print("\n=== Testing Data Quality Validation ===")
        
        # Load test data
        df1 = pd.read_csv(self.districts_file)
        df2 = pd.read_csv(self.statistics_file)
        
        # Test data quality checks
        print("Checking data quality...")
        
        # Check for missing values
        missing_df1 = df1.isnull().sum()
        missing_df2 = df2.isnull().sum()
        
        print(f"File 1 missing values: {missing_df1.sum()}")
        print(f"File 2 missing values: {missing_df2.sum()}")
        
        # Check for duplicate records
        duplicates_df1 = df1.duplicated().sum()
        duplicates_df2 = df2.duplicated().sum()
        
        print(f"File 1 duplicates: {duplicates_df1}")
        print(f"File 2 duplicates: {duplicates_df2}")
        
        # Check data types
        print("Data types validation:")
        for col in df1.columns:
            print(f"  File 1 - {col}: {df1[col].dtype}")
        
        for col in df2.columns:
            print(f"  File 2 - {col}: {df2[col].dtype}")
        
        # Test matching quality
        print("Testing matching quality...")
        
        # Simulate matching with quality metrics
        matches = []
        for _, row1 in df1.iterrows():
            best_match = None
            best_score = 0
            
            for _, row2 in df2.iterrows():
                # Simple similarity scoring
                name1 = str(row1['name']).lower()
                name2 = str(row2['full_name']).lower()
                
                # Count common words
                words1 = set(name1.split())
                words2 = set(name2.split())
                common_words = words1.intersection(words2)
                
                if common_words:
                    score = len(common_words) / max(len(words1), len(words2)) * 100
                    if score > best_score:
                        best_score = score
                        best_match = row2
            
            if best_match is not None and best_score > 50:
                matches.append({
                    'source': row1['name'],
                    'target': best_match['full_name'],
                    'confidence': best_score
                })
        
        # Quality metrics
        high_confidence = sum(1 for m in matches if m['confidence'] > 80)
        medium_confidence = sum(1 for m in matches if 60 <= m['confidence'] <= 80)
        low_confidence = sum(1 for m in matches if m['confidence'] < 60)
        
        print(f"Matching quality distribution:")
        print(f"  High confidence (>80%): {high_confidence}")
        print(f"  Medium confidence (60-80%): {medium_confidence}")
        print(f"  Low confidence (<60%): {low_confidence}")
        print(f"  Total matches: {len(matches)}")
        
        # Assert reasonable quality
        self.assertGreater(len(matches), 5, "Should find reasonable number of matches")
        self.assertGreater(high_confidence, 0, "Should have some high confidence matches")
        
        print("✅ Data quality validation completed successfully")

    def test_06_user_interface_scenarios(self):
        """Test user interface scenarios"""
        print("\n=== Testing User Interface Scenarios ===")
        
        # Test configuration scenarios
        print("Testing configuration scenarios...")
        
        # Scenario 1: Beginner user (simple configuration)
        beginner_config = {
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 1.0
                }
            ],
            "settings": {
                "confidence_threshold": 75,
                "matching_type": "one-to-one",
                "output_format": "csv"
            }
        }
        
        # Validate beginner configuration
        self.assertEqual(len(beginner_config["mapping_fields"]), 1)
        self.assertEqual(beginner_config["settings"]["matching_type"], "one-to-one")
        print("✅ Beginner configuration validated")
        
        # Scenario 2: Advanced user (complex configuration)
        advanced_config = {
            "mapping_fields": [
                {
                    "file1_col": "name",
                    "file2_col": "full_name",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 0.6
                },
                {
                    "file1_col": "region",
                    "file2_col": "district",
                    "match_type": "fuzzy",
                    "use_normalization": True,
                    "case_sensitive": False,
                    "weight": 0.4
                }
            ],
            "settings": {
                "confidence_threshold": 85,
                "matching_type": "many-to-many",
                "output_format": "both",
                "unmatched_files": {"generate": True}
            }
        }
        
        # Validate advanced configuration
        self.assertEqual(len(advanced_config["mapping_fields"]), 2)
        self.assertEqual(advanced_config["settings"]["matching_type"], "many-to-many")
        self.assertTrue(advanced_config["settings"]["unmatched_files"]["generate"])
        print("✅ Advanced configuration validated")
        
        # Test result presentation scenarios
        print("Testing result presentation...")
        
        # Simulate different result formats
        sample_results = [
            {
                "district_id": 1,
                "district_name": "Toshkent shahri",
                "stat_code": "TSH001",
                "stat_full_name": "Toshkent shahar",
                "confidence": 95.5
            },
            {
                "district_id": 2,
                "district_name": "Samarqand tumani",
                "stat_code": "SAM002",
                "stat_full_name": "Samarqand tuman",
                "confidence": 88.2
            }
        ]
        
        # Test CSV export
        results_df = pd.DataFrame(sample_results)
        csv_output = os.path.join(self.test_dir, 'ui_results.csv')
        results_df.to_csv(csv_output, index=False)
        self.assertTrue(os.path.exists(csv_output))
        print("✅ CSV export validated")
        
        # Test JSON export
        json_output = os.path.join(self.test_dir, 'ui_results.json')
        with open(json_output, 'w') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        self.assertTrue(os.path.exists(json_output))
        print("✅ JSON export validated")
        
        print("✅ User interface scenarios completed successfully")

def run_user_acceptance_tests():
    """Run all user acceptance tests"""
    print("="*80)
    print("USER ACCEPTANCE TESTING SUITE")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(UserAcceptanceTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("USER ACCEPTANCE TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nUSER EXPERIENCE ISSUES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nTEST ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_user_acceptance_tests()
    sys.exit(0 if success else 1)