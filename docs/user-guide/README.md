# User Guide

## Welcome to File Processing Optimization System

This comprehensive user guide will help you get started with the File Processing Optimization system, whether you're using the web interface, command-line tool, or API. The system is designed to help you match records between two datasets using advanced algorithms optimized for various text types, including Uzbek language processing.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web Interface Guide](#web-interface-guide)
3. [Command Line Interface](#command-line-interface)
4. [Configuration Guide](#configuration-guide)
5. [Matching Algorithms](#matching-algorithms)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Getting Started

### What is File Processing Optimization?

The File Processing Optimization system is a powerful tool that helps you:

- **Match Records**: Find corresponding records between two datasets
- **Handle Multiple Formats**: Process CSV, JSON, and Excel files
- **Use Advanced Algorithms**: Exact, fuzzy, and phonetic matching
- **Process Uzbek Text**: Specialized handling for Uzbek language
- **Scale Efficiently**: Handle large datasets with optimized performance
- **Track Progress**: Real-time updates on processing status

### System Requirements

**Minimum Requirements:**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for web interface
- Files in supported formats (CSV, JSON, Excel)

**Recommended:**
- High-speed internet connection
- Files with consistent structure and encoding
- UTF-8 encoded text files for best results

### Quick Start

1. **Access the System**
   - Web Interface: Open your browser and navigate to the system URL
   - Command Line: Install the CLI tool and run commands
   - API: Use HTTP requests to interact programmatically

2. **Prepare Your Data**
   - Ensure files are in supported formats
   - Check that column names are descriptive
   - Verify text encoding (UTF-8 recommended)

3. **Upload and Configure**
   - Upload your two files
   - Map columns between files
   - Choose matching algorithms
   - Set confidence thresholds

4. **Process and Review**
   - Start the matching process
   - Monitor progress in real-time
   - Review and download results

## Web Interface Guide

### Accessing the Web Interface

1. Open your web browser
2. Navigate to the system URL (e.g., `http://localhost:5000`)
3. You'll see the main dashboard with upload options

### Step-by-Step Walkthrough

#### Step 1: File Upload

![File Upload Interface](images/file-upload.png)

1. **Select Files**
   - Click "Choose File 1" and select your first dataset
   - Click "Choose File 2" and select your second dataset
   - Supported formats: `.csv`, `.json`, `.xlsx`

2. **File Validation**
   - The system automatically validates your files
   - Green checkmarks indicate successful validation
   - Red warnings show issues that need attention

3. **Preview Data**
   - Click "Preview" to see the first few rows
   - Verify that columns are correctly detected
   - Check for encoding issues or formatting problems

**Example Upload Process:**
```
✅ File 1: customers.csv (1,000 rows, 5 columns)
   Columns: name, email, phone, address, city

✅ File 2: users.json (850 records, 4 fields)  
   Fields: full_name, email_address, mobile, location
```

#### Step 2: Column Mapping

![Column Mapping Interface](images/column-mapping.png)

1. **Map Corresponding Columns**
   - Select columns from File 1 and File 2 that should be compared
   - Choose the matching algorithm for each column pair
   - Set the weight (importance) for each mapping

2. **Matching Types Available**
   - **Exact**: Perfect character-by-character match
   - **Fuzzy**: Similarity-based matching (handles typos, variations)
   - **Phonetic**: Sound-based matching (similar pronunciation)

3. **Example Mapping Configuration**
   ```
   File 1 Column    →    File 2 Column    →    Algorithm    →    Weight
   name            →    full_name        →    Fuzzy        →    2.0
   email           →    email_address    →    Exact        →    3.0
   phone           →    mobile           →    Phonetic     →    1.5
   ```

#### Step 3: Advanced Configuration

![Advanced Configuration](images/advanced-config.png)

1. **Matching Thresholds**
   - **Minimum Confidence**: Overall threshold for accepting matches (0-100%)
   - **Fuzzy Threshold**: Specific threshold for fuzzy matching
   - **Algorithm Weights**: Relative importance of different algorithms

2. **Output Options**
   - **Include Unmatched Records**: Export records that didn't match
   - **Include Confidence Scores**: Show matching confidence for each result
   - **Output Format**: Choose CSV, JSON, or Excel for results

3. **Uzbek Text Processing** (if applicable)
   - **Enable Uzbek Processing**: Specialized handling for Uzbek text
   - **Script Conversion**: Convert between Cyrillic and Latin scripts
   - **Phonetic Matching**: Use Uzbek-specific phonetic rules

#### Step 4: Processing and Monitoring

![Processing Monitor](images/processing-monitor.png)

1. **Start Processing**
   - Click "Start Matching" to begin
   - Processing starts immediately with real-time updates

2. **Monitor Progress**
   - **Progress Bar**: Visual indication of completion percentage
   - **Status Messages**: Current processing step and details
   - **Time Estimates**: Elapsed time and estimated completion
   - **Cancel Option**: Stop processing if needed

3. **Real-time Updates**
   ```
   Status: Processing...
   Progress: 45% (4,500 of 10,000 records processed)
   Current Step: Fuzzy matching on 'name' field
   Elapsed Time: 2m 15s
   Estimated Remaining: 3m 10s
   ```

#### Step 5: Results Review

![Results Interface](images/results-review.png)

1. **Results Summary**
   - **Total Matches Found**: Number of successful matches
   - **Average Confidence**: Overall matching quality
   - **Unmatched Records**: Records that didn't find matches

2. **Browse Results**
   - **Matched Records**: Paginated table of successful matches
   - **Confidence Scores**: Quality indicator for each match
   - **Search and Filter**: Find specific records quickly
   - **Sort Options**: Order by confidence, name, or other fields

3. **Example Results Table**
   ```
   File 1 Name     | File 2 Name      | Confidence | Email Match | Phone Match
   John Smith      | John Smith       | 95.5%      | ✅ Exact    | ✅ Exact
   Jane Doe        | Jane M. Doe      | 87.2%      | ✅ Exact    | ❌ No Match
   Bob Johnson     | Robert Johnson   | 82.1%      | ✅ Exact    | ✅ Fuzzy
   ```

#### Step 6: Download Results

![Download Options](images/download-options.png)

1. **Available Downloads**
   - **Matched Results**: Successfully matched records
   - **Low Confidence Matches**: Matches below threshold (for review)
   - **Unmatched from File 1**: Records from first file without matches
   - **Unmatched from File 2**: Records from second file without matches

2. **Format Options**
   - **CSV**: Comma-separated values (Excel compatible)
   - **JSON**: JavaScript Object Notation (programming friendly)
   - **Excel**: Native Excel format with formatting

3. **Download Process**
   - Click the download button for desired file type
   - Files are generated on-demand
   - Large files may take a moment to prepare

### Web Interface Tips

**Performance Tips:**
- Use smaller files for initial testing
- Enable parallel processing for large datasets
- Close other browser tabs to free memory
- Use a stable internet connection

**Quality Tips:**
- Preview data before processing to check formatting
- Use exact matching for unique identifiers (emails, IDs)
- Use fuzzy matching for names and addresses
- Adjust thresholds based on data quality

**Troubleshooting:**
- If upload fails, check file size and format
- If processing is slow, reduce file size or increase thresholds
- If results are poor, adjust algorithm weights and thresholds
- Use browser developer tools (F12) to check for errors

## Command Line Interface

### Installation

```bash
# Install the CLI tool
pip install file-processing-optimization

# Verify installation
file-processor --version

# Get help
file-processor --help
```

### Basic Usage

#### Simple File Processing

```bash
# Basic matching with default settings
file-processor \
  --file1 customers.csv \
  --file2 users.json \
  --output results.csv

# With custom configuration
file-processor \
  --config matching_config.json \
  --file1 data1.csv \
  --file2 data2.csv \
  --output-dir ./results/
```

#### Configuration File Example

Create `matching_config.json`:

```json
{
  "mappings": [
    {
      "file1_col": "name",
      "file2_col": "full_name",
      "algorithm": "fuzzy",
      "weight": 2.0,
      "threshold": 80
    },
    {
      "file1_col": "email",
      "file2_col": "email_address", 
      "algorithm": "exact",
      "weight": 3.0
    }
  ],
  "output": {
    "format": "csv",
    "include_unmatched": true,
    "include_confidence": true
  },
  "processing": {
    "parallel": true,
    "max_workers": 4,
    "chunk_size": 1000
  }
}
```

### Advanced CLI Usage

#### Batch Processing

```bash
# Process multiple file pairs
file-processor batch \
  --config batch_config.json \
  --input-dir ./input_files/ \
  --output-dir ./results/

# Batch configuration
{
  "file_pairs": [
    {
      "file1": "customers_2023.csv",
      "file2": "users_2023.json",
      "output": "matches_2023.csv"
    },
    {
      "file1": "customers_2024.csv", 
      "file2": "users_2024.json",
      "output": "matches_2024.csv"
    }
  ],
  "default_config": {
    "threshold": 75,
    "algorithms": ["exact", "fuzzy"]
  }
}
```

#### Uzbek Text Processing

```bash
# Enable Uzbek-specific processing
file-processor \
  --file1 uzbek_names.csv \
  --file2 uzbek_database.json \
  --uzbek-processing \
  --script-conversion \
  --phonetic-uzbek \
  --output uzbek_matches.csv
```

#### Performance Optimization

```bash
# High-performance processing
file-processor \
  --file1 large_file1.csv \
  --file2 large_file2.csv \
  --parallel \
  --max-workers 8 \
  --memory-limit 4096 \
  --chunk-size 5000 \
  --cache-size 50000 \
  --output results.csv
```

### CLI Command Reference

#### Global Options

```bash
file-processor [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

Global Options:
  --config FILE         Configuration file path
  --verbose, -v         Enable verbose logging
  --quiet, -q          Suppress output except errors
  --log-file FILE      Write logs to file
  --help, -h           Show help message
  --version            Show version information
```

#### Process Command

```bash
file-processor process [OPTIONS]

Options:
  --file1 FILE         First input file (required)
  --file2 FILE         Second input file (required)
  --output FILE        Output file path
  --output-dir DIR     Output directory
  --format FORMAT      Output format (csv, json, excel)
  --threshold FLOAT    Minimum confidence threshold (0-100)
  --algorithm ALGO     Matching algorithm (exact, fuzzy, phonetic)
  --parallel           Enable parallel processing
  --max-workers INT    Maximum worker processes
  --uzbek-processing   Enable Uzbek text processing
  --preview            Show preview without processing
```

#### Validate Command

```bash
file-processor validate [OPTIONS]

Options:
  --file FILE          File to validate
  --format FORMAT      Expected format
  --encoding ENCODING  Text encoding
  --delimiter CHAR     CSV delimiter
  --report             Generate validation report
```

#### Examples with Output

```bash
# Example 1: Basic processing
$ file-processor process --file1 data1.csv --file2 data2.json --output results.csv
Loading files...
✅ File 1: 1,000 records loaded
✅ File 2: 850 records loaded
Processing matches...
████████████████████████████████ 100% | 850/850 | 2m 15s
✅ Processing complete!
📊 Results: 742 matches found (87.3% match rate)
💾 Output saved to: results.csv

# Example 2: Validation
$ file-processor validate --file problematic.csv --report
Validating file: problematic.csv
❌ Encoding issue detected (line 45): Invalid UTF-8 sequence
⚠️  Empty values found in required column 'email' (15 rows)
⚠️  Inconsistent date format in column 'created_date'
📋 Validation report saved to: problematic_validation_report.txt
```

### CLI Best Practices

**File Preparation:**
```bash
# Check file encoding
file-processor validate --file data.csv --encoding utf-8

# Preview data structure
file-processor process --file1 data1.csv --file2 data2.json --preview

# Test with small sample first
head -100 large_file.csv > sample.csv
file-processor process --file1 sample.csv --file2 data2.json
```

**Performance Optimization:**
```bash
# For large files
file-processor process \
  --file1 large1.csv \
  --file2 large2.csv \
  --parallel \
  --max-workers $(nproc) \
  --memory-limit 8192 \
  --chunk-size 10000

# Monitor system resources
htop  # In another terminal
```

**Automation Scripts:**
```bash
#!/bin/bash
# automated_matching.sh

# Set variables
INPUT_DIR="/data/input"
OUTPUT_DIR="/data/output"
CONFIG_FILE="/config/production.json"

# Process all file pairs
for file1 in "$INPUT_DIR"/*_customers.csv; do
    file2="${file1/_customers/_users}"
    output="$OUTPUT_DIR/$(basename "$file1" .csv)_matches.csv"
    
    echo "Processing: $file1 + $file2 → $output"
    
    file-processor process \
        --config "$CONFIG_FILE" \
        --file1 "$file1" \
        --file2 "$file2" \
        --output "$output" \
        --verbose
        
    if [ $? -eq 0 ]; then
        echo "✅ Success: $output"
    else
        echo "❌ Failed: $file1"
    fi
done
```

## Configuration Guide

### Configuration File Structure

The system uses JSON configuration files to define matching behavior, performance settings, and output options.

#### Complete Configuration Example

```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  
  "file_processing": {
    "max_file_size_mb": 500,
    "allowed_extensions": [".csv", ".json", ".xlsx"],
    "encoding": "utf-8",
    "csv_settings": {
      "delimiter": "auto",
      "quotechar": "\"",
      "escapechar": "\\",
      "skip_blank_lines": true
    },
    "json_settings": {
      "array_path": null,
      "flatten_nested": true
    }
  },
  
  "matching": {
    "mappings": [
      {
        "file1_col": "customer_name",
        "file2_col": "user_full_name",
        "algorithm": "fuzzy",
        "weight": 2.0,
        "normalization": {
          "enabled": true,
          "lowercase": true,
          "remove_punctuation": true,
          "remove_extra_spaces": true
        },
        "fuzzy_settings": {
          "threshold": 80,
          "algorithm": "levenshtein"
        }
      },
      {
        "file1_col": "email",
        "file2_col": "email_address",
        "algorithm": "exact",
        "weight": 3.0,
        "normalization": {
          "enabled": true,
          "lowercase": true
        }
      }
    ],
    
    "algorithms": {
      "exact": {
        "enabled": true,
        "case_sensitive": false
      },
      "fuzzy": {
        "enabled": true,
        "default_threshold": 75,
        "algorithm": "levenshtein"
      },
      "phonetic": {
        "enabled": true,
        "algorithm": "soundex",
        "uzbek_specific": true
      }
    },
    
    "thresholds": {
      "minimum_confidence": 70.0,
      "auto_accept_threshold": 95.0,
      "manual_review_threshold": 80.0
    },
    
    "uzbek_processing": {
      "enabled": false,
      "script_conversion": true,
      "cyrillic_to_latin": true,
      "phonetic_rules": true,
      "common_variations": {
        "enabled": true,
        "custom_rules": {
          "ў": ["u", "w"],
          "ғ": ["g", "gh"],
          "қ": ["q", "k"]
        }
      }
    }
  },
  
  "performance": {
    "parallel_processing": true,
    "max_workers": 4,
    "chunk_size": 1000,
    "memory_limit_mb": 2048,
    "timeout_seconds": 3600,
    "cache": {
      "enabled": true,
      "type": "memory",
      "size": 10000,
      "ttl_seconds": 3600
    }
  },
  
  "output": {
    "format": "csv",
    "include_unmatched": true,
    "include_confidence_scores": true,
    "include_algorithm_details": false,
    "column_prefix": {
      "file1": "f1_",
      "file2": "f2_"
    },
    "export_options": {
      "csv": {
        "delimiter": ",",
        "quoting": "minimal"
      },
      "excel": {
        "sheet_names": {
          "matched": "Matched Records",
          "unmatched1": "Unmatched File 1",
          "unmatched2": "Unmatched File 2"
        }
      }
    }
  },
  
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "logs/application.log",
    "max_size_mb": 100,
    "backup_count": 5
  }
}
```

### Configuration Sections Explained

#### File Processing Settings

```json
{
  "file_processing": {
    "max_file_size_mb": 500,           // Maximum file size allowed
    "allowed_extensions": [".csv", ".json", ".xlsx"],  // Supported formats
    "encoding": "utf-8",               // Default text encoding
    "auto_detect_encoding": true,      // Try to detect encoding automatically
    
    "csv_settings": {
      "delimiter": "auto",             // Auto-detect or specify: ",", ";", "\t"
      "quotechar": "\"",              // Quote character for CSV
      "escapechar": "\\",             // Escape character
      "skip_blank_lines": true,       // Skip empty rows
      "header_row": 0                 // Header row index (0-based)
    },
    
    "json_settings": {
      "array_path": null,             // JSONPath to array (null for root)
      "flatten_nested": true,         // Flatten nested objects
      "max_depth": 5                  // Maximum nesting depth
    }
  }
}
```

#### Matching Configuration

```json
{
  "matching": {
    "mappings": [
      {
        "file1_col": "name",           // Column name in first file
        "file2_col": "full_name",      // Column name in second file
        "algorithm": "fuzzy",          // Matching algorithm to use
        "weight": 2.0,                 // Importance weight (higher = more important)
        "required": true,              // Must have a value to be considered
        
        "normalization": {
          "enabled": true,             // Enable text normalization
          "lowercase": true,           // Convert to lowercase
          "remove_punctuation": true,  // Remove punctuation marks
          "remove_extra_spaces": true, // Normalize whitespace
          "trim_whitespace": true      // Remove leading/trailing spaces
        },
        
        "fuzzy_settings": {
          "threshold": 80,             // Minimum similarity percentage
          "algorithm": "levenshtein"   // Fuzzy algorithm: levenshtein, jaro, jaro_winkler
        }
      }
    ],
    
    "thresholds": {
      "minimum_confidence": 70.0,      // Overall minimum confidence
      "auto_accept_threshold": 95.0,   // Auto-accept above this score
      "manual_review_threshold": 80.0  // Flag for manual review below this
    }
  }
}
```

#### Performance Tuning

```json
{
  "performance": {
    "parallel_processing": true,       // Enable multiprocessing
    "max_workers": 4,                 // Number of worker processes
    "chunk_size": 1000,               // Records per processing chunk
    "memory_limit_mb": 2048,          // Maximum memory usage
    "timeout_seconds": 3600,          // Processing timeout
    
    "cache": {
      "enabled": true,                // Enable result caching
      "type": "memory",               // Cache type: memory, redis, file
      "size": 10000,                  // Maximum cache entries
      "ttl_seconds": 3600             // Cache expiration time
    },
    
    "optimization": {
      "blocking_enabled": true,       // Use blocking to reduce comparisons
      "blocking_strategy": "phonetic", // Blocking method: exact, phonetic, ngram
      "block_size": 1000,            // Maximum records per block
      "early_termination": true      // Stop when confidence is very low
    }
  }
}
```

### Environment-Specific Configurations

#### Development Configuration

```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  
  "file_processing": {
    "max_file_size_mb": 50,
    "allowed_extensions": [".csv", ".json"]
  },
  
  "performance": {
    "parallel_processing": false,
    "max_workers": 1,
    "chunk_size": 100,
    "memory_limit_mb": 512
  },
  
  "output": {
    "include_algorithm_details": true,
    "include_debug_info": true
  }
}
```

#### Production Configuration

```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  
  "file_processing": {
    "max_file_size_mb": 1000,
    "allowed_extensions": [".csv", ".json", ".xlsx"]
  },
  
  "performance": {
    "parallel_processing": true,
    "max_workers": 8,
    "chunk_size": 5000,
    "memory_limit_mb": 8192,
    "cache": {
      "type": "redis",
      "url": "redis://localhost:6379/0"
    }
  },
  
  "security": {
    "file_validation": true,
    "virus_scanning": true,
    "content_filtering": true
  }
}
```

### Configuration Validation

The system automatically validates configuration files and provides helpful error messages:

```bash
# Validate configuration
file-processor validate-config --config my_config.json

# Example validation output
✅ Configuration is valid
📋 Summary:
   - 2 column mappings configured
   - Fuzzy matching enabled with 80% threshold
   - Parallel processing: 4 workers
   - Output format: CSV with confidence scores

# Example validation errors
❌ Configuration validation failed:
   - mappings[0].algorithm: 'fuzzy_advanced' is not a valid algorithm
   - performance.max_workers: must be between 1 and 16
   - thresholds.minimum_confidence: must be between 0 and 100
```

This user guide provides comprehensive information for users to effectively use the File Processing Optimization system across all interfaces and use cases.