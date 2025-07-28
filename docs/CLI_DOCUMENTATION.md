# Command Line Interface Documentation

## Overview

The File Processing Optimization CLI tool provides a powerful command-line interface for batch processing, automation, and integration with existing workflows. This document covers installation, usage, configuration, and advanced features.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Methods

#### Method 1: Install from PyPI (Recommended)

```bash
# Install the package
pip install file-processing-optimization

# Verify installation
file-processor --version
```

#### Method 2: Install from Source

```bash
# Clone repository
git clone https://github.com/company/file-processing-optimization.git
cd file-processing-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Verify installation
file-processor --version
```

#### Method 3: Docker Installation

```bash
# Pull Docker image
docker pull file-processor:latest

# Create alias for easy usage
alias file-processor='docker run --rm -v $(pwd):/data file-processor:latest'

# Verify installation
file-processor --version
```

## Basic Usage

### Command Structure

```bash
file-processor [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Global Options

```bash
--config FILE         Configuration file path
--verbose, -v         Enable verbose logging
--quiet, -q          Suppress output except errors
--log-file FILE      Write logs to file
--log-level LEVEL    Set logging level (DEBUG, INFO, WARN, ERROR)
--help, -h           Show help message
--version            Show version information
```

### Quick Start Examples

#### Simple File Matching

```bash
# Basic matching with default settings
file-processor process \
  --file1 customers.csv \
  --file2 users.json \
  --output matched_results.csv

# With custom threshold
file-processor process \
  --file1 data1.csv \
  --file2 data2.csv \
  --threshold 80 \
  --output results.json
```

#### Using Configuration File

```bash
# Create configuration file
cat > config.json << 'EOF'
{
  "mappings": [
    {
      "file1_col": "name",
      "file2_col": "full_name",
      "algorithm": "fuzzy",
      "weight": 2.0
    }
  ],
  "threshold": 75,
  "output_format": "csv"
}
EOF

# Process with configuration
file-processor process \
  --config config.json \
  --file1 data1.csv \
  --file2 data2.csv \
  --output results.csv
```

## Commands Reference

### 1. Process Command

The main command for file processing and matching.

```bash
file-processor process [OPTIONS]
```

#### Required Options

```bash
--file1 FILE         First input file (required)
--file2 FILE         Second input file (required)
```

#### Output Options

```bash
--output FILE        Output file path
--output-dir DIR     Output directory (creates multiple files)
--format FORMAT      Output format: csv, json, excel (default: csv)
--include-unmatched  Include unmatched records in output
--include-confidence Include confidence scores in output
```

#### Matching Options

```bash
--threshold FLOAT    Minimum confidence threshold (0-100, default: 75)
--algorithm ALGO     Default matching algorithm: exact, fuzzy, phonetic
--fuzzy-threshold N  Fuzzy matching threshold (default: 80)
--exact-weight N     Weight for exact matches (default: 3.0)
--fuzzy-weight N     Weight for fuzzy matches (default: 2.0)
--phonetic-weight N  Weight for phonetic matches (default: 1.0)
```

#### Performance Options

```bash
--parallel           Enable parallel processing
--max-workers N      Maximum worker processes (default: CPU count)
--chunk-size N       Records per processing chunk (default: 1000)
--memory-limit N     Memory limit in MB (default: 2048)
--timeout N          Processing timeout in seconds (default: 3600)
```

#### Text Processing Options

```bash
--normalize          Enable text normalization
--lowercase          Convert text to lowercase
--remove-punctuation Remove punctuation marks
--uzbek-processing   Enable Uzbek text processing
--script-conversion  Convert between Cyrillic/Latin scripts
--phonetic-uzbek     Use Uzbek-specific phonetic rules
```

#### Examples

```bash
# Basic processing
file-processor process \
  --file1 customers.csv \
  --file2 users.json \
  --output results.csv \
  --threshold 80

# High-performance processing
file-processor process \
  --file1 large_file1.csv \
  --file2 large_file2.csv \
  --parallel \
  --max-workers 8 \
  --chunk-size 5000 \
  --memory-limit 4096 \
  --output results.csv

# Uzbek text processing
file-processor process \
  --file1 uzbek_names.csv \
  --file2 uzbek_database.json \
  --uzbek-processing \
  --script-conversion \
  --phonetic-uzbek \
  --threshold 70 \
  --output uzbek_matches.csv

# Multiple output formats
file-processor process \
  --file1 data1.csv \
  --file2 data2.json \
  --output-dir ./results/ \
  --format excel \
  --include-unmatched \
  --include-confidence
```

### 2. Validate Command

Validate files and configurations before processing.

```bash
file-processor validate [OPTIONS]
```

#### Options

```bash
--file FILE          File to validate (required)
--format FORMAT      Expected format: csv, json, excel
--encoding ENCODING  Text encoding (default: utf-8)
--delimiter CHAR     CSV delimiter (default: auto-detect)
--report             Generate detailed validation report
--fix-issues         Attempt to fix common issues
```

#### Examples

```bash
# Validate single file
file-processor validate --file data.csv

# Validate with specific format
file-processor validate \
  --file data.csv \
  --format csv \
  --encoding utf-8 \
  --delimiter ";"

# Generate detailed report
file-processor validate \
  --file problematic.csv \
  --report \
  --output validation_report.txt

# Validate and fix issues
file-processor validate \
  --file data.csv \
  --fix-issues \
  --output cleaned_data.csv
```

### 3. Config Command

Manage configuration files and templates.

```bash
file-processor config [SUBCOMMAND] [OPTIONS]
```

#### Subcommands

```bash
create               Create configuration template
validate             Validate configuration file
show                 Display current configuration
examples             Show configuration examples
```

#### Examples

```bash
# Create configuration template
file-processor config create \
  --template basic \
  --output config.json

# Validate configuration
file-processor config validate \
  --config my_config.json

# Show configuration examples
file-processor config examples \
  --type fuzzy-matching

# Display current configuration
file-processor config show \
  --config config.json \
  --format yaml
```

### 4. Batch Command

Process multiple file pairs in batch mode.

```bash
file-processor batch [OPTIONS]
```

#### Options

```bash
--config FILE        Batch configuration file (required)
--input-dir DIR      Input directory path
--output-dir DIR     Output directory path
--parallel           Process batches in parallel
--max-concurrent N   Maximum concurrent batch jobs
--continue-on-error  Continue processing if one batch fails
```

#### Batch Configuration Example

```json
{
  "file_pairs": [
    {
      "file1": "customers_2023.csv",
      "file2": "users_2023.json",
      "output": "matches_2023.csv",
      "config": {
        "threshold": 80,
        "algorithms": ["exact", "fuzzy"]
      }
    },
    {
      "file1": "customers_2024.csv",
      "file2": "users_2024.json", 
      "output": "matches_2024.csv",
      "config": {
        "threshold": 75,
        "uzbek_processing": true
      }
    }
  ],
  "default_config": {
    "parallel_processing": true,
    "max_workers": 4,
    "include_unmatched": true
  }
}
```

#### Examples

```bash
# Process batch with configuration
file-processor batch \
  --config batch_config.json \
  --input-dir ./data/ \
  --output-dir ./results/

# Parallel batch processing
file-processor batch \
  --config batch_config.json \
  --parallel \
  --max-concurrent 3 \
  --continue-on-error
```

### 5. Stats Command

Generate statistics and reports about processing results.

```bash
file-processor stats [OPTIONS]
```

#### Options

```bash
--input FILE         Input results file (required)
--format FORMAT      Input format: csv, json, excel
--output FILE        Output report file
--report-type TYPE   Report type: summary, detailed, quality
--charts             Generate charts (requires matplotlib)
```

#### Examples

```bash
# Generate summary statistics
file-processor stats \
  --input results.csv \
  --report-type summary

# Detailed quality report
file-processor stats \
  --input results.json \
  --report-type quality \
  --output quality_report.html \
  --charts

# Compare multiple results
file-processor stats \
  --input results1.csv \
  --compare results2.csv \
  --output comparison_report.pdf
```

## Configuration Files

### Basic Configuration Structure

```json
{
  "file_processing": {
    "encoding": "utf-8",
    "max_file_size_mb": 500,
    "csv_settings": {
      "delimiter": "auto",
      "quotechar": "\"",
      "skip_blank_lines": true
    }
  },
  
  "matching": {
    "mappings": [
      {
        "file1_col": "name",
        "file2_col": "full_name",
        "algorithm": "fuzzy",
        "weight": 2.0,
        "normalization": {
          "enabled": true,
          "lowercase": true,
          "remove_punctuation": true
        }
      }
    ],
    
    "thresholds": {
      "minimum_confidence": 75.0,
      "auto_accept_threshold": 95.0
    }
  },
  
  "performance": {
    "parallel_processing": true,
    "max_workers": 4,
    "chunk_size": 1000,
    "memory_limit_mb": 2048
  },
  
  "output": {
    "format": "csv",
    "include_unmatched": true,
    "include_confidence_scores": true
  }
}
```

### Configuration Templates

#### Template 1: High Accuracy Matching

```json
{
  "name": "high_accuracy_template",
  "description": "Configuration for high-accuracy matching (financial, legal)",
  
  "matching": {
    "mappings": [
      {
        "file1_col": "id",
        "file2_col": "customer_id",
        "algorithm": "exact",
        "weight": 5.0,
        "required": true
      },
      {
        "file1_col": "email",
        "file2_col": "email_address",
        "algorithm": "exact",
        "weight": 4.0,
        "normalization": {
          "lowercase": true,
          "trim_whitespace": true
        }
      },
      {
        "file1_col": "name",
        "file2_col": "full_name",
        "algorithm": "fuzzy",
        "weight": 2.0,
        "fuzzy_settings": {
          "threshold": 90
        }
      }
    ],
    
    "thresholds": {
      "minimum_confidence": 85.0,
      "auto_accept_threshold": 95.0,
      "manual_review_threshold": 90.0
    }
  }
}
```

#### Template 2: Uzbek Text Processing

```json
{
  "name": "uzbek_processing_template",
  "description": "Optimized for Uzbek language text processing",
  
  "matching": {
    "mappings": [
      {
        "file1_col": "ism",
        "file2_col": "to_liq_ism",
        "algorithm": "fuzzy",
        "weight": 2.0,
        "uzbek_specific": {
          "script_conversion": true,
          "phonetic_rules": true,
          "common_variations": true
        }
      }
    ],
    
    "uzbek_processing": {
      "enabled": true,
      "script_conversion": true,
      "cyrillic_to_latin": true,
      "phonetic_matching": true,
      "regional_variations": true
    },
    
    "thresholds": {
      "minimum_confidence": 70.0
    }
  }
}
```

#### Template 3: High Performance

```json
{
  "name": "high_performance_template",
  "description": "Optimized for processing large datasets quickly",
  
  "performance": {
    "parallel_processing": true,
    "max_workers": 8,
    "chunk_size": 10000,
    "memory_limit_mb": 8192,
    
    "optimization": {
      "blocking_enabled": true,
      "blocking_strategy": "phonetic",
      "block_size": 2000,
      "early_termination": true
    },
    
    "cache": {
      "enabled": true,
      "type": "memory",
      "size": 50000,
      "ttl_seconds": 3600
    }
  },
  
  "matching": {
    "thresholds": {
      "minimum_confidence": 75.0
    }
  }
}
```

## Advanced Usage Examples

### 1. Automated Processing Pipeline

```bash
#!/bin/bash
# automated_pipeline.sh

set -e  # Exit on any error

# Configuration
INPUT_DIR="/data/input"
OUTPUT_DIR="/data/output"
CONFIG_FILE="/config/production.json"
LOG_FILE="/logs/processing.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to process file pair
process_files() {
    local file1="$1"
    local file2="$2"
    local output="$3"
    
    echo "Processing: $(basename "$file1") + $(basename "$file2")"
    
    file-processor process \
        --config "$CONFIG_FILE" \
        --file1 "$file1" \
        --file2 "$file2" \
        --output "$output" \
        --log-file "$LOG_FILE" \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $output"
        return 0
    else
        echo "❌ Failed: $file1"
        return 1
    fi
}

# Process all customer files
for customer_file in "$INPUT_DIR"/*_customers.csv; do
    if [ -f "$customer_file" ]; then
        base_name=$(basename "$customer_file" _customers.csv)
        user_file="$INPUT_DIR/${base_name}_users.json"
        output_file="$OUTPUT_DIR/${base_name}_matches.csv"
        
        if [ -f "$user_file" ]; then
            process_files "$customer_file" "$user_file" "$output_file"
        else
            echo "⚠️  Warning: No matching user file for $customer_file"
        fi
    fi
done

echo "Pipeline completed"
```

### 2. Data Quality Assessment

```bash
#!/bin/bash
# quality_assessment.sh

# Validate all input files
echo "=== File Validation ==="
for file in data/*.csv data/*.json; do
    if [ -f "$file" ]; then
        echo "Validating: $(basename "$file")"
        file-processor validate \
            --file "$file" \
            --report \
            --output "validation_$(basename "$file").txt"
    fi
done

# Process with quality monitoring
echo "=== Processing with Quality Monitoring ==="
file-processor process \
    --file1 data/customers.csv \
    --file2 data/users.json \
    --output results.csv \
    --include-confidence \
    --verbose

# Generate quality report
echo "=== Quality Report ==="
file-processor stats \
    --input results.csv \
    --report-type quality \
    --output quality_report.html \
    --charts

echo "Quality assessment completed"
```

### 3. Performance Benchmarking

```bash
#!/bin/bash
# benchmark.sh

# Test different configurations
configs=("basic.json" "optimized.json" "high_performance.json")
test_files=("small_test.csv" "medium_test.csv" "large_test.csv")

echo "=== Performance Benchmarking ==="

for config in "${configs[@]}"; do
    for test_file in "${test_files[@]}"; do
        echo "Testing: $config with $test_file"
        
        # Run with timing
        start_time=$(date +%s)
        
        file-processor process \
            --config "configs/$config" \
            --file1 "test_data/$test_file" \
            --file2 "test_data/reference.json" \
            --output "results/${config}_${test_file}_results.csv" \
            --quiet
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo "  Duration: ${duration}s"
        
        # Generate stats
        file-processor stats \
            --input "results/${config}_${test_file}_results.csv" \
            --report-type summary \
            >> "benchmark_results.txt"
    done
done

echo "Benchmarking completed. Results in benchmark_results.txt"
```

### 4. Continuous Integration Integration

```yaml
# .github/workflows/file-processing.yml
name: File Processing CI

on:
  push:
    paths:
      - 'data/**'
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  process-files:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install file-processing-optimization
    
    - name: Validate input files
      run: |
        file-processor validate --file data/customers.csv --report
        file-processor validate --file data/users.json --report
    
    - name: Process files
      run: |
        file-processor process \
          --config .github/config/production.json \
          --file1 data/customers.csv \
          --file2 data/users.json \
          --output results/matches.csv \
          --include-confidence
    
    - name: Generate quality report
      run: |
        file-processor stats \
          --input results/matches.csv \
          --report-type quality \
          --output results/quality_report.html
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: processing-results
        path: results/
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Command not found: file-processor"

**Cause**: CLI tool not installed or not in PATH

**Solutions:**
```bash
# Check if installed
pip list | grep file-processing-optimization

# Reinstall if needed
pip install --force-reinstall file-processing-optimization

# Check PATH
echo $PATH
which file-processor

# Use full path if needed
python -m src.application.cli --help
```

#### Issue: "Permission denied" when processing files

**Cause**: Insufficient file permissions

**Solutions:**
```bash
# Check file permissions
ls -la your_file.csv

# Fix permissions
chmod 644 your_file.csv

# Check directory permissions
chmod 755 /path/to/directory

# Run with sudo (not recommended)
sudo file-processor process --file1 file1.csv --file2 file2.csv
```

#### Issue: "Memory error" with large files

**Cause**: Insufficient memory or inefficient processing

**Solutions:**
```bash
# Reduce memory usage
file-processor process \
  --file1 large_file.csv \
  --file2 large_file2.csv \
  --memory-limit 1024 \
  --chunk-size 500 \
  --max-workers 2

# Use streaming mode
file-processor process \
  --file1 large_file.csv \
  --file2 large_file2.csv \
  --streaming \
  --output results.csv

# Split large files first
split -l 10000 large_file.csv chunk_
```

#### Issue: "Encoding error" when processing files

**Cause**: Incorrect file encoding

**Solutions:**
```bash
# Detect encoding
file -i your_file.csv

# Specify encoding
file-processor process \
  --file1 file1.csv \
  --file2 file2.csv \
  --encoding cp1251 \
  --output results.csv

# Convert encoding first
iconv -f cp1251 -t utf-8 input.csv > output.csv
```

### Performance Optimization Tips

#### For Large Files (> 100MB)

```bash
file-processor process \
  --file1 large_file1.csv \
  --file2 large_file2.csv \
  --parallel \
  --max-workers 8 \
  --chunk-size 10000 \
  --memory-limit 8192 \
  --timeout 7200 \
  --output results.csv
```

#### For High-Accuracy Requirements

```bash
file-processor process \
  --file1 file1.csv \
  --file2 file2.csv \
  --algorithm exact \
  --threshold 95 \
  --exact-weight 5.0 \
  --fuzzy-weight 1.0 \
  --output results.csv
```

#### For Uzbek Text Processing

```bash
file-processor process \
  --file1 uzbek_file1.csv \
  --file2 uzbek_file2.json \
  --uzbek-processing \
  --script-conversion \
  --phonetic-uzbek \
  --threshold 70 \
  --normalize \
  --output uzbek_results.csv
```

## Integration Examples

### 1. Python Script Integration

```python
#!/usr/bin/env python3
import subprocess
import json
import sys
from pathlib import Path

def run_file_processor(file1, file2, config, output):
    """Run file processor with error handling."""
    
    cmd = [
        'file-processor', 'process',
        '--file1', str(file1),
        '--file2', str(file2),
        '--config', str(config),
        '--output', str(output),
        '--verbose'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✅ Success: {output}")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {file1} + {file2}")
        print(f"Exit code: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False

def main():
    # Configuration
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    config_file = Path("config/production.json")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process file pairs
    file_pairs = [
        ("customers_2023.csv", "users_2023.json", "matches_2023.csv"),
        ("customers_2024.csv", "users_2024.json", "matches_2024.csv"),
    ]
    
    success_count = 0
    total_count = len(file_pairs)
    
    for file1_name, file2_name, output_name in file_pairs:
        file1 = input_dir / file1_name
        file2 = input_dir / file2_name
        output = output_dir / output_name
        
        if file1.exists() and file2.exists():
            if run_file_processor(file1, file2, config_file, output):
                success_count += 1
        else:
            print(f"⚠️  Missing files: {file1_name} or {file2_name}")
    
    print(f"\n📊 Summary: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 2. Makefile Integration

```makefile
# Makefile for file processing automation

# Configuration
INPUT_DIR := data/input
OUTPUT_DIR := data/output
CONFIG_FILE := config/production.json
LOG_DIR := logs

# Default target
.PHONY: all
all: validate process report

# Create directories
$(OUTPUT_DIR) $(LOG_DIR):
	mkdir -p $@

# Validate input files
.PHONY: validate
validate: $(LOG_DIR)
	@echo "Validating input files..."
	@for file in $(INPUT_DIR)/*.csv $(INPUT_DIR)/*.json; do \
		if [ -f "$$file" ]; then \
			echo "Validating $$file"; \
			file-processor validate --file "$$file" --report > $(LOG_DIR)/validation_$$(basename "$$file").log; \
		fi \
	done

# Process files
.PHONY: process
process: $(OUTPUT_DIR) validate
	@echo "Processing files..."
	file-processor process \
		--config $(CONFIG_FILE) \
		--file1 $(INPUT_DIR)/customers.csv \
		--file2 $(INPUT_DIR)/users.json \
		--output $(OUTPUT_DIR)/matches.csv \
		--include-confidence \
		--log-file $(LOG_DIR)/processing.log

# Generate reports
.PHONY: report
report: process
	@echo "Generating reports..."
	file-processor stats \
		--input $(OUTPUT_DIR)/matches.csv \
		--report-type quality \
		--output $(OUTPUT_DIR)/quality_report.html \
		--charts

# Clean up
.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR) $(LOG_DIR)

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all      - Run complete pipeline (validate, process, report)"
	@echo "  validate - Validate input files"
	@echo "  process  - Process files and generate matches"
	@echo "  report   - Generate quality reports"
	@echo "  clean    - Clean up output and log directories"
	@echo "  help     - Show this help message"
```

This comprehensive CLI documentation provides users with all the information needed to effectively use the command-line interface for the File Processing Optimization system.