# Configuration Reference

## Overview

This document provides a comprehensive reference for all configuration options available in the File Processing Optimization system. Configuration files use JSON format and support environment-specific settings, validation, and hot-reloading.

## Configuration File Structure

### Root Configuration Schema

```json
{
  "environment": "string",           // Environment name (development, production, etc.)
  "debug": "boolean",               // Enable debug mode
  "log_level": "string",            // Logging level (DEBUG, INFO, WARN, ERROR)
  
  "file_processing": { ... },       // File processing settings
  "matching": { ... },              // Matching algorithm configuration
  "performance": { ... },           // Performance and optimization settings
  "output": { ... },                // Output format and options
  "security": { ... },              // Security and validation settings
  "logging": { ... },               // Logging configuration
  "cache": { ... },                 // Caching configuration
  "uzbek_processing": { ... }       // Uzbek-specific processing options
}
```

## File Processing Configuration

### file_processing

Controls how files are loaded, validated, and processed.

```json
{
  "file_processing": {
    "max_file_size_mb": 500,                    // Maximum file size in MB
    "allowed_extensions": [".csv", ".json", ".xlsx"],  // Supported file formats
    "encoding": "utf-8",                        // Default text encoding
    "auto_detect_encoding": true,               // Auto-detect file encoding
    "validate_on_upload": true,                 // Validate files immediately on upload
    "temp_directory": "/tmp/file_processor",    // Temporary file storage location
    "cleanup_temp_files": true,                 // Auto-cleanup temporary files
    "file_timeout_seconds": 300,                // File operation timeout
    
    "csv_settings": {
      "delimiter": "auto",                      // CSV delimiter: "auto", ",", ";", "\t", "|"
      "quotechar": "\"",                       // Quote character
      "escapechar": "\\",                      // Escape character
      "skip_blank_lines": true,                // Skip empty rows
      "header_row": 0,                         // Header row index (0-based, null for no header)
      "skip_rows": 0,                          // Number of rows to skip at start
      "max_columns": 1000,                     // Maximum number of columns
      "date_format": "auto",                   // Date format detection
      "decimal_separator": ".",                // Decimal separator for numbers
      "thousands_separator": ",",              // Thousands separator for numbers
      "true_values": ["true", "True", "1", "yes", "Yes"],    // Values treated as True
      "false_values": ["false", "False", "0", "no", "No"],   // Values treated as False
      "na_values": ["", "NULL", "null", "N/A", "n/a", "NaN"] // Values treated as missing
    },
    
    "json_settings": {
      "array_path": null,                      // JSONPath to array (null for root array)
      "flatten_nested": true,                  // Flatten nested objects
      "max_depth": 10,                         // Maximum nesting depth to flatten
      "separator": "_",                        // Separator for flattened keys
      "preserve_arrays": false,                // Keep arrays as arrays (don't flatten)
      "date_parsing": true,                    // Parse ISO date strings
      "number_parsing": true                   // Parse numeric strings
    },
    
    "excel_settings": {
      "sheet_name": null,                      // Sheet name or index (null for first sheet)
      "header_row": 0,                         // Header row index
      "skip_rows": 0,                          // Rows to skip at start
      "max_rows": null,                        // Maximum rows to read (null for all)
      "use_column_names": true,                // Use first row as column names
      "convert_float": true,                   // Convert float columns
      "date_format": null                      // Date format for parsing
    }
  }
}
```

#### Example Configurations

**CSV with Semicolon Delimiter:**
```json
{
  "file_processing": {
    "csv_settings": {
      "delimiter": ";",
      "encoding": "cp1251",
      "decimal_separator": ",",
      "thousands_separator": " "
    }
  }
}
```

**JSON with Nested Structure:**
```json
{
  "file_processing": {
    "json_settings": {
      "array_path": "$.data.records",
      "flatten_nested": true,
      "max_depth": 5,
      "separator": "__"
    }
  }
}
```

**Excel Multi-sheet:**
```json
{
  "file_processing": {
    "excel_settings": {
      "sheet_name": "Customer Data",
      "header_row": 2,
      "skip_rows": 1,
      "max_rows": 10000
    }
  }
}
```

## Matching Configuration

### matching

Defines how records are matched between datasets.

```json
{
  "matching": {
    "mappings": [
      {
        "file1_col": "string",                 // Column name in first file
        "file2_col": "string",                 // Column name in second file
        "algorithm": "string",                 // Matching algorithm
        "weight": "number",                    // Importance weight (0.1-10.0)
        "required": "boolean",                 // Must have value to match
        "case_sensitive": "boolean",           // Case-sensitive matching
        
        "normalization": {
          "enabled": "boolean",                // Enable text normalization
          "lowercase": "boolean",              // Convert to lowercase
          "uppercase": "boolean",              // Convert to uppercase
          "remove_punctuation": "boolean",     // Remove punctuation marks
          "remove_extra_spaces": "boolean",    // Normalize whitespace
          "trim_whitespace": "boolean",        // Remove leading/trailing spaces
          "remove_accents": "boolean",         // Remove accent marks
          "expand_contractions": "boolean",    // Expand contractions (don't -> do not)
          "custom_replacements": {             // Custom text replacements
            "pattern": "replacement"
          }
        },
        
        "exact_settings": {
          "ignore_case": "boolean",            // Case-insensitive exact matching
          "ignore_whitespace": "boolean",      // Ignore whitespace differences
          "ignore_punctuation": "boolean"      // Ignore punctuation differences
        },
        
        "fuzzy_settings": {
          "algorithm": "string",               // levenshtein, jaro, jaro_winkler, ratio
          "threshold": "number",               // Minimum similarity (0-100)
          "max_distance": "number",            // Maximum edit distance
          "prefix_weight": "number",           // Weight for matching prefixes
          "case_sensitive": "boolean"          // Case-sensitive fuzzy matching
        },
        
        "phonetic_settings": {
          "algorithm": "string",               // soundex, metaphone, double_metaphone
          "max_codes": "number",               // Maximum phonetic codes to generate
          "min_length": "number"               // Minimum string length for phonetic matching
        }
      }
    ],
    
    "algorithms": {
      "exact": {
        "enabled": "boolean",                  // Enable exact matching
        "priority": "number",                  // Algorithm priority (1-10)
        "case_sensitive": "boolean",           // Default case sensitivity
        "ignore_whitespace": "boolean"         // Ignore whitespace by default
      },
      
      "fuzzy": {
        "enabled": "boolean",                  // Enable fuzzy matching
        "priority": "number",                  // Algorithm priority
        "default_threshold": "number",         // Default similarity threshold
        "algorithm": "string",                 // Default fuzzy algorithm
        "max_comparisons": "number",           // Maximum comparisons per record
        "early_termination": "boolean",        // Stop at very low scores
        "cache_results": "boolean"             // Cache similarity calculations
      },
      
      "phonetic": {
        "enabled": "boolean",                  // Enable phonetic matching
        "priority": "number",                  // Algorithm priority
        "algorithm": "string",                 // Default phonetic algorithm
        "min_string_length": "number",         // Minimum length for phonetic matching
        "max_codes_per_string": "number"       // Maximum phonetic codes
      }
    },
    
    "thresholds": {
      "minimum_confidence": "number",          // Overall minimum confidence (0-100)
      "auto_accept_threshold": "number",       // Auto-accept above this score
      "manual_review_threshold": "number",     // Flag for manual review
      "reject_threshold": "number",            // Auto-reject below this score
      "confidence_calculation": "string"       // weighted_average, maximum, minimum
    },
    
    "matching_strategy": {
      "type": "string",                        // one-to-one, one-to-many, many-to-many
      "allow_duplicates": "boolean",           // Allow duplicate matches
      "max_matches_per_record": "number",      // Maximum matches per record
      "tie_breaking": "string"                 // highest_confidence, first_found, all
    },
    
    "blocking": {
      "enabled": "boolean",                    // Enable blocking optimization
      "strategy": "string",                    // exact, phonetic, ngram, custom
      "block_size": "number",                  // Maximum records per block
      "max_blocks": "number",                  // Maximum number of blocks
      "overlap_percentage": "number",          // Block overlap percentage
      "custom_blocking_function": "string"     // Custom blocking function name
    }
  }
}
```

#### Algorithm Options

**Available Algorithms:**
- `exact` - Character-by-character exact matching
- `fuzzy` - Similarity-based matching using edit distance
- `phonetic` - Sound-based matching using phonetic algorithms
- `jaro` - Jaro similarity algorithm
- `jaro_winkler` - Jaro-Winkler similarity algorithm
- `ratio` - Simple ratio-based similarity

**Fuzzy Algorithm Options:**
- `levenshtein` - Edit distance (insertions, deletions, substitutions)
- `damerau_levenshtein` - Includes transpositions
- `hamming` - Character position differences (same length strings)
- `jaro` - Jaro similarity
- `jaro_winkler` - Jaro-Winkler with prefix bonus

**Phonetic Algorithm Options:**
- `soundex` - Traditional Soundex algorithm
- `metaphone` - Metaphone algorithm
- `double_metaphone` - Double Metaphone (more accurate)
- `nysiis` - New York State Identification and Intelligence System
- `match_rating` - Match Rating Approach

#### Example Matching Configurations

**High-Accuracy Financial Matching:**
```json
{
  "matching": {
    "mappings": [
      {
        "file1_col": "account_number",
        "file2_col": "account_id",
        "algorithm": "exact",
        "weight": 5.0,
        "required": true
      },
      {
        "file1_col": "customer_name",
        "file2_col": "account_holder",
        "algorithm": "fuzzy",
        "weight": 2.0,
        "fuzzy_settings": {
          "algorithm": "jaro_winkler",
          "threshold": 90
        }
      }
    ],
    "thresholds": {
      "minimum_confidence": 85.0,
      "auto_accept_threshold": 95.0
    }
  }
}
```

**Flexible Name Matching:**
```json
{
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
          "remove_punctuation": true,
          "remove_extra_spaces": true
        },
        "fuzzy_settings": {
          "algorithm": "jaro_winkler",
          "threshold": 75,
          "prefix_weight": 0.1
        }
      }
    ],
    "blocking": {
      "enabled": true,
      "strategy": "phonetic",
      "block_size": 1000
    }
  }
}
```

## Performance Configuration

### performance

Controls system performance, memory usage, and optimization.

```json
{
  "performance": {
    "parallel_processing": "boolean",          // Enable multiprocessing
    "max_workers": "number",                   // Number of worker processes
    "chunk_size": "number",                    // Records per processing chunk
    "memory_limit_mb": "number",               // Maximum memory usage in MB
    "timeout_seconds": "number",               // Processing timeout
    "streaming_mode": "boolean",               // Enable streaming for large files
    "progress_reporting_interval": "number",   // Progress update interval (seconds)
    
    "optimization": {
      "blocking_enabled": "boolean",           // Use blocking to reduce comparisons
      "blocking_strategy": "string",           // Blocking method
      "block_size": "number",                  // Maximum records per block
      "early_termination": "boolean",          // Stop at very low confidence
      "max_comparisons_per_record": "number",  // Limit comparisons per record
      "similarity_cache_size": "number",       // Cache size for similarity calculations
      "precompute_features": "boolean",        // Precompute text features
      "use_gpu": "boolean",                    // Enable GPU acceleration (if available)
      "gpu_batch_size": "number"               // GPU batch size
    },
    
    "memory_management": {
      "garbage_collection_threshold": "number", // GC threshold (MB)
      "max_cache_memory_mb": "number",          // Maximum cache memory
      "memory_monitoring": "boolean",           // Monitor memory usage
      "memory_warning_threshold": "number",     // Warning threshold (%)
      "memory_error_threshold": "number",       // Error threshold (%)
      "swap_to_disk": "boolean",               // Use disk for overflow
      "temp_directory": "string"               // Temporary directory for disk swap
    },
    
    "io_optimization": {
      "read_buffer_size": "number",            // File read buffer size (bytes)
      "write_buffer_size": "number",           // File write buffer size (bytes)
      "compression": "boolean",                // Compress temporary files
      "compression_level": "number",           // Compression level (1-9)
      "async_io": "boolean",                   // Use asynchronous I/O
      "prefetch_size": "number"                // Prefetch buffer size
    }
  }
}
```

#### Performance Tuning Examples

**High-Performance Configuration:**
```json
{
  "performance": {
    "parallel_processing": true,
    "max_workers": 8,
    "chunk_size": 10000,
    "memory_limit_mb": 8192,
    "streaming_mode": true,
    
    "optimization": {
      "blocking_enabled": true,
      "blocking_strategy": "phonetic",
      "block_size": 2000,
      "early_termination": true,
      "similarity_cache_size": 50000,
      "precompute_features": true
    }
  }
}
```

**Memory-Constrained Configuration:**
```json
{
  "performance": {
    "parallel_processing": false,
    "max_workers": 1,
    "chunk_size": 500,
    "memory_limit_mb": 512,
    "streaming_mode": true,
    
    "memory_management": {
      "garbage_collection_threshold": 100,
      "max_cache_memory_mb": 50,
      "memory_monitoring": true,
      "swap_to_disk": true
    }
  }
}
```

## Output Configuration

### output

Controls result formatting and export options.

```json
{
  "output": {
    "format": "string",                        // csv, json, excel, parquet
    "include_unmatched": "boolean",            // Include unmatched records
    "include_confidence_scores": "boolean",    // Include confidence scores
    "include_algorithm_details": "boolean",    // Include algorithm breakdown
    "include_processing_metadata": "boolean",  // Include processing info
    "separate_unmatched_files": "boolean",     // Create separate files for unmatched
    "output_directory": "string",              // Output directory path
    "filename_template": "string",             // Filename template
    "overwrite_existing": "boolean",           // Overwrite existing files
    
    "column_naming": {
      "file1_prefix": "string",                // Prefix for file1 columns
      "file2_prefix": "string",                // Prefix for file2 columns
      "confidence_column": "string",           // Confidence score column name
      "match_id_column": "string",             // Match ID column name
      "algorithm_column": "string",            // Algorithm used column name
      "preserve_original_names": "boolean"     // Keep original column names
    },
    
    "csv_options": {
      "delimiter": "string",                   // Output delimiter
      "quoting": "string",                     // minimal, all, nonnumeric, none
      "line_terminator": "string",             // Line ending
      "encoding": "string",                    // Output encoding
      "include_header": "boolean",             // Include header row
      "date_format": "string",                 // Date format for output
      "float_format": "string"                 // Float number format
    },
    
    "json_options": {
      "indent": "number",                      // JSON indentation
      "ensure_ascii": "boolean",               // Ensure ASCII output
      "sort_keys": "boolean",                  // Sort JSON keys
      "date_format": "string",                 // Date serialization format
      "orient": "string"                       // records, index, values, split
    },
    
    "excel_options": {
      "sheet_names": {
        "matched": "string",                   // Matched records sheet name
        "unmatched1": "string",                // Unmatched file1 sheet name
        "unmatched2": "string",                // Unmatched file2 sheet name
        "summary": "string"                    // Summary sheet name
      },
      "include_charts": "boolean",             // Include charts in Excel
      "freeze_panes": "boolean",               // Freeze header row
      "auto_filter": "boolean",                // Enable auto-filter
      "column_width": "string"                 // auto, fixed, content
    },
    
    "filtering": {
      "min_confidence": "number",              // Minimum confidence to include
      "max_confidence": "number",              // Maximum confidence to include
      "include_algorithms": ["string"],        // Only include specific algorithms
      "exclude_algorithms": ["string"],        // Exclude specific algorithms
      "custom_filter": "string"                // Custom filter expression
    }
  }
}
```

#### Output Examples

**CSV with Custom Formatting:**
```json
{
  "output": {
    "format": "csv",
    "include_confidence_scores": true,
    "column_naming": {
      "file1_prefix": "customer_",
      "file2_prefix": "user_",
      "confidence_column": "match_confidence"
    },
    "csv_options": {
      "delimiter": ";",
      "encoding": "utf-8-sig",
      "date_format": "%Y-%m-%d",
      "float_format": "%.2f"
    }
  }
}
```

**Excel with Multiple Sheets:**
```json
{
  "output": {
    "format": "excel",
    "separate_unmatched_files": false,
    "excel_options": {
      "sheet_names": {
        "matched": "Matched Records",
        "unmatched1": "Unmatched Customers",
        "unmatched2": "Unmatched Users",
        "summary": "Processing Summary"
      },
      "include_charts": true,
      "freeze_panes": true,
      "auto_filter": true
    }
  }
}
```

## Uzbek Processing Configuration

### uzbek_processing

Specialized configuration for Uzbek language text processing.

```json
{
  "uzbek_processing": {
    "enabled": "boolean",                      // Enable Uzbek processing
    "script_conversion": "boolean",            // Convert between scripts
    "cyrillic_to_latin": "boolean",           // Convert Cyrillic to Latin
    "latin_to_cyrillic": "boolean",           // Convert Latin to Cyrillic
    "phonetic_matching": "boolean",            // Use Uzbek phonetic rules
    "regional_variations": "boolean",          // Handle regional variations
    
    "normalization": {
      "standardize_spelling": "boolean",       // Standardize common spellings
      "remove_diacritics": "boolean",         // Remove accent marks
      "normalize_apostrophes": "boolean",      // Normalize apostrophe usage
      "expand_abbreviations": "boolean"        // Expand common abbreviations
    },
    
    "script_mapping": {
      "ў": ["u", "w"],                        // Character mapping rules
      "ғ": ["g", "gh"],
      "қ": ["q", "k"],
      "ҳ": ["h", "kh"],
      "ё": ["yo", "e"],
      "ю": ["yu", "u"],
      "я": ["ya", "a"]
    },
    
    "phonetic_rules": {
      "vowel_groups": [
        ["а", "о"],                           // Vowel similarity groups
        ["и", "ы", "ий"],
        ["у", "ў"]
      ],
      "consonant_groups": [
        ["к", "қ", "q"],                      // Consonant similarity groups
        ["г", "ғ", "gh"],
        ["х", "ҳ", "h", "kh"]
      ]
    },
    
    "common_variations": {
      "enabled": "boolean",                    // Enable common variations
      "place_names": {                        // Place name variations
        "Тошкент": ["Toshkent", "Tashkent"],
        "Самарқанд": ["Samarqand", "Samarkand"],
        "Бухоро": ["Buxoro", "Bukhara"]
      },
      "personal_names": {                     // Personal name variations
        "Муҳаммад": ["Muhammad", "Mohammed", "Mukhammad"],
        "Аҳмад": ["Ahmad", "Ahmed", "Akhmed"]
      },
      "custom_rules": {                       // Custom variation rules
        "pattern": ["variation1", "variation2"]
      }
    }
  }
}
```

#### Uzbek Processing Examples

**Full Uzbek Processing:**
```json
{
  "uzbek_processing": {
    "enabled": true,
    "script_conversion": true,
    "cyrillic_to_latin": true,
    "phonetic_matching": true,
    "regional_variations": true,
    
    "normalization": {
      "standardize_spelling": true,
      "remove_diacritics": true,
      "normalize_apostrophes": true
    },
    
    "common_variations": {
      "enabled": true,
      "place_names": {
        "Тошкент": ["Toshkent", "Tashkent"],
        "Самарқанд": ["Samarqand", "Samarkand"]
      }
    }
  }
}
```

## Security Configuration

### security

Security and validation settings.

```json
{
  "security": {
    "file_validation": {
      "enabled": "boolean",                    // Enable file validation
      "max_file_size_mb": "number",           // Maximum file size
      "allowed_extensions": ["string"],        // Allowed file extensions
      "scan_for_malware": "boolean",          // Enable malware scanning
      "content_type_validation": "boolean",    // Validate MIME types
      "filename_sanitization": "boolean"       // Sanitize filenames
    },
    
    "data_protection": {
      "pii_detection": "boolean",             // Detect personally identifiable information
      "pii_masking": "boolean",               // Mask PII in logs and outputs
      "data_anonymization": "boolean",        // Anonymize sensitive data
      "encryption_at_rest": "boolean",        // Encrypt temporary files
      "secure_deletion": "boolean"            // Secure file deletion
    },
    
    "access_control": {
      "authentication_required": "boolean",   // Require authentication
      "session_timeout_minutes": "number",    // Session timeout
      "max_concurrent_sessions": "number",    // Maximum concurrent sessions
      "ip_whitelist": ["string"],             // Allowed IP addresses
      "rate_limiting": "boolean"              // Enable rate limiting
    },
    
    "audit_logging": {
      "enabled": "boolean",                   // Enable audit logging
      "log_file_access": "boolean",          // Log file access
      "log_data_processing": "boolean",       // Log data processing events
      "log_user_actions": "boolean",         // Log user actions
      "retention_days": "number"             // Log retention period
    }
  }
}
```

## Logging Configuration

### logging

Comprehensive logging configuration.

```json
{
  "logging": {
    "level": "string",                        // DEBUG, INFO, WARN, ERROR, CRITICAL
    "format": "string",                       // text, json, structured
    "output": "string",                       // console, file, both
    "file": "string",                         // Log file path
    "max_size_mb": "number",                  // Maximum log file size
    "backup_count": "number",                 // Number of backup files
    "rotation": "string",                     // time, size, both
    
    "structured_logging": {
      "enabled": "boolean",                   // Enable structured logging
      "include_timestamp": "boolean",         // Include timestamps
      "include_level": "boolean",             // Include log level
      "include_module": "boolean",            // Include module name
      "include_function": "boolean",          // Include function name
      "include_line_number": "boolean",       // Include line number
      "correlation_id": "boolean"             // Include correlation ID
    },
    
    "loggers": {
      "file_processing": "string",            // File processing logger level
      "matching_engine": "string",            // Matching engine logger level
      "performance": "string",                // Performance logger level
      "security": "string",                   // Security logger level
      "api": "string"                        // API logger level
    },
    
    "filters": {
      "exclude_patterns": ["string"],         // Patterns to exclude from logs
      "include_patterns": ["string"],         // Patterns to include in logs
      "sensitive_data_patterns": ["string"]   // Patterns for sensitive data masking
    }
  }
}
```

## Cache Configuration

### cache

Caching configuration for performance optimization.

```json
{
  "cache": {
    "enabled": "boolean",                     // Enable caching
    "type": "string",                         // memory, redis, file, hybrid
    "size": "number",                         // Maximum cache entries
    "ttl_seconds": "number",                  // Time to live
    "cleanup_interval_seconds": "number",     // Cleanup interval
    
    "memory_cache": {
      "max_memory_mb": "number",              // Maximum memory usage
      "eviction_policy": "string",            // lru, lfu, fifo, random
      "compression": "boolean"                // Compress cached data
    },
    
    "redis_cache": {
      "host": "string",                       // Redis host
      "port": "number",                       // Redis port
      "db": "number",                         // Redis database number
      "password": "string",                   // Redis password
      "ssl": "boolean",                       // Use SSL connection
      "connection_pool_size": "number",       // Connection pool size
      "socket_timeout": "number",             // Socket timeout
      "key_prefix": "string"                  // Key prefix for namespacing
    },
    
    "file_cache": {
      "directory": "string",                  // Cache directory
      "max_size_mb": "number",               // Maximum cache size
      "compression": "boolean",               // Compress cache files
      "encryption": "boolean"                 // Encrypt cache files
    }
  }
}
```

## Environment-Specific Configurations

### Development Environment

```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  
  "file_processing": {
    "max_file_size_mb": 50,
    "validate_on_upload": true
  },
  
  "performance": {
    "parallel_processing": false,
    "max_workers": 1,
    "chunk_size": 100
  },
  
  "output": {
    "include_algorithm_details": true,
    "include_processing_metadata": true
  },
  
  "logging": {
    "level": "DEBUG",
    "output": "console",
    "structured_logging": {
      "enabled": true,
      "include_function": true,
      "include_line_number": true
    }
  }
}
```

### Production Environment

```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  
  "file_processing": {
    "max_file_size_mb": 1000,
    "validate_on_upload": true,
    "cleanup_temp_files": true
  },
  
  "performance": {
    "parallel_processing": true,
    "max_workers": 8,
    "chunk_size": 5000,
    "memory_limit_mb": 8192,
    "optimization": {
      "blocking_enabled": true,
      "similarity_cache_size": 50000
    }
  },
  
  "security": {
    "file_validation": {
      "enabled": true,
      "scan_for_malware": true
    },
    "data_protection": {
      "pii_detection": true,
      "encryption_at_rest": true
    },
    "audit_logging": {
      "enabled": true,
      "retention_days": 90
    }
  },
  
  "cache": {
    "enabled": true,
    "type": "redis",
    "size": 100000,
    "ttl_seconds": 3600
  },
  
  "logging": {
    "level": "INFO",
    "format": "json",
    "output": "file",
    "file": "/var/log/file-processor/application.log",
    "max_size_mb": 100,
    "backup_count": 10
  }
}
```

## Configuration Validation

The system automatically validates configuration files and provides detailed error messages:

### Validation Rules

- **Required Fields**: Certain fields are mandatory
- **Type Validation**: Values must match expected types
- **Range Validation**: Numeric values must be within valid ranges
- **Enum Validation**: String values must be from allowed options
- **Dependency Validation**: Some options require others to be enabled

### Example Validation Errors

```json
{
  "errors": [
    {
      "field": "performance.max_workers",
      "message": "Value must be between 1 and 32",
      "current_value": 50,
      "valid_range": "1-32"
    },
    {
      "field": "matching.algorithms.fuzzy.default_threshold",
      "message": "Value must be between 0 and 100",
      "current_value": 150,
      "valid_range": "0-100"
    },
    {
      "field": "output.format",
      "message": "Invalid format specified",
      "current_value": "xml",
      "valid_options": ["csv", "json", "excel", "parquet"]
    }
  ],
  "warnings": [
    {
      "field": "performance.chunk_size",
      "message": "Large chunk size may cause memory issues",
      "current_value": 50000,
      "recommended_max": 10000
    }
  ]
}
```

This configuration reference provides comprehensive documentation for all available options in the File Processing Optimization system, enabling users to customize the system for their specific needs and environments.