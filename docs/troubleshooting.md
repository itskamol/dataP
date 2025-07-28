# Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when using the File Processing Optimization system. Issues are organized by category with step-by-step resolution instructions.

## Quick Diagnosis

### System Health Check

```bash
# Check application health
curl http://localhost:5000/health

# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"

# Check log files
tail -f logs/application.log
tail -f logs/errors.log
```

### Common Error Patterns

**Memory Errors:**
- `MemoryError: Unable to allocate array`
- `OutOfMemoryError: Java heap space`
- `Process killed (OOM)`

**File Errors:**
- `FileNotFoundError: No such file or directory`
- `PermissionError: Access denied`
- `UnicodeDecodeError: Invalid encoding`

**Processing Errors:**
- `TimeoutError: Operation timed out`
- `ValidationError: Invalid configuration`
- `MatchingError: Algorithm failed`

## File Processing Issues

### 1. File Upload Problems

#### Issue: File Upload Fails
**Symptoms:**
- Upload button doesn't respond
- "File too large" error
- "Invalid file format" error

**Diagnosis:**
```bash
# Check file size
ls -lh your_file.csv

# Check file format
file your_file.csv

# Check upload directory permissions
ls -la uploads/
```

**Solutions:**

**File Too Large:**
```json
// Increase limits in config.json
{
  "file_upload": {
    "max_file_size_mb": 500,  // Increase from default
    "chunk_size_mb": 10       // Enable chunked upload
  }
}
```

**Invalid Format:**
```python
# Validate file format manually
import pandas as pd

try:
    df = pd.read_csv('your_file.csv', nrows=5)
    print("File format is valid")
    print(df.head())
except Exception as e:
    print(f"Format error: {e}")
```

**Permission Issues:**
```bash
# Fix upload directory permissions
sudo chown -R $USER:$USER uploads/
chmod 755 uploads/
```

#### Issue: Encoding Problems
**Symptoms:**
- `UnicodeDecodeError` during file processing
- Garbled text in results
- Missing characters in Uzbek text

**Diagnosis:**
```python
# Detect file encoding
import chardet

with open('your_file.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(f"Detected encoding: {result['encoding']}")
    print(f"Confidence: {result['confidence']}")
```

**Solutions:**

**Specify Encoding:**
```json
{
  "file1": {
    "encoding": "utf-8",  // or "cp1251", "iso-8859-1"
    "encoding_errors": "replace"  // or "ignore", "strict"
  }
}
```

**Convert Encoding:**
```bash
# Convert file encoding
iconv -f cp1251 -t utf-8 input.csv > output.csv

# Or using Python
python -c "
import codecs
with codecs.open('input.csv', 'r', 'cp1251') as f:
    content = f.read()
with codecs.open('output.csv', 'w', 'utf-8') as f:
    f.write(content)
"
```

### 2. CSV Parsing Issues

#### Issue: Delimiter Detection Fails
**Symptoms:**
- All data appears in single column
- Incorrect column separation
- "Unable to detect delimiter" error

**Diagnosis:**
```python
# Manual delimiter detection
import csv

with open('your_file.csv', 'r') as f:
    sample = f.read(1024)
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
    print(f"Detected delimiter: '{delimiter}'")
```

**Solutions:**

**Specify Delimiter:**
```json
{
  "file1": {
    "delimiter": ";",  // or "\t", "|", etc.
    "quotechar": "\"",
    "escapechar": "\\"
  }
}
```

**Handle Complex Delimiters:**
```python
# For files with multiple delimiters
import pandas as pd

# Try different delimiters
delimiters = [',', ';', '\t', '|']
for delim in delimiters:
    try:
        df = pd.read_csv('your_file.csv', delimiter=delim, nrows=5)
        if len(df.columns) > 1:
            print(f"Working delimiter: '{delim}'")
            break
    except:
        continue
```

#### Issue: Header Detection Problems
**Symptoms:**
- Column names appear as data rows
- Missing column headers
- Incorrect column mapping

**Solutions:**

**Specify Header Row:**
```json
{
  "file1": {
    "header_row": 0,  // 0-based index, or null for no header
    "skip_rows": 2    // Skip first 2 rows
  }
}
```

**Manual Header Specification:**
```json
{
  "file1": {
    "column_names": ["name", "age", "city", "email"],
    "header_row": null
  }
}
```

### 3. JSON Processing Issues

#### Issue: Invalid JSON Structure
**Symptoms:**
- `JSONDecodeError: Expecting value`
- "Invalid JSON format" error
- Partial data loading

**Diagnosis:**
```python
import json

try:
    with open('your_file.json', 'r') as f:
        data = json.load(f)
    print("JSON is valid")
except json.JSONDecodeError as e:
    print(f"JSON error at line {e.lineno}, column {e.colno}: {e.msg}")
```

**Solutions:**

**Fix JSON Format:**
```bash
# Validate and format JSON
python -m json.tool your_file.json > formatted.json

# Or use jq
jq '.' your_file.json > formatted.json
```

**Handle Large JSON Files:**
```python
# Stream large JSON files
import ijson

def stream_json(filename):
    with open(filename, 'rb') as f:
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if event == 'start_array':
                # Process array items
                pass
```

## Memory and Performance Issues

### 1. Out of Memory Errors

#### Issue: System Runs Out of Memory
**Symptoms:**
- `MemoryError` exceptions
- System becomes unresponsive
- Process killed by OS

**Diagnosis:**
```python
# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
print(f"Memory percent: {process.memory_percent():.2f}%")

# Check system memory
mem = psutil.virtual_memory()
print(f"Total: {mem.total / 1024 / 1024 / 1024:.2f} GB")
print(f"Available: {mem.available / 1024 / 1024 / 1024:.2f} GB")
print(f"Used: {mem.percent}%")
```

**Solutions:**

**Reduce Memory Usage:**
```json
{
  "processing": {
    "memory_limit_mb": 1024,    // Reduce from default
    "chunk_size": 1000,         // Process smaller chunks
    "streaming_mode": true,     // Enable streaming
    "cache_size": 5000         // Reduce cache size
  }
}
```

**Enable Streaming Processing:**
```python
# Process files in chunks
def process_large_file(filename, chunk_size=1000):
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process each chunk
        yield process_chunk(chunk)
```

**Use Memory Mapping:**
```python
# Memory-mapped file access
import mmap

with open('large_file.csv', 'r') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # Process memory-mapped file
        pass
```

### 2. Performance Issues

#### Issue: Slow Processing Speed
**Symptoms:**
- Processing takes hours for medium datasets
- High CPU usage with low throughput
- Frequent timeouts

**Diagnosis:**
```python
# Profile performance
import cProfile
import pstats

# Profile your processing function
cProfile.run('your_processing_function()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

**Solutions:**

**Enable Parallel Processing:**
```json
{
  "processing": {
    "max_workers": 4,           // Use multiple CPU cores
    "parallel_matching": true,  // Enable parallel matching
    "batch_size": 10000        // Optimize batch size
  }
}
```

**Optimize Blocking Strategy:**
```json
{
  "matching": {
    "blocking": {
      "enabled": true,
      "strategy": "phonetic",    // or "ngram", "exact"
      "block_size": 1000,
      "max_comparisons": 100000
    }
  }
}
```

**Use Caching:**
```json
{
  "cache": {
    "type": "redis",           // or "memory"
    "url": "redis://localhost:6379/0",
    "size": 50000,
    "ttl_seconds": 3600
  }
}
```

## Matching Algorithm Issues

### 1. Poor Matching Accuracy

#### Issue: Low Match Quality
**Symptoms:**
- Many false positives/negatives
- Low confidence scores
- Missing obvious matches

**Diagnosis:**
```python
# Analyze matching results
import pandas as pd

results = pd.read_json('matched_results.json')
print(f"Average confidence: {results['confidence_score'].mean():.2f}")
print(f"Matches below 80%: {(results['confidence_score'] < 80).sum()}")

# Check specific examples
low_confidence = results[results['confidence_score'] < 70]
print(low_confidence[['f1_name', 'f2_name', 'confidence_score']].head())
```

**Solutions:**

**Adjust Thresholds:**
```json
{
  "matching": {
    "thresholds": {
      "minimum_confidence": 70,  // Lower for more matches
      "fuzzy_threshold": 75,     // Adjust fuzzy matching
      "exact_weight": 2.0,       // Increase exact match weight
      "fuzzy_weight": 1.0
    }
  }
}
```

**Enable Text Normalization:**
```json
{
  "matching": {
    "normalization": {
      "enabled": true,
      "lowercase": true,
      "remove_punctuation": true,
      "remove_extra_spaces": true,
      "uzbek_specific": true     // Enable Uzbek text processing
    }
  }
}
```

**Use Multiple Algorithms:**
```json
{
  "matching": {
    "algorithms": [
      {
        "name": "exact",
        "weight": 2.0,
        "enabled": true
      },
      {
        "name": "fuzzy",
        "weight": 1.0,
        "threshold": 80,
        "enabled": true
      },
      {
        "name": "phonetic",
        "weight": 0.5,
        "enabled": true
      }
    ]
  }
}
```

### 2. Uzbek Text Processing Issues

#### Issue: Incorrect Uzbek Text Matching
**Symptoms:**
- Similar Uzbek words not matching
- Encoding issues with Cyrillic/Latin scripts
- Poor phonetic matching

**Solutions:**

**Enable Uzbek Normalization:**
```json
{
  "matching": {
    "uzbek_processing": {
      "enabled": true,
      "script_conversion": true,    // Convert between Cyrillic/Latin
      "phonetic_matching": true,    // Enable phonetic rules
      "common_variations": true     // Handle common spelling variations
    }
  }
}
```

**Custom Uzbek Rules:**
```python
# Add custom normalization rules
uzbek_rules = {
    "replacements": {
        "ў": "u",
        "ғ": "g",
        "қ": "q",
        "ҳ": "h"
    },
    "phonetic_groups": [
        ["и", "ий", "ы"],
        ["у", "ў"],
        ["о", "а"]
    ]
}
```

## Web Interface Issues

### 1. UI Not Loading

#### Issue: Web Interface Doesn't Load
**Symptoms:**
- Blank page or loading spinner
- JavaScript errors in browser console
- 500 Internal Server Error

**Diagnosis:**
```bash
# Check web server logs
tail -f logs/application.log | grep ERROR

# Check browser console (F12)
# Look for JavaScript errors

# Test API endpoints directly
curl http://localhost:5000/health
curl http://localhost:5000/api/v1/status
```

**Solutions:**

**Check Dependencies:**
```bash
# Verify Flask installation
python -c "import flask; print(flask.__version__)"

# Check static files
ls -la src/web/static/
ls -la src/web/templates/
```

**Fix Static File Issues:**
```python
# In Flask app configuration
app.static_folder = 'static'
app.static_url_path = '/static'

# Or serve static files with nginx in production
```

### 2. WebSocket Connection Issues

#### Issue: Real-time Updates Not Working
**Symptoms:**
- Progress bar doesn't update
- "WebSocket connection failed" error
- Stale progress information

**Diagnosis:**
```javascript
// Check WebSocket connection in browser console
const ws = new WebSocket('ws://localhost:5000/socket.io/');
ws.onopen = () => console.log('Connected');
ws.onerror = (error) => console.log('Error:', error);
```

**Solutions:**

**Check WebSocket Configuration:**
```python
# Ensure SocketIO is properly configured
from flask_socketio import SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")
```

**Firewall/Proxy Issues:**
```nginx
# Nginx configuration for WebSocket
location /socket.io/ {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

## API Issues

### 1. Authentication Problems

#### Issue: API Authentication Fails
**Symptoms:**
- 401 Unauthorized errors
- "Invalid token" messages
- Token expiration issues

**Diagnosis:**
```bash
# Test authentication
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Verify token
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:5000/api/v1/auth/verify
```

**Solutions:**

**Check Credentials:**
```json
// Default credentials (change in production)
{
  "username": "admin",
  "password": "admin123"
}
```

**Token Configuration:**
```json
{
  "jwt": {
    "secret_key": "your-secret-key",
    "expiration_hours": 24,
    "algorithm": "HS256"
  }
}
```

### 2. Rate Limiting Issues

#### Issue: API Rate Limits Exceeded
**Symptoms:**
- 429 Too Many Requests errors
- Requests being blocked
- Slow API responses

**Solutions:**

**Adjust Rate Limits:**
```json
{
  "rate_limiting": {
    "default": "1000 per hour",
    "authenticated": "2000 per hour",
    "upload": "50 per hour"
  }
}
```

**Implement Backoff:**
```python
import time
import requests

def api_request_with_backoff(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
            continue
        return response
    raise Exception("Max retries exceeded")
```

## Configuration Issues

### 1. Invalid Configuration

#### Issue: Configuration Validation Fails
**Symptoms:**
- "Invalid configuration" errors
- Application won't start
- Missing required parameters

**Diagnosis:**
```python
# Validate configuration manually
from src.application.services.config_service import ConfigurationManager

try:
    config_manager = ConfigurationManager('config.json')
    config = config_manager.load_config()
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

**Solutions:**

**Use Configuration Template:**
```json
{
  "environment": "development",
  "debug": true,
  "log_level": "INFO",
  "file_upload": {
    "max_file_size_mb": 100,
    "allowed_extensions": [".csv", ".json", ".xlsx"],
    "upload_directory": "uploads"
  },
  "processing": {
    "max_workers": 2,
    "memory_limit_mb": 1024,
    "timeout_seconds": 3600
  },
  "matching": {
    "mappings": [],
    "algorithms": [
      {
        "name": "exact",
        "enabled": true,
        "priority": 1
      }
    ],
    "thresholds": {
      "minimum_confidence": 75.0
    }
  }
}
```

**Validate Schema:**
```python
# Use JSON schema validation
import jsonschema

schema = {
    "type": "object",
    "required": ["environment", "file_upload", "processing"],
    "properties": {
        "environment": {"type": "string"},
        "file_upload": {
            "type": "object",
            "required": ["max_file_size_mb"]
        }
    }
}

jsonschema.validate(config_data, schema)
```

## Deployment Issues

### 1. Docker Container Problems

#### Issue: Container Won't Start
**Symptoms:**
- Container exits immediately
- "No such file or directory" errors
- Permission denied errors

**Diagnosis:**
```bash
# Check container logs
docker logs container_name

# Run container interactively
docker run -it --entrypoint /bin/bash image_name

# Check file permissions
docker exec container_name ls -la /app/
```

**Solutions:**

**Fix File Permissions:**
```dockerfile
# In Dockerfile
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser
```

**Check Dependencies:**
```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

### 2. Kubernetes Deployment Issues

#### Issue: Pods Not Starting
**Symptoms:**
- Pods stuck in Pending state
- ImagePullBackOff errors
- CrashLoopBackOff errors

**Diagnosis:**
```bash
# Check pod status
kubectl describe pod pod_name

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Check logs
kubectl logs pod_name
```

**Solutions:**

**Resource Issues:**
```yaml
# Adjust resource requests
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

**Image Issues:**
```yaml
# Use specific image tag
image: file-processor:v1.0.0
imagePullPolicy: Always
```

## Emergency Procedures

### 1. System Recovery

**High Memory Usage:**
```bash
# Kill high-memory processes
pkill -f "python.*file.*processor"

# Clear caches
echo 3 > /proc/sys/vm/drop_caches

# Restart services
systemctl restart file-processor
```

**Disk Space Issues:**
```bash
# Clean up old files
find uploads/ -type f -mtime +7 -delete
find results/ -type f -mtime +7 -delete
find logs/ -type f -mtime +30 -delete

# Compress old logs
gzip logs/*.log
```

### 2. Data Recovery

**Recover Interrupted Processing:**
```python
# Check for partial results
import os
import json

def recover_partial_results(operation_id):
    temp_file = f"temp/partial_{operation_id}.json"
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            return json.load(f)
    return None
```

**Backup Critical Data:**
```bash
# Backup configuration
cp config.json config.json.backup.$(date +%Y%m%d)

# Backup results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/

# Backup database
pg_dump fileprocessor > backup_$(date +%Y%m%d).sql
```

## Getting Help

### 1. Log Analysis

**Enable Debug Logging:**
```json
{
  "log_level": "DEBUG",
  "debug": true
}
```

**Collect System Information:**
```bash
# System info script
cat > collect_info.sh << 'EOF'
#!/bin/bash
echo "=== System Information ==="
uname -a
python --version
pip list | grep -E "(pandas|flask|redis)"

echo "=== Memory Usage ==="
free -h

echo "=== Disk Usage ==="
df -h

echo "=== Process Information ==="
ps aux | grep python

echo "=== Recent Errors ==="
tail -50 logs/errors.log
EOF

chmod +x collect_info.sh
./collect_info.sh > system_info.txt
```

### 2. Support Channels

**Before Contacting Support:**
1. Check this troubleshooting guide
2. Review application logs
3. Verify configuration
4. Test with minimal dataset
5. Collect system information

**Information to Provide:**
- Error messages and stack traces
- Configuration files (remove sensitive data)
- System information
- Steps to reproduce the issue
- Expected vs actual behavior

This troubleshooting guide covers the most common issues encountered with the File Processing Optimization system. For issues not covered here, please check the application logs and contact the development team with detailed information about the problem.