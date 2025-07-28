# Quick Start Guide

Get up and running with the File Processing API in minutes. This guide covers the essential steps to upload files, process them, and download results.

## Prerequisites

- API server running (default: `http://localhost:5000`)
- Two data files (CSV or JSON format) for matching
- Basic understanding of REST APIs
- Tool for making HTTP requests (curl, Postman, or programming language)

## Step-by-Step Tutorial

### Step 1: Authenticate

First, get a JWT token by logging in:

```bash
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 86400
}
```

Save the `access_token` - you'll need it for all subsequent requests.

### Step 2: Upload Files

Upload two files for processing:

```bash
curl -X POST http://localhost:5000/api/v1/files/upload \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "file1=@customers.csv" \
  -F "file2=@contacts.json"
```

**Response:**
```json
[
  {
    "original_name": "customers.csv",
    "unique_name": "customers_abc123.csv",
    "file_type": "csv",
    "delimiter": ",",
    "upload_timestamp": "2025-01-27T10:00:00Z"
  },
  {
    "original_name": "contacts.json",
    "unique_name": "contacts_def456.json",
    "file_type": "json",
    "upload_timestamp": "2025-01-27T10:00:00Z"
  }
]
```

### Step 3: Validate Files

Check that your files were uploaded correctly and see their structure:

```bash
curl -X GET http://localhost:5000/api/v1/files/validate \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Response:**
```json
{
  "file1": {
    "validation": {
      "valid": true,
      "errors": [],
      "warnings": [],
      "info": {
        "rows": 1000,
        "columns": 4,
        "column_names": ["name", "email", "phone", "city"]
      }
    },
    "preview": {
      "columns": ["name", "email", "phone", "city"],
      "preview": [
        {
          "name": "John Smith",
          "email": "john@example.com",
          "phone": "555-0123",
          "city": "New York"
        }
      ],
      "total_rows": 1000
    }
  },
  "file2": {
    "validation": {...},
    "preview": {...}
  }
}
```

### Step 4: Start Processing

Configure and start the matching process:

```bash
curl -X POST http://localhost:5000/api/v1/processing/start \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "mappings": [
      {
        "file1_col": "name",
        "file2_col": "full_name",
        "match_type": "fuzzy",
        "use_normalization": true,
        "weight": 1.0
      },
      {
        "file1_col": "email",
        "file2_col": "email_address",
        "match_type": "exact",
        "use_normalization": true,
        "weight": 2.0
      }
    ],
    "output_cols1": ["name", "email", "phone"],
    "output_cols2": ["full_name", "email_address", "address"],
    "threshold": 80,
    "generate_unmatched": true
  }'
```

**Response:**
```json
{
  "operation_id": "op_12345678-1234-1234-1234-123456789abc",
  "status": "starting",
  "progress": 0,
  "message": "Initializing processing...",
  "can_cancel": true
}
```

### Step 5: Monitor Progress

Check the processing status:

```bash
curl -X GET http://localhost:5000/api/v1/processing/status \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Response:**
```json
{
  "operation_id": "op_12345678-1234-1234-1234-123456789abc",
  "status": "processing",
  "progress": 65,
  "message": "Matching records...",
  "elapsed_time": 45,
  "estimated_remaining": 25,
  "can_cancel": true
}
```

Keep checking until `status` becomes `"completed"`.

### Step 6: Get Results

Once processing is complete, get the list of result files:

```bash
curl -X GET http://localhost:5000/api/v1/results/files \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

**Response:**
```json
[
  {
    "name": "Matched Results",
    "path": "/results/matched_results.json",
    "type": "matched",
    "count": 850,
    "columns": ["name", "email", "phone", "full_name", "email_address", "address", "confidence_score"]
  },
  {
    "name": "Unmatched from File 1",
    "path": "/results/unmatched_1.json",
    "type": "unmatched_1",
    "count": 150,
    "columns": ["name", "email", "phone"]
  }
]
```

### Step 7: Download Results

Download the matched results:

```bash
curl -X GET http://localhost:5000/api/v1/results/download/matched \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -o matched_results.json
```

Download unmatched records:

```bash
curl -X GET http://localhost:5000/api/v1/results/download/unmatched_1 \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -o unmatched_from_file1.json
```

### Step 8: Cleanup (Optional)

Clean up uploaded files and results to free space:

```bash
curl -X DELETE http://localhost:5000/api/v1/results/cleanup \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Complete Example Script

Here's a complete bash script that performs the entire workflow:

```bash
#!/bin/bash

# Configuration
BASE_URL="http://localhost:5000/api/v1"
USERNAME="admin"
PASSWORD="admin123"
FILE1="customers.csv"
FILE2="contacts.json"

echo "=== File Processing API Quick Start ==="

# Step 1: Authenticate
echo "1. Authenticating..."
TOKEN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')

if [ "$TOKEN" = "null" ]; then
    echo "❌ Authentication failed"
    echo "Response: $TOKEN_RESPONSE"
    exit 1
fi

echo "✅ Authentication successful"

# Step 2: Upload files
echo "2. Uploading files..."
UPLOAD_RESPONSE=$(curl -s -X POST "$BASE_URL/files/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file1=@$FILE1" \
  -F "file2=@$FILE2")

echo "✅ Files uploaded"
echo "Response: $UPLOAD_RESPONSE"

# Step 3: Validate files
echo "3. Validating files..."
VALIDATION_RESPONSE=$(curl -s -X GET "$BASE_URL/files/validate" \
  -H "Authorization: Bearer $TOKEN")

FILE1_VALID=$(echo $VALIDATION_RESPONSE | jq -r '.file1.validation.valid')
FILE2_VALID=$(echo $VALIDATION_RESPONSE | jq -r '.file2.validation.valid')

if [ "$FILE1_VALID" = "true" ] && [ "$FILE2_VALID" = "true" ]; then
    echo "✅ Files validated successfully"
    
    # Show file info
    FILE1_ROWS=$(echo $VALIDATION_RESPONSE | jq -r '.file1.validation.info.rows')
    FILE1_COLS=$(echo $VALIDATION_RESPONSE | jq -r '.file1.validation.info.columns')
    FILE2_ROWS=$(echo $VALIDATION_RESPONSE | jq -r '.file2.validation.info.rows')
    FILE2_COLS=$(echo $VALIDATION_RESPONSE | jq -r '.file2.validation.info.columns')
    
    echo "  File 1: $FILE1_ROWS rows, $FILE1_COLS columns"
    echo "  File 2: $FILE2_ROWS rows, $FILE2_COLS columns"
else
    echo "❌ File validation failed"
    echo "Response: $VALIDATION_RESPONSE"
    exit 1
fi

# Step 4: Start processing
echo "4. Starting processing..."
PROCESSING_CONFIG='{
  "mappings": [
    {
      "file1_col": "name",
      "file2_col": "full_name",
      "match_type": "fuzzy",
      "use_normalization": true,
      "weight": 1.0
    }
  ],
  "output_cols1": ["name", "email"],
  "output_cols2": ["full_name", "phone"],
  "threshold": 80,
  "generate_unmatched": true
}'

START_RESPONSE=$(curl -s -X POST "$BASE_URL/processing/start" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$PROCESSING_CONFIG")

OPERATION_ID=$(echo $START_RESPONSE | jq -r '.operation_id')

if [ "$OPERATION_ID" = "null" ]; then
    echo "❌ Failed to start processing"
    echo "Response: $START_RESPONSE"
    exit 1
fi

echo "✅ Processing started"
echo "Operation ID: $OPERATION_ID"

# Step 5: Monitor progress
echo "5. Monitoring progress..."
while true; do
    STATUS_RESPONSE=$(curl -s -X GET "$BASE_URL/processing/status" \
      -H "Authorization: Bearer $TOKEN")
    
    STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
    PROGRESS=$(echo $STATUS_RESPONSE | jq -r '.progress')
    MESSAGE=$(echo $STATUS_RESPONSE | jq -r '.message')
    
    echo "  Status: $STATUS, Progress: $PROGRESS%, Message: $MESSAGE"
    
    if [ "$STATUS" = "completed" ]; then
        echo "✅ Processing completed successfully"
        break
    elif [ "$STATUS" = "failed" ] || [ "$STATUS" = "cancelled" ]; then
        echo "❌ Processing failed or was cancelled"
        echo "Response: $STATUS_RESPONSE"
        exit 1
    fi
    
    sleep 5
done

# Step 6: Get results
echo "6. Getting results..."
RESULTS_RESPONSE=$(curl -s -X GET "$BASE_URL/results/files" \
  -H "Authorization: Bearer $TOKEN")

echo "✅ Results available:"
echo $RESULTS_RESPONSE | jq -r '.[] | "  - \(.name): \(.count) records"'

# Step 7: Download results
echo "7. Downloading results..."

# Download matched results
curl -s -X GET "$BASE_URL/results/download/matched" \
  -H "Authorization: Bearer $TOKEN" \
  -o "matched_results.json"

if [ -f "matched_results.json" ]; then
    echo "✅ Downloaded matched_results.json"
fi

# Download unmatched results
curl -s -X GET "$BASE_URL/results/download/unmatched_1" \
  -H "Authorization: Bearer $TOKEN" \
  -o "unmatched_1.json"

if [ -f "unmatched_1.json" ]; then
    echo "✅ Downloaded unmatched_1.json"
fi

curl -s -X GET "$BASE_URL/results/download/unmatched_2" \
  -H "Authorization: Bearer $TOKEN" \
  -o "unmatched_2.json"

if [ -f "unmatched_2.json" ]; then
    echo "✅ Downloaded unmatched_2.json"
fi

# Step 8: Cleanup
echo "8. Cleaning up..."
CLEANUP_RESPONSE=$(curl -s -X DELETE "$BASE_URL/results/cleanup" \
  -H "Authorization: Bearer $TOKEN")

REMOVED_FILES=$(echo $CLEANUP_RESPONSE | jq -r '.removed_result_files + .removed_upload_files')
echo "✅ Cleaned up $REMOVED_FILES files"

echo ""
echo "🎉 Quick start completed successfully!"
echo "Check the downloaded files for your results."
```

## Python Quick Start

```python
import requests
import time
import json

def quick_start_example():
    """Complete quick start example in Python."""
    
    base_url = "http://localhost:5000/api/v1"
    
    # Step 1: Authenticate
    print("1. Authenticating...")
    auth_response = requests.post(f"{base_url}/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    auth_response.raise_for_status()
    
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful")
    
    # Step 2: Upload files
    print("2. Uploading files...")
    with open("customers.csv", "rb") as f1, open("contacts.json", "rb") as f2:
        files = {"file1": f1, "file2": f2}
        upload_response = requests.post(f"{base_url}/files/upload", 
                                      files=files, headers=headers)
        upload_response.raise_for_status()
    
    print("✅ Files uploaded")
    
    # Step 3: Validate files
    print("3. Validating files...")
    validation_response = requests.get(f"{base_url}/files/validate", headers=headers)
    validation_response.raise_for_status()
    
    validation_data = validation_response.json()
    if validation_data["file1"]["validation"]["valid"] and validation_data["file2"]["validation"]["valid"]:
        print("✅ Files validated successfully")
        print(f"  File 1: {validation_data['file1']['validation']['info']['rows']} rows")
        print(f"  File 2: {validation_data['file2']['validation']['info']['rows']} rows")
    else:
        raise Exception("File validation failed")
    
    # Step 4: Start processing
    print("4. Starting processing...")
    config = {
        "mappings": [
            {
                "file1_col": "name",
                "file2_col": "full_name",
                "match_type": "fuzzy",
                "use_normalization": True,
                "weight": 1.0
            }
        ],
        "output_cols1": ["name", "email"],
        "output_cols2": ["full_name", "phone"],
        "threshold": 80,
        "generate_unmatched": True
    }
    
    start_response = requests.post(f"{base_url}/processing/start", 
                                 json=config, headers=headers)
    start_response.raise_for_status()
    
    operation_id = start_response.json()["operation_id"]
    print(f"✅ Processing started (ID: {operation_id})")
    
    # Step 5: Monitor progress
    print("5. Monitoring progress...")
    while True:
        status_response = requests.get(f"{base_url}/processing/status", headers=headers)
        status_response.raise_for_status()
        
        status_data = status_response.json()
        status = status_data["status"]
        progress = status_data["progress"]
        message = status_data["message"]
        
        print(f"  Status: {status}, Progress: {progress}%, Message: {message}")
        
        if status == "completed":
            print("✅ Processing completed successfully")
            break
        elif status in ["failed", "cancelled"]:
            raise Exception(f"Processing {status}")
        
        time.sleep(5)
    
    # Step 6: Get results
    print("6. Getting results...")
    results_response = requests.get(f"{base_url}/results/files", headers=headers)
    results_response.raise_for_status()
    
    results_data = results_response.json()
    print("✅ Results available:")
    for result in results_data:
        print(f"  - {result['name']}: {result['count']} records")
    
    # Step 7: Download results
    print("7. Downloading results...")
    
    # Download matched results
    matched_response = requests.get(f"{base_url}/results/download/matched", 
                                  headers=headers)
    matched_response.raise_for_status()
    
    with open("matched_results.json", "wb") as f:
        f.write(matched_response.content)
    print("✅ Downloaded matched_results.json")
    
    # Step 8: Cleanup
    print("8. Cleaning up...")
    cleanup_response = requests.delete(f"{base_url}/results/cleanup", headers=headers)
    cleanup_response.raise_for_status()
    
    cleanup_data = cleanup_response.json()
    total_removed = cleanup_data["removed_result_files"] + cleanup_data["removed_upload_files"]
    print(f"✅ Cleaned up {total_removed} files")
    
    print("\n🎉 Quick start completed successfully!")
    print("Check matched_results.json for your results.")

if __name__ == "__main__":
    try:
        quick_start_example()
    except Exception as e:
        print(f"❌ Error: {e}")
```

## Common Configuration Options

### Basic Fuzzy Matching
```json
{
  "mappings": [
    {
      "file1_col": "name",
      "file2_col": "full_name",
      "match_type": "fuzzy",
      "use_normalization": true,
      "weight": 1.0
    }
  ],
  "threshold": 80
}
```

### Exact Email Matching
```json
{
  "mappings": [
    {
      "file1_col": "email",
      "file2_col": "email_address",
      "match_type": "exact",
      "case_sensitive": false,
      "weight": 2.0
    }
  ],
  "threshold": 100
}
```

### Multi-field Matching
```json
{
  "mappings": [
    {
      "file1_col": "name",
      "file2_col": "full_name",
      "match_type": "fuzzy",
      "weight": 1.0
    },
    {
      "file1_col": "email",
      "file2_col": "email_address",
      "match_type": "exact",
      "weight": 2.0
    },
    {
      "file1_col": "phone",
      "file2_col": "phone_number",
      "match_type": "phonetic",
      "weight": 1.5
    }
  ],
  "threshold": 75
}
```

## Next Steps

- Read the [Authentication Guide](authentication.md) for advanced token management
- Check out [Job Management](job-management.md) for batch processing
- Explore the [complete API documentation](README.md) for all available endpoints
- Review error handling patterns and best practices
- Set up monitoring and logging for production use

## Troubleshooting

**File Upload Issues:**
- Check file size limits (default: 100MB)
- Ensure files are valid CSV or JSON format
- Verify file permissions and accessibility

**Authentication Problems:**
- Confirm username/password are correct
- Check token expiration (default: 24 hours)
- Ensure Authorization header format: `Bearer <token>`

**Processing Failures:**
- Verify column names exist in both files
- Check threshold values (0-100)
- Ensure sufficient system resources

**Download Issues:**
- Wait for processing to complete before downloading
- Check that result files were generated
- Verify network connectivity and timeouts