# File Processing API Documentation

## Overview

The File Processing API provides comprehensive endpoints for uploading, processing, and matching data between two files (CSV or JSON format). This RESTful API supports various matching algorithms including exact, fuzzy, and phonetic matching with real-time progress tracking, advanced configuration options, and robust error handling.

## API Specification

**Version:** 1.0.0  
**Base URL:** `/api/v1`  
**Content-Type:** `application/json`  
**Authentication:** JWT Bearer Token  

### OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Interactive Docs:** `http://localhost:5000/api/v1/docs/`
- **JSON Spec:** `http://localhost:5000/api/v1/swagger.json`
- **YAML Spec:** `http://localhost:5000/api/v1/swagger.yaml`

## Base URL

- Development: `http://localhost:5000/api/v1`
- Production: `https://your-domain.com/api/v1`

## Authentication

All API endpoints (except `/auth/login`) require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

```bash
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 86400
}
```

## Rate Limits

- Default: 1000 requests per hour, 100 per minute
- Authenticated users: 2000 requests per hour, 200 per minute
- File uploads: 50 requests per hour, 10 per minute
- Processing operations: 20 requests per hour, 5 per minute

## Documentation Structure

This API documentation is organized into several guides:

- **[Quick Start Guide](quick-start.md)** - Get started in minutes with a complete example
- **[Authentication Guide](authentication.md)** - Detailed JWT authentication and security
- **[Job Management Guide](job-management.md)** - Batch processing and queue management
- **[Complete API Reference](#api-endpoints)** - All endpoints with examples (below)

## API Endpoints

### Authentication

#### POST /auth/login
Authenticate user and get JWT token.

**Request:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "jwt-token-here",
  "expires_in": 86400
}
```

#### GET /auth/verify
Verify JWT token validity.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "valid": true,
  "user": "admin",
  "expires_at": 1643723400
}
```

### File Management

#### POST /files/upload
Upload two files for processing.

**Headers:** 
- `Authorization: Bearer <token>`
- `Content-Type: multipart/form-data`

**Form Data:**
- `file1`: First file (CSV or JSON)
- `file2`: Second file (CSV or JSON)

**Response:**
```json
[
  {
    "original_name": "file1.csv",
    "unique_name": "file1_uuid.csv",
    "file_type": "csv",
    "delimiter": ",",
    "upload_timestamp": "2025-01-27T00:00:00Z"
  },
  {
    "original_name": "file2.json",
    "unique_name": "file2_uuid.json",
    "file_type": "json",
    "upload_timestamp": "2025-01-27T00:00:00Z"
  }
]
```

#### GET /files/validate
Validate uploaded files and return structure information.

**Headers:** `Authorization: Bearer <token>`

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
        "columns": 5,
        "column_names": ["name", "age", "city", "email", "phone"]
      }
    },
    "preview": {
      "columns": ["name", "age", "city", "email", "phone"],
      "preview": [
        {"name": "John", "age": 25, "city": "NYC", "email": "john@example.com", "phone": "123-456-7890"}
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

#### GET /files/preview
Get preview of uploaded files.

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**
- `rows` (optional): Number of preview rows (default: 5, max: 100)

### Processing

#### POST /processing/start
Start file processing operation.

**Headers:** 
- `Authorization: Bearer <token>`
- `Content-Type: application/json`

**Request:**
```json
{
  "mappings": [
    {
      "file1_col": "name",
      "file2_col": "full_name",
      "match_type": "fuzzy",
      "use_normalization": true,
      "case_sensitive": false,
      "weight": 1.0
    }
  ],
  "output_cols1": ["name", "age", "city"],
  "output_cols2": ["full_name", "email", "phone"],
  "output_format": "json",
  "output_path": "matched_results",
  "threshold": 80,
  "matching_type": "one-to-one",
  "generate_unmatched": true
}
```

**Response:**
```json
{
  "operation_id": "uuid-here",
  "status": "starting",
  "progress": 0,
  "message": "Initializing processing...",
  "can_cancel": true
}
```

#### GET /processing/status
Get current processing status.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "operation_id": "uuid-here",
  "status": "processing",
  "progress": 45,
  "message": "Matching records...",
  "elapsed_time": 120,
  "estimated_remaining": 150,
  "can_cancel": true
}
```

#### POST /processing/cancel
Cancel current processing operation.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "success": true,
  "message": "Processing cancelled successfully"
}
```

### Results

#### GET /results/files
Get list of result files.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
[
  {
    "name": "Matched Results",
    "path": "/path/to/matched.json",
    "type": "matched",
    "count": 850,
    "columns": ["f1_name", "f1_age", "f2_full_name", "f2_email", "confidence_score"]
  },
  {
    "name": "Unmatched from File 1",
    "path": "/path/to/unmatched_1.json",
    "type": "unmatched_1",
    "count": 150,
    "columns": ["name", "age", "city"]
  }
]
```

#### GET /results/data/{file_type}
Get paginated result data.

**Headers:** `Authorization: Bearer <token>`

**Path Parameters:**
- `file_type`: One of `matched`, `low_confidence`, `unmatched_1`, `unmatched_2`

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Records per page (default: 50, max: 1000)
- `search` (optional): Search term to filter results

**Response:**
```json
{
  "data": [
    {
      "f1_name": "John Smith",
      "f1_age": 25,
      "f2_full_name": "John Smith",
      "f2_email": "john@example.com",
      "confidence_score": 95.5
    }
  ],
  "total": 850,
  "page": 1,
  "per_page": 50,
  "total_pages": 17
}
```

#### GET /results/download/{file_type}
Download result file.

**Headers:** `Authorization: Bearer <token>`

**Path Parameters:**
- `file_type`: One of `matched`, `low_confidence`, `unmatched_1`, `unmatched_2`

**Response:** File download

#### DELETE /results/cleanup
Clean up result files and session data.

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "success": true,
  "removed_result_files": 4,
  "removed_upload_files": 2,
  "message": "Cleaned up 6 files"
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error Type",
  "details": "Detailed error message",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (missing or invalid token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource not found)
- `413` - Request Entity Too Large (file too large)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

## Workflow Example

1. **Authenticate:**
   ```bash
   curl -X POST http://localhost:5000/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'
   ```

2. **Upload Files:**
   ```bash
   curl -X POST http://localhost:5000/api/v1/files/upload \
     -H "Authorization: Bearer <token>" \
     -F "file1=@data1.csv" \
     -F "file2=@data2.json"
   ```

3. **Validate Files:**
   ```bash
   curl -X GET http://localhost:5000/api/v1/files/validate \
     -H "Authorization: Bearer <token>"
   ```

4. **Start Processing:**
   ```bash
   curl -X POST http://localhost:5000/api/v1/processing/start \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "mappings": [
         {
           "file1_col": "name",
           "file2_col": "full_name",
           "match_type": "fuzzy"
         }
       ],
       "output_cols1": ["name", "age"],
       "output_cols2": ["full_name", "email"],
       "threshold": 80
     }'
   ```

5. **Monitor Progress:**
   ```bash
   curl -X GET http://localhost:5000/api/v1/processing/status \
     -H "Authorization: Bearer <token>"
   ```

6. **Download Results:**
   ```bash
   curl -X GET http://localhost:5000/api/v1/results/download/matched \
     -H "Authorization: Bearer <token>" \
     -o matched_results.json
   ```

7. **Cleanup:**
   ```bash
   curl -X DELETE http://localhost:5000/api/v1/results/cleanup \
     -H "Authorization: Bearer <token>"
   ```

## Interactive Documentation

When the API server is running, you can access interactive Swagger UI documentation at:

- `http://localhost:5000/api/v1/docs/`

The OpenAPI specification is available at:

- `http://localhost:5000/api/v1/swagger.json`

## Health Check

Check API health status:

```bash
curl -X GET http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T00:00:00Z",
  "version": "1.0.0",
  "components": {
    "upload_directory": "healthy",
    "rate_limiter": "healthy",
    "authentication": "healthy"
  }
}
```
## C
ode Examples

### Python Client Example

```python
import requests
import json
from typing import Dict, Any, Optional

class FileProcessingClient:
    """Python client for File Processing API."""
    
    def __init__(self, base_url: str = "http://localhost:5000/api/v1"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.session = requests.Session()
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate and store JWT token."""
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            return True
        return False
    
    def upload_files(self, file1_path: str, file2_path: str) -> Dict[str, Any]:
        """Upload two files for processing."""
        with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
            files = {
                'file1': f1,
                'file2': f2
            }
            response = self.session.post(
                f"{self.base_url}/files/upload",
                files=files
            )
        
        response.raise_for_status()
        return response.json()
    
    def start_processing(self, config: Dict[str, Any]) -> str:
        """Start file processing and return operation ID."""
        response = self.session.post(
            f"{self.base_url}/processing/start",
            json=config
        )
        
        response.raise_for_status()
        return response.json()["operation_id"]
    
    def wait_for_completion(self, operation_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for processing to complete."""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/processing/status")
            response.raise_for_status()
            
            status = response.json()
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            time.sleep(5)  # Poll every 5 seconds
        
        raise TimeoutError("Processing timed out")
    
    def download_results(self, file_type: str, output_path: str):
        """Download results to local file."""
        response = self.session.get(
            f"{self.base_url}/results/download/{file_type}",
            stream=True
        )
        
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Usage example
def main():
    client = FileProcessingClient()
    
    # Authenticate
    if not client.authenticate("admin", "admin123"):
        print("Authentication failed")
        return
    
    # Upload files
    print("Uploading files...")
    upload_result = client.upload_files("data1.csv", "data2.json")
    print(f"Files uploaded: {upload_result}")
    
    # Configure processing
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
        "output_cols1": ["name", "age", "city"],
        "output_cols2": ["full_name", "email", "phone"],
        "threshold": 80,
        "generate_unmatched": True
    }
    
    # Start processing
    print("Starting processing...")
    operation_id = client.start_processing(config)
    print(f"Operation ID: {operation_id}")
    
    # Wait for completion
    print("Waiting for completion...")
    final_status = client.wait_for_completion(operation_id)
    print(f"Processing completed: {final_status}")
    
    # Download results
    if final_status["status"] == "completed":
        print("Downloading results...")
        client.download_results("matched", "matched_results.json")
        client.download_results("unmatched_1", "unmatched_1.json")
        print("Results downloaded successfully")

if __name__ == "__main__":
    main()
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class FileProcessingClient {
    constructor(baseUrl = 'http://localhost:5000/api/v1') {
        this.baseUrl = baseUrl;
        this.token = null;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000
        });
    }

    async authenticate(username, password) {
        try {
            const response = await this.client.post('/auth/login', {
                username,
                password
            });
            
            this.token = response.data.access_token;
            this.client.defaults.headers.common['Authorization'] = `Bearer ${this.token}`;
            return true;
        } catch (error) {
            console.error('Authentication failed:', error.response?.data);
            return false;
        }
    }

    async uploadFiles(file1Path, file2Path) {
        const formData = new FormData();
        formData.append('file1', fs.createReadStream(file1Path));
        formData.append('file2', fs.createReadStream(file2Path));

        const response = await this.client.post('/files/upload', formData, {
            headers: formData.getHeaders()
        });

        return response.data;
    }

    async startProcessing(config) {
        const response = await this.client.post('/processing/start', config);
        return response.data.operation_id;
    }

    async waitForCompletion(operationId, timeout = 3600000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const response = await this.client.get('/processing/status');
            const status = response.data;
            
            if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                return status;
            }
            
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        
        throw new Error('Processing timed out');
    }

    async downloadResults(fileType, outputPath) {
        const response = await this.client.get(`/results/download/${fileType}`, {
            responseType: 'stream'
        });

        const writer = fs.createWriteStream(outputPath);
        response.data.pipe(writer);

        return new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });
    }
}

// Usage example
async function main() {
    const client = new FileProcessingClient();
    
    try {
        // Authenticate
        console.log('Authenticating...');
        const authenticated = await client.authenticate('admin', 'admin123');
        if (!authenticated) {
            console.error('Authentication failed');
            return;
        }
        
        // Upload files
        console.log('Uploading files...');
        const uploadResult = await client.uploadFiles('data1.csv', 'data2.json');
        console.log('Files uploaded:', uploadResult);
        
        // Configure processing
        const config = {
            mappings: [{
                file1_col: 'name',
                file2_col: 'full_name',
                match_type: 'fuzzy',
                use_normalization: true,
                weight: 1.0
            }],
            output_cols1: ['name', 'age', 'city'],
            output_cols2: ['full_name', 'email', 'phone'],
            threshold: 80,
            generate_unmatched: true
        };
        
        // Start processing
        console.log('Starting processing...');
        const operationId = await client.startProcessing(config);
        console.log('Operation ID:', operationId);
        
        // Wait for completion
        console.log('Waiting for completion...');
        const finalStatus = await client.waitForCompletion(operationId);
        console.log('Processing completed:', finalStatus);
        
        // Download results
        if (finalStatus.status === 'completed') {
            console.log('Downloading results...');
            await client.downloadResults('matched', 'matched_results.json');
            await client.downloadResults('unmatched_1', 'unmatched_1.json');
            console.log('Results downloaded successfully');
        }
        
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

main();
```

### cURL Examples

#### Complete Workflow

```bash
#!/bin/bash

# Configuration
BASE_URL="http://localhost:5000/api/v1"
USERNAME="admin"
PASSWORD="admin123"

# Step 1: Authenticate
echo "Authenticating..."
TOKEN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")

TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')

if [ "$TOKEN" = "null" ]; then
    echo "Authentication failed"
    exit 1
fi

echo "Authentication successful"

# Step 2: Upload files
echo "Uploading files..."
UPLOAD_RESPONSE=$(curl -s -X POST "$BASE_URL/files/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file1=@data1.csv" \
  -F "file2=@data2.json")

echo "Upload response: $UPLOAD_RESPONSE"

# Step 3: Validate files
echo "Validating files..."
VALIDATION_RESPONSE=$(curl -s -X GET "$BASE_URL/files/validate" \
  -H "Authorization: Bearer $TOKEN")

echo "Validation response: $VALIDATION_RESPONSE"

# Step 4: Start processing
echo "Starting processing..."
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
  "output_cols1": ["name", "age", "city"],
  "output_cols2": ["full_name", "email", "phone"],
  "threshold": 80,
  "generate_unmatched": true
}'

START_RESPONSE=$(curl -s -X POST "$BASE_URL/processing/start" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$PROCESSING_CONFIG")

OPERATION_ID=$(echo $START_RESPONSE | jq -r '.operation_id')
echo "Operation ID: $OPERATION_ID"

# Step 5: Monitor progress
echo "Monitoring progress..."
while true; do
    STATUS_RESPONSE=$(curl -s -X GET "$BASE_URL/processing/status" \
      -H "Authorization: Bearer $TOKEN")
    
    STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
    PROGRESS=$(echo $STATUS_RESPONSE | jq -r '.progress')
    MESSAGE=$(echo $STATUS_RESPONSE | jq -r '.message')
    
    echo "Status: $STATUS, Progress: $PROGRESS%, Message: $MESSAGE"
    
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] || [ "$STATUS" = "cancelled" ]; then
        break
    fi
    
    sleep 5
done

# Step 6: Download results (if completed)
if [ "$STATUS" = "completed" ]; then
    echo "Downloading results..."
    
    curl -X GET "$BASE_URL/results/download/matched" \
      -H "Authorization: Bearer $TOKEN" \
      -o "matched_results.json"
    
    curl -X GET "$BASE_URL/results/download/unmatched_1" \
      -H "Authorization: Bearer $TOKEN" \
      -o "unmatched_1.json"
    
    curl -X GET "$BASE_URL/results/download/unmatched_2" \
      -H "Authorization: Bearer $TOKEN" \
      -o "unmatched_2.json"
    
    echo "Results downloaded successfully"
    
    # Step 7: Cleanup
    echo "Cleaning up..."
    CLEANUP_RESPONSE=$(curl -s -X DELETE "$BASE_URL/results/cleanup" \
      -H "Authorization: Bearer $TOKEN")
    
    echo "Cleanup response: $CLEANUP_RESPONSE"
else
    echo "Processing failed or was cancelled"
fi
```

## Advanced Configuration Examples

### Complex Matching Configuration

```json
{
  "mappings": [
    {
      "file1_col": "full_name",
      "file2_col": "name",
      "match_type": "fuzzy",
      "use_normalization": true,
      "case_sensitive": false,
      "weight": 2.0,
      "fuzzy_threshold": 85
    },
    {
      "file1_col": "email",
      "file2_col": "email_address",
      "match_type": "exact",
      "use_normalization": true,
      "case_sensitive": false,
      "weight": 3.0
    },
    {
      "file1_col": "phone",
      "file2_col": "phone_number",
      "match_type": "phonetic",
      "use_normalization": true,
      "weight": 1.5
    }
  ],
  "output_cols1": ["full_name", "email", "phone", "address"],
  "output_cols2": ["name", "email_address", "phone_number", "location"],
  "output_format": "csv",
  "threshold": 75,
  "matching_type": "one-to-many",
  "generate_unmatched": true,
  "include_confidence_scores": true,
  "uzbek_processing": {
    "enabled": true,
    "script_conversion": true,
    "phonetic_matching": true
  },
  "performance": {
    "parallel_processing": true,
    "max_workers": 4,
    "chunk_size": 1000,
    "memory_limit_mb": 2048
  }
}
```

### Uzbek Text Processing Configuration

```json
{
  "mappings": [
    {
      "file1_col": "ism",
      "file2_col": "to_liq_ism",
      "match_type": "fuzzy",
      "use_normalization": true,
      "uzbek_specific": {
        "script_conversion": true,
        "phonetic_rules": true,
        "common_variations": true
      },
      "weight": 1.0
    }
  ],
  "text_processing": {
    "uzbek_normalization": {
      "enabled": true,
      "cyrillic_to_latin": true,
      "remove_diacritics": true,
      "standardize_spelling": true
    }
  },
  "threshold": 70
}
```

## WebSocket Real-time Updates

### JavaScript WebSocket Client

```javascript
// Connect to WebSocket for real-time progress updates
const socket = io('http://localhost:5000', {
    auth: {
        token: 'your-jwt-token'
    }
});

// Listen for progress updates
socket.on('progress_update', (data) => {
    console.log('Progress:', data);
    updateProgressBar(data.progress);
    updateStatusMessage(data.message);
});

// Listen for completion
socket.on('processing_complete', (data) => {
    console.log('Processing completed:', data);
    showResults(data.results);
});

// Listen for errors
socket.on('processing_error', (error) => {
    console.error('Processing error:', error);
    showError(error.message);
});

// Start processing with WebSocket updates
function startProcessingWithUpdates(config) {
    socket.emit('start_processing', config);
}

// Cancel processing
function cancelProcessing() {
    socket.emit('cancel_processing');
}
```

## Error Handling Examples

### Python Error Handling

```python
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

class APIError(Exception):
    """Custom API error with detailed information."""
    
    def __init__(self, message, status_code=None, response_data=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

def handle_api_request(func):
    """Decorator for handling API requests with proper error handling."""
    
    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            response.raise_for_status()
            return response.json()
            
        except Timeout:
            raise APIError("Request timed out")
            
        except HTTPError as e:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                message = error_data.get('details', str(e))
            except:
                message = str(e)
            
            if status_code == 401:
                raise APIError("Authentication failed", status_code, error_data)
            elif status_code == 429:
                raise APIError("Rate limit exceeded", status_code, error_data)
            elif status_code == 413:
                raise APIError("File too large", status_code, error_data)
            else:
                raise APIError(message, status_code, error_data)
                
        except RequestException as e:
            raise APIError(f"Network error: {str(e)}")
    
    return wrapper

# Usage with error handling
@handle_api_request
def upload_files_safe(file1_path, file2_path):
    with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
        files = {'file1': f1, 'file2': f2}
        return requests.post(
            "http://localhost:5000/api/v1/files/upload",
            files=files,
            headers={"Authorization": f"Bearer {token}"},
            timeout=300
        )

try:
    result = upload_files_safe("data1.csv", "data2.json")
    print("Upload successful:", result)
except APIError as e:
    if e.status_code == 413:
        print("File too large. Please reduce file size or contact administrator.")
    elif e.status_code == 401:
        print("Authentication failed. Please login again.")
    else:
        print(f"API error: {e}")
```

## Performance Optimization

### Batch Processing Example

```python
import asyncio
import aiohttp
from typing import List, Dict

class AsyncFileProcessingClient:
    """Async client for batch processing multiple file pairs."""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    async def process_file_batch(self, file_pairs: List[Dict[str, str]]) -> List[Dict]:
        """Process multiple file pairs concurrently."""
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._process_single_pair(session, pair)
                for pair in file_pairs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
    
    async def _process_single_pair(self, session: aiohttp.ClientSession, pair: Dict[str, str]):
        """Process a single file pair."""
        
        try:
            # Upload files
            upload_result = await self._upload_files(session, pair['file1'], pair['file2'])
            
            # Start processing
            config = pair.get('config', self._default_config())
            operation_id = await self._start_processing(session, config)
            
            # Wait for completion
            result = await self._wait_for_completion(session, operation_id)
            
            return {
                'pair': pair,
                'status': 'success',
                'result': result
            }
            
        except Exception as e:
            return {
                'pair': pair,
                'status': 'error',
                'error': str(e)
            }

# Usage
async def main():
    client = AsyncFileProcessingClient("http://localhost:5000/api/v1", token)
    
    file_pairs = [
        {'file1': 'batch1_file1.csv', 'file2': 'batch1_file2.json'},
        {'file1': 'batch2_file1.csv', 'file2': 'batch2_file2.json'},
        {'file1': 'batch3_file1.csv', 'file2': 'batch3_file2.json'},
    ]
    
    results = await client.process_file_batch(file_pairs)
    
    for result in results:
        if result['status'] == 'success':
            print(f"Processed {result['pair']} successfully")
        else:
            print(f"Failed to process {result['pair']}: {result['error']}")

asyncio.run(main())
```

This comprehensive API documentation provides detailed examples, error handling patterns, and advanced usage scenarios for the File Processing Optimization system API.