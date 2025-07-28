# Authentication Guide

This guide covers authentication and authorization for the File Processing API, including JWT token management, error handling, and security best practices.

## Overview

The API uses JWT (JSON Web Token) based authentication with the following features:
- Stateless authentication
- Token-based authorization
- Configurable token expiration
- Role-based access control
- Rate limiting per user

## Authentication Flow

### 1. Login and Token Generation

**Endpoint:** `POST /auth/login`

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
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTY0MzcyMzQwMCwianRpIjoiYWJjZGVmZ2gtaWprbC1tbm9wLXFyc3QtdXZ3eHl6MTIzNCIsInR5cCI6ImFjY2VzcyIsInN1YiI6ImFkbWluIiwibmJmIjoxNjQzNzIzNDAwLCJleHAiOjE2NDM4MDk4MDB9.example_signature",
  "expires_in": 86400
}
```

### 2. Using the Token

Include the JWT token in the Authorization header for all API requests:

```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### 3. Token Verification

**Endpoint:** `GET /auth/verify`

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "valid": true,
  "user": "admin",
  "expires_at": 1643809800
}
```

## Demo Users

The system includes demo users for testing:

| Username | Password | Role  | Description |
|----------|----------|-------|-------------|
| admin    | admin123 | admin | Full access to all features |
| user     | user123  | user  | Standard user access |
| demo     | demo123  | user  | Demo account |

## Token Configuration

### Environment Variables

Configure JWT settings using environment variables:

```bash
# JWT Secret Key (change in production!)
JWT_SECRET_KEY=your-super-secret-jwt-key-here

# Token expiration (in hours)
JWT_EXPIRES_HOURS=24

# Redis URL for token blacklisting (optional)
REDIS_URL=redis://localhost:6379/0
```

### Token Properties

- **Algorithm:** HS256
- **Default Expiration:** 24 hours
- **Blacklist Support:** Yes (requires Redis)
- **Refresh Tokens:** Not implemented (use re-authentication)

## Error Handling

### Authentication Errors

The API returns specific error responses for different authentication scenarios:

#### Missing Authorization Header (401)
```json
{
  "error": "Unauthorized",
  "details": "Authorization header is required. Please provide a valid JWT token.",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

#### Invalid Token Format (401)
```json
{
  "error": "Unauthorized",
  "details": "Invalid token format or content.",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

#### Expired Token (401)
```json
{
  "error": "Unauthorized",
  "details": "Token has expired. Please login again.",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

#### Invalid Credentials (401)
```json
{
  "error": "Unauthorized",
  "details": "Invalid username or password",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

## Client Implementation Examples

### Python Client with Token Management

```python
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class AuthenticatedAPIClient:
    """API client with automatic token management."""
    
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.session = requests.Session()
    
    def authenticate(self) -> bool:
        """Authenticate and store token."""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={
                    "username": self.username,
                    "password": self.password
                }
            )
            response.raise_for_status()
            
            data = response.json()
            self.token = data["access_token"]
            
            # Calculate expiration time
            expires_in = data["expires_in"]
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            # Update session headers
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            
            print(f"Authentication successful. Token expires at: {self.token_expires_at}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            return False
    
    def is_token_valid(self) -> bool:
        """Check if current token is still valid."""
        if not self.token or not self.token_expires_at:
            return False
        
        # Check if token expires within next 5 minutes
        buffer_time = timedelta(minutes=5)
        return datetime.now() + buffer_time < self.token_expires_at
    
    def ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, re-authenticate if needed."""
        if self.is_token_valid():
            return True
        
        print("Token expired or missing, re-authenticating...")
        return self.authenticate()
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated API request with automatic token refresh."""
        if not self.ensure_authenticated():
            raise Exception("Failed to authenticate")
        
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        # If we get 401, try to re-authenticate once
        if response.status_code == 401:
            print("Received 401, attempting re-authentication...")
            if self.authenticate():
                # Retry the request with new token
                response = self.session.request(method, url, **kwargs)
        
        return response
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make GET request."""
        return self.make_request("GET", endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make POST request."""
        return self.make_request("POST", endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """Make PUT request."""
        return self.make_request("PUT", endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make DELETE request."""
        return self.make_request("DELETE", endpoint, **kwargs)

# Usage example
def main():
    client = AuthenticatedAPIClient(
        base_url="http://localhost:5000/api/v1",
        username="admin",
        password="admin123"
    )
    
    # The client will automatically authenticate on first request
    try:
        # Upload files
        with open("file1.csv", "rb") as f1, open("file2.json", "rb") as f2:
            files = {"file1": f1, "file2": f2}
            response = client.post("/files/upload", files=files)
            response.raise_for_status()
            print("Files uploaded successfully")
        
        # Get file validation
        response = client.get("/files/validate")
        response.raise_for_status()
        validation_data = response.json()
        print("File validation:", validation_data)
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")

if __name__ == "__main__":
    main()
```

### JavaScript/Node.js Client with Token Management

```javascript
const axios = require('axios');

class AuthenticatedAPIClient {
    constructor(baseUrl, username, password) {
        this.baseUrl = baseUrl;
        this.username = username;
        this.password = password;
        this.token = null;
        this.tokenExpiresAt = null;
        
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000
        });
        
        // Add request interceptor for automatic token refresh
        this.client.interceptors.request.use(
            async (config) => {
                await this.ensureAuthenticated();
                return config;
            },
            (error) => Promise.reject(error)
        );
        
        // Add response interceptor for handling 401 errors
        this.client.interceptors.response.use(
            (response) => response,
            async (error) => {
                if (error.response?.status === 401 && !error.config._retry) {
                    error.config._retry = true;
                    
                    console.log('Received 401, attempting re-authentication...');
                    const authenticated = await this.authenticate();
                    
                    if (authenticated) {
                        return this.client.request(error.config);
                    }
                }
                
                return Promise.reject(error);
            }
        );
    }
    
    async authenticate() {
        try {
            const response = await axios.post(`${this.baseUrl}/auth/login`, {
                username: this.username,
                password: this.password
            });
            
            const { access_token, expires_in } = response.data;
            this.token = access_token;
            this.tokenExpiresAt = new Date(Date.now() + expires_in * 1000);
            
            // Update default headers
            this.client.defaults.headers.common['Authorization'] = `Bearer ${this.token}`;
            
            console.log(`Authentication successful. Token expires at: ${this.tokenExpiresAt}`);
            return true;
            
        } catch (error) {
            console.error('Authentication failed:', error.response?.data || error.message);
            return false;
        }
    }
    
    isTokenValid() {
        if (!this.token || !this.tokenExpiresAt) {
            return false;
        }
        
        // Check if token expires within next 5 minutes
        const bufferTime = 5 * 60 * 1000; // 5 minutes in milliseconds
        return Date.now() + bufferTime < this.tokenExpiresAt.getTime();
    }
    
    async ensureAuthenticated() {
        if (this.isTokenValid()) {
            return true;
        }
        
        console.log('Token expired or missing, re-authenticating...');
        return await this.authenticate();
    }
    
    // Convenience methods
    async get(endpoint, config = {}) {
        return this.client.get(endpoint, config);
    }
    
    async post(endpoint, data, config = {}) {
        return this.client.post(endpoint, data, config);
    }
    
    async put(endpoint, data, config = {}) {
        return this.client.put(endpoint, data, config);
    }
    
    async delete(endpoint, config = {}) {
        return this.client.delete(endpoint, config);
    }
}

// Usage example
async function main() {
    const client = new AuthenticatedAPIClient(
        'http://localhost:5000/api/v1',
        'admin',
        'admin123'
    );
    
    try {
        // The client will automatically authenticate on first request
        
        // Get job statistics
        const statsResponse = await client.get('/jobs/stats');
        console.log('Job statistics:', statsResponse.data);
        
        // Submit a job
        const jobConfig = {
            name: 'Test Job',
            config: {
                mappings: [{
                    file1_col: 'name',
                    file2_col: 'full_name',
                    match_type: 'fuzzy'
                }],
                threshold: 80
            },
            priority: 2
        };
        
        const submitResponse = await client.post('/jobs/submit', jobConfig);
        console.log('Job submitted:', submitResponse.data);
        
    } catch (error) {
        console.error('API request failed:', error.response?.data || error.message);
    }
}

main();
```

### cURL Examples with Token Management

```bash
#!/bin/bash

# Configuration
BASE_URL="http://localhost:5000/api/v1"
USERNAME="admin"
PASSWORD="admin123"
TOKEN_FILE="/tmp/api_token"

# Function to authenticate and save token
authenticate() {
    echo "Authenticating..."
    
    RESPONSE=$(curl -s -X POST "$BASE_URL/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\": \"$USERNAME\", \"password\": \"$PASSWORD\"}")
    
    TOKEN=$(echo "$RESPONSE" | jq -r '.access_token')
    EXPIRES_IN=$(echo "$RESPONSE" | jq -r '.expires_in')
    
    if [ "$TOKEN" = "null" ]; then
        echo "Authentication failed: $RESPONSE"
        return 1
    fi
    
    # Save token and expiration time
    EXPIRES_AT=$(($(date +%s) + EXPIRES_IN))
    echo "$TOKEN" > "$TOKEN_FILE"
    echo "$EXPIRES_AT" > "${TOKEN_FILE}.expires"
    
    echo "Authentication successful, token expires at $(date -d @$EXPIRES_AT)"
    return 0
}

# Function to check if token is still valid
is_token_valid() {
    if [ ! -f "$TOKEN_FILE" ] || [ ! -f "${TOKEN_FILE}.expires" ]; then
        return 1
    fi
    
    EXPIRES_AT=$(cat "${TOKEN_FILE}.expires")
    CURRENT_TIME=$(date +%s)
    BUFFER_TIME=300  # 5 minutes buffer
    
    if [ $((CURRENT_TIME + BUFFER_TIME)) -lt "$EXPIRES_AT" ]; then
        return 0
    else
        return 1
    fi
}

# Function to ensure we have a valid token
ensure_authenticated() {
    if is_token_valid; then
        return 0
    fi
    
    echo "Token expired or missing, re-authenticating..."
    authenticate
}

# Function to make authenticated API requests
api_request() {
    local method="$1"
    local endpoint="$2"
    shift 2
    
    ensure_authenticated || {
        echo "Failed to authenticate"
        return 1
    }
    
    TOKEN=$(cat "$TOKEN_FILE")
    
    curl -X "$method" "$BASE_URL$endpoint" \
        -H "Authorization: Bearer $TOKEN" \
        "$@"
}

# Usage examples
echo "=== API Request Examples ==="

# Get job statistics
echo "Getting job statistics..."
api_request GET "/jobs/stats" -s | jq '.'

# Submit a job
echo "Submitting a job..."
JOB_CONFIG='{
    "name": "Test Job from Script",
    "config": {
        "mappings": [{
            "file1_col": "name",
            "file2_col": "full_name",
            "match_type": "fuzzy"
        }],
        "threshold": 80
    },
    "priority": 2
}'

JOB_RESPONSE=$(api_request POST "/jobs/submit" \
    -H "Content-Type: application/json" \
    -d "$JOB_CONFIG" -s)

echo "Job submission response: $JOB_RESPONSE"

JOB_ID=$(echo "$JOB_RESPONSE" | jq -r '.job_id')

if [ "$JOB_ID" != "null" ]; then
    echo "Job ID: $JOB_ID"
    
    # Get job details
    echo "Getting job details..."
    api_request GET "/jobs/$JOB_ID" -s | jq '.'
fi

# Cleanup
rm -f "$TOKEN_FILE" "${TOKEN_FILE}.expires"
```

## Security Best Practices

### Token Security
- **Never log tokens** in application logs or error messages
- **Use HTTPS** in production to protect tokens in transit
- **Store tokens securely** on the client side (avoid localStorage for sensitive apps)
- **Implement token rotation** for long-running applications

### Password Security
- **Change default passwords** immediately in production
- **Use strong passwords** with mixed case, numbers, and symbols
- **Implement password policies** for user accounts
- **Consider multi-factor authentication** for enhanced security

### Environment Configuration
```bash
# Production environment variables
JWT_SECRET_KEY=your-super-secret-256-bit-key-here-change-this
JWT_EXPIRES_HOURS=8
REDIS_URL=redis://localhost:6379/0

# Enable HTTPS in production
PREFERRED_URL_SCHEME=https
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Lax
```

### Rate Limiting
The API implements rate limiting per user:
- **Authenticated users:** 2000 requests/hour, 200/minute
- **Anonymous users:** 1000 requests/hour, 100/minute
- **File uploads:** 50 requests/hour, 10/minute
- **Processing operations:** 20 requests/hour, 5/minute

### Monitoring and Logging
- Monitor failed authentication attempts
- Log token usage patterns
- Set up alerts for unusual activity
- Regularly audit user access logs

## Troubleshooting

### Common Issues

**"Missing Authorization Header" Error:**
- Ensure the Authorization header is included in all requests
- Check that the header format is: `Authorization: Bearer <token>`

**"Token has expired" Error:**
- Implement automatic token refresh in your client
- Check system clock synchronization
- Verify JWT_EXPIRES_HOURS configuration

**"Invalid token format" Error:**
- Ensure the token is not truncated or modified
- Check for extra whitespace or characters
- Verify the token was obtained from the correct endpoint

**Rate Limit Exceeded:**
- Implement exponential backoff in your client
- Check the Retry-After header for wait time
- Consider caching responses to reduce API calls

### Debug Mode

Enable debug logging to troubleshoot authentication issues:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

This will provide detailed logs about token validation and authentication processes.