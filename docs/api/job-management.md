# Job Management API

The Job Management API provides endpoints for submitting, monitoring, and managing batch processing jobs. This allows for asynchronous processing of large datasets with queue management and priority handling.

## Overview

Jobs are processed asynchronously in a queue system with the following features:
- Priority-based job scheduling (1=low, 2=normal, 3=high, 4=urgent)
- Real-time progress tracking
- Job cancellation support
- Automatic cleanup of old jobs
- User-specific job isolation

## Authentication

All job management endpoints require JWT authentication:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Submit Job

Submit a new job for batch processing.

**Endpoint:** `POST /jobs/submit`

**Request Body:**
```json
{
  "name": "Customer Data Matching - Batch 1",
  "config": {
    "mappings": [
      {
        "file1_col": "customer_name",
        "file2_col": "full_name",
        "match_type": "fuzzy",
        "use_normalization": true,
        "weight": 1.0
      }
    ],
    "output_cols1": ["customer_name", "email", "phone"],
    "output_cols2": ["full_name", "address", "city"],
    "threshold": 80,
    "generate_unmatched": true
  },
  "priority": 2
}
```

**Response:**
```json
{
  "job_id": "job_12345678-1234-1234-1234-123456789abc",
  "message": "Job submitted successfully"
}
```

**Priority Levels:**
- `1` - Low priority
- `2` - Normal priority (default)
- `3` - High priority
- `4` - Urgent priority

### List Jobs

Get a list of jobs with optional filtering.

**Endpoint:** `GET /jobs/list`

**Query Parameters:**
- `status` (optional): Filter by job status (`pending`, `running`, `completed`, `failed`, `cancelled`)
- `limit` (optional): Maximum number of jobs to return

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/v1/jobs/list?status=running&limit=10" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
[
  {
    "id": "job_12345678-1234-1234-1234-123456789abc",
    "name": "Customer Data Matching - Batch 1",
    "status": "running",
    "priority": 2,
    "created_at": "2025-01-27T10:00:00Z",
    "started_at": "2025-01-27T10:05:00Z",
    "completed_at": null,
    "progress": 45,
    "message": "Processing records...",
    "error_message": null,
    "result_files": [],
    "user_id": "admin",
    "duration": 300,
    "estimated_remaining": 360
  }
]
```

### Get Job Details

Get detailed information about a specific job.

**Endpoint:** `GET /jobs/{job_id}`

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/v1/jobs/job_12345678-1234-1234-1234-123456789abc" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "id": "job_12345678-1234-1234-1234-123456789abc",
  "name": "Customer Data Matching - Batch 1",
  "status": "completed",
  "priority": 2,
  "created_at": "2025-01-27T10:00:00Z",
  "started_at": "2025-01-27T10:05:00Z",
  "completed_at": "2025-01-27T10:15:00Z",
  "progress": 100,
  "message": "Processing completed successfully",
  "error_message": null,
  "result_files": [
    "/results/job_12345678/matched_results.json",
    "/results/job_12345678/unmatched_1.json",
    "/results/job_12345678/unmatched_2.json"
  ],
  "user_id": "admin",
  "duration": 600,
  "estimated_remaining": 0
}
```

### Cancel Job

Cancel a running or pending job.

**Endpoint:** `POST /jobs/{job_id}/cancel`

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/v1/jobs/job_12345678-1234-1234-1234-123456789abc/cancel" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "success": true,
  "message": "Job cancelled successfully"
}
```

### Delete Job

Delete a completed job and its associated files.

**Endpoint:** `DELETE /jobs/{job_id}`

**Example Request:**
```bash
curl -X DELETE "http://localhost:5000/api/v1/jobs/job_12345678-1234-1234-1234-123456789abc" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "success": true,
  "message": "Job deleted successfully"
}
```

### Get Job Statistics

Get queue statistics and system status.

**Endpoint:** `GET /jobs/stats`

**Example Request:**
```bash
curl -X GET "http://localhost:5000/api/v1/jobs/stats" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "total_jobs": 150,
  "pending": 5,
  "running": 2,
  "completed": 140,
  "failed": 2,
  "cancelled": 1,
  "active_workers": 2,
  "total_workers": 4
}
```

### Cleanup Old Jobs

Clean up old completed jobs to free up storage space.

**Endpoint:** `POST /jobs/cleanup`

**Query Parameters:**
- `max_age_hours` (optional): Maximum age of jobs to keep (default: 24, range: 1-168)

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/v1/jobs/cleanup?max_age_hours=48" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "success": true,
  "cleaned_jobs": 25,
  "freed_space_mb": 1024,
  "message": "Cleaned up 25 old jobs, freed 1024 MB"
}
```

## Job Status Flow

Jobs progress through the following states:

1. **pending** - Job is queued and waiting to be processed
2. **running** - Job is currently being processed
3. **completed** - Job finished successfully
4. **failed** - Job encountered an error and failed
5. **cancelled** - Job was cancelled by user or system

## Error Handling

### Common Error Responses

**Job Not Found (404):**
```json
{
  "error": "Not Found",
  "details": "Job not found: job_invalid_id",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

**Access Denied (403):**
```json
{
  "error": "Forbidden",
  "details": "Access denied to this job",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

**Invalid Priority (400):**
```json
{
  "error": "Bad Request",
  "details": "Priority must be between 1 (low) and 4 (urgent)",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

## Usage Examples

### Python Client Example

```python
import requests
import time
from typing import Dict, Any, Optional

class JobManagementClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def submit_job(self, name: str, config: Dict[str, Any], priority: int = 2) -> str:
        """Submit a new job and return job ID."""
        response = requests.post(
            f"{self.base_url}/jobs/submit",
            json={
                "name": name,
                "config": config,
                "priority": priority
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["job_id"]
    
    def wait_for_job_completion(self, job_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for job to complete and return final status."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job = self.get_job(job_id)
            
            if job["status"] in ["completed", "failed", "cancelled"]:
                return job
            
            print(f"Job {job_id}: {job['status']} - {job['progress']}% - {job['message']}")
            time.sleep(10)  # Check every 10 seconds
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job details."""
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self, status: Optional[str] = None, limit: Optional[int] = None) -> list:
        """List jobs with optional filtering."""
        params = {}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit
        
        response = requests.get(
            f"{self.base_url}/jobs/list",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/cancel",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["success"]
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        response = requests.delete(
            f"{self.base_url}/jobs/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["success"]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        response = requests.get(
            f"{self.base_url}/jobs/stats",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
def main():
    client = JobManagementClient("http://localhost:5000/api/v1", "your-jwt-token")
    
    # Submit a high-priority job
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
    
    job_id = client.submit_job(
        name="Urgent Customer Matching",
        config=config,
        priority=3  # High priority
    )
    
    print(f"Submitted job: {job_id}")
    
    # Wait for completion
    try:
        final_job = client.wait_for_job_completion(job_id)
        
        if final_job["status"] == "completed":
            print("Job completed successfully!")
            print(f"Result files: {final_job['result_files']}")
        else:
            print(f"Job failed: {final_job['error_message']}")
    
    except TimeoutError as e:
        print(f"Job timed out: {e}")
        # Optionally cancel the job
        client.cancel_job(job_id)

if __name__ == "__main__":
    main()
```

### Batch Job Processing Example

```python
import asyncio
import aiohttp
from typing import List, Dict

async def submit_batch_jobs(base_url: str, token: str, job_configs: List[Dict]) -> List[str]:
    """Submit multiple jobs concurrently."""
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for config in job_configs:
            task = submit_single_job(session, base_url, headers, config)
            tasks.append(task)
        
        job_ids = await asyncio.gather(*tasks)
        return job_ids

async def submit_single_job(session, base_url: str, headers: Dict, config: Dict) -> str:
    """Submit a single job."""
    async with session.post(
        f"{base_url}/jobs/submit",
        json=config,
        headers=headers
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data["job_id"]

# Usage
async def main():
    job_configs = [
        {
            "name": "Batch Job 1",
            "config": {...},  # Your processing config
            "priority": 2
        },
        {
            "name": "Batch Job 2", 
            "config": {...},  # Your processing config
            "priority": 2
        },
        # ... more jobs
    ]
    
    job_ids = await submit_batch_jobs(
        "http://localhost:5000/api/v1",
        "your-jwt-token",
        job_configs
    )
    
    print(f"Submitted {len(job_ids)} jobs: {job_ids}")

asyncio.run(main())
```

## Best Practices

### Job Naming
- Use descriptive names that identify the purpose and data source
- Include batch numbers or dates for tracking
- Example: "Customer_Data_Matching_2025-01-27_Batch_001"

### Priority Management
- Use priority 4 (urgent) sparingly for truly time-critical jobs
- Most jobs should use priority 2 (normal)
- Use priority 1 (low) for background maintenance tasks

### Resource Management
- Monitor queue statistics regularly
- Clean up old completed jobs to free storage space
- Cancel jobs that are no longer needed

### Error Handling
- Always check job status after submission
- Implement retry logic for failed jobs with transient errors
- Log job IDs for troubleshooting

### Monitoring
- Set up alerts for jobs that run longer than expected
- Monitor queue depth to prevent backlog buildup
- Track job success/failure rates over time