"""
Job management service for batch processing operations.
Implements requirements 6.3, 6.4, 3.4: Job queue system with status tracking and cancellation.
"""

import os
import json
import uuid
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from queue import PriorityQueue, Empty
import sqlite3

from src.web.models.web_models import ProcessingConfig
from src.web.services.processing_service import ProcessingService
from src.infrastructure.logging import get_logger

logger = get_logger('job_service')


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Job:
    """Job data model."""
    id: str
    name: str
    config: Dict[str, Any]
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    message: str = ""
    error_message: Optional[str] = None
    result_files: List[str] = None
    user_id: str = "anonymous"
    
    def __post_init__(self):
        if self.result_files is None:
            self.result_files = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'config': self.config,
            'status': self.status.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'message': self.message,
            'error_message': self.error_message,
            'result_files': self.result_files,
            'user_id': self.user_id,
            'duration': self._calculate_duration(),
            'estimated_remaining': self._estimate_remaining_time()
        }
    
    def _calculate_duration(self) -> Optional[int]:
        """Calculate job duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now()
        return int((end_time - self.started_at).total_seconds())
    
    def _estimate_remaining_time(self) -> Optional[int]:
        """Estimate remaining time based on progress."""
        if not self.started_at or self.progress <= 0 or self.status != JobStatus.RUNNING:
            return None
        
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if elapsed <= 0:
            return None
        
        progress_rate = self.progress / elapsed
        if progress_rate <= 0:
            return None
        
        remaining_progress = 100 - self.progress
        return int(remaining_progress / progress_rate)


class JobQueue:
    """Priority-based job queue."""
    
    def __init__(self):
        """Initialize job queue."""
        self._queue = PriorityQueue()
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
    
    def add_job(self, job: Job) -> None:
        """Add job to queue."""
        with self._lock:
            self._jobs[job.id] = job
            # Use negative priority for max-heap behavior (higher priority first)
            self._queue.put((-job.priority.value, job.created_at, job.id))
            logger.info(f"Added job {job.id} to queue with priority {job.priority.value}")
    
    def get_next_job(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Get next job from queue."""
        try:
            _, _, job_id = self._queue.get(timeout=timeout)
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.PENDING:
                    return job
                # Job was cancelled or removed, try next
                return self.get_next_job(timeout=0.1)
        except Empty:
            return None
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, **updates) -> bool:
        """Update job properties."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            return True
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.message = "Job cancelled by user"
                logger.info(f"Cancelled job {job_id}")
                return True
            
            return False
    
    def list_jobs(self, user_id: Optional[str] = None, 
                  status: Optional[JobStatus] = None,
                  limit: Optional[int] = None) -> List[Job]:
        """List jobs with optional filtering."""
        with self._lock:
            jobs = list(self._jobs.values())
            
            # Filter by user
            if user_id:
                jobs = [job for job in jobs if job.user_id == user_id]
            
            # Filter by status
            if status:
                jobs = [job for job in jobs if job.status == status]
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            # Apply limit
            if limit:
                jobs = jobs[:limit]
            
            return jobs
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed/failed jobs."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        with self._lock:
            jobs_to_remove = []
            
            for job_id, job in self._jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old jobs")
            return removed_count


class JobPersistence:
    """Job persistence using SQLite."""
    
    def __init__(self, db_path: str = "jobs.db"):
        """Initialize job persistence."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    progress INTEGER DEFAULT 0,
                    message TEXT DEFAULT '',
                    error_message TEXT,
                    result_files TEXT,
                    user_id TEXT DEFAULT 'anonymous'
                )
            """)
            conn.commit()
    
    def save_job(self, job: Job) -> None:
        """Save job to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs 
                (id, name, config, status, priority, created_at, started_at, 
                 completed_at, progress, message, error_message, result_files, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id,
                job.name,
                json.dumps(job.config),
                job.status.value,
                job.priority.value,
                job.created_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.progress,
                job.message,
                job.error_message,
                json.dumps(job.result_files),
                job.user_id
            ))
            conn.commit()
    
    def load_job(self, job_id: str) -> Optional[Job]:
        """Load job from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return Job(
                id=row['id'],
                name=row['name'],
                config=json.loads(row['config']),
                status=JobStatus(row['status']),
                priority=JobPriority(row['priority']),
                created_at=datetime.fromisoformat(row['created_at']),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                progress=row['progress'],
                message=row['message'],
                error_message=row['error_message'],
                result_files=json.loads(row['result_files']),
                user_id=row['user_id']
            )
    
    def load_all_jobs(self) -> List[Job]:
        """Load all jobs from database."""
        jobs = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            
            for row in cursor.fetchall():
                job = Job(
                    id=row['id'],
                    name=row['name'],
                    config=json.loads(row['config']),
                    status=JobStatus(row['status']),
                    priority=JobPriority(row['priority']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    progress=row['progress'],
                    message=row['message'],
                    error_message=row['error_message'],
                    result_files=json.loads(row['result_files']),
                    user_id=row['user_id']
                )
                jobs.append(job)
        
        return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0


class JobWorker:
    """Job worker thread."""
    
    def __init__(self, worker_id: str, job_queue: JobQueue, 
                 persistence: JobPersistence):
        """Initialize job worker."""
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.persistence = persistence
        self.processing_service = ProcessingService()
        self.running = False
        self.current_job: Optional[Job] = None
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start worker thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started job worker {self.worker_id}")
    
    def stop(self):
        """Stop worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info(f"Stopped job worker {self.worker_id}")
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                # Get next job
                job = self.job_queue.get_next_job(timeout=1.0)
                if not job:
                    continue
                
                # Process job
                self.current_job = job
                self._process_job(job)
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {str(e)}")
                if self.current_job:
                    self._handle_job_error(self.current_job, str(e))
    
    def _process_job(self, job: Job):
        """Process a single job."""
        logger.info(f"Worker {self.worker_id} processing job {job.id}")
        
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            job.message = "Processing started"
            self.persistence.save_job(job)
            
            # Create processing configuration
            processing_config = ProcessingConfig.from_dict(job.config)
            
            # Create progress callback
            def progress_callback(status: str, progress: int, message: str):
                job.progress = progress
                job.message = message
                self.job_queue.update_job(job.id, progress=progress, message=message)
                self.persistence.save_job(job)
            
            # Start processing
            success = self.processing_service.start_processing(processing_config)
            
            if success:
                # Monitor processing
                while self.processing_service.is_processing():
                    if job.status == JobStatus.CANCELLED:
                        logger.info(f"Job {job.id} was cancelled")
                        return
                    
                    # Update progress
                    progress_status = self.processing_service.get_progress()
                    job.progress = progress_status.progress
                    job.message = progress_status.message
                    
                    if progress_status.error:
                        raise Exception(progress_status.error)
                    
                    self.persistence.save_job(job)
                    time.sleep(1)
                
                # Processing completed
                result_files = self.processing_service.get_result_files()
                job.result_files = [rf.path for rf in result_files]
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.progress = 100
                job.message = "Processing completed successfully"
                
                logger.info(f"Job {job.id} completed successfully")
                
            else:
                raise Exception("Failed to start processing")
        
        except Exception as e:
            self._handle_job_error(job, str(e))
        
        finally:
            self.persistence.save_job(job)
    
    def _handle_job_error(self, job: Job, error_message: str):
        """Handle job processing error."""
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now()
        job.error_message = error_message
        job.message = f"Processing failed: {error_message}"
        
        logger.error(f"Job {job.id} failed: {error_message}")


class JobService:
    """Main job management service."""
    
    def __init__(self, max_workers: int = 2, db_path: str = "jobs.db"):
        """Initialize job service."""
        self.max_workers = max_workers
        self.job_queue = JobQueue()
        self.persistence = JobPersistence(db_path)
        self.workers: List[JobWorker] = []
        self.running = False
        
        # Load existing jobs from database
        self._load_existing_jobs()
    
    def start(self):
        """Start job service."""
        if self.running:
            return
        
        self.running = True
        
        # Start workers
        for i in range(self.max_workers):
            worker = JobWorker(f"worker-{i}", self.job_queue, self.persistence)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started job service with {self.max_workers} workers")
    
    def stop(self):
        """Stop job service."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        self.workers.clear()
        logger.info("Stopped job service")
    
    def submit_job(self, name: str, config: Dict[str, Any], 
                   priority: JobPriority = JobPriority.NORMAL,
                   user_id: str = "anonymous") -> str:
        """Submit a new job."""
        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            config=config,
            status=JobStatus.PENDING,
            priority=priority,
            created_at=datetime.now(),
            user_id=user_id
        )
        
        # Save to database
        self.persistence.save_job(job)
        
        # Add to queue
        self.job_queue.add_job(job)
        
        logger.info(f"Submitted job {job.id}: {name}")
        return job.id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.job_queue.get_job(job_id)
    
    def list_jobs(self, user_id: Optional[str] = None,
                  status: Optional[JobStatus] = None,
                  limit: Optional[int] = None) -> List[Job]:
        """List jobs with optional filtering."""
        return self.job_queue.list_jobs(user_id, status, limit)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        success = self.job_queue.cancel_job(job_id)
        if success:
            job = self.job_queue.get_job(job_id)
            if job:
                self.persistence.save_job(job)
        return success
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        # Only allow deletion of completed/failed/cancelled jobs
        job = self.job_queue.get_job(job_id)
        if not job or job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            return False
        
        # Remove from database
        success = self.persistence.delete_job(job_id)
        
        # Remove from memory (if exists)
        if success and job_id in self.job_queue._jobs:
            del self.job_queue._jobs[job_id]
        
        return success
    
    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old jobs."""
        return self.job_queue.cleanup_old_jobs(max_age_hours)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        jobs = self.job_queue.list_jobs()
        
        stats = {
            'total_jobs': len(jobs),
            'pending': len([j for j in jobs if j.status == JobStatus.PENDING]),
            'running': len([j for j in jobs if j.status == JobStatus.RUNNING]),
            'completed': len([j for j in jobs if j.status == JobStatus.COMPLETED]),
            'failed': len([j for j in jobs if j.status == JobStatus.FAILED]),
            'cancelled': len([j for j in jobs if j.status == JobStatus.CANCELLED]),
            'active_workers': len([w for w in self.workers if w.current_job]),
            'total_workers': len(self.workers)
        }
        
        return stats
    
    def _load_existing_jobs(self):
        """Load existing jobs from database."""
        try:
            jobs = self.persistence.load_all_jobs()
            
            for job in jobs:
                # Reset running jobs to pending (they were interrupted)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.PENDING
                    job.started_at = None
                    job.message = "Job requeued after restart"
                    self.persistence.save_job(job)
                
                # Add pending jobs back to queue
                if job.status == JobStatus.PENDING:
                    self.job_queue.add_job(job)
                else:
                    # Add to memory for status queries
                    self.job_queue._jobs[job.id] = job
            
            logger.info(f"Loaded {len(jobs)} existing jobs from database")
            
        except Exception as e:
            logger.error(f"Error loading existing jobs: {str(e)}")


# Global job service instance
job_service = JobService()