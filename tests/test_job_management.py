"""
Tests for job management API endpoints.
Tests requirements 6.3, 6.4, 3.4: Job queue system with status tracking and cancellation.
"""

import os
import json
import tempfile
import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.web.services.job_service import JobService, Job, JobStatus, JobPriority, JobQueue, JobPersistence
from src.web.models.web_models import ProcessingConfig, FileUpload, FieldMapping


class TestJobQueue:
    """Test job queue functionality."""
    
    def test_add_job(self):
        """Test adding job to queue."""
        queue = JobQueue()
        
        job = Job(
            id="test-job-1",
            name="Test Job",
            config={},
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=datetime.now()
        )
        
        queue.add_job(job)
        
        # Check job was added
        retrieved_job = queue.get_job("test-job-1")
        assert retrieved_job is not None
        assert retrieved_job.id == "test-job-1"
        assert retrieved_job.name == "Test Job"
    
    def test_priority_ordering(self):
        """Test that jobs are retrieved in priority order."""
        queue = JobQueue()
        
        # Add jobs with different priorities
        low_job = Job(
            id="low-job",
            name="Low Priority",
            config={},
            status=JobStatus.PENDING,
            priority=JobPriority.LOW,
            created_at=datetime.now()
        )
        
        high_job = Job(
            id="high-job",
            name="High Priority",
            config={},
            status=JobStatus.PENDING,
            priority=JobPriority.HIGH,
            created_at=datetime.now()
        )
        
        normal_job = Job(
            id="normal-job",
            name="Normal Priority",
            config={},
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=datetime.now()
        )
        
        # Add in random order
        queue.add_job(low_job)
        queue.add_job(normal_job)
        queue.add_job(high_job)
        
        # Should get high priority first
        next_job = queue.get_next_job(timeout=0.1)
        assert next_job is not None
        assert next_job.id == "high-job"
        
        # Then normal priority
        next_job = queue.get_next_job(timeout=0.1)
        assert next_job is not None
        assert next_job.id == "normal-job"
        
        # Finally low priority
        next_job = queue.get_next_job(timeout=0.1)
        assert next_job is not None
        assert next_job.id == "low-job"
    
    def test_cancel_job(self):
        """Test job cancellation."""
        queue = JobQueue()
        
        job = Job(
            id="cancel-test",
            name="Cancel Test",
            config={},
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=datetime.now()
        )
        
        queue.add_job(job)
        
        # Cancel job
        success = queue.cancel_job("cancel-test")
        assert success is True
        
        # Check job status
        cancelled_job = queue.get_job("cancel-test")
        assert cancelled_job.status == JobStatus.CANCELLED
        assert cancelled_job.completed_at is not None
    
    def test_list_jobs_filtering(self):
        """Test job listing with filters."""
        queue = JobQueue()
        
        # Add jobs with different statuses and users
        jobs = [
            Job("job1", "Job 1", {}, JobStatus.PENDING, JobPriority.NORMAL, datetime.now(), user_id="user1"),
            Job("job2", "Job 2", {}, JobStatus.RUNNING, JobPriority.NORMAL, datetime.now(), user_id="user1"),
            Job("job3", "Job 3", {}, JobStatus.COMPLETED, JobPriority.NORMAL, datetime.now(), user_id="user2"),
            Job("job4", "Job 4", {}, JobStatus.PENDING, JobPriority.NORMAL, datetime.now(), user_id="user2"),
        ]
        
        for job in jobs:
            queue.add_job(job)
        
        # Test filtering by user
        user1_jobs = queue.list_jobs(user_id="user1")
        assert len(user1_jobs) == 2
        assert all(job.user_id == "user1" for job in user1_jobs)
        
        # Test filtering by status
        pending_jobs = queue.list_jobs(status=JobStatus.PENDING)
        assert len(pending_jobs) == 2
        assert all(job.status == JobStatus.PENDING for job in pending_jobs)
        
        # Test filtering by both user and status
        user1_pending = queue.list_jobs(user_id="user1", status=JobStatus.PENDING)
        assert len(user1_pending) == 1
        assert user1_pending[0].id == "job1"
        
        # Test limit
        limited_jobs = queue.list_jobs(limit=2)
        assert len(limited_jobs) == 2


class TestJobPersistence:
    """Test job persistence functionality."""
    
    def test_save_and_load_job(self):
        """Test saving and loading jobs."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            persistence = JobPersistence(db_path)
            
            job = Job(
                id="persist-test",
                name="Persistence Test",
                config={"test": "data"},
                status=JobStatus.PENDING,
                priority=JobPriority.HIGH,
                created_at=datetime.now(),
                user_id="test-user"
            )
            
            # Save job
            persistence.save_job(job)
            
            # Load job
            loaded_job = persistence.load_job("persist-test")
            
            assert loaded_job is not None
            assert loaded_job.id == job.id
            assert loaded_job.name == job.name
            assert loaded_job.config == job.config
            assert loaded_job.status == job.status
            assert loaded_job.priority == job.priority
            assert loaded_job.user_id == job.user_id
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_load_all_jobs(self):
        """Test loading all jobs."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            persistence = JobPersistence(db_path)
            
            # Save multiple jobs
            jobs = [
                Job("job1", "Job 1", {}, JobStatus.PENDING, JobPriority.NORMAL, datetime.now()),
                Job("job2", "Job 2", {}, JobStatus.COMPLETED, JobPriority.HIGH, datetime.now()),
                Job("job3", "Job 3", {}, JobStatus.FAILED, JobPriority.LOW, datetime.now()),
            ]
            
            for job in jobs:
                persistence.save_job(job)
            
            # Load all jobs
            loaded_jobs = persistence.load_all_jobs()
            
            assert len(loaded_jobs) == 3
            loaded_ids = {job.id for job in loaded_jobs}
            assert loaded_ids == {"job1", "job2", "job3"}
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_delete_job(self):
        """Test job deletion."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            persistence = JobPersistence(db_path)
            
            job = Job(
                id="delete-test",
                name="Delete Test",
                config={},
                status=JobStatus.COMPLETED,
                priority=JobPriority.NORMAL,
                created_at=datetime.now()
            )
            
            # Save job
            persistence.save_job(job)
            
            # Verify it exists
            loaded_job = persistence.load_job("delete-test")
            assert loaded_job is not None
            
            # Delete job
            success = persistence.delete_job("delete-test")
            assert success is True
            
            # Verify it's gone
            deleted_job = persistence.load_job("delete-test")
            assert deleted_job is None
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestJobService:
    """Test job service functionality."""
    
    def test_submit_job(self):
        """Test job submission."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            service = JobService(max_workers=1, db_path=db_path)
            
            config = {
                "file1": {"path": "/tmp/test1.csv", "type": "csv"},
                "file2": {"path": "/tmp/test2.csv", "type": "csv"},
                "mappings": [{"file1_col": "name", "file2_col": "name"}]
            }
            
            job_id = service.submit_job(
                name="Test Job",
                config=config,
                priority=JobPriority.HIGH,
                user_id="test-user"
            )
            
            assert job_id is not None
            
            # Get job
            job = service.get_job(job_id)
            assert job is not None
            assert job.name == "Test Job"
            assert job.config == config
            assert job.priority == JobPriority.HIGH
            assert job.user_id == "test-user"
            assert job.status == JobStatus.PENDING
            
        finally:
            service.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_list_jobs(self):
        """Test job listing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            service = JobService(max_workers=1, db_path=db_path)
            
            # Submit multiple jobs
            job_ids = []
            for i in range(3):
                job_id = service.submit_job(
                    name=f"Test Job {i}",
                    config={"test": i},
                    user_id="test-user"
                )
                job_ids.append(job_id)
            
            # List jobs
            jobs = service.list_jobs(user_id="test-user")
            assert len(jobs) == 3
            
            # Check they're sorted by creation time (newest first)
            for i in range(len(jobs) - 1):
                assert jobs[i].created_at >= jobs[i + 1].created_at
            
        finally:
            service.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_cancel_job(self):
        """Test job cancellation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            service = JobService(max_workers=1, db_path=db_path)
            
            job_id = service.submit_job(
                name="Cancel Test",
                config={"test": "data"},
                user_id="test-user"
            )
            
            # Cancel job
            success = service.cancel_job(job_id)
            assert success is True
            
            # Check job status
            job = service.get_job(job_id)
            assert job.status == JobStatus.CANCELLED
            
        finally:
            service.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_delete_job(self):
        """Test job deletion."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            service = JobService(max_workers=1, db_path=db_path)
            
            job_id = service.submit_job(
                name="Delete Test",
                config={"test": "data"},
                user_id="test-user"
            )
            
            # Cancel job first (so it can be deleted)
            service.cancel_job(job_id)
            
            # Delete job
            success = service.delete_job(job_id)
            assert success is True
            
            # Check job is gone
            job = service.get_job(job_id)
            assert job is None
            
        finally:
            service.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_get_queue_stats(self):
        """Test queue statistics."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            service = JobService(max_workers=2, db_path=db_path)
            
            # Submit jobs with different statuses
            job_id1 = service.submit_job("Job 1", {"test": 1}, user_id="user1")
            job_id2 = service.submit_job("Job 2", {"test": 2}, user_id="user1")
            
            # Cancel one job
            service.cancel_job(job_id2)
            
            # Get stats
            stats = service.get_queue_stats()
            
            assert stats['total_jobs'] == 2
            assert stats['pending'] == 1
            assert stats['cancelled'] == 1
            assert stats['total_workers'] == 2
            
        finally:
            service.stop()
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestJobModel:
    """Test job data model."""
    
    def test_job_to_dict(self):
        """Test job serialization."""
        job = Job(
            id="test-job",
            name="Test Job",
            config={"test": "data"},
            status=JobStatus.RUNNING,
            priority=JobPriority.HIGH,
            created_at=datetime(2025, 1, 27, 12, 0, 0),
            started_at=datetime(2025, 1, 27, 12, 1, 0),
            progress=50,
            message="Processing...",
            user_id="test-user"
        )
        
        job_dict = job.to_dict()
        
        assert job_dict['id'] == "test-job"
        assert job_dict['name'] == "Test Job"
        assert job_dict['config'] == {"test": "data"}
        assert job_dict['status'] == "running"
        assert job_dict['priority'] == 3
        assert job_dict['progress'] == 50
        assert job_dict['message'] == "Processing..."
        assert job_dict['user_id'] == "test-user"
        assert 'created_at' in job_dict
        assert 'started_at' in job_dict
        assert 'duration' in job_dict
        assert 'estimated_remaining' in job_dict
    
    def test_duration_calculation(self):
        """Test job duration calculation."""
        start_time = datetime(2025, 1, 27, 12, 0, 0)
        end_time = datetime(2025, 1, 27, 12, 5, 30)  # 5 minutes 30 seconds later
        
        job = Job(
            id="duration-test",
            name="Duration Test",
            config={},
            status=JobStatus.COMPLETED,
            priority=JobPriority.NORMAL,
            created_at=start_time,
            started_at=start_time,
            completed_at=end_time
        )
        
        job_dict = job.to_dict()
        assert job_dict['duration'] == 330  # 5 minutes 30 seconds = 330 seconds
    
    def test_estimated_remaining_time(self):
        """Test estimated remaining time calculation."""
        start_time = datetime(2025, 1, 27, 12, 0, 0)
        
        job = Job(
            id="estimate-test",
            name="Estimate Test",
            config={},
            status=JobStatus.RUNNING,
            priority=JobPriority.NORMAL,
            created_at=start_time,
            started_at=start_time,
            progress=25  # 25% complete
        )
        
        # Mock current time to be 1 minute after start
        with patch('src.web.services.job_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 27, 12, 1, 0)
            
            job_dict = job.to_dict()
            # If 25% took 60 seconds, remaining 75% should take 180 seconds
            assert job_dict['estimated_remaining'] == 180


if __name__ == '__main__':
    pytest.main([__file__, '-v'])