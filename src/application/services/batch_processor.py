"""
Batch processing service for handling multiple file matching operations.
Supports parallel processing, progress tracking, and error handling.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp

from .cli_config_service import CLIConfigurationManager
from ...infrastructure.progress_tracker import ProgressTracker
from ...domain.exceptions import ConfigurationError, FileProcessingError


class BatchStatus(Enum):
    """Batch processing status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Represents a single batch job"""
    job_id: str
    config_path: str
    config: Dict[str, Any]
    output_path: str
    status: BatchStatus = BatchStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed (successfully or with error)"""
        return self.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]


@dataclass
class BatchResult:
    """Results of batch processing operation"""
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    total_duration: float
    job_results: List[BatchJob]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_jobs == 0:
            return 0.0
        return (self.successful_jobs / self.total_jobs) * 100


class BatchProcessor:
    """Handles batch processing of multiple file matching operations"""
    
    def __init__(self, max_workers: Optional[int] = None, use_multiprocessing: bool = True):
        self.config_manager = CLIConfigurationManager()
        self.progress_tracker = ProgressTracker()
        self.logger = logging.getLogger(__name__)
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 4)  # Reasonable default
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        
        self.jobs: List[BatchJob] = []
        self.is_cancelled = False
    
    def discover_config_files(self, config_dir: str, pattern: str = "*.json") -> List[str]:
        """Discover configuration files in directory"""
        config_path = Path(config_dir)
        
        if not config_path.exists():
            raise FileProcessingError(f"Configuration directory not found: {config_dir}")
        
        if not config_path.is_dir():
            raise FileProcessingError(f"Path is not a directory: {config_dir}")
        
        config_files = list(config_path.glob(pattern))
        config_files.extend(config_path.glob("**/" + pattern))  # Recursive search
        
        return [str(f) for f in config_files if f.is_file()]
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate a single configuration file"""
        try:
            config = self.config_manager.load_and_validate_config(config_path)
            
            # Additional validation
            validation_errors = self.config_manager.comprehensive_config_validation(config)
            if validation_errors:
                return False, f"Validation errors: {'; '.join(validation_errors)}", None
            
            return True, None, config
            
        except Exception as e:
            return False, str(e), None
    
    def prepare_batch_jobs(self, config_dir: str, output_dir: str, 
                          config_pattern: str = "*.json") -> List[BatchJob]:
        """Prepare batch jobs from configuration directory"""
        config_files = self.discover_config_files(config_dir, config_pattern)
        
        if not config_files:
            raise FileProcessingError(f"No configuration files found in {config_dir}")
        
        self.logger.info(f"Found {len(config_files)} configuration files")
        
        jobs = []
        for config_path in config_files:
            try:
                is_valid, error_msg, config = self.validate_config_file(config_path)
                
                if not is_valid:
                    self.logger.warning(f"Skipping invalid config {config_path}: {error_msg}")
                    continue
                
                # Generate job ID and output path
                config_name = Path(config_path).stem
                job_id = f"batch_{config_name}_{int(time.time())}"
                output_path = os.path.join(output_dir, config_name)
                
                # Update config with batch-specific output path
                config['settings']['matched_output_path'] = output_path
                
                job = BatchJob(
                    job_id=job_id,
                    config_path=config_path,
                    config=config,
                    output_path=output_path
                )
                
                jobs.append(job)
                
            except Exception as e:
                self.logger.error(f"Error preparing job for {config_path}: {str(e)}")
        
        self.jobs = jobs
        return jobs
    
    def process_single_job(self, job: BatchJob, 
                          progress_callback: Optional[Callable] = None) -> BatchJob:
        """Process a single batch job"""
        job.status = BatchStatus.RUNNING
        job.start_time = time.time()
        
        try:
            self.logger.info(f"Starting job {job.job_id}")
            
            # Create job-specific progress callback
            def job_progress_callback(current, total, message=""):
                job.progress = (current / total * 100) if total > 0 else 0
                if progress_callback:
                    progress_callback(job.job_id, job.progress, message)
            
            # Import and run processing (avoiding circular imports)
            try:
                from main import run_processing_optimized
                run_processing_optimized(job.config, job_progress_callback)
                
                job.status = BatchStatus.COMPLETED
                job.progress = 100.0
                self.logger.info(f"Job {job.job_id} completed successfully")
                
            except ImportError:
                # Fallback to main.py if main.py is not available
                from main import run_processing_optimized
                run_processing_optimized(job.config, job_progress_callback)
                
                job.status = BatchStatus.COMPLETED
                job.progress = 100.0
                self.logger.info(f"Job {job.job_id} completed successfully")
                
        except Exception as e:
            job.status = BatchStatus.FAILED
            job.error_message = str(e)
            self.logger.error(f"Job {job.job_id} failed: {str(e)}")
        
        finally:
            job.end_time = time.time()
        
        return job
    
    def run_batch_sequential(self, jobs: List[BatchJob], 
                           progress_callback: Optional[Callable] = None) -> BatchResult:
        """Run batch jobs sequentially"""
        start_time = time.time()
        successful_jobs = 0
        failed_jobs = 0
        cancelled_jobs = 0
        
        for i, job in enumerate(jobs):
            if self.is_cancelled:
                job.status = BatchStatus.CANCELLED
                cancelled_jobs += 1
                continue
            
            # Overall progress callback
            if progress_callback:
                overall_progress = (i / len(jobs)) * 100
                progress_callback("batch_overall", overall_progress, f"Processing job {i+1}/{len(jobs)}")
            
            processed_job = self.process_single_job(job, progress_callback)
            
            if processed_job.status == BatchStatus.COMPLETED:
                successful_jobs += 1
            elif processed_job.status == BatchStatus.FAILED:
                failed_jobs += 1
            elif processed_job.status == BatchStatus.CANCELLED:
                cancelled_jobs += 1
        
        total_duration = time.time() - start_time
        
        return BatchResult(
            total_jobs=len(jobs),
            successful_jobs=successful_jobs,
            failed_jobs=failed_jobs,
            cancelled_jobs=cancelled_jobs,
            total_duration=total_duration,
            job_results=jobs
        )
    
    def run_batch_parallel(self, jobs: List[BatchJob], 
                          progress_callback: Optional[Callable] = None) -> BatchResult:
        """Run batch jobs in parallel"""
        start_time = time.time()
        successful_jobs = 0
        failed_jobs = 0
        cancelled_jobs = 0
        
        # Choose executor type based on configuration
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_single_job, job, progress_callback): job 
                for job in jobs
            }
            
            completed_jobs = 0
            for future in as_completed(future_to_job):
                if self.is_cancelled:
                    # Cancel remaining futures
                    for f in future_to_job:
                        f.cancel()
                    break
                
                job = future_to_job[future]
                completed_jobs += 1
                
                try:
                    processed_job = future.result()
                    
                    if processed_job.status == BatchStatus.COMPLETED:
                        successful_jobs += 1
                    elif processed_job.status == BatchStatus.FAILED:
                        failed_jobs += 1
                    
                    # Overall progress callback
                    if progress_callback:
                        overall_progress = (completed_jobs / len(jobs)) * 100
                        progress_callback("batch_overall", overall_progress, 
                                        f"Completed {completed_jobs}/{len(jobs)} jobs")
                
                except Exception as e:
                    job.status = BatchStatus.FAILED
                    job.error_message = f"Executor error: {str(e)}"
                    failed_jobs += 1
                    self.logger.error(f"Executor error for job {job.job_id}: {str(e)}")
            
            # Handle cancelled jobs
            for job in jobs:
                if job.status == BatchStatus.PENDING:
                    job.status = BatchStatus.CANCELLED
                    cancelled_jobs += 1
        
        total_duration = time.time() - start_time
        
        return BatchResult(
            total_jobs=len(jobs),
            successful_jobs=successful_jobs,
            failed_jobs=failed_jobs,
            cancelled_jobs=cancelled_jobs,
            total_duration=total_duration,
            job_results=jobs
        )
    
    def run_batch(self, config_dir: str, output_dir: str, 
                  parallel: bool = True,
                  progress_callback: Optional[Callable] = None) -> BatchResult:
        """Run batch processing operation"""
        self.logger.info(f"Starting batch processing: {config_dir} -> {output_dir}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare jobs
        jobs = self.prepare_batch_jobs(config_dir, output_dir)
        
        if not jobs:
            raise FileProcessingError("No valid configuration files found for batch processing")
        
        self.logger.info(f"Prepared {len(jobs)} batch jobs")
        
        # Run jobs
        if parallel and len(jobs) > 1:
            result = self.run_batch_parallel(jobs, progress_callback)
        else:
            result = self.run_batch_sequential(jobs, progress_callback)
        
        # Log results
        self.logger.info(f"Batch processing completed:")
        self.logger.info(f"  Total jobs: {result.total_jobs}")
        self.logger.info(f"  Successful: {result.successful_jobs}")
        self.logger.info(f"  Failed: {result.failed_jobs}")
        self.logger.info(f"  Cancelled: {result.cancelled_jobs}")
        self.logger.info(f"  Success rate: {result.success_rate:.1f}%")
        self.logger.info(f"  Total duration: {result.total_duration:.2f}s")
        
        return result
    
    def cancel_batch(self):
        """Cancel batch processing"""
        self.is_cancelled = True
        self.logger.info("Batch processing cancellation requested")
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a specific job"""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of current batch processing state"""
        if not self.jobs:
            return {"status": "no_jobs", "jobs": []}
        
        completed_jobs = [j for j in self.jobs if j.is_completed]
        running_jobs = [j for j in self.jobs if j.status == BatchStatus.RUNNING]
        pending_jobs = [j for j in self.jobs if j.status == BatchStatus.PENDING]
        
        return {
            "status": "cancelled" if self.is_cancelled else "running" if running_jobs else "completed",
            "total_jobs": len(self.jobs),
            "completed_jobs": len(completed_jobs),
            "running_jobs": len(running_jobs),
            "pending_jobs": len(pending_jobs),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "config_path": job.config_path,
                    "status": job.status.value,
                    "progress": job.progress,
                    "duration": job.duration,
                    "error_message": job.error_message
                }
                for job in self.jobs
            ]
        }
    
    def save_batch_report(self, result: BatchResult, report_path: str):
        """Save batch processing report to file"""
        report = {
            "batch_summary": {
                "total_jobs": result.total_jobs,
                "successful_jobs": result.successful_jobs,
                "failed_jobs": result.failed_jobs,
                "cancelled_jobs": result.cancelled_jobs,
                "success_rate": result.success_rate,
                "total_duration": result.total_duration,
                "timestamp": time.time()
            },
            "job_details": [
                {
                    "job_id": job.job_id,
                    "config_path": job.config_path,
                    "output_path": job.output_path,
                    "status": job.status.value,
                    "start_time": job.start_time,
                    "end_time": job.end_time,
                    "duration": job.duration,
                    "error_message": job.error_message
                }
                for job in result.job_results
            ]
        }
        
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Batch report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save batch report: {str(e)}")


class BatchProgressReporter:
    """Helper class for reporting batch processing progress"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.job_progress = {}
        self.start_time = time.time()
    
    def __call__(self, job_id: str, progress: float, message: str = ""):
        """Progress callback function"""
        self.job_progress[job_id] = progress
        
        if job_id == "batch_overall":
            elapsed = time.time() - self.start_time
            if self.verbose:
                print(f"\r[BATCH {progress:6.2f}%] {message} ({elapsed:.1f}s)", end="", flush=True)
            else:
                print(f"\r[{progress:6.2f}%] {message}", end="", flush=True)
        else:
            if self.verbose:
                print(f"\n  [{job_id}] {progress:6.2f}% - {message}")
    
    def print_summary(self, result: BatchResult):
        """Print batch processing summary"""
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total jobs: {result.total_jobs}")
        print(f"Successful: {result.successful_jobs}")
        print(f"Failed: {result.failed_jobs}")
        print(f"Cancelled: {result.cancelled_jobs}")
        print(f"Success rate: {result.success_rate:.1f}%")
        print(f"Total duration: {result.total_duration:.2f}s")
        
        if result.failed_jobs > 0:
            print(f"\nFAILED JOBS:")
            for job in result.job_results:
                if job.status == BatchStatus.FAILED:
                    print(f"  - {job.job_id}: {job.error_message}")
        
        print(f"{'='*60}")