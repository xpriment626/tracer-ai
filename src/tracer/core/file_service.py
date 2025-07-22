"""
Async file processing service for data ingestion
"""

import os
import uuid
import pandas as pd
import aiofiles
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, BinaryIO
from pathlib import Path
import structlog
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from .models import DataIngestionJob, ProcessingStatus, ValidationResult, PipelineConfig
from .validation import DataValidator, BatchValidator
from .preprocessing import AsyncDataProcessor
from ..api.schemas import FileProcessingOptions

logger = structlog.get_logger(__name__)


class FileProcessingService:
    """
    Async file processing service for customer data ingestion
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, storage_dir: Optional[str] = None):
        """
        Initialize file processing service
        
        Args:
            config: Pipeline configuration
            storage_dir: Directory for temporary file storage
        """
        self.config = config or PipelineConfig()
        self.logger = logger.bind(component="FileProcessingService")
        
        # Setup storage directory
        self.storage_dir = Path(storage_dir) if storage_dir else Path(tempfile.gettempdir()) / "tracer_uploads"
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize services
        self.data_validator = DataValidator(
            strict_mode=False,
            auto_correct=self.config.auto_correct_data
        )
        self.batch_validator = BatchValidator(batch_size=1000)
        self.data_processor = AsyncDataProcessor()
        
        # Job tracking
        self.active_jobs: Dict[str, DataIngestionJob] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Start background processor
        self._background_task: Optional[asyncio.Task] = None
    
    async def start_service(self):
        """Start the background processing service"""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._process_queue())
            self.logger.info("File processing service started")
    
    async def stop_service(self):
        """Stop the background processing service"""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
        self.logger.info("File processing service stopped")
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str = "text/csv",
        options: Optional[FileProcessingOptions] = None
    ) -> DataIngestionJob:
        """
        Upload and queue a file for processing
        
        Args:
            file_content: Raw file content
            filename: Original filename
            content_type: MIME content type
            options: Processing options
            
        Returns:
            DataIngestionJob with job tracking information
        """
        self.logger.info("Uploading file", filename=filename, size=len(file_content))
        
        # Validate file
        self._validate_file_upload(file_content, filename, content_type)
        
        # Create job
        job = DataIngestionJob(
            filename=filename,
            file_size=len(file_content),
            status=ProcessingStatus.PENDING
        )
        
        # Save file to storage
        file_path = self.storage_dir / f"{job.job_id}_{filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Add processing options to job metadata
        job.processing_options = options.dict() if options else {}
        job.file_path = str(file_path)
        
        # Track job and queue for processing
        self.active_jobs[job.job_id] = job
        await self.processing_queue.put(job.job_id)
        
        self.logger.info("File queued for processing", job_id=job.job_id, filename=filename)
        return job
    
    async def get_job_status(self, job_id: str) -> Optional[DataIngestionJob]:
        """Get current job status"""
        return self.active_jobs.get(job_id)
    
    async def get_validation_results(self, job_id: str) -> Optional[ValidationResult]:
        """Get detailed validation results for a job"""
        job = self.active_jobs.get(job_id)
        if job and job.validation_result:
            return job.validation_result
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job"""
        job = self.active_jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
            job.status = ProcessingStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.completed_at = datetime.utcnow()
            
            # Clean up file
            if hasattr(job, 'file_path') and os.path.exists(job.file_path):
                os.remove(job.file_path)
            
            self.logger.info("Job cancelled", job_id=job_id)
            return True
        
        return False
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        jobs_to_remove = []
        
        for job_id, job in self.active_jobs.items():
            if (job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED] and
                job.completed_at and job.completed_at.timestamp() < cutoff_time):
                
                # Clean up file
                if hasattr(job, 'file_path') and os.path.exists(job.file_path):
                    try:
                        os.remove(job.file_path)
                    except OSError:
                        pass
                
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        if jobs_to_remove:
            self.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def _validate_file_upload(self, content: bytes, filename: str, content_type: str):
        """Validate file upload constraints"""
        
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise ValueError(f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({self.config.max_file_size_mb}MB)")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config.allowed_extensions:
            raise ValueError(f"File extension '{file_ext}' not allowed. Supported: {self.config.allowed_extensions}")
        
        # Basic content validation
        if not content:
            raise ValueError("File is empty")
        
        # Check if content looks like CSV
        if file_ext == '.csv':
            try:
                content_str = content.decode('utf-8')[:1000]  # Check first 1000 chars
                if not any(delimiter in content_str for delimiter in [',', ';', '\t']):
                    raise ValueError("File does not appear to contain delimited data")
            except UnicodeDecodeError:
                raise ValueError("File encoding is not supported (expected UTF-8)")
    
    async def _process_queue(self):
        """Background task to process queued jobs"""
        self.logger.info("Starting background job processor")
        
        while True:
            try:
                # Wait for job with timeout
                job_id = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                if job_id in self.active_jobs:
                    await self._process_job(job_id)
                else:
                    self.logger.warning("Job not found", job_id=job_id)
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue loop
                continue
            except asyncio.CancelledError:
                # Service is shutting down
                break
            except Exception as e:
                self.logger.error("Error in background processor", error=str(e))
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _process_job(self, job_id: str):
        """Process a single job"""
        job = self.active_jobs[job_id]
        self.logger.info("Starting job processing", job_id=job_id, filename=job.filename)
        
        try:
            # Update job status
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.utcnow()
            
            # Load file
            df = await self._load_file(job.file_path)
            self.logger.info("File loaded", job_id=job_id, rows=len(df), columns=len(df.columns))
            
            # Get processing options
            options = FileProcessingOptions(**job.processing_options)
            
            # Sample data if requested
            if options.sample_size and len(df) > options.sample_size:
                df = df.sample(n=options.sample_size, random_state=42)
                self.logger.info("Data sampled", job_id=job_id, sample_size=len(df))
            
            # Validate data
            validation_result = await self._validate_data(df, job.filename, options)
            job.validation_result = validation_result
            
            # Process data if validation passed or not in strict mode
            processed_df = None
            target = None
            
            if (validation_result.is_valid or not options.strict_validation) and not options.skip_preprocessing:
                try:
                    processed_df, target = await self._preprocess_data(df, options, job_id)
                    self.logger.info("Data preprocessing completed", 
                                   job_id=job_id, 
                                   processed_rows=len(processed_df) if processed_df is not None else 0,
                                   features=len(processed_df.columns) if processed_df is not None else 0)
                except Exception as e:
                    self.logger.error("Preprocessing failed", job_id=job_id, error=str(e))
                    # Continue with validation results only
            
            # Store processed data if successful
            if processed_df is not None:
                await self._store_processed_data(job_id, processed_df, target)
            
            # Complete job
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            self.logger.info("Job processing completed", 
                           job_id=job_id, 
                           valid_records=validation_result.valid_records,
                           invalid_records=validation_result.invalid_records)
            
        except Exception as e:
            # Handle job failure
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            
            self.logger.error("Job processing failed", job_id=job_id, error=str(e))
        
        finally:
            # Clean up original file
            if hasattr(job, 'file_path') and os.path.exists(job.file_path):
                try:
                    os.remove(job.file_path)
                except OSError:
                    pass
    
    async def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load file into DataFrame"""
        file_ext = Path(file_path).suffix.lower()
        
        loop = asyncio.get_event_loop()
        
        if file_ext == '.csv':
            # Try different encodings and delimiters
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for delimiter in [',', ';', '\t']:
                    try:
                        df = await loop.run_in_executor(
                            self.executor,
                            lambda: pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                        )
                        if len(df.columns) > 1:  # Successfully parsed with multiple columns
                            return df
                    except Exception:
                        continue
            
            # Fallback to default pandas behavior
            df = await loop.run_in_executor(self.executor, pd.read_csv, file_path)
            
        elif file_ext == '.xlsx':
            df = await loop.run_in_executor(
                self.executor,
                lambda: pd.read_excel(file_path, engine='openpyxl')
            )
            
        elif file_ext == '.json':
            df = await loop.run_in_executor(self.executor, pd.read_json, file_path)
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if df.empty:
            raise ValueError("File contains no data")
        
        return df
    
    async def _validate_data(
        self, 
        df: pd.DataFrame, 
        filename: str, 
        options: FileProcessingOptions
    ) -> ValidationResult:
        """Validate DataFrame"""
        
        # Use batch validator for large datasets
        if len(df) > 10000:
            return await self.batch_validator.validate_large_dataset(df, filename)
        else:
            # Configure validator based on options
            self.data_validator.strict_mode = options.strict_validation
            self.data_validator.auto_correct = options.auto_correction
            
            return await self.data_validator.validate_dataframe(df, filename)
    
    async def _preprocess_data(
        self, 
        df: pd.DataFrame, 
        options: FileProcessingOptions,
        job_id: str
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess DataFrame"""
        
        target_column = options.target_column or 'churn_status'
        
        if len(df) > 5000:
            # Use batch processing for large datasets
            return await self.data_processor.process_dataframe_batch(df, batch_size=1000)
        else:
            # Direct processing for smaller datasets
            return await self.data_processor.preprocessor.preprocess_data(
                df, fit_transforms=True, target_column=target_column
            )
    
    async def _store_processed_data(
        self, 
        job_id: str, 
        df: pd.DataFrame, 
        target: Optional[pd.Series]
    ):
        """Store processed data for later retrieval"""
        
        output_dir = self.storage_dir / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # Store features
        features_path = output_dir / f"{job_id}_features.parquet"
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: df.to_parquet(features_path, index=False)
        )
        
        # Store target if available
        if target is not None:
            target_path = output_dir / f"{job_id}_target.parquet"
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: target.to_frame().to_parquet(target_path, index=False)
            )
        
        self.logger.info("Processed data stored", job_id=job_id)
    
    async def get_processed_data(self, job_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Retrieve processed data for a job"""
        
        output_dir = self.storage_dir / "processed"
        features_path = output_dir / f"{job_id}_features.parquet"
        target_path = output_dir / f"{job_id}_target.parquet"
        
        features_df = None
        target_series = None
        
        loop = asyncio.get_event_loop()
        
        if features_path.exists():
            features_df = await loop.run_in_executor(
                self.executor,
                lambda: pd.read_parquet(features_path)
            )
        
        if target_path.exists():
            target_df = await loop.run_in_executor(
                self.executor,
                lambda: pd.read_parquet(target_path)
            )
            target_series = target_df.iloc[:, 0] if not target_df.empty else None
        
        return features_df, target_series
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        total_jobs = len(self.active_jobs)
        status_counts = {}
        
        for status in ProcessingStatus:
            status_counts[status.value] = sum(
                1 for job in self.active_jobs.values() if job.status == status
            )
        
        return {
            'total_jobs': total_jobs,
            'status_distribution': status_counts,
            'queue_size': self.processing_queue.qsize(),
            'storage_dir': str(self.storage_dir),
            'active_background_task': self._background_task is not None and not self._background_task.done()
        }