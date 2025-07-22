"""
FastAPI endpoints for data ingestion and validation
"""

import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog

from ..core.file_service import FileProcessingService
from ..core.models import ProcessingStatus, PipelineConfig
from .schemas import (
    HealthResponse, ErrorResponse, FileUploadResponse, ValidationSummaryResponse,
    DetailedValidationResponse, ProcessingStatusResponse, DataProfileResponse,
    BatchProcessingRequest, BatchProcessingResponse, SystemStatusResponse,
    PipelineMetrics, FileProcessingOptions, APIConfiguration
)

logger = structlog.get_logger(__name__)

# Global service instance
file_service: Optional[FileProcessingService] = None
api_config = APIConfiguration()
start_time = time.time()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Tracer AI - Data Ingestion API",
        description="Customer churn prediction data ingestion and validation API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add custom middleware for request logging and error handling
    @app.middleware("http")
    async def logging_middleware(request, call_next):
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info("Request started",
                   method=request.method,
                   url=str(request.url),
                   request_id=request_id)
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info("Request completed",
                       method=request.method,
                       url=str(request.url),
                       status_code=response.status_code,
                       process_time=process_time,
                       request_id=request_id)
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error("Request failed",
                        method=request.method,
                        url=str(request.url),
                        error=str(e),
                        process_time=process_time,
                        request_id=request_id)
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="InternalServerError",
                    message="An internal server error occurred",
                    request_id=request_id,
                    timestamp=datetime.utcnow()
                ).dict()
            )
    
    return app


app = create_app()


async def get_file_service() -> FileProcessingService:
    """Dependency to get file service instance"""
    global file_service
    if file_service is None:
        config = PipelineConfig()
        file_service = FileProcessingService(config=config)
        await file_service.start_service()
    return file_service


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Tracer AI Data Ingestion API")
    
    # Initialize file service
    await get_file_service()
    
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Tracer AI Data Ingestion API")
    
    global file_service
    if file_service:
        await file_service.stop_service()
    
    logger.info("API shutdown completed")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.get("/status", response_model=SystemStatusResponse)
async def system_status(service: FileProcessingService = Depends(get_file_service)):
    """Get comprehensive system status"""
    
    stats = service.get_service_stats()
    
    # Calculate metrics
    completed_jobs = stats['status_distribution'].get('completed', 0)
    failed_jobs = stats['status_distribution'].get('failed', 0)
    total_processed = completed_jobs + failed_jobs
    
    success_rate = (completed_jobs / max(total_processed, 1)) * 100
    error_rate = (failed_jobs / max(total_processed, 1)) * 100
    
    metrics = PipelineMetrics(
        total_files_processed=total_processed,
        total_records_processed=0,  # Would need to track this separately
        average_processing_time_seconds=0.0,  # Would need to track this separately
        success_rate_percentage=success_rate,
        error_rate_percentage=error_rate,
        throughput_records_per_second=0.0,  # Would need to calculate
        uptime_hours=(time.time() - start_time) / 3600
    )
    
    return SystemStatusResponse(
        service_status="operational",
        api_version="1.0.0",
        timestamp=datetime.utcnow(),
        pipeline_metrics=metrics,
        active_jobs=stats['status_distribution'].get('processing', 0),
        queued_jobs=stats['status_distribution'].get('pending', 0),
        system_resources={
            "storage_directory": stats['storage_dir'],
            "background_processor_active": stats['active_background_task']
        }
    )


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strict_validation: bool = Query(False, description="Enable strict validation mode"),
    auto_correction: bool = Query(True, description="Enable automatic data correction"),
    skip_preprocessing: bool = Query(False, description="Skip data preprocessing"),
    sample_size: Optional[int] = Query(None, description="Process only a sample of records"),
    target_column: Optional[str] = Query("churn_status", description="Name of target column"),
    service: FileProcessingService = Depends(get_file_service)
):
    """
    Upload CSV file for processing and validation
    
    This endpoint accepts CSV files containing customer data and processes them
    through validation and preprocessing pipelines.
    """
    
    try:
        # Read file content
        content = await file.read()
        
        # Create processing options
        options = FileProcessingOptions(
            strict_validation=strict_validation,
            auto_correction=auto_correction,
            skip_preprocessing=skip_preprocessing,
            sample_size=sample_size,
            target_column=target_column
        )
        
        # Upload file for processing
        job = await service.upload_file(
            file_content=content,
            filename=file.filename or "uploaded_file.csv",
            content_type=file.content_type or "text/csv",
            options=options
        )
        
        # Estimate processing time based on file size
        estimated_time = min(max(len(content) / 100000, 10), 300)  # 10s to 5min
        
        # Schedule cleanup task
        background_tasks.add_task(service.cleanup_completed_jobs)
        
        return FileUploadResponse(
            job_id=job.job_id,
            filename=job.filename,
            file_size=job.file_size,
            status=job.status,
            message="File uploaded successfully and queued for processing",
            estimated_processing_time_seconds=int(estimated_time)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("File upload failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail="File upload failed")


@app.get("/jobs/{job_id}/status", response_model=ProcessingStatusResponse)
async def get_job_status(
    job_id: str = Path(..., description="Job identifier"),
    service: FileProcessingService = Depends(get_file_service)
):
    """Get processing status for a specific job"""
    
    job = await service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate progress
    progress = None
    if job.status == ProcessingStatus.PROCESSING:
        progress = 50.0  # Simplified progress calculation
    elif job.status == ProcessingStatus.COMPLETED:
        progress = 100.0
    elif job.status == ProcessingStatus.FAILED:
        progress = 0.0
    
    return ProcessingStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_percentage=progress,
        current_step=job.status.value.replace('_', ' ').title(),
        estimated_completion_time=None,  # Could be calculated based on queue
        errors=[job.error_message] if job.error_message else []
    )


@app.get("/jobs/{job_id}/validation", response_model=ValidationSummaryResponse)
async def get_validation_summary(
    job_id: str = Path(..., description="Job identifier"),
    service: FileProcessingService = Depends(get_file_service)
):
    """Get validation summary for a completed job"""
    
    job = await service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not job.validation_result:
        if job.status == ProcessingStatus.PENDING:
            raise HTTPException(status_code=202, detail="Job is still pending")
        elif job.status == ProcessingStatus.PROCESSING:
            raise HTTPException(status_code=202, detail="Job is still processing")
        else:
            raise HTTPException(status_code=404, detail="Validation results not available")
    
    validation = job.validation_result
    critical_errors = len([e for e in validation.errors if e.row_number is None])
    
    return ValidationSummaryResponse(
        job_id=job.job_id,
        is_valid=validation.is_valid,
        total_records=validation.total_records,
        valid_records=validation.valid_records,
        invalid_records=validation.invalid_records,
        error_rate_percentage=validation.error_rate,
        warning_count=len(validation.warnings),
        critical_error_count=critical_errors,
        processing_duration_seconds=job.processing_duration
    )


@app.get("/jobs/{job_id}/validation/details", response_model=DetailedValidationResponse)
async def get_validation_details(
    job_id: str = Path(..., description="Job identifier"),
    max_samples: int = Query(10, ge=1, le=100, description="Maximum number of sample records"),
    service: FileProcessingService = Depends(get_file_service)
):
    """Get detailed validation results including sample records and errors"""
    
    validation_result = await service.get_validation_results(job_id)
    if not validation_result:
        raise HTTPException(status_code=404, detail="Validation results not found")
    
    # Get sample records (this would need to be implemented in the service)
    sample_valid = []  # Would load sample valid records
    sample_invalid = []  # Would load sample invalid records
    
    return DetailedValidationResponse(
        job_id=job_id,
        validation_result=validation_result,
        sample_valid_records=sample_valid[:max_samples],
        sample_invalid_records=sample_invalid[:max_samples]
    )


@app.get("/jobs/{job_id}/profile", response_model=DataProfileResponse)
async def get_data_profile(
    job_id: str = Path(..., description="Job identifier"),
    service: FileProcessingService = Depends(get_file_service)
):
    """Get data profiling information for processed data"""
    
    # Get processed data
    features_df, target_series = await service.get_processed_data(job_id)
    
    if features_df is None:
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    # Create basic data profile
    numeric_columns = features_df.select_dtypes(include=['number']).columns
    
    # Basic statistics
    summary = {
        'total_customers': len(features_df),
        'churn_distribution': {},
        'age_statistics': {},
        'tenure_statistics': {},
        'revenue_statistics': {},
        'service_tier_distribution': {},
        'customer_type_distribution': {},
        'data_quality_score': 85.0  # Simplified calculation
    }
    
    if target_series is not None:
        summary['churn_distribution'] = target_series.value_counts().to_dict()
    
    # Column profiles
    column_profiles = {}
    for col in features_df.columns[:10]:  # Limit to first 10 columns
        if col in numeric_columns:
            column_profiles[col] = {
                'type': 'numeric',
                'mean': float(features_df[col].mean()),
                'std': float(features_df[col].std()),
                'min': float(features_df[col].min()),
                'max': float(features_df[col].max()),
                'null_count': int(features_df[col].isnull().sum())
            }
        else:
            column_profiles[col] = {
                'type': 'categorical',
                'unique_count': int(features_df[col].nunique()),
                'null_count': int(features_df[col].isnull().sum()),
                'top_values': features_df[col].value_counts().head(5).to_dict()
            }
    
    return DataProfileResponse(
        job_id=job_id,
        summary=summary,
        column_profiles=column_profiles,
        correlations={},  # Would calculate correlations
        missing_data_analysis={},  # Would analyze missing data
        outlier_analysis={}  # Would analyze outliers
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str = Path(..., description="Job identifier"),
    service: FileProcessingService = Depends(get_file_service)
):
    """Cancel a pending or processing job"""
    
    success = await service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    
    return {"message": "Job cancelled successfully", "job_id": job_id}


@app.get("/jobs", response_model=List[ProcessingStatusResponse])
async def list_jobs(
    status: Optional[ProcessingStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of jobs to return"),
    service: FileProcessingService = Depends(get_file_service)
):
    """List all jobs with optional status filtering"""
    
    stats = service.get_service_stats()
    all_jobs = service.active_jobs
    
    # Filter by status if specified
    filtered_jobs = all_jobs
    if status:
        filtered_jobs = {k: v for k, v in all_jobs.items() if v.status == status}
    
    # Sort by creation time (most recent first)
    sorted_jobs = sorted(filtered_jobs.values(), key=lambda x: x.created_at, reverse=True)
    
    # Limit results
    limited_jobs = sorted_jobs[:limit]
    
    # Convert to response format
    responses = []
    for job in limited_jobs:
        progress = None
        if job.status == ProcessingStatus.PROCESSING:
            progress = 50.0
        elif job.status == ProcessingStatus.COMPLETED:
            progress = 100.0
        elif job.status == ProcessingStatus.FAILED:
            progress = 0.0
        
        responses.append(ProcessingStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress_percentage=progress,
            current_step=job.status.value.replace('_', ' ').title(),
            estimated_completion_time=None,
            errors=[job.error_message] if job.error_message else []
        ))
    
    return responses


@app.post("/batch/process", response_model=BatchProcessingResponse)
async def batch_process_jobs(
    request: BatchProcessingRequest,
    service: FileProcessingService = Depends(get_file_service)
):
    """Process multiple jobs as a batch operation"""
    
    # Validate job IDs
    accepted_jobs = []
    rejected_jobs = []
    
    for job_id in request.job_ids:
        job = await service.get_job_status(job_id)
        if job and job.status in [ProcessingStatus.PENDING, ProcessingStatus.FAILED]:
            accepted_jobs.append(job_id)
        else:
            rejected_jobs.append(job_id)
    
    batch_id = str(uuid.uuid4())
    
    return BatchProcessingResponse(
        batch_id=batch_id,
        accepted_jobs=accepted_jobs,
        rejected_jobs=rejected_jobs,
        estimated_completion_time=None  # Would calculate based on queue
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValueError",
            message=str(exc),
            request_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)