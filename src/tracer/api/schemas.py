"""
API response schemas and data transfer objects
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field
from ..core.models import (
    ValidationResult, ValidationError, ProcessingStatus, 
    DataIngestionJob, ChurnStatus, CustomerType, ServiceTier
)


class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FileUploadResponse(BaseModel):
    """Response for file upload operations"""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    status: ProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    estimated_processing_time_seconds: Optional[int] = Field(None, description="Estimated processing time")
    
    class Config:
        use_enum_values = True


class ValidationSummaryResponse(BaseModel):
    """Summary response for validation results"""
    job_id: str = Field(..., description="Job identifier")
    is_valid: bool = Field(..., description="Overall validation status")
    total_records: int = Field(..., description="Total number of records")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")
    error_rate_percentage: float = Field(..., description="Error rate as percentage")
    warning_count: int = Field(..., description="Number of warnings")
    critical_error_count: int = Field(..., description="Number of critical errors")
    processing_duration_seconds: Optional[float] = Field(None, description="Processing time")


class DetailedValidationResponse(BaseModel):
    """Detailed validation response with all errors and warnings"""
    job_id: str = Field(..., description="Job identifier")
    validation_result: ValidationResult = Field(..., description="Complete validation result")
    sample_valid_records: List[Dict[str, Any]] = Field(default_factory=list, description="Sample of valid records")
    sample_invalid_records: List[Dict[str, Any]] = Field(default_factory=list, description="Sample of invalid records")


class ProcessingStatusResponse(BaseModel):
    """Processing status response"""
    job_id: str = Field(..., description="Job identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress_percentage: Optional[float] = Field(None, description="Processing progress (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CustomerDataSummary(BaseModel):
    """Summary statistics for customer data"""
    total_customers: int = Field(..., description="Total number of customers")
    churn_distribution: Dict[str, int] = Field(..., description="Distribution of churn status")
    age_statistics: Dict[str, float] = Field(..., description="Age statistics")
    tenure_statistics: Dict[str, float] = Field(..., description="Tenure statistics")
    revenue_statistics: Dict[str, float] = Field(..., description="Revenue statistics")
    service_tier_distribution: Dict[str, int] = Field(..., description="Service tier distribution")
    customer_type_distribution: Dict[str, int] = Field(..., description="Customer type distribution")
    data_quality_score: float = Field(..., description="Overall data quality score (0-100)")


class DataProfileResponse(BaseModel):
    """Data profiling response"""
    job_id: str = Field(..., description="Job identifier")
    summary: CustomerDataSummary = Field(..., description="Data summary statistics")
    column_profiles: Dict[str, Dict[str, Any]] = Field(..., description="Individual column profiles")
    correlations: Dict[str, Dict[str, float]] = Field(..., description="Feature correlations")
    missing_data_analysis: Dict[str, float] = Field(..., description="Missing data percentages")
    outlier_analysis: Dict[str, int] = Field(..., description="Outlier counts by column")


class BatchProcessingRequest(BaseModel):
    """Request schema for batch processing"""
    job_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of job IDs to process")
    processing_options: Optional[Dict[str, Any]] = Field(None, description="Additional processing options")
    priority: Optional[str] = Field("normal", description="Processing priority")
    
    class Config:
        schema_extra = {
            "example": {
                "job_ids": ["job-123", "job-456"],
                "processing_options": {
                    "strict_validation": False,
                    "auto_correction": True
                },
                "priority": "high"
            }
        }


class BatchProcessingResponse(BaseModel):
    """Response for batch processing operations"""
    batch_id: str = Field(..., description="Batch processing identifier")
    accepted_jobs: List[str] = Field(..., description="Successfully accepted job IDs")
    rejected_jobs: List[str] = Field(default_factory=list, description="Rejected job IDs")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated batch completion time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PipelineMetrics(BaseModel):
    """Pipeline performance metrics"""
    total_files_processed: int = Field(..., description="Total files processed")
    total_records_processed: int = Field(..., description="Total records processed")
    average_processing_time_seconds: float = Field(..., description="Average processing time per file")
    success_rate_percentage: float = Field(..., description="Success rate percentage")
    error_rate_percentage: float = Field(..., description="Error rate percentage")
    throughput_records_per_second: float = Field(..., description="Processing throughput")
    uptime_hours: float = Field(..., description="Service uptime in hours")


class SystemStatusResponse(BaseModel):
    """Comprehensive system status response"""
    service_status: str = Field(..., description="Overall service status")
    api_version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Status timestamp")
    pipeline_metrics: PipelineMetrics = Field(..., description="Pipeline performance metrics")
    active_jobs: int = Field(..., description="Number of active jobs")
    queued_jobs: int = Field(..., description="Number of queued jobs")
    system_resources: Dict[str, Any] = Field(..., description="System resource utilization")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CustomerRecordResponse(BaseModel):
    """Response schema for individual customer records"""
    customer_id: str = Field(..., description="Customer identifier")
    validation_status: str = Field(..., description="Validation status")
    data_quality_score: float = Field(..., description="Individual record quality score")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing fields")
    invalid_fields: List[str] = Field(default_factory=list, description="List of invalid fields")
    corrected_fields: List[str] = Field(default_factory=list, description="List of auto-corrected fields")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class BulkValidationResponse(BaseModel):
    """Response for bulk validation operations"""
    job_id: str = Field(..., description="Bulk validation job ID")
    total_records: int = Field(..., description="Total records in bulk operation")
    processed_records: int = Field(..., description="Number of processed records")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")
    validation_summary: ValidationSummaryResponse = Field(..., description="Overall validation summary")
    sample_results: List[CustomerRecordResponse] = Field(default_factory=list, description="Sample validation results")


# Configuration models for API operations
class APIConfiguration(BaseModel):
    """API configuration settings"""
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size in MB")
    max_batch_size: int = Field(default=10000, ge=1, le=100000, description="Maximum batch size")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Request timeout in seconds")
    enable_async_processing: bool = Field(default=True, description="Enable asynchronous processing")
    enable_detailed_logging: bool = Field(default=False, description="Enable detailed request logging")
    rate_limit_per_minute: int = Field(default=100, ge=1, le=1000, description="Rate limit per minute")


class FileProcessingOptions(BaseModel):
    """Options for file processing"""
    strict_validation: bool = Field(default=False, description="Enable strict validation mode")
    auto_correction: bool = Field(default=True, description="Enable automatic data correction")
    skip_preprocessing: bool = Field(default=False, description="Skip data preprocessing")
    sample_size: Optional[int] = Field(None, ge=1, le=10000, description="Process only a sample of records")
    target_column: Optional[str] = Field(None, description="Name of target column for supervised learning")
    
    class Config:
        schema_extra = {
            "example": {
                "strict_validation": False,
                "auto_correction": True,
                "skip_preprocessing": False,
                "sample_size": 1000,
                "target_column": "churn_status"
            }
        }