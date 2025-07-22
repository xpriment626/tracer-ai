"""
Core data models for the Tracer data pipeline system
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import pandas as pd


class DataQualityLevel(Enum):
    """Data quality severity levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


class DataType(Enum):
    """Supported data types for schema validation"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string" 
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class ProcessingStatus(Enum):
    """Data processing pipeline status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ColumnSpec:
    """Column specification for data schema"""
    name: str
    data_type: DataType
    required: bool = True
    nullable: bool = False
    description: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


@dataclass 
class DataQualityIssue:
    """Represents a data quality issue found during validation"""
    column: str
    issue_type: str
    severity: DataQualityLevel
    message: str
    failed_count: int = 0
    total_count: int = 0
    examples: Optional[List[Any]] = None
    
    @property
    def failure_rate(self) -> float:
        """Calculate the failure rate for this issue"""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count


class DatasetMetadata(BaseModel):
    """Metadata about an uploaded dataset"""
    id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns") 
    column_names: List[str] = Field(..., description="List of column names")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str = Field(..., description="SHA-256 hash of file content")
    mime_type: str = Field(default="text/csv")
    encoding: str = Field(default="utf-8")
    
    @validator('size_bytes')
    def validate_file_size(cls, v):
        # 100MB limit
        max_size = 100 * 1024 * 1024
        if v > max_size:
            raise ValueError(f"File size {v} exceeds maximum allowed size {max_size}")
        return v


class ValidationReport(BaseModel):
    """Data validation report"""
    dataset_id: str
    schema_name: str
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    passed: bool
    issues: List[DataQualityIssue] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def critical_issues(self) -> List[DataQualityIssue]:
        """Get critical severity issues"""
        return [issue for issue in self.issues if issue.severity == DataQualityLevel.CRITICAL]
    
    @property
    def high_issues(self) -> List[DataQualityIssue]:
        """Get high severity issues"""
        return [issue for issue in self.issues if issue.severity == DataQualityLevel.HIGH]


class ProcessingResult(BaseModel):
    """Result of data processing operation"""
    dataset_id: str
    operation: str
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    rows_processed: int = 0
    rows_output: int = 0
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_completed(self, rows_output: int = None):
        """Mark processing as completed"""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = ProcessingStatus.COMPLETED
        if rows_output is not None:
            self.rows_output = rows_output
    
    def mark_failed(self, error_message: str):
        """Mark processing as failed"""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = ProcessingStatus.FAILED
        self.errors.append(error_message)


@dataclass
class FeatureEngineering:
    """Feature engineering configuration and results"""
    dataset_id: str
    features_created: List[str] = field(default_factory=list)
    features_dropped: List[str] = field(default_factory=list)
    transformations_applied: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None
    
    def add_feature(self, feature_name: str, transformation: str):
        """Add a newly created feature"""
        self.features_created.append(feature_name)
        self.transformations_applied.append(f"{feature_name}: {transformation}")
    
    def drop_feature(self, feature_name: str, reason: str = ""):
        """Record a dropped feature"""
        self.features_dropped.append(feature_name)
        if reason:
            self.transformations_applied.append(f"DROPPED {feature_name}: {reason}")


class PipelineConfig(BaseModel):
    """Configuration for data pipeline execution"""
    pipeline_id: str
    dataset_id: str
    blueprint_name: str
    validation_config: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = Field(default_factory=dict)
    feature_engineering_config: Dict[str, Any] = Field(default_factory=dict)
    quality_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        'completeness': 0.95,
        'uniqueness': 1.0,
        'validity': 0.98
    })
    
    class Config:
        # Allow extra fields for blueprint-specific configurations
        extra = "allow"


class DataPipelineState(BaseModel):
    """Represents the current state of a data pipeline execution"""
    pipeline_id: str
    dataset_id: str
    blueprint_name: str
    current_stage: str
    status: ProcessingStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    stages_completed: List[str] = Field(default_factory=list)
    results: Dict[str, ProcessingResult] = Field(default_factory=dict)
    validation_report: Optional[ValidationReport] = None
    feature_engineering: Optional[FeatureEngineering] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_stage(self, stage: str, status: ProcessingStatus = ProcessingStatus.IN_PROGRESS):
        """Update the current processing stage"""
        if self.current_stage and self.current_stage not in self.stages_completed:
            self.stages_completed.append(self.current_stage)
        
        self.current_stage = stage
        self.status = status
        self.updated_at = datetime.utcnow()
    
    def add_result(self, stage: str, result: ProcessingResult):
        """Add a processing result for a stage"""
        self.results[stage] = result
        self.updated_at = datetime.utcnow()
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline execution is complete"""
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def has_failed(self) -> bool:
        """Check if pipeline execution has failed"""
        return self.status == ProcessingStatus.FAILED