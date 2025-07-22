"""
Tracer Framework Core Data Pipeline Components

This package contains the core data processing components for the Tracer ML framework:
- Data ingestion pipelines
- Data validation and quality checks
- Data preprocessing workflows
- Pipeline orchestration
- Schema definitions
"""

import pandas as pd

# New pipeline system components
from .pipeline import DataPipeline
from .ingestion import DataIngestionService
from .validation import DataValidationService, ChurnDetectionSchema
from .preprocessing import DataPreprocessingService
from .schemas import schema_registry
from .orchestrator import PipelineOrchestrator
from .models import (
    DatasetMetadata, 
    DataPipelineState, 
    ProcessingResult,
    ValidationReport,
    FeatureEngineering,
    ProcessingStatus,
    PipelineConfig,
    DataQualityIssue,
    DataQualityLevel
)

__all__ = [
    # Main pipeline interface
    'DataPipeline',
    
    # Core services
    'DataIngestionService', 
    'DataValidationService',
    'DataPreprocessingService',
    'PipelineOrchestrator',
    
    # Schemas and registry
    'ChurnDetectionSchema',
    'schema_registry',
    
    # Data models
    'DatasetMetadata',
    'DataPipelineState',
    'ProcessingResult',
    'ValidationReport', 
    'FeatureEngineering',
    'PipelineConfig',
    'DataQualityIssue',
    'DataQualityLevel',
    'ProcessingStatus',
    
    # Dependencies
    'pd'  # pandas for external use
]