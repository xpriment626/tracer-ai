"""
Tracer AI - Customer Churn Detection Data Pipeline

A comprehensive data ingestion and validation system for customer churn prediction.
"""

__version__ = "1.0.0"
__author__ = "Tracer AI Team"
__description__ = "Customer churn detection data pipeline"

from .core.config import settings
from .core.logging_config import setup_logging


def initialize_tracer():
    """Initialize Tracer AI system"""
    # Setup logging
    setup_logging(
        log_level=settings.logging.log_level,
        log_format=settings.logging.log_format,
        log_file=settings.logging.log_file_path,
        enable_console=settings.logging.console_logging
    )
    
    # Setup directories
    settings.setup_directories()
    
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("Tracer AI initialized", version=__version__)


# Expose key components
from .core.models import (
    CustomerRecord,
    CustomerBatch,
    ValidationResult,
    DataIngestionJob,
    ProcessingStatus,
    PipelineConfig
)

from .core.validation import DataValidator, BatchValidator
from .core.preprocessing import DataPreprocessor, AsyncDataProcessor
from .core.file_service import FileProcessingService
from .core.exceptions import (
    TracerBaseException,
    ValidationError,
    FileProcessingError,
    DataIngestionError,
    PreprocessingError
)

__all__ = [
    '__version__',
    '__author__',
    '__description__',
    'initialize_tracer',
    'settings',
    'CustomerRecord',
    'CustomerBatch',
    'ValidationResult',
    'DataIngestionJob',
    'ProcessingStatus',
    'PipelineConfig',
    'DataValidator',
    'BatchValidator',
    'DataPreprocessor',
    'AsyncDataProcessor',
    'FileProcessingService',
    'TracerBaseException',
    'ValidationError',
    'FileProcessingError',
    'DataIngestionError',
    'PreprocessingError'
]