"""
Custom exception classes for Tracer AI
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class TracerBaseException(Exception):
    """Base exception class for all Tracer AI exceptions"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.context = context or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'exception_type': self.__class__.__name__
        }


class ValidationError(TracerBaseException):
    """Raised when data validation fails"""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        row_number: Optional[int] = None,
        validation_errors: Optional[List[str]] = None
    ):
        details = {}
        if field:
            details['field'] = field
        if invalid_value is not None:
            details['invalid_value'] = str(invalid_value)
        if row_number is not None:
            details['row_number'] = row_number
        if validation_errors:
            details['validation_errors'] = validation_errors
        
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)


class FileProcessingError(TracerBaseException):
    """Raised when file processing fails"""
    
    def __init__(
        self, 
        message: str, 
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        processing_step: Optional[str] = None
    ):
        details = {}
        if filename:
            details['filename'] = filename
        if file_size is not None:
            details['file_size'] = file_size
        if processing_step:
            details['processing_step'] = processing_step
        
        super().__init__(message, error_code="FILE_PROCESSING_ERROR", details=details)


class DataIngestionError(TracerBaseException):
    """Raised when data ingestion fails"""
    
    def __init__(
        self, 
        message: str, 
        job_id: Optional[str] = None,
        stage: Optional[str] = None
    ):
        details = {}
        if job_id:
            details['job_id'] = job_id
        if stage:
            details['stage'] = stage
        
        super().__init__(message, error_code="DATA_INGESTION_ERROR", details=details)


class PreprocessingError(TracerBaseException):
    """Raised when data preprocessing fails"""
    
    def __init__(
        self, 
        message: str, 
        preprocessing_step: Optional[str] = None,
        feature_name: Optional[str] = None,
        data_shape: Optional[tuple] = None
    ):
        details = {}
        if preprocessing_step:
            details['preprocessing_step'] = preprocessing_step
        if feature_name:
            details['feature_name'] = feature_name
        if data_shape:
            details['data_shape'] = data_shape
        
        super().__init__(message, error_code="PREPROCESSING_ERROR", details=details)


class ConfigurationError(TracerBaseException):
    """Raised when there's a configuration issue"""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None
    ):
        details = {}
        if config_key:
            details['config_key'] = config_key
        if expected_type:
            details['expected_type'] = expected_type
        if actual_value is not None:
            details['actual_value'] = str(actual_value)
        
        super().__init__(message, error_code="CONFIGURATION_ERROR", details=details)


class ResourceError(TracerBaseException):
    """Raised when there's a resource-related issue"""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        operation: Optional[str] = None
    ):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        if operation:
            details['operation'] = operation
        
        super().__init__(message, error_code="RESOURCE_ERROR", details=details)


class SecurityError(TracerBaseException):
    """Raised when there's a security-related issue"""
    
    def __init__(
        self, 
        message: str, 
        security_context: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        details = {}
        if security_context:
            details['security_context'] = security_context
        if user_id:
            details['user_id'] = user_id
        if ip_address:
            details['ip_address'] = ip_address
        
        super().__init__(message, error_code="SECURITY_ERROR", details=details)


class RateLimitError(TracerBaseException):
    """Raised when rate limits are exceeded"""
    
    def __init__(
        self, 
        message: str, 
        rate_limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None
    ):
        details = {}
        if rate_limit is not None:
            details['rate_limit'] = rate_limit
        if window_seconds is not None:
            details['window_seconds'] = window_seconds
        if retry_after is not None:
            details['retry_after'] = retry_after
        
        super().__init__(message, error_code="RATE_LIMIT_ERROR", details=details)


class ServiceUnavailableError(TracerBaseException):
    """Raised when a required service is unavailable"""
    
    def __init__(
        self, 
        message: str, 
        service_name: Optional[str] = None,
        expected_availability: Optional[str] = None
    ):
        details = {}
        if service_name:
            details['service_name'] = service_name
        if expected_availability:
            details['expected_availability'] = expected_availability
        
        super().__init__(message, error_code="SERVICE_UNAVAILABLE_ERROR", details=details)


class DataQualityError(TracerBaseException):
    """Raised when data quality issues are detected"""
    
    def __init__(
        self, 
        message: str, 
        quality_check: Optional[str] = None,
        threshold: Optional[float] = None,
        actual_value: Optional[float] = None,
        affected_columns: Optional[List[str]] = None
    ):
        details = {}
        if quality_check:
            details['quality_check'] = quality_check
        if threshold is not None:
            details['threshold'] = threshold
        if actual_value is not None:
            details['actual_value'] = actual_value
        if affected_columns:
            details['affected_columns'] = affected_columns
        
        super().__init__(message, error_code="DATA_QUALITY_ERROR", details=details)


class JobNotFoundError(TracerBaseException):
    """Raised when a requested job cannot be found"""
    
    def __init__(self, job_id: str):
        super().__init__(
            f"Job with ID '{job_id}' not found",
            error_code="JOB_NOT_FOUND",
            details={'job_id': job_id}
        )


class InvalidJobStateError(TracerBaseException):
    """Raised when attempting invalid operations on jobs"""
    
    def __init__(
        self, 
        message: str, 
        job_id: str, 
        current_state: str, 
        expected_states: Optional[List[str]] = None
    ):
        details = {
            'job_id': job_id,
            'current_state': current_state
        }
        if expected_states:
            details['expected_states'] = expected_states
        
        super().__init__(message, error_code="INVALID_JOB_STATE", details=details)


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    ValidationError: 400,
    FileProcessingError: 400,
    ConfigurationError: 400,
    DataQualityError: 400,
    SecurityError: 403,
    RateLimitError: 429,
    JobNotFoundError: 404,
    InvalidJobStateError: 409,
    ResourceError: 500,
    DataIngestionError: 500,
    PreprocessingError: 500,
    ServiceUnavailableError: 503,
    TracerBaseException: 500  # Default for any other Tracer exception
}


def get_http_status_for_exception(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception"""
    for exc_type, status_code in EXCEPTION_STATUS_MAP.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default for unknown exceptions


class ExceptionHandler:
    """Centralized exception handling and logging"""
    
    def __init__(self):
        from .logging_config import get_error_tracker, get_audit_logger
        self.error_tracker = get_error_tracker()
        self.audit_logger = get_audit_logger()
    
    def handle_exception(
        self, 
        exception: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle and log an exception, return formatted error response
        
        Args:
            exception: The exception to handle
            context: Additional context information
            
        Returns:
            Dictionary with error information for API response
        """
        
        # Log the exception
        if isinstance(exception, TracerBaseException):
            # Use the exception's built-in details
            error_response = exception.to_dict()
            
            # Track the error
            self.error_tracker.track_error(
                error_type=exception.error_code,
                error_message=exception.message,
                context={**exception.context, **(context or {})}
            )
            
            # Audit log for security errors
            if isinstance(exception, SecurityError):
                self.audit_logger.log_security_event(
                    event_type="security_exception",
                    details=exception.details
                )
        
        else:
            # Handle non-Tracer exceptions
            error_response = {
                'error_code': 'INTERNAL_ERROR',
                'message': str(exception),
                'details': {},
                'context': context or {},
                'timestamp': datetime.utcnow().isoformat(),
                'exception_type': exception.__class__.__name__
            }
            
            # Track the error
            self.error_tracker.track_error(
                error_type='INTERNAL_ERROR',
                error_message=str(exception),
                context=context
            )
        
        return error_response
    
    def create_validation_error(
        self, 
        errors: List[str], 
        field: Optional[str] = None
    ) -> ValidationError:
        """Create a validation error from a list of error messages"""
        message = f"Validation failed: {'; '.join(errors)}"
        return ValidationError(
            message=message,
            field=field,
            validation_errors=errors
        )
    
    def create_file_processing_error(
        self, 
        message: str, 
        filename: str, 
        step: str
    ) -> FileProcessingError:
        """Create a file processing error"""
        return FileProcessingError(
            message=message,
            filename=filename,
            processing_step=step
        )


# Global exception handler instance
exception_handler = ExceptionHandler()


def get_exception_handler() -> ExceptionHandler:
    """Get the global exception handler instance"""
    return exception_handler