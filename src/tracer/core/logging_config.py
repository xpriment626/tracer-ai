"""
Comprehensive logging configuration for Tracer AI
"""

import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import structlog
from datetime import datetime
import json


class TracerJSONProcessor:
    """Custom JSON processor for Tracer logs"""
    
    def __call__(self, logger, method_name, event_dict):
        """Process log entry and add custom fields"""
        
        # Add timestamp if not present
        if 'timestamp' not in event_dict:
            event_dict['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service identifier
        event_dict['service'] = 'tracer-ai'
        event_dict['component'] = event_dict.get('component', 'unknown')
        
        # Add log level
        event_dict['level'] = method_name.upper()
        
        # Format exception information if present
        if 'exc_info' in event_dict:
            exc_info = event_dict.pop('exc_info')
            if exc_info:
                event_dict['exception'] = {
                    'type': exc_info[0].__name__ if exc_info[0] else None,
                    'message': str(exc_info[1]) if exc_info[1] else None,
                    'traceback': structlog.processors.format_exc_info(logger, method_name, {'exc_info': exc_info})
                }
        
        return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, pretty, simple)
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure processors based on format
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            parameters=[structlog.processors.CallsiteParameter.FILENAME,
                       structlog.processors.CallsiteParameter.LINENO]
        ),
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if log_format == "json":
        processors.extend([
            TracerJSONProcessor(),
            structlog.processors.JSONRenderer()
        ])
    elif log_format == "pretty":
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:  # simple format
        processors.extend([
            structlog.processors.KeyValueRenderer(key_order=['timestamp', 'level', 'event'])
        ])
    
    # Setup handlers
    handlers = []
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        handlers.append(console_handler)
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always capture all levels to file
        handlers.append(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    
    # Set specific logger levels
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    

class ErrorTracker:
    """Track and analyze errors across the system"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_details: Dict[str, Dict[str, Any]] = {}
        self.logger = structlog.get_logger(__name__).bind(component="ErrorTracker")
    
    def track_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Track an error occurrence"""
        
        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Store details
        if error_type not in self.error_details:
            self.error_details[error_type] = {
                'first_seen': datetime.utcnow().isoformat(),
                'count': 0,
                'recent_messages': [],
                'recent_contexts': []
            }
        
        details = self.error_details[error_type]
        details['count'] += 1
        details['last_seen'] = datetime.utcnow().isoformat()
        
        # Keep last 5 messages and contexts
        if error_message not in details['recent_messages']:
            details['recent_messages'].append(error_message)
            if len(details['recent_messages']) > 5:
                details['recent_messages'].pop(0)
        
        if context:
            details['recent_contexts'].append(context)
            if len(details['recent_contexts']) > 5:
                details['recent_contexts'].pop(0)
        
        self.logger.error("Error tracked",
                         error_type=error_type,
                         message=error_message,
                         context=context,
                         total_count=details['count'])
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked errors"""
        return {
            'error_counts': self.error_counts,
            'error_details': self.error_details,
            'total_unique_errors': len(self.error_counts),
            'total_error_occurrences': sum(self.error_counts.values())
        }
    
    def get_frequent_errors(self, top_n: int = 5) -> Dict[str, int]:
        """Get most frequent errors"""
        sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_errors[:top_n])


class PerformanceTracker:
    """Track performance metrics across the system"""
    
    def __init__(self):
        self.request_times: Dict[str, list] = {}
        self.operation_times: Dict[str, list] = {}
        self.logger = structlog.get_logger(__name__).bind(component="PerformanceTracker")
    
    def track_request(self, endpoint: str, duration: float):
        """Track API request performance"""
        if endpoint not in self.request_times:
            self.request_times[endpoint] = []
        
        self.request_times[endpoint].append(duration)
        
        # Keep only last 100 measurements
        if len(self.request_times[endpoint]) > 100:
            self.request_times[endpoint].pop(0)
        
        # Log slow requests
        if duration > 5.0:  # More than 5 seconds
            self.logger.warning("Slow request detected",
                              endpoint=endpoint,
                              duration=duration)
    
    def track_operation(self, operation: str, duration: float):
        """Track internal operation performance"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        self.operation_times[operation].append(duration)
        
        # Keep only last 100 measurements
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation].pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        summary = {
            'request_metrics': {},
            'operation_metrics': {}
        }
        
        # Calculate request metrics
        for endpoint, times in self.request_times.items():
            if times:
                summary['request_metrics'][endpoint] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times),
                    'p95_duration': sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0
                }
        
        # Calculate operation metrics
        for operation, times in self.operation_times.items():
            if times:
                summary['operation_metrics'][operation] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times),
                    'p95_duration': sorted(times)[int(len(times) * 0.95)] if len(times) > 0 else 0
                }
        
        return summary


class AuditLogger:
    """Security and compliance audit logging"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__).bind(component="AuditLogger")
    
    def log_file_upload(self, user_id: Optional[str], filename: str, file_size: int, job_id: str):
        """Log file upload events"""
        self.logger.info("File uploaded",
                        event_type="file_upload",
                        user_id=user_id,
                        filename=filename,
                        file_size=file_size,
                        job_id=job_id)
    
    def log_data_access(self, user_id: Optional[str], job_id: str, access_type: str):
        """Log data access events"""
        self.logger.info("Data accessed",
                        event_type="data_access",
                        user_id=user_id,
                        job_id=job_id,
                        access_type=access_type)
    
    def log_validation_results(self, job_id: str, valid_records: int, invalid_records: int):
        """Log validation results"""
        self.logger.info("Data validated",
                        event_type="data_validation",
                        job_id=job_id,
                        valid_records=valid_records,
                        invalid_records=invalid_records)
    
    def log_processing_error(self, job_id: str, error_type: str, error_message: str):
        """Log processing errors"""
        self.logger.error("Processing error occurred",
                         event_type="processing_error",
                         job_id=job_id,
                         error_type=error_type,
                         error_message=error_message)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        self.logger.warning("Security event",
                          event_type="security",
                          security_event_type=event_type,
                          details=details)


# Global instances
error_tracker = ErrorTracker()
performance_tracker = PerformanceTracker()
audit_logger = AuditLogger()


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    return error_tracker


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance"""
    return performance_tracker


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    return audit_logger