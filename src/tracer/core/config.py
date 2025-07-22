"""
Configuration management for Tracer AI
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from pydantic.types import PositiveInt, NonNegativeFloat
import structlog

logger = structlog.get_logger(__name__)


class APISettings(BaseSettings):
    """API server configuration settings"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    
    # Security settings
    allowed_hosts: List[str] = Field(default=["*"], env="API_ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="API_CORS_ALLOW_CREDENTIALS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="API_RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=100, env="API_RATE_LIMIT_RPM")
    rate_limit_burst_size: int = Field(default=200, env="API_RATE_LIMIT_BURST")
    
    # Request/response settings
    max_request_size_mb: int = Field(default=100, env="API_MAX_REQUEST_SIZE_MB")
    request_timeout_seconds: int = Field(default=300, env="API_REQUEST_TIMEOUT")
    
    # API versioning and documentation
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    docs_enabled: bool = Field(default=True, env="API_DOCS_ENABLED")
    docs_url: str = Field(default="/docs", env="API_DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="API_REDOC_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class DataProcessingSettings(BaseSettings):
    """Data processing pipeline configuration"""
    
    # File upload constraints
    max_file_size_mb: int = Field(default=100, env="DATA_MAX_FILE_SIZE_MB")
    max_batch_size: int = Field(default=10000, env="DATA_MAX_BATCH_SIZE")
    allowed_extensions: List[str] = Field(
        default=[".csv", ".xlsx", ".json"], 
        env="DATA_ALLOWED_EXTENSIONS"
    )
    
    # Processing settings
    validation_strict_mode: bool = Field(default=False, env="DATA_VALIDATION_STRICT")
    auto_correct_data: bool = Field(default=True, env="DATA_AUTO_CORRECT")
    enable_preprocessing: bool = Field(default=True, env="DATA_ENABLE_PREPROCESSING")
    
    # Performance tuning
    max_workers: int = Field(default=4, env="DATA_MAX_WORKERS")
    chunk_size: int = Field(default=1000, env="DATA_CHUNK_SIZE")
    batch_processing_threshold: int = Field(default=5000, env="DATA_BATCH_THRESHOLD")
    
    # Data quality thresholds
    missing_data_threshold: float = Field(default=0.3, env="DATA_MISSING_THRESHOLD")
    outlier_z_score_threshold: float = Field(default=3.5, env="DATA_OUTLIER_THRESHOLD")
    min_customer_records: int = Field(default=10, env="DATA_MIN_RECORDS")
    max_customer_records: int = Field(default=1000000, env="DATA_MAX_RECORDS")
    
    # Feature engineering settings
    enable_feature_engineering: bool = Field(default=True, env="DATA_FEATURE_ENGINEERING")
    feature_selection_enabled: bool = Field(default=True, env="DATA_FEATURE_SELECTION")
    correlation_threshold: float = Field(default=0.95, env="DATA_CORRELATION_THRESHOLD")
    
    @validator('allowed_extensions')
    def validate_extensions(cls, v):
        """Ensure extensions start with dot"""
        return [ext if ext.startswith('.') else f'.{ext}' for ext in v]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class StorageSettings(BaseSettings):
    """Storage and file management configuration"""
    
    # Storage directories
    upload_directory: str = Field(default="/tmp/tracer_uploads", env="STORAGE_UPLOAD_DIR")
    processed_directory: str = Field(default="/tmp/tracer_processed", env="STORAGE_PROCESSED_DIR")
    logs_directory: str = Field(default="/tmp/tracer_logs", env="STORAGE_LOGS_DIR")
    temp_directory: str = Field(default="/tmp/tracer_temp", env="STORAGE_TEMP_DIR")
    
    # Retention policies
    temp_file_retention_hours: int = Field(default=24, env="STORAGE_TEMP_RETENTION_HOURS")
    processed_data_retention_days: int = Field(default=30, env="STORAGE_PROCESSED_RETENTION_DAYS")
    log_retention_days: int = Field(default=7, env="STORAGE_LOG_RETENTION_DAYS")
    
    # Storage limits
    max_storage_usage_gb: float = Field(default=10.0, env="STORAGE_MAX_USAGE_GB")
    cleanup_threshold_percent: float = Field(default=80.0, env="STORAGE_CLEANUP_THRESHOLD")
    
    # Backup settings
    backup_enabled: bool = Field(default=False, env="STORAGE_BACKUP_ENABLED")
    backup_schedule: str = Field(default="daily", env="STORAGE_BACKUP_SCHEDULE")
    
    def create_directories(self):
        """Create necessary storage directories"""
        directories = [
            self.upload_directory,
            self.processed_directory,
            self.logs_directory,
            self.temp_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    
    # Logging levels and formats
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json, pretty, simple
    
    # Output destinations
    console_logging: bool = Field(default=True, env="LOG_CONSOLE")
    file_logging: bool = Field(default=True, env="LOG_FILE")
    log_file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    
    # Log rotation
    max_log_file_mb: int = Field(default=100, env="LOG_MAX_FILE_MB")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Structured logging
    enable_request_logging: bool = Field(default=True, env="LOG_REQUESTS")
    enable_performance_logging: bool = Field(default=True, env="LOG_PERFORMANCE")
    enable_audit_logging: bool = Field(default=True, env="LOG_AUDIT")
    
    # External logging services
    external_logging_enabled: bool = Field(default=False, env="LOG_EXTERNAL_ENABLED")
    external_logging_endpoint: Optional[str] = Field(default=None, env="LOG_EXTERNAL_ENDPOINT")
    external_logging_api_key: Optional[str] = Field(default=None, env="LOG_EXTERNAL_API_KEY")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @validator('log_format')
    def validate_log_format(cls, v):
        """Validate log format"""
        valid_formats = ['json', 'pretty', 'simple']
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid log format. Must be one of: {valid_formats}")
        return v.lower()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    # Authentication
    auth_enabled: bool = Field(default=False, env="SECURITY_AUTH_ENABLED")
    jwt_secret_key: Optional[str] = Field(default=None, env="SECURITY_JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="SECURITY_JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="SECURITY_JWT_EXPIRATION")
    
    # API Security
    api_key_required: bool = Field(default=False, env="SECURITY_API_KEY_REQUIRED")
    valid_api_keys: List[str] = Field(default=[], env="SECURITY_API_KEYS")
    
    # Data security
    encrypt_sensitive_data: bool = Field(default=True, env="SECURITY_ENCRYPT_DATA")
    data_encryption_key: Optional[str] = Field(default=None, env="SECURITY_ENCRYPTION_KEY")
    
    # Request validation
    validate_file_headers: bool = Field(default=True, env="SECURITY_VALIDATE_HEADERS")
    scan_for_malicious_content: bool = Field(default=True, env="SECURITY_SCAN_CONTENT")
    
    # Rate limiting and DDoS protection
    enable_ddos_protection: bool = Field(default=True, env="SECURITY_DDOS_PROTECTION")
    max_requests_per_ip_per_minute: int = Field(default=60, env="SECURITY_MAX_REQUESTS_PER_IP")
    
    # Audit and compliance
    audit_all_requests: bool = Field(default=False, env="SECURITY_AUDIT_ALL")
    compliance_mode: str = Field(default="standard", env="SECURITY_COMPLIANCE_MODE")  # standard, hipaa, gdpr
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration settings (for future use)"""
    
    # Connection settings
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_type: str = Field(default="sqlite", env="DATABASE_TYPE")  # sqlite, postgresql, mysql
    
    # Connection pool settings
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    
    # Query settings
    query_timeout_seconds: int = Field(default=30, env="DATABASE_QUERY_TIMEOUT")
    enable_query_logging: bool = Field(default=False, env="DATABASE_QUERY_LOGGING")
    
    # Backup settings
    auto_backup: bool = Field(default=True, env="DATABASE_AUTO_BACKUP")
    backup_retention_days: int = Field(default=7, env="DATABASE_BACKUP_RETENTION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""
    
    # Health checks
    health_check_enabled: bool = Field(default=True, env="MONITORING_HEALTH_CHECK")
    health_check_interval_seconds: int = Field(default=30, env="MONITORING_HEALTH_INTERVAL")
    
    # Metrics collection
    metrics_enabled: bool = Field(default=True, env="MONITORING_METRICS_ENABLED")
    metrics_endpoint: str = Field(default="/metrics", env="MONITORING_METRICS_ENDPOINT")
    
    # Performance monitoring
    track_request_duration: bool = Field(default=True, env="MONITORING_REQUEST_DURATION")
    track_memory_usage: bool = Field(default=True, env="MONITORING_MEMORY_USAGE")
    track_cpu_usage: bool = Field(default=True, env="MONITORING_CPU_USAGE")
    
    # Alerting
    alerting_enabled: bool = Field(default=False, env="MONITORING_ALERTING_ENABLED")
    alert_webhook_url: Optional[str] = Field(default=None, env="MONITORING_ALERT_WEBHOOK")
    
    # Error tracking
    error_tracking_enabled: bool = Field(default=True, env="MONITORING_ERROR_TRACKING")
    max_error_samples: int = Field(default=1000, env="MONITORING_MAX_ERROR_SAMPLES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class TracerSettings(BaseSettings):
    """Main application settings container"""
    
    # Environment
    environment: str = Field(default="development", env="TRACER_ENVIRONMENT")
    debug: bool = Field(default=False, env="TRACER_DEBUG")
    testing: bool = Field(default=False, env="TRACER_TESTING")
    
    # Feature flags
    enable_async_processing: bool = Field(default=True, env="TRACER_ASYNC_PROCESSING")
    enable_batch_processing: bool = Field(default=True, env="TRACER_BATCH_PROCESSING")
    enable_data_profiling: bool = Field(default=True, env="TRACER_DATA_PROFILING")
    
    # Service discovery (for future microservices)
    service_name: str = Field(default="tracer-ai", env="TRACER_SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="TRACER_SERVICE_VERSION")
    
    # Nested settings
    api: APISettings = Field(default_factory=APISettings)
    data_processing: DataProcessingSettings = Field(default_factory=DataProcessingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    def setup_directories(self):
        """Setup all required directories"""
        self.storage.create_directories()
        logger.info("Storage directories created")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary (excluding sensitive data)"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'service_name': self.service_name,
            'service_version': self.service_version,
            'api_host': self.api.host,
            'api_port': self.api.port,
            'max_file_size_mb': self.data_processing.max_file_size_mb,
            'max_batch_size': self.data_processing.max_batch_size,
            'log_level': self.logging.log_level,
            'log_format': self.logging.log_format,
            'auth_enabled': self.security.auth_enabled,
            'metrics_enabled': self.monitoring.metrics_enabled
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = TracerSettings()


def get_settings() -> TracerSettings:
    """Get global settings instance"""
    return settings


def load_settings(env_file: Optional[str] = None) -> TracerSettings:
    """Load settings from environment file"""
    if env_file:
        return TracerSettings(_env_file=env_file)
    return settings


def create_env_template(output_path: str = ".env.template"):
    """Create environment variable template file"""
    template_content = '''# Tracer AI Configuration Template

# Environment
TRACER_ENVIRONMENT=development
TRACER_DEBUG=false
TRACER_TESTING=false

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=false
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_RPM=100
API_MAX_REQUEST_SIZE_MB=100
API_REQUEST_TIMEOUT=300

# Data Processing
DATA_MAX_FILE_SIZE_MB=100
DATA_MAX_BATCH_SIZE=10000
DATA_ALLOWED_EXTENSIONS=.csv,.xlsx,.json
DATA_VALIDATION_STRICT=false
DATA_AUTO_CORRECT=true
DATA_MAX_WORKERS=4
DATA_MISSING_THRESHOLD=0.3
DATA_OUTLIER_THRESHOLD=3.5

# Storage
STORAGE_UPLOAD_DIR=/tmp/tracer_uploads
STORAGE_PROCESSED_DIR=/tmp/tracer_processed
STORAGE_LOGS_DIR=/tmp/tracer_logs
STORAGE_TEMP_RETENTION_HOURS=24
STORAGE_MAX_USAGE_GB=10.0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_CONSOLE=true
LOG_FILE=true
LOG_REQUESTS=true
LOG_PERFORMANCE=true

# Security
SECURITY_AUTH_ENABLED=false
SECURITY_API_KEY_REQUIRED=false
SECURITY_ENCRYPT_DATA=true
SECURITY_VALIDATE_HEADERS=true
SECURITY_DDOS_PROTECTION=true

# Monitoring
MONITORING_HEALTH_CHECK=true
MONITORING_METRICS_ENABLED=true
MONITORING_ERROR_TRACKING=true

# Database (Future use)
DATABASE_TYPE=sqlite
DATABASE_POOL_SIZE=10
'''
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    logger.info(f"Environment template created: {output_path}")


if __name__ == "__main__":
    # Create environment template if run directly
    create_env_template()
    print("Environment template created: .env.template")