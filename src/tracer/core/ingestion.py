"""
Data Ingestion Service for Tracer Framework

Handles secure, scalable ingestion of CSV files with:
- Async file processing
- Content validation and sanitization
- Metadata extraction
- Error handling and recovery
- Memory-efficient streaming for large files
"""

import asyncio
import hashlib
import io
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, BinaryIO, Any
import aiofiles
import pandas as pd
from datetime import datetime
import logging
import structlog
import chardet
import tempfile
import os
from contextlib import asynccontextmanager

from .models import (
    DatasetMetadata, 
    ProcessingResult, 
    ProcessingStatus,
    DataQualityLevel
)

logger = structlog.get_logger(__name__)


class DataIngestionError(Exception):
    """Base exception for data ingestion errors"""
    pass


class UnsupportedFileFormatError(DataIngestionError):
    """Raised when file format is not supported"""
    pass


class FileSizeLimitExceededError(DataIngestionError):
    """Raised when file exceeds size limit"""
    pass


class FileCorruptionError(DataIngestionError):
    """Raised when file is corrupted or unreadable"""
    pass


class DataIngestionService:
    """
    High-performance async data ingestion service optimized for ML workflows
    
    Features:
    - Streaming file processing for memory efficiency
    - Automatic encoding detection
    - Content validation and sanitization
    - Duplicate detection
    - Schema inference
    - Async processing with proper error handling
    """
    
    def __init__(
        self,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
        chunk_size: int = 8192,
        temp_dir: Optional[str] = None,
        enable_compression: bool = True
    ):
        self.max_file_size = max_file_size
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.enable_compression = enable_compression
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(exist_ok=True)
        
        # Supported MIME types
        self.supported_formats = {
            'text/csv': self._process_csv,
            'application/csv': self._process_csv,
            'text/plain': self._process_csv  # CSV sometimes detected as plain text
        }
        
        logger.info(
            "DataIngestionService initialized",
            max_file_size=self.max_file_size,
            chunk_size=self.chunk_size,
            temp_dir=self.temp_dir
        )

    async def ingest_file(
        self, 
        file_path: str, 
        dataset_id: Optional[str] = None
    ) -> Tuple[DatasetMetadata, pd.DataFrame]:
        """
        Main entry point for file ingestion
        
        Args:
            file_path: Path to the file to ingest
            dataset_id: Optional custom dataset ID
            
        Returns:
            Tuple of (metadata, dataframe)
            
        Raises:
            DataIngestionError: For various ingestion failures
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate file exists and is readable
            path_obj = Path(file_path)
            if not path_obj.exists():
                raise DataIngestionError(f"File not found: {file_path}")
            
            if not path_obj.is_file():
                raise DataIngestionError(f"Path is not a file: {file_path}")
                
            # Check file size
            file_size = path_obj.stat().st_size
            if file_size > self.max_file_size:
                raise FileSizeLimitExceededError(
                    f"File size {file_size} exceeds limit {self.max_file_size}"
                )
            
            logger.info(
                "Starting file ingestion",
                file_path=file_path,
                file_size=file_size,
                dataset_id=dataset_id
            )
            
            # Generate content hash and detect encoding
            content_hash, encoding = await self._analyze_file(file_path)
            
            # Detect MIME type and process accordingly
            mime_type = self._detect_mime_type(file_path)
            if mime_type not in self.supported_formats:
                raise UnsupportedFileFormatError(f"Unsupported format: {mime_type}")
            
            # Process the file using appropriate handler
            processor = self.supported_formats[mime_type]
            df, column_info = await processor(file_path, encoding)
            
            # Generate dataset ID if not provided
            if dataset_id is None:
                dataset_id = self._generate_dataset_id(path_obj.name, content_hash)
            
            # Create metadata
            metadata = DatasetMetadata(
                id=dataset_id,
                filename=path_obj.name,
                size_bytes=file_size,
                rows=len(df),
                columns=len(df.columns),
                column_names=list(df.columns),
                content_hash=content_hash,
                mime_type=mime_type,
                encoding=encoding
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "File ingestion completed successfully",
                dataset_id=dataset_id,
                rows=len(df),
                columns=len(df.columns),
                duration_seconds=duration
            )
            
            return metadata, df
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "File ingestion failed",
                file_path=file_path,
                error=str(e),
                duration_seconds=duration,
                exc_info=True
            )
            
            if isinstance(e, DataIngestionError):
                raise
            else:
                raise DataIngestionError(f"Ingestion failed: {str(e)}") from e

    async def ingest_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        content_length: Optional[int] = None,
        dataset_id: Optional[str] = None
    ) -> Tuple[DatasetMetadata, pd.DataFrame]:
        """
        Ingest data from a file stream (useful for web uploads)
        
        Args:
            file_stream: Binary file stream
            filename: Original filename
            content_length: Optional content length for progress tracking
            dataset_id: Optional custom dataset ID
            
        Returns:
            Tuple of (metadata, dataframe)
        """
        start_time = datetime.utcnow()
        temp_file_path = None
        
        try:
            # Create temporary file
            temp_file_path = Path(self.temp_dir) / f"upload_{datetime.utcnow().timestamp()}_{filename}"
            
            logger.info(
                "Starting stream ingestion",
                filename=filename,
                content_length=content_length,
                temp_file=str(temp_file_path)
            )
            
            # Stream to temporary file with size limit checking
            bytes_written = 0
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                while True:
                    chunk = file_stream.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    bytes_written += len(chunk)
                    if bytes_written > self.max_file_size:
                        raise FileSizeLimitExceededError(
                            f"Stream size exceeds limit {self.max_file_size}"
                        )
                    
                    await temp_file.write(chunk)
            
            # Process the temporary file
            result = await self.ingest_file(str(temp_file_path), dataset_id)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Stream ingestion completed",
                filename=filename,
                bytes_processed=bytes_written,
                duration_seconds=duration
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Stream ingestion failed",
                filename=filename,
                error=str(e),
                exc_info=True
            )
            raise
            
        finally:
            # Clean up temporary file
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to clean up temporary file",
                        temp_file=str(temp_file_path),
                        error=str(cleanup_error)
                    )

    async def _analyze_file(self, file_path: str) -> Tuple[str, str]:
        """
        Analyze file to generate hash and detect encoding
        
        Returns:
            Tuple of (content_hash, encoding)
        """
        hasher = hashlib.sha256()
        encoding_detector = chardet.UniversalDetector()
        
        async with aiofiles.open(file_path, 'rb') as file:
            while True:
                chunk = await file.read(self.chunk_size)
                if not chunk:
                    break
                
                hasher.update(chunk)
                encoding_detector.feed(chunk)
                
                if encoding_detector.done:
                    break
        
        encoding_detector.close()
        
        content_hash = hasher.hexdigest()
        encoding_result = encoding_detector.result
        encoding = encoding_result.get('encoding', 'utf-8') if encoding_result else 'utf-8'
        
        # Fallback to utf-8 for common cases
        if not encoding or encoding.lower() in ['ascii']:
            encoding = 'utf-8'
        
        logger.debug(
            "File analysis completed",
            file_path=file_path,
            content_hash=content_hash[:16] + "...",
            encoding=encoding,
            encoding_confidence=encoding_result.get('confidence', 0) if encoding_result else 0
        )
        
        return content_hash, encoding

    async def _process_csv(
        self, 
        file_path: str, 
        encoding: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process CSV file with intelligent parsing
        
        Args:
            file_path: Path to CSV file
            encoding: Detected file encoding
            
        Returns:
            Tuple of (dataframe, column_info)
        """
        try:
            # First, detect CSV dialect and structure
            dialect_info = await self._detect_csv_dialect(file_path, encoding)
            
            # Read CSV with pandas using detected parameters
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=dialect_info['delimiter'],
                quotechar=dialect_info['quotechar'],
                escapechar=dialect_info['escapechar'],
                skipinitialspace=True,
                na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a', '#N/A'],
                keep_default_na=True,
                low_memory=False  # Ensure consistent data types
            )
            
            # Basic data cleaning
            df = await self._clean_dataframe(df)
            
            # Extract column information
            column_info = await self._analyze_columns(df)
            
            logger.info(
                "CSV processing completed",
                file_path=file_path,
                rows=len(df),
                columns=len(df.columns),
                delimiter=dialect_info['delimiter'],
                encoding=encoding
            )
            
            return df, column_info
            
        except pd.errors.EmptyDataError:
            raise FileCorruptionError("CSV file is empty or corrupted")
        except pd.errors.ParserError as e:
            raise FileCorruptionError(f"Failed to parse CSV file: {str(e)}")
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                try:
                    return await self._process_csv(file_path, alt_encoding)
                except:
                    continue
            raise FileCorruptionError("Unable to decode file with any supported encoding")

    async def _detect_csv_dialect(self, file_path: str, encoding: str) -> Dict[str, str]:
        """
        Detect CSV dialect (delimiter, quote char, etc.)
        
        Returns:
            Dictionary with dialect parameters
        """
        sample_size = min(8192, os.path.getsize(file_path))
        
        async with aiofiles.open(file_path, 'r', encoding=encoding) as file:
            sample = await file.read(sample_size)
        
        try:
            # Use Python's CSV sniffer
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=',;\t|')
            
            return {
                'delimiter': dialect.delimiter,
                'quotechar': dialect.quotechar,
                'escapechar': getattr(dialect, 'escapechar', None),
                'skipinitialspace': dialect.skipinitialspace,
                'lineterminator': dialect.lineterminator
            }
        except csv.Error:
            # Fallback to default CSV parameters
            logger.warning(
                "Could not detect CSV dialect, using defaults",
                file_path=file_path
            )
            return {
                'delimiter': ',',
                'quotechar': '"',
                'escapechar': None,
                'skipinitialspace': True,
                'lineterminator': '\r\n'
            }

    async def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic data cleaning operations
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all')  # Remove empty rows
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]  # Remove unnamed columns
        
        # Clean column names
        df.columns = df.columns.str.strip()  # Remove whitespace
        df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars
        df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscore
        df.columns = df.columns.str.lower()  # Convert to lowercase
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.debug(
            "DataFrame cleaned",
            original_shape=df.shape,
            final_shape=df.shape,
            columns=list(df.columns)[:10]  # Log first 10 columns
        )
        
        return df

    async def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze column characteristics for metadata
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column analysis results
        """
        column_info = {}
        
        for column in df.columns:
            series = df[column]
            
            column_info[column] = {
                'dtype': str(series.dtype),
                'non_null_count': int(series.notna().sum()),
                'null_count': int(series.isna().sum()),
                'unique_count': int(series.nunique()),
                'memory_usage': int(series.memory_usage(deep=True)),
                'is_numeric': pd.api.types.is_numeric_dtype(series),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(series),
                'is_categorical': pd.api.types.is_categorical_dtype(series)
            }
            
            # Add statistics for numeric columns
            if column_info[column]['is_numeric']:
                column_info[column].update({
                    'min': float(series.min()) if pd.notna(series.min()) else None,
                    'max': float(series.max()) if pd.notna(series.max()) else None,
                    'mean': float(series.mean()) if pd.notna(series.mean()) else None,
                    'std': float(series.std()) if pd.notna(series.std()) else None
                })
                
            # Add sample values for inspection
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                sample_size = min(5, len(non_null_values))
                column_info[column]['sample_values'] = non_null_values.head(sample_size).tolist()
        
        return column_info

    def _detect_mime_type(self, file_path: str) -> str:
        """
        Detect MIME type based on file extension and content
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        
        # Simple extension-based detection
        extension_mapping = {
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values',
            '.txt': 'text/plain'
        }
        
        return extension_mapping.get(extension, 'text/csv')  # Default to CSV

    def _generate_dataset_id(self, filename: str, content_hash: str) -> str:
        """
        Generate a unique dataset ID
        
        Args:
            filename: Original filename
            content_hash: SHA-256 hash of file content
            
        Returns:
            Unique dataset ID
        """
        # Use first 8 characters of hash plus timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        hash_prefix = content_hash[:8]
        clean_filename = Path(filename).stem.lower().replace(' ', '_')[:20]
        
        return f"{clean_filename}_{timestamp}_{hash_prefix}"

    async def validate_dataset_compatibility(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset compatibility with blueprint requirements
        
        Args:
            df: Input dataframe
            required_columns: List of required column names
            optional_columns: List of optional column names
            
        Returns:
            Compatibility report dictionary
        """
        optional_columns = optional_columns or []
        
        # Check for required columns
        missing_required = set(required_columns) - set(df.columns)
        available_optional = set(optional_columns) & set(df.columns)
        
        # Calculate compatibility score
        required_score = 1.0 if not missing_required else 0.0
        optional_score = len(available_optional) / len(optional_columns) if optional_columns else 1.0
        overall_score = (required_score * 0.8) + (optional_score * 0.2)
        
        compatibility_report = {
            'compatible': len(missing_required) == 0,
            'score': overall_score,
            'required_columns_status': {
                'found': list(set(required_columns) & set(df.columns)),
                'missing': list(missing_required)
            },
            'optional_columns_status': {
                'found': list(available_optional),
                'missing': list(set(optional_columns) - available_optional)
            },
            'recommendations': []
        }
        
        # Add recommendations
        if missing_required:
            compatibility_report['recommendations'].append(
                f"Add missing required columns: {', '.join(missing_required)}"
            )
        
        if len(available_optional) < len(optional_columns):
            missing_optional = set(optional_columns) - available_optional
            compatibility_report['recommendations'].append(
                f"Consider adding optional columns for better accuracy: {', '.join(missing_optional)}"
            )
        
        return compatibility_report