"""
Main Data Pipeline Interface for Tracer Framework

High-level interface for data processing pipelines providing:
- Simple API for processing datasets
- Blueprint-specific configurations
- Progress monitoring and status updates
- Error handling and recovery
- Integration with all core services
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from datetime import datetime
from pathlib import Path
import structlog
import pandas as pd

from .models import (
    DataPipelineState,
    PipelineConfig, 
    DatasetMetadata,
    ProcessingStatus
)
from .orchestrator import PipelineOrchestrator, PipelineOrchestrationError
from .schemas import schema_registry

logger = structlog.get_logger(__name__)


class DataPipeline:
    """
    High-level interface for Tracer data processing pipelines
    
    This is the main entry point for data processing in the Tracer framework.
    It provides a simple, intuitive API while leveraging all the underlying
    services for ingestion, validation, and preprocessing.
    
    Example Usage:
    ```python
    pipeline = DataPipeline()
    
    # Process a dataset file
    processed_data, state = await pipeline.process_file(
        file_path="customer_data.csv",
        blueprint_name="customer_churn"
    )
    
    # Check processing status
    status = pipeline.get_status(state.pipeline_id)
    
    # Process multiple files
    results = await pipeline.process_batch([
        ("file1.csv", "customer_churn"),
        ("file2.csv", "revenue_projection")
    ])
    ```
    """
    
    def __init__(
        self,
        max_concurrent_pipelines: int = 3,
        enable_monitoring: bool = True
    ):
        self.orchestrator = PipelineOrchestrator(
            max_concurrent_pipelines=max_concurrent_pipelines,
            enable_monitoring=enable_monitoring
        )
        
        logger.info(
            "DataPipeline initialized",
            max_concurrent_pipelines=max_concurrent_pipelines
        )
    
    async def process_file(
        self,
        file_path: Union[str, Path],
        blueprint_name: str,
        dataset_id: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, DataPipelineState]:
        """
        Process a single dataset file
        
        Args:
            file_path: Path to the dataset file
            blueprint_name: Target blueprint (e.g., 'customer_churn')
            dataset_id: Optional custom dataset ID
            preprocessing_config: Optional preprocessing configuration
            validation_config: Optional validation configuration
            
        Returns:
            Tuple of (processed_dataframe, pipeline_state)
            
        Raises:
            PipelineOrchestrationError: If processing fails
        """
        
        # Validate inputs
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if blueprint_name not in schema_registry.marshmallow_schemas:
            available = list(schema_registry.marshmallow_schemas.keys())
            raise ValueError(f"Unknown blueprint '{blueprint_name}'. Available: {available}")
        
        # Create pipeline configuration
        config = PipelineConfig(
            pipeline_id="",  # Will be set by orchestrator
            dataset_id=dataset_id or "",  # Will be set by orchestrator
            blueprint_name=blueprint_name,
            preprocessing_config=preprocessing_config or {},
            validation_config=validation_config or {}
        )
        
        try:
            logger.info(
                "Starting file processing",
                file_path=str(file_path),
                blueprint_name=blueprint_name,
                dataset_id=dataset_id
            )
            
            # Process through orchestrator
            processed_df, pipeline_state = await self.orchestrator.process_dataset(
                file_path=str(file_path),
                blueprint_name=blueprint_name,
                config=config,
                dataset_id=dataset_id
            )
            
            logger.info(
                "File processing completed successfully",
                pipeline_id=pipeline_state.pipeline_id,
                dataset_id=pipeline_state.dataset_id,
                final_shape=processed_df.shape
            )
            
            return processed_df, pipeline_state
            
        except Exception as e:
            logger.error(
                "File processing failed",
                file_path=str(file_path),
                blueprint_name=blueprint_name,
                error=str(e)
            )
            raise
    
    async def process_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        blueprint_name: str,
        content_length: Optional[int] = None,
        dataset_id: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, DataPipelineState]:
        """
        Process a file from a stream (useful for web uploads)
        
        Args:
            file_stream: Binary file stream
            filename: Original filename
            blueprint_name: Target blueprint name
            content_length: Optional content length for progress tracking
            dataset_id: Optional custom dataset ID
            preprocessing_config: Optional preprocessing configuration
            validation_config: Optional validation configuration
            
        Returns:
            Tuple of (processed_dataframe, pipeline_state)
        """
        
        # Validate blueprint
        if blueprint_name not in schema_registry.marshmallow_schemas:
            available = list(schema_registry.marshmallow_schemas.keys())
            raise ValueError(f"Unknown blueprint '{blueprint_name}'. Available: {available}")
        
        # Create pipeline configuration
        config = PipelineConfig(
            pipeline_id="",  # Will be set by orchestrator
            dataset_id=dataset_id or "",  # Will be set by orchestrator
            blueprint_name=blueprint_name,
            preprocessing_config=preprocessing_config or {},
            validation_config=validation_config or {}
        )
        
        try:
            logger.info(
                "Starting stream processing",
                filename=filename,
                blueprint_name=blueprint_name,
                content_length=content_length
            )
            
            # Process through orchestrator
            processed_df, pipeline_state = await self.orchestrator.process_file_stream(
                file_stream=file_stream,
                filename=filename,
                blueprint_name=blueprint_name,
                content_length=content_length,
                config=config
            )
            
            logger.info(
                "Stream processing completed successfully",
                pipeline_id=pipeline_state.pipeline_id,
                filename=filename,
                final_shape=processed_df.shape
            )
            
            return processed_df, pipeline_state
            
        except Exception as e:
            logger.error(
                "Stream processing failed",
                filename=filename,
                blueprint_name=blueprint_name,
                error=str(e)
            )
            raise
    
    async def process_batch(
        self,
        file_configs: List[Tuple[Union[str, Path], str]],
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, DataPipelineState]]:
        """
        Process multiple files concurrently
        
        Args:
            file_configs: List of (file_path, blueprint_name) tuples
            max_concurrent: Optional override for max concurrent processing
            
        Returns:
            List of (processed_dataframe, pipeline_state) tuples
        """
        
        if not file_configs:
            return []
        
        # Limit concurrency
        if max_concurrent is None:
            max_concurrent = min(len(file_configs), 3)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(file_path: Union[str, Path], blueprint_name: str):
            async with semaphore:
                return await self.process_file(file_path, blueprint_name)
        
        logger.info(
            "Starting batch processing",
            total_files=len(file_configs),
            max_concurrent=max_concurrent
        )
        
        # Process all files concurrently
        tasks = [
            process_single(file_path, blueprint_name)
            for file_path, blueprint_name in file_configs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append((file_configs[i], result))
                logger.error(
                    "Batch processing item failed",
                    file_path=str(file_configs[i][0]),
                    blueprint_name=file_configs[i][1],
                    error=str(result)
                )
            else:
                successful_results.append(result)
        
        logger.info(
            "Batch processing completed",
            total_files=len(file_configs),
            successful=len(successful_results),
            failed=len(failed_results)
        )
        
        if failed_results:
            logger.warning(
                "Some batch processing items failed",
                failed_count=len(failed_results),
                failed_items=[(str(cfg[0]), str(err)) for cfg, err in failed_results[:5]]
            )
        
        return successful_results
    
    def get_status(self, pipeline_id: str) -> Optional[DataPipelineState]:
        """
        Get the current status of a pipeline
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Pipeline state or None if not found
        """
        return self.orchestrator.get_pipeline_status(pipeline_id)
    
    def get_all_pipelines(self) -> Dict[str, DataPipelineState]:
        """Get all currently tracked pipelines"""
        return self.orchestrator.get_active_pipelines()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about pipeline processing"""
        return self.orchestrator.get_pipeline_summary()
    
    def get_available_blueprints(self) -> List[str]:
        """Get list of available blueprint names"""
        return list(schema_registry.marshmallow_schemas.keys())
    
    def get_blueprint_info(self, blueprint_name: str) -> Dict[str, Any]:
        """
        Get information about a specific blueprint
        
        Args:
            blueprint_name: Blueprint name
            
        Returns:
            Dictionary with blueprint information
        """
        if blueprint_name not in schema_registry.marshmallow_schemas:
            raise ValueError(f"Unknown blueprint: {blueprint_name}")
        
        required_columns = schema_registry.get_required_columns(blueprint_name)
        optional_columns = schema_registry.get_optional_columns(blueprint_name)
        column_specs = schema_registry.get_column_specs(blueprint_name)
        
        return {
            'blueprint_name': blueprint_name,
            'required_columns': required_columns,
            'optional_columns': optional_columns,
            'total_columns': len(required_columns) + len(optional_columns),
            'column_specifications': {
                name: {
                    'data_type': spec.data_type.value,
                    'required': spec.required,
                    'nullable': spec.nullable,
                    'description': spec.description,
                    'constraints': spec.constraints
                }
                for name, spec in column_specs.items()
            } if column_specs else {}
        }
    
    async def validate_file_compatibility(
        self,
        file_path: Union[str, Path],
        blueprint_name: str
    ) -> Dict[str, Any]:
        """
        Check if a file is compatible with a blueprint without full processing
        
        Args:
            file_path: Path to the dataset file
            blueprint_name: Blueprint name to check against
            
        Returns:
            Compatibility report dictionary
        """
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if blueprint_name not in schema_registry.marshmallow_schemas:
            available = list(schema_registry.marshmallow_schemas.keys())
            raise ValueError(f"Unknown blueprint '{blueprint_name}'. Available: {available}")
        
        try:
            # Quick ingestion to get column information
            from .ingestion import DataIngestionService
            ingestion_service = DataIngestionService()
            metadata, df = await ingestion_service.ingest_file(str(file_path))
            
            # Check compatibility using validation service
            from .validation import DataValidationService
            validation_service = DataValidationService()
            
            required_columns = schema_registry.get_required_columns(blueprint_name)
            optional_columns = schema_registry.get_optional_columns(blueprint_name)
            
            compatibility_report = await validation_service.validate_blueprint_compatibility(
                df=df,
                blueprint_name=blueprint_name,
                required_columns=required_columns,
                optional_columns=optional_columns
            )
            
            # Add file metadata to the report
            compatibility_report['file_metadata'] = {
                'filename': file_path.name,
                'size_mb': metadata.size_bytes / 1024 / 1024,
                'rows': metadata.rows,
                'columns': metadata.columns,
                'column_names': metadata.column_names
            }
            
            return compatibility_report
            
        except Exception as e:
            logger.error(
                "File compatibility check failed",
                file_path=str(file_path),
                blueprint_name=blueprint_name,
                error=str(e)
            )
            return {
                'compatible': False,
                'score': 0.0,
                'error': str(e),
                'blueprint_name': blueprint_name
            }
    
    async def cleanup_old_pipelines(self, older_than_hours: int = 24) -> int:
        """
        Clean up completed pipelines older than specified hours
        
        Args:
            older_than_hours: Remove pipelines older than this many hours
            
        Returns:
            Number of pipelines removed
        """
        return await self.orchestrator.cleanup_completed_pipelines(older_than_hours)
    
    def get_processing_stages(self) -> List[str]:
        """Get list of processing stages in order"""
        return [
            "initializing",
            "ingestion", 
            "validation",
            "compatibility_check",
            "preprocessing",
            "quality_assurance",
            "completed"
        ]
    
    def estimate_processing_time(
        self,
        file_size_mb: float,
        blueprint_name: str
    ) -> Dict[str, float]:
        """
        Estimate processing time based on file size and blueprint
        
        Args:
            file_size_mb: File size in megabytes
            blueprint_name: Blueprint name
            
        Returns:
            Dictionary with time estimates for each stage
        """
        
        # Base processing rates (MB/second) - these would be tuned based on actual performance
        base_rates = {
            'ingestion': 50.0,      # 50 MB/s
            'validation': 30.0,     # 30 MB/s
            'preprocessing': 20.0,  # 20 MB/s (varies by blueprint complexity)
            'quality_assurance': 100.0  # 100 MB/s
        }
        
        # Blueprint complexity multipliers
        complexity_multipliers = {
            'customer_churn': 1.0,
            'revenue_projection': 0.8,  # Simpler processing
            'price_optimization': 1.2   # More complex feature engineering
        }
        
        multiplier = complexity_multipliers.get(blueprint_name, 1.0)
        
        estimates = {}
        for stage, rate in base_rates.items():
            adjusted_rate = rate * multiplier
            estimates[stage] = file_size_mb / adjusted_rate
        
        # Add overhead for compatibility check (constant time)
        estimates['compatibility_check'] = 0.5  # 0.5 seconds
        
        # Total estimate
        estimates['total'] = sum(estimates.values())
        
        return estimates