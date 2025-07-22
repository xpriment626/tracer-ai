"""
Pipeline Orchestrator for Tracer Framework

Coordinates the entire data processing pipeline:
- Ingestion -> Validation -> Preprocessing 
- Error handling and recovery
- Progress tracking and monitoring
- Blueprint-specific workflow management
- Async processing with proper resource management
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import structlog
import pandas as pd
from pathlib import Path
import uuid

from .models import (
    DataPipelineState,
    ProcessingResult,
    ProcessingStatus,
    ValidationReport,
    DatasetMetadata,
    PipelineConfig,
    FeatureEngineering
)
from .ingestion import DataIngestionService, DataIngestionError
from .validation import DataValidationService
from .preprocessing import DataPreprocessingService, PreprocessingError
from .schemas import schema_registry

logger = structlog.get_logger(__name__)


class PipelineOrchestrationError(Exception):
    """Base exception for pipeline orchestration errors"""
    pass


class PipelineOrchestrator:
    """
    Main orchestrator for data processing pipelines
    
    Manages the complete workflow:
    1. Data ingestion (CSV -> DataFrame)
    2. Data validation (quality checks + schema validation)
    3. Blueprint compatibility assessment  
    4. Data preprocessing (cleaning + feature engineering)
    5. Final quality assurance
    
    Features:
    - Async processing with proper error handling
    - Progress tracking and state management
    - Blueprint-specific configurations
    - Resource cleanup and recovery
    - Detailed logging and monitoring
    """
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        max_concurrent_pipelines: int = 5
    ):
        # Initialize services
        self.ingestion_service = DataIngestionService()
        self.validation_service = DataValidationService()
        self.preprocessing_service = DataPreprocessingService()
        
        # Pipeline state management
        self.active_pipelines: Dict[str, DataPipelineState] = {}
        self.pipeline_history: List[str] = []
        
        # Configuration
        self.enable_monitoring = enable_monitoring
        self.max_concurrent_pipelines = max_concurrent_pipelines
        
        # Semaphore for controlling concurrent pipelines
        self.pipeline_semaphore = asyncio.Semaphore(max_concurrent_pipelines)
        
        logger.info(
            "PipelineOrchestrator initialized",
            max_concurrent_pipelines=max_concurrent_pipelines,
            monitoring_enabled=enable_monitoring
        )
    
    async def process_dataset(
        self,
        file_path: str,
        blueprint_name: str,
        config: Optional[PipelineConfig] = None,
        dataset_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, DataPipelineState]:
        """
        Main entry point for dataset processing
        
        Args:
            file_path: Path to the dataset file
            blueprint_name: Target blueprint name
            config: Optional pipeline configuration
            dataset_id: Optional custom dataset ID
            
        Returns:
            Tuple of (processed_dataframe, pipeline_state)
        """
        
        # Generate pipeline ID and dataset ID
        pipeline_id = str(uuid.uuid4())
        if dataset_id is None:
            dataset_id = f"{blueprint_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create default config if not provided
        if config is None:
            config = PipelineConfig(
                pipeline_id=pipeline_id,
                dataset_id=dataset_id,
                blueprint_name=blueprint_name
            )
        else:
            config.pipeline_id = pipeline_id
            config.dataset_id = dataset_id
            config.blueprint_name = blueprint_name
        
        # Initialize pipeline state
        pipeline_state = DataPipelineState(
            pipeline_id=pipeline_id,
            dataset_id=dataset_id,
            blueprint_name=blueprint_name,
            current_stage="initializing",
            status=ProcessingStatus.PENDING
        )
        
        # Register pipeline
        self.active_pipelines[pipeline_id] = pipeline_state
        
        async with self.pipeline_semaphore:
            try:
                logger.info(
                    "Starting dataset processing pipeline",
                    pipeline_id=pipeline_id,
                    dataset_id=dataset_id,
                    blueprint_name=blueprint_name,
                    file_path=file_path
                )
                
                # Execute pipeline stages
                processed_df = await self._execute_pipeline(
                    file_path, config, pipeline_state
                )
                
                # Mark pipeline as completed
                pipeline_state.update_stage("completed", ProcessingStatus.COMPLETED)
                
                logger.info(
                    "Dataset processing pipeline completed successfully",
                    pipeline_id=pipeline_id,
                    dataset_id=dataset_id,
                    final_shape=processed_df.shape,
                    total_duration_seconds=(
                        datetime.utcnow() - pipeline_state.created_at
                    ).total_seconds()
                )
                
                return processed_df, pipeline_state
                
            except Exception as e:
                # Mark pipeline as failed
                pipeline_state.update_stage("failed", ProcessingStatus.FAILED)
                pipeline_state.metadata['error'] = str(e)
                
                logger.error(
                    "Dataset processing pipeline failed",
                    pipeline_id=pipeline_id,
                    dataset_id=dataset_id,
                    error=str(e),
                    exc_info=True
                )
                
                raise PipelineOrchestrationError(
                    f"Pipeline {pipeline_id} failed: {str(e)}"
                ) from e
                
            finally:
                # Move to history and cleanup
                self.pipeline_history.append(pipeline_id)
                if len(self.pipeline_history) > 1000:  # Keep last 1000 pipelines
                    self.pipeline_history = self.pipeline_history[-1000:]
    
    async def _execute_pipeline(
        self,
        file_path: str,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> pd.DataFrame:
        """Execute the complete data processing pipeline"""
        
        # Stage 1: Data Ingestion
        state.update_stage("ingestion", ProcessingStatus.IN_PROGRESS)
        metadata, df = await self._execute_ingestion_stage(file_path, config, state)
        
        # Stage 2: Data Validation
        state.update_stage("validation", ProcessingStatus.IN_PROGRESS)  
        df, validation_report = await self._execute_validation_stage(df, config, state)
        
        # Stage 3: Blueprint Compatibility Check
        state.update_stage("compatibility_check", ProcessingStatus.IN_PROGRESS)
        compatibility_report = await self._execute_compatibility_stage(df, config, state)
        
        # Stage 4: Data Preprocessing
        state.update_stage("preprocessing", ProcessingStatus.IN_PROGRESS)
        df = await self._execute_preprocessing_stage(df, config, state)
        
        # Stage 5: Final Quality Assurance
        state.update_stage("quality_assurance", ProcessingStatus.IN_PROGRESS)
        await self._execute_final_qa_stage(df, config, state)
        
        return df
    
    async def _execute_ingestion_stage(
        self,
        file_path: str,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> Tuple[DatasetMetadata, pd.DataFrame]:
        """Execute data ingestion stage"""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Executing ingestion stage",
                pipeline_id=config.pipeline_id,
                file_path=file_path
            )
            
            # Ingest the file
            metadata, df = await self.ingestion_service.ingest_file(
                file_path, config.dataset_id
            )
            
            # Create processing result
            result = ProcessingResult(
                dataset_id=config.dataset_id,
                operation="ingestion",
                status=ProcessingStatus.COMPLETED,
                start_time=start_time,
                rows_processed=0,
                rows_output=len(df),
                metadata={
                    'file_size_mb': metadata.size_bytes / 1024 / 1024,
                    'encoding': metadata.encoding,
                    'columns': metadata.column_names
                }
            )
            result.mark_completed(len(df))
            
            state.add_result("ingestion", result)
            state.metadata['dataset_metadata'] = metadata.dict()
            
            logger.info(
                "Ingestion stage completed",
                pipeline_id=config.pipeline_id,
                rows_ingested=len(df),
                columns=len(df.columns),
                duration_seconds=result.duration_seconds
            )
            
            return metadata, df
            
        except DataIngestionError as e:
            logger.error(
                "Ingestion stage failed",
                pipeline_id=config.pipeline_id,
                error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in ingestion stage",
                pipeline_id=config.pipeline_id,
                error=str(e),
                exc_info=True
            )
            raise PipelineOrchestrationError(f"Ingestion failed: {str(e)}")
    
    async def _execute_validation_stage(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Execute data validation stage"""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Executing validation stage",
                pipeline_id=config.pipeline_id,
                input_shape=df.shape
            )
            
            # Validate dataset
            validation_report = await self.validation_service.validate_dataset(
                df=df,
                dataset_id=config.dataset_id,
                schema_name=config.blueprint_name,
                custom_rules=None
            )
            
            # Create processing result
            result = ProcessingResult(
                dataset_id=config.dataset_id,
                operation="validation",
                status=ProcessingStatus.COMPLETED,
                start_time=start_time,
                rows_processed=len(df),
                rows_output=len(df),
                metadata={
                    'validation_passed': validation_report.passed,
                    'total_issues': len(validation_report.issues),
                    'critical_issues': len(validation_report.critical_issues),
                    'high_issues': len(validation_report.high_issues),
                    'data_quality_score': validation_report.summary.get('data_quality_score', 0)
                }
            )
            result.mark_completed(len(df))
            
            state.add_result("validation", result)
            state.validation_report = validation_report
            
            # Handle validation failures
            if not validation_report.passed:
                critical_issues = validation_report.critical_issues
                if critical_issues:
                    error_msg = f"Critical data quality issues found: {len(critical_issues)} issues"
                    logger.error(
                        "Validation stage failed due to critical issues",
                        pipeline_id=config.pipeline_id,
                        critical_issues=[issue.message for issue in critical_issues]
                    )
                    raise PipelineOrchestrationError(error_msg)
                else:
                    logger.warning(
                        "Validation stage passed with warnings",
                        pipeline_id=config.pipeline_id,
                        high_issues=len(validation_report.high_issues),
                        total_issues=len(validation_report.issues)
                    )
            
            logger.info(
                "Validation stage completed",
                pipeline_id=config.pipeline_id,
                validation_passed=validation_report.passed,
                data_quality_score=validation_report.summary.get('data_quality_score', 0),
                duration_seconds=result.duration_seconds
            )
            
            return df, validation_report
            
        except Exception as e:
            logger.error(
                "Validation stage failed",
                pipeline_id=config.pipeline_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _execute_compatibility_stage(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> Dict[str, Any]:
        """Execute blueprint compatibility check"""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Executing compatibility check stage",
                pipeline_id=config.pipeline_id,
                blueprint_name=config.blueprint_name
            )
            
            # Get required and optional columns for the blueprint
            required_columns = schema_registry.get_required_columns(config.blueprint_name)
            optional_columns = schema_registry.get_optional_columns(config.blueprint_name)
            
            # Check compatibility
            compatibility_report = await self.validation_service.validate_blueprint_compatibility(
                df=df,
                blueprint_name=config.blueprint_name,
                required_columns=required_columns,
                optional_columns=optional_columns
            )
            
            # Create processing result
            result = ProcessingResult(
                dataset_id=config.dataset_id,
                operation="compatibility_check",
                status=ProcessingStatus.COMPLETED,
                start_time=start_time,
                rows_processed=len(df),
                rows_output=len(df),
                metadata={
                    'compatibility_score': compatibility_report['score'],
                    'compatible': compatibility_report['compatible'],
                    'required_columns_found': len(compatibility_report['column_analysis']['required_found']),
                    'required_columns_missing': len(compatibility_report['column_analysis']['required_missing']),
                    'optional_columns_found': len(compatibility_report['column_analysis']['optional_found'])
                }
            )
            result.mark_completed(len(df))
            
            state.add_result("compatibility_check", result)
            state.metadata['compatibility_report'] = compatibility_report
            
            # Handle incompatibility
            if not compatibility_report['compatible']:
                error_msg = f"Dataset not compatible with {config.blueprint_name} blueprint"
                logger.error(
                    "Compatibility check failed",
                    pipeline_id=config.pipeline_id,
                    compatibility_score=compatibility_report['score'],
                    missing_required=compatibility_report['column_analysis']['required_missing'],
                    recommendations=compatibility_report['recommendations']
                )
                raise PipelineOrchestrationError(error_msg)
            
            logger.info(
                "Compatibility check completed",
                pipeline_id=config.pipeline_id,
                compatibility_score=compatibility_report['score'],
                blueprint_compatible=compatibility_report['compatible'],
                duration_seconds=result.duration_seconds
            )
            
            return compatibility_report
            
        except Exception as e:
            logger.error(
                "Compatibility check stage failed",
                pipeline_id=config.pipeline_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _execute_preprocessing_stage(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> pd.DataFrame:
        """Execute data preprocessing stage"""
        
        try:
            logger.info(
                "Executing preprocessing stage",
                pipeline_id=config.pipeline_id,
                input_shape=df.shape,
                blueprint_name=config.blueprint_name
            )
            
            # Preprocess the dataset
            processed_df, result = await self.preprocessing_service.preprocess_dataset(
                df=df,
                dataset_id=config.dataset_id,
                blueprint_name=config.blueprint_name,
                config=config.preprocessing_config
            )
            
            state.add_result("preprocessing", result)
            
            # Extract feature engineering information
            if 'features_engineered' in result.metadata:
                feature_info = result.metadata['features_engineered']
                state.feature_engineering = FeatureEngineering(
                    dataset_id=config.dataset_id,
                    features_created=feature_info.get('features_created', []),
                    transformations_applied=feature_info.get('transformations', [])
                )
            
            logger.info(
                "Preprocessing stage completed",
                pipeline_id=config.pipeline_id,
                input_shape=df.shape,
                output_shape=processed_df.shape,
                features_created=len(state.feature_engineering.features_created) if state.feature_engineering else 0,
                duration_seconds=result.duration_seconds
            )
            
            return processed_df
            
        except PreprocessingError as e:
            logger.error(
                "Preprocessing stage failed",
                pipeline_id=config.pipeline_id,
                error=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in preprocessing stage",
                pipeline_id=config.pipeline_id,
                error=str(e),
                exc_info=True
            )
            raise PipelineOrchestrationError(f"Preprocessing failed: {str(e)}")
    
    async def _execute_final_qa_stage(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> None:
        """Execute final quality assurance checks"""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Executing final QA stage",
                pipeline_id=config.pipeline_id,
                final_shape=df.shape
            )
            
            # Final quality checks
            qa_issues = []
            
            # Check for empty dataset
            if len(df) == 0:
                qa_issues.append("Dataset is empty after processing")
            
            # Check for missing columns
            if len(df.columns) == 0:
                qa_issues.append("Dataset has no columns after processing")
            
            # Check for all-NaN columns
            nan_columns = df.columns[df.isna().all()].tolist()
            if nan_columns:
                qa_issues.append(f"Columns with all NaN values: {nan_columns}")
            
            # Check data types
            invalid_dtypes = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Object columns should be categorical or properly encoded
                    if df[col].nunique() > len(df) * 0.5:  # High cardinality
                        invalid_dtypes.append(col)
            
            if invalid_dtypes:
                qa_issues.append(f"High cardinality object columns: {invalid_dtypes}")
            
            # Create processing result
            result = ProcessingResult(
                dataset_id=config.dataset_id,
                operation="final_qa",
                status=ProcessingStatus.COMPLETED,
                start_time=start_time,
                rows_processed=len(df),
                rows_output=len(df),
                metadata={
                    'qa_issues': qa_issues,
                    'qa_passed': len(qa_issues) == 0,
                    'final_memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'final_dtypes': df.dtypes.astype(str).to_dict()
                }
            )
            result.mark_completed(len(df))
            
            state.add_result("final_qa", result)
            
            # Handle QA failures
            if qa_issues:
                logger.warning(
                    "Final QA stage found issues",
                    pipeline_id=config.pipeline_id,
                    qa_issues=qa_issues
                )
                # For now, we'll warn but not fail - in production, this might be configurable
            
            logger.info(
                "Final QA stage completed",
                pipeline_id=config.pipeline_id,
                qa_passed=len(qa_issues) == 0,
                issues_found=len(qa_issues),
                duration_seconds=result.duration_seconds
            )
            
        except Exception as e:
            logger.error(
                "Final QA stage failed",
                pipeline_id=config.pipeline_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[DataPipelineState]:
        """Get the current status of a pipeline"""
        return self.active_pipelines.get(pipeline_id)
    
    def get_active_pipelines(self) -> Dict[str, DataPipelineState]:
        """Get all currently active pipelines"""
        return self.active_pipelines.copy()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary statistics about pipeline processing"""
        active_count = len(self.active_pipelines)
        completed_count = sum(
            1 for p in self.active_pipelines.values() 
            if p.status == ProcessingStatus.COMPLETED
        )
        failed_count = sum(
            1 for p in self.active_pipelines.values() 
            if p.status == ProcessingStatus.FAILED
        )
        in_progress_count = sum(
            1 for p in self.active_pipelines.values() 
            if p.status == ProcessingStatus.IN_PROGRESS
        )
        
        return {
            'active_pipelines': active_count,
            'completed_pipelines': completed_count,
            'failed_pipelines': failed_count,
            'in_progress_pipelines': in_progress_count,
            'total_processed': len(self.pipeline_history),
            'success_rate': completed_count / active_count if active_count > 0 else 0
        }
    
    async def cleanup_completed_pipelines(self, older_than_hours: int = 24) -> int:
        """Clean up completed pipelines older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        pipelines_to_remove = []
        for pipeline_id, state in self.active_pipelines.items():
            if (state.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED] and
                state.updated_at < cutoff_time):
                pipelines_to_remove.append(pipeline_id)
        
        for pipeline_id in pipelines_to_remove:
            del self.active_pipelines[pipeline_id]
        
        if pipelines_to_remove:
            logger.info(
                "Cleaned up completed pipelines",
                removed_count=len(pipelines_to_remove),
                cutoff_hours=older_than_hours
            )
        
        return len(pipelines_to_remove)
    
    async def process_file_stream(
        self,
        file_stream,
        filename: str,
        blueprint_name: str,
        content_length: Optional[int] = None,
        config: Optional[PipelineConfig] = None
    ) -> Tuple[pd.DataFrame, DataPipelineState]:
        """
        Process a file from a stream (useful for web uploads)
        
        Args:
            file_stream: File stream object
            filename: Original filename
            blueprint_name: Target blueprint name
            content_length: Optional content length
            config: Optional pipeline configuration
            
        Returns:
            Tuple of (processed_dataframe, pipeline_state)
        """
        
        # Generate pipeline and dataset IDs
        pipeline_id = str(uuid.uuid4())
        dataset_id = f"{blueprint_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create config if not provided
        if config is None:
            config = PipelineConfig(
                pipeline_id=pipeline_id,
                dataset_id=dataset_id,
                blueprint_name=blueprint_name
            )
        
        # Initialize pipeline state
        pipeline_state = DataPipelineState(
            pipeline_id=pipeline_id,
            dataset_id=dataset_id,
            blueprint_name=blueprint_name,
            current_stage="initializing",
            status=ProcessingStatus.PENDING
        )
        
        self.active_pipelines[pipeline_id] = pipeline_state
        
        async with self.pipeline_semaphore:
            try:
                logger.info(
                    "Starting file stream processing pipeline",
                    pipeline_id=pipeline_id,
                    filename=filename,
                    blueprint_name=blueprint_name
                )
                
                # Stage 1: Stream Ingestion
                pipeline_state.update_stage("stream_ingestion", ProcessingStatus.IN_PROGRESS)
                metadata, df = await self.ingestion_service.ingest_stream(
                    file_stream=file_stream,
                    filename=filename,
                    content_length=content_length,
                    dataset_id=dataset_id
                )
                
                # Continue with normal pipeline execution
                pipeline_state.metadata['dataset_metadata'] = metadata.dict()
                
                # Execute remaining stages
                processed_df = await self._execute_pipeline_from_dataframe(
                    df, config, pipeline_state
                )
                
                pipeline_state.update_stage("completed", ProcessingStatus.COMPLETED)
                
                logger.info(
                    "File stream processing pipeline completed",
                    pipeline_id=pipeline_id,
                    final_shape=processed_df.shape
                )
                
                return processed_df, pipeline_state
                
            except Exception as e:
                pipeline_state.update_stage("failed", ProcessingStatus.FAILED)
                pipeline_state.metadata['error'] = str(e)
                
                logger.error(
                    "File stream processing pipeline failed",
                    pipeline_id=pipeline_id,
                    error=str(e),
                    exc_info=True
                )
                
                raise PipelineOrchestrationError(
                    f"Stream pipeline {pipeline_id} failed: {str(e)}"
                ) from e
    
    async def _execute_pipeline_from_dataframe(
        self,
        df: pd.DataFrame,
        config: PipelineConfig,
        state: DataPipelineState
    ) -> pd.DataFrame:
        """Execute pipeline stages starting from an already loaded DataFrame"""
        
        # Stage 2: Data Validation
        state.update_stage("validation", ProcessingStatus.IN_PROGRESS)  
        df, validation_report = await self._execute_validation_stage(df, config, state)
        
        # Stage 3: Blueprint Compatibility Check
        state.update_stage("compatibility_check", ProcessingStatus.IN_PROGRESS)
        compatibility_report = await self._execute_compatibility_stage(df, config, state)
        
        # Stage 4: Data Preprocessing
        state.update_stage("preprocessing", ProcessingStatus.IN_PROGRESS)
        df = await self._execute_preprocessing_stage(df, config, state)
        
        # Stage 5: Final Quality Assurance
        state.update_stage("quality_assurance", ProcessingStatus.IN_PROGRESS)
        await self._execute_final_qa_stage(df, config, state)
        
        return df