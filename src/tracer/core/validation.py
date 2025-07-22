"""
Data Validation Service for Tracer Framework

Comprehensive data validation and quality assurance system with:
- Schema validation using marshmallow and jsonschema
- Data quality checks (completeness, uniqueness, validity)  
- Business rule validation
- Statistical anomaly detection
- Performance monitoring and reporting
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from marshmallow import Schema, fields, ValidationError, validate, post_load
import jsonschema
import structlog
from abc import ABC, abstractmethod

from .models import (
    DataQualityIssue,
    DataQualityLevel, 
    ValidationReport,
    ProcessingResult,
    ProcessingStatus,
    ColumnSpec,
    DataType
)

logger = structlog.get_logger(__name__)


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, description: str, severity: DataQualityLevel):
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        """Execute the validation rule"""
        pass


class CompletenessRule(ValidationRule):
    """Validate data completeness (non-null values)"""
    
    def __init__(self, threshold: float = 0.95, severity: DataQualityLevel = DataQualityLevel.HIGH):
        super().__init__(
            name="completeness_check",
            description=f"Check that columns have at least {threshold*100}% non-null values",
            severity=severity
        )
        self.threshold = threshold
    
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        issues = []
        columns_to_check = [column] if column else df.columns
        
        for col in columns_to_check:
            if col not in df.columns:
                continue
                
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness_rate = non_null_count / total_count if total_count > 0 else 0
            
            if completeness_rate < self.threshold:
                null_examples = df[df[col].isna()].index.tolist()[:5]
                
                issues.append(DataQualityIssue(
                    column=col,
                    issue_type="completeness",
                    severity=self.severity,
                    message=f"Column '{col}' has {completeness_rate:.2%} completeness (threshold: {self.threshold:.2%})",
                    failed_count=total_count - non_null_count,
                    total_count=total_count,
                    examples=null_examples
                ))
        
        return issues


class UniquenessRule(ValidationRule):
    """Validate data uniqueness"""
    
    def __init__(self, severity: DataQualityLevel = DataQualityLevel.MEDIUM):
        super().__init__(
            name="uniqueness_check",
            description="Check for duplicate values in columns that should be unique",
            severity=severity
        )
    
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        issues = []
        columns_to_check = [column] if column else df.columns
        
        for col in columns_to_check:
            if col not in df.columns:
                continue
                
            duplicate_mask = df[col].duplicated()
            duplicate_count = duplicate_mask.sum()
            
            if duplicate_count > 0:
                duplicate_values = df[duplicate_mask][col].unique()[:5]
                
                issues.append(DataQualityIssue(
                    column=col,
                    issue_type="uniqueness",
                    severity=self.severity,
                    message=f"Column '{col}' has {duplicate_count} duplicate values",
                    failed_count=duplicate_count,
                    total_count=len(df),
                    examples=duplicate_values.tolist()
                ))
        
        return issues


class DataTypeRule(ValidationRule):
    """Validate expected data types"""
    
    def __init__(self, expected_types: Dict[str, str], severity: DataQualityLevel = DataQualityLevel.MEDIUM):
        super().__init__(
            name="data_type_check",
            description="Check that columns have expected data types",
            severity=severity
        )
        self.expected_types = expected_types
    
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        issues = []
        columns_to_check = [column] if column else list(self.expected_types.keys())
        
        for col in columns_to_check:
            if col not in df.columns or col not in self.expected_types:
                continue
                
            expected_type = self.expected_types[col]
            actual_type = str(df[col].dtype)
            
            # Type compatibility mapping
            type_mapping = {
                'integer': ['int64', 'int32', 'Int64'],
                'float': ['float64', 'float32'],
                'string': ['object', 'string'],
                'datetime': ['datetime64[ns]', 'datetime64'],
                'boolean': ['bool', 'boolean']
            }
            
            compatible_types = type_mapping.get(expected_type, [expected_type])
            
            if actual_type not in compatible_types:
                issues.append(DataQualityIssue(
                    column=col,
                    issue_type="data_type",
                    severity=self.severity,
                    message=f"Column '{col}' has type '{actual_type}', expected '{expected_type}'",
                    failed_count=1,
                    total_count=1,
                    examples=[f"actual: {actual_type}, expected: {expected_type}"]
                ))
        
        return issues


class RangeRule(ValidationRule):
    """Validate numeric ranges"""
    
    def __init__(
        self, 
        ranges: Dict[str, Dict[str, float]], 
        severity: DataQualityLevel = DataQualityLevel.MEDIUM
    ):
        super().__init__(
            name="range_check",
            description="Check that numeric values fall within expected ranges",
            severity=severity
        )
        self.ranges = ranges
    
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        issues = []
        columns_to_check = [column] if column else list(self.ranges.keys())
        
        for col in columns_to_check:
            if col not in df.columns or col not in self.ranges:
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            range_config = self.ranges[col]
            min_val = range_config.get('min')
            max_val = range_config.get('max')
            
            violations = []
            violation_examples = []
            
            if min_val is not None:
                below_min = df[df[col] < min_val]
                violations.extend(below_min.index.tolist())
                violation_examples.extend(below_min[col].head(3).tolist())
            
            if max_val is not None:
                above_max = df[df[col] > max_val]
                violations.extend(above_max.index.tolist())
                violation_examples.extend(above_max[col].head(3).tolist())
            
            if violations:
                issues.append(DataQualityIssue(
                    column=col,
                    issue_type="range",
                    severity=self.severity,
                    message=f"Column '{col}' has {len(violations)} values outside range [{min_val}, {max_val}]",
                    failed_count=len(violations),
                    total_count=len(df),
                    examples=violation_examples[:5]
                ))
        
        return issues


class PatternRule(ValidationRule):
    """Validate data patterns using regex"""
    
    def __init__(
        self, 
        patterns: Dict[str, str], 
        severity: DataQualityLevel = DataQualityLevel.LOW
    ):
        super().__init__(
            name="pattern_check",
            description="Check that text values match expected patterns",
            severity=severity
        )
        self.patterns = {col: re.compile(pattern) for col, pattern in patterns.items()}
    
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        issues = []
        columns_to_check = [column] if column else list(self.patterns.keys())
        
        for col in columns_to_check:
            if col not in df.columns or col not in self.patterns:
                continue
                
            pattern = self.patterns[col]
            non_null_series = df[col].dropna().astype(str)
            
            if len(non_null_series) == 0:
                continue
            
            matches = non_null_series.str.match(pattern)
            invalid_count = (~matches).sum()
            
            if invalid_count > 0:
                invalid_values = non_null_series[~matches].head(5).tolist()
                
                issues.append(DataQualityIssue(
                    column=col,
                    issue_type="pattern",
                    severity=self.severity,
                    message=f"Column '{col}' has {invalid_count} values not matching expected pattern",
                    failed_count=invalid_count,
                    total_count=len(non_null_series),
                    examples=invalid_values
                ))
        
        return issues


class StatisticalAnomalyRule(ValidationRule):
    """Detect statistical anomalies using IQR and Z-score methods"""
    
    def __init__(
        self, 
        method: str = "iqr",
        threshold: float = 3.0,
        severity: DataQualityLevel = DataQualityLevel.LOW
    ):
        super().__init__(
            name="statistical_anomaly",
            description=f"Detect statistical anomalies using {method} method",
            severity=severity
        )
        self.method = method
        self.threshold = threshold
    
    async def validate(self, df: pd.DataFrame, column: Optional[str] = None) -> List[DataQualityIssue]:
        issues = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        columns_to_check = [column] if column else numeric_columns
        
        for col in columns_to_check:
            if col not in df.columns or col not in numeric_columns:
                continue
            
            series = df[col].dropna()
            if len(series) < 10:  # Need sufficient data for anomaly detection
                continue
            
            anomaly_indices = []
            
            if self.method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.threshold * IQR)
                upper_bound = Q3 + (self.threshold * IQR)
                anomaly_mask = (series < lower_bound) | (series > upper_bound)
                
            elif self.method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                anomaly_mask = z_scores > self.threshold
            
            else:
                continue  # Unknown method
            
            anomaly_count = anomaly_mask.sum()
            
            if anomaly_count > 0:
                anomaly_values = series[anomaly_mask].head(5).tolist()
                
                issues.append(DataQualityIssue(
                    column=col,
                    issue_type="statistical_anomaly",
                    severity=self.severity,
                    message=f"Column '{col}' has {anomaly_count} statistical anomalies ({self.method} method)",
                    failed_count=anomaly_count,
                    total_count=len(series),
                    examples=anomaly_values
                ))
        
        return issues


class ChurnDetectionSchema(Schema):
    """Marshmallow schema for customer churn detection data"""
    
    # Required fields
    customer_id = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    last_activity_date = fields.DateTime(required=True)
    account_created_date = fields.DateTime(required=True)
    total_value = fields.Float(required=True, validate=validate.Range(min=0))
    
    # Optional fields
    support_tickets = fields.Int(missing=0, validate=validate.Range(min=0))
    feature_usage_count = fields.Int(missing=0, validate=validate.Range(min=0))
    plan_type = fields.Str(missing="unknown", validate=validate.OneOf([
        "basic", "premium", "enterprise", "trial", "unknown"
    ]))
    
    @post_load
    def validate_business_rules(self, data, **kwargs):
        """Apply business rule validations"""
        # Account creation date should be before last activity
        if data['account_created_date'] > data['last_activity_date']:
            raise ValidationError("Account creation date cannot be after last activity date")
        
        # Check for reasonable account age
        account_age = (datetime.now() - data['account_created_date']).days
        if account_age > 10 * 365:  # 10 years
            raise ValidationError("Account age appears unreasonably old")
        
        return data


class DataValidationService:
    """
    Comprehensive data validation service for ML pipelines
    
    Provides:
    - Schema validation using marshmallow
    - Data quality rule engine
    - Statistical anomaly detection
    - Performance monitoring
    - Detailed reporting
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.schemas: Dict[str, Schema] = {
            'customer_churn': ChurnDetectionSchema()
        }
        
        # Register default quality rules
        self._register_default_rules()
        
        logger.info("DataValidationService initialized")
    
    def _register_default_rules(self):
        """Register default data quality rules"""
        # Completeness rules
        self.rules.append(CompletenessRule(threshold=0.95, severity=DataQualityLevel.HIGH))
        
        # Statistical anomaly detection
        self.rules.append(StatisticalAnomalyRule(
            method="iqr", 
            threshold=3.0, 
            severity=DataQualityLevel.LOW
        ))
        self.rules.append(StatisticalAnomalyRule(
            method="zscore", 
            threshold=3.0, 
            severity=DataQualityLevel.LOW
        ))
        
        # Email pattern validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.rules.append(PatternRule(
            patterns={'email': email_pattern},
            severity=DataQualityLevel.LOW
        ))
    
    async def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        schema_name: Optional[str] = None,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> ValidationReport:
        """
        Comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            dataset_id: Unique identifier for the dataset
            schema_name: Name of schema to validate against
            custom_rules: Additional validation rules to apply
            
        Returns:
            ValidationReport with all issues found
        """
        start_time = datetime.utcnow()
        all_issues = []
        
        logger.info(
            "Starting dataset validation",
            dataset_id=dataset_id,
            rows=len(df),
            columns=len(df.columns),
            schema_name=schema_name
        )
        
        try:
            # Schema validation
            if schema_name and schema_name in self.schemas:
                schema_issues = await self._validate_schema(df, schema_name)
                all_issues.extend(schema_issues)
            
            # Apply data quality rules
            quality_rules = self.rules + (custom_rules or [])
            for rule in quality_rules:
                try:
                    rule_issues = await rule.validate(df)
                    all_issues.extend(rule_issues)
                except Exception as e:
                    logger.error(
                        "Error executing validation rule",
                        rule_name=rule.name,
                        error=str(e),
                        exc_info=True
                    )
            
            # Determine overall pass/fail
            critical_issues = [i for i in all_issues if i.severity == DataQualityLevel.CRITICAL]
            high_issues = [i for i in all_issues if i.severity == DataQualityLevel.HIGH]
            
            # Fail if any critical issues or too many high issues
            passed = len(critical_issues) == 0 and len(high_issues) <= 5
            
            # Generate summary statistics
            summary = self._generate_summary(df, all_issues)
            
            report = ValidationReport(
                dataset_id=dataset_id,
                schema_name=schema_name or "none",
                passed=passed,
                issues=all_issues,
                summary=summary
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Dataset validation completed",
                dataset_id=dataset_id,
                passed=passed,
                total_issues=len(all_issues),
                critical_issues=len(critical_issues),
                high_issues=len(high_issues),
                duration_seconds=duration
            )
            
            return report
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "Dataset validation failed",
                dataset_id=dataset_id,
                error=str(e),
                duration_seconds=duration,
                exc_info=True
            )
            raise
    
    async def _validate_schema(self, df: pd.DataFrame, schema_name: str) -> List[DataQualityIssue]:
        """Validate DataFrame against marshmallow schema"""
        schema = self.schemas[schema_name]
        issues = []
        
        # Convert DataFrame to dict records for validation
        records = df.to_dict('records')
        
        # Validate each record
        validation_errors = []
        for idx, record in enumerate(records[:1000]):  # Limit to first 1000 records
            try:
                schema.load(record)
            except ValidationError as e:
                validation_errors.append({
                    'row': idx,
                    'errors': e.messages
                })
                
                if len(validation_errors) >= 100:  # Limit error collection
                    break
        
        # Convert validation errors to DataQualityIssue objects
        if validation_errors:
            error_summary = defaultdict(list)
            for error in validation_errors:
                for field, messages in error['errors'].items():
                    error_summary[field].extend(messages)
            
            for field, messages in error_summary.items():
                unique_messages = list(set(messages))
                issues.append(DataQualityIssue(
                    column=field,
                    issue_type="schema_validation",
                    severity=DataQualityLevel.HIGH,
                    message=f"Schema validation failed: {'; '.join(unique_messages)}",
                    failed_count=len(messages),
                    total_count=len(records),
                    examples=unique_messages[:5]
                ))
        
        return issues
    
    def _generate_summary(self, df: pd.DataFrame, issues: List[DataQualityIssue]) -> Dict[str, Any]:
        """Generate validation summary statistics"""
        # Group issues by severity
        severity_counts = defaultdict(int)
        for issue in issues:
            severity_counts[issue.severity.value] += 1
        
        # Column-level statistics
        column_stats = {}
        for col in df.columns:
            column_issues = [i for i in issues if i.column == col]
            column_stats[col] = {
                'issues_count': len(column_issues),
                'completeness': (df[col].notna().sum() / len(df)) if len(df) > 0 else 0,
                'uniqueness': (df[col].nunique() / len(df)) if len(df) > 0 else 0,
                'data_type': str(df[col].dtype)
            }
        
        return {
            'total_issues': len(issues),
            'severity_breakdown': dict(severity_counts),
            'data_quality_score': self._calculate_quality_score(issues, len(df)),
            'column_statistics': column_stats,
            'dataset_shape': list(df.shape),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def _calculate_quality_score(self, issues: List[DataQualityIssue], total_rows: int) -> float:
        """Calculate overall data quality score (0-100)"""
        if not issues or total_rows == 0:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            DataQualityLevel.CRITICAL: 10,
            DataQualityLevel.HIGH: 5,
            DataQualityLevel.MEDIUM: 2,
            DataQualityLevel.LOW: 1
        }
        
        weighted_issues = sum(
            severity_weights.get(issue.severity, 1) * (issue.failed_count / total_rows)
            for issue in issues
        )
        
        # Convert to 0-100 scale (lower is better for issues)
        quality_score = max(0, 100 - (weighted_issues * 100))
        return round(quality_score, 2)
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add a custom validation rule"""
        self.rules.append(rule)
        logger.info("Custom validation rule added", rule_name=rule.name)
    
    def add_schema(self, name: str, schema: Schema):
        """Add a custom schema for validation"""
        self.schemas[name] = schema
        logger.info("Custom schema added", schema_name=name)
    
    async def validate_blueprint_compatibility(
        self,
        df: pd.DataFrame,
        blueprint_name: str,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset compatibility with a specific blueprint
        
        Args:
            df: Input DataFrame
            blueprint_name: Name of the blueprint
            required_columns: List of required column names
            optional_columns: List of optional column names
            
        Returns:
            Compatibility assessment
        """
        optional_columns = optional_columns or []
        
        # Check column availability
        available_columns = set(df.columns)
        missing_required = set(required_columns) - available_columns
        available_optional = set(optional_columns) & available_columns
        
        # Data quality assessment for required columns
        quality_issues = []
        for col in required_columns:
            if col in df.columns:
                # Check completeness
                completeness = df[col].notna().sum() / len(df) if len(df) > 0 else 0
                if completeness < 0.8:  # 80% threshold for blueprint compatibility
                    quality_issues.append(f"{col}: {completeness:.1%} completeness")
        
        # Calculate compatibility score
        required_score = 1.0 if not missing_required else 0.0
        optional_score = len(available_optional) / len(optional_columns) if optional_columns else 1.0
        quality_penalty = len(quality_issues) * 0.1  # Penalize quality issues
        
        compatibility_score = max(0, (required_score * 0.7) + (optional_score * 0.2) + 
                                 (0.1 if required_score == 1.0 else 0) - quality_penalty)
        
        return {
            'compatible': len(missing_required) == 0 and len(quality_issues) <= 2,
            'score': round(compatibility_score, 3),
            'blueprint_name': blueprint_name,
            'column_analysis': {
                'required_found': list(set(required_columns) & available_columns),
                'required_missing': list(missing_required),
                'optional_found': list(available_optional),
                'optional_missing': list(set(optional_columns) - available_optional)
            },
            'quality_issues': quality_issues,
            'recommendations': self._generate_compatibility_recommendations(
                missing_required, quality_issues, blueprint_name
            )
        }
    
    def _generate_compatibility_recommendations(
        self,
        missing_required: set,
        quality_issues: List[str],
        blueprint_name: str
    ) -> List[str]:
        """Generate recommendations for improving blueprint compatibility"""
        recommendations = []
        
        if missing_required:
            recommendations.append(
                f"Add missing required columns: {', '.join(missing_required)}"
            )
        
        if quality_issues:
            recommendations.append(
                "Improve data quality in the following columns: " + 
                "; ".join(quality_issues)
            )
        
        recommendations.append(
            f"Ensure data follows the expected format for {blueprint_name} blueprint"
        )
        
        return recommendations