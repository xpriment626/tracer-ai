"""
Unit tests for data validation modules.

Tests CSV validation, data quality checks, and schema validation
for customer churn detection pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from tracer.core.data_validation import (
    CSVValidator,
    DataQualityChecker,
    SchemaValidator,
    ValidationResult,
    ValidationError
)

class TestCSVValidator:
    """Test cases for CSV file validation."""
    
    @pytest.fixture
    def validator(self):
        """Create CSVValidator instance."""
        return CSVValidator()
    
    def test_valid_csv_file(self, validator: CSVValidator, valid_csv_file: Path):
        """Test validation of a valid CSV file."""
        result = validator.validate_file(valid_csv_file)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.row_count > 0
        assert result.column_count > 0
        assert 'customer_id' in result.columns
        assert 'churn' in result.columns
    
    def test_invalid_csv_file(self, validator: CSVValidator, invalid_csv_file: Path):
        """Test validation of an invalid CSV file."""
        result = validator.validate_file(invalid_csv_file)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any('column count' in error.lower() for error in result.errors)
    
    def test_malformed_csv_file(self, validator: CSVValidator, malformed_csv_file: Path):
        """Test validation of a malformed CSV file."""
        result = validator.validate_file(malformed_csv_file)
        
        # Should detect malformed data but still be readable
        assert result.row_count > 0
        assert len(result.warnings) > 0  # Should have warnings about data quality
    
    def test_nonexistent_file(self, validator: CSVValidator, temp_dir: Path):
        """Test validation of non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            validator.validate_file(nonexistent_file)
    
    def test_empty_file(self, validator: CSVValidator, temp_dir: Path):
        """Test validation of empty CSV file."""
        empty_file = temp_dir / "empty.csv"
        empty_file.touch()
        
        result = validator.validate_file(empty_file)
        
        assert not result.is_valid
        assert 'empty' in ' '.join(result.errors).lower()
    
    def test_file_encoding_detection(self, validator: CSVValidator, temp_dir: Path):
        """Test automatic encoding detection."""
        from tests.conftest import TestDataGenerator
        
        # Create file with different encoding
        encoded_file = temp_dir / "encoded.csv"
        TestDataGenerator.create_csv_with_issues(encoded_file, "encoding")
        
        result = validator.validate_file(encoded_file)
        
        # Should handle encoding gracefully
        assert result.encoding in ['utf-8', 'latin-1', 'cp1252']
    
    def test_large_file_validation(self, validator: CSVValidator, large_csv_file: Path):
        """Test validation of large CSV files."""
        result = validator.validate_file(large_csv_file)
        
        assert result.is_valid
        assert result.row_count >= 10000
        assert result.file_size > 0
    
    @pytest.mark.parametrize("delimiter", [',', ';', '\t', '|'])
    def test_delimiter_detection(self, validator: CSVValidator, temp_dir: Path, delimiter: str):
        """Test automatic delimiter detection."""
        csv_file = temp_dir / f"delimited_{delimiter.replace('|', 'pipe')}.csv"
        
        with open(csv_file, 'w') as f:
            f.write(f"col1{delimiter}col2{delimiter}col3\n")
            f.write(f"val1{delimiter}val2{delimiter}val3\n")
        
        result = validator.validate_file(csv_file)
        
        assert result.is_valid
        assert result.delimiter == delimiter


class TestDataQualityChecker:
    """Test cases for data quality checking."""
    
    @pytest.fixture
    def quality_checker(self):
        """Create DataQualityChecker instance."""
        return DataQualityChecker()
    
    def test_missing_value_detection(self, quality_checker: DataQualityChecker, sample_csv_data: Dict[str, Any]):
        """Test detection of missing values."""
        # Add missing values to test data
        data = sample_csv_data.copy()
        data['age'][1] = None
        data['monthly_charges'][2] = np.nan
        
        df = pd.DataFrame(data)
        result = quality_checker.check_missing_values(df)
        
        assert not result.is_valid
        assert len(result.missing_columns) == 2
        assert 'age' in result.missing_columns
        assert 'monthly_charges' in result.missing_columns
    
    def test_data_type_validation(self, quality_checker: DataQualityChecker, sample_csv_data: Dict[str, Any]):
        """Test data type validation."""
        df = pd.DataFrame(sample_csv_data)
        
        expected_types = {
            'customer_id': 'object',
            'age': 'int64',
            'tenure': 'int64',
            'monthly_charges': 'float64',
            'total_charges': 'float64',
            'churn': 'int64'
        }
        
        result = quality_checker.validate_data_types(df, expected_types)
        
        assert result.is_valid
        assert len(result.type_mismatches) == 0
    
    def test_outlier_detection(self, quality_checker: DataQualityChecker):
        """Test outlier detection using IQR method."""
        # Create data with outliers
        data = {
            'normal_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'outlier_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        }
        df = pd.DataFrame(data)
        
        outliers = quality_checker.detect_outliers(df, ['normal_column', 'outlier_column'])
        
        assert 'outlier_column' in outliers
        assert len(outliers['outlier_column']) > 0
        assert 'normal_column' not in outliers or len(outliers['normal_column']) == 0
    
    def test_duplicate_detection(self, quality_checker: DataQualityChecker):
        """Test duplicate row detection."""
        data = {
            'customer_id': ['CUST001', 'CUST002', 'CUST001'],  # Duplicate customer_id
            'age': [25, 35, 25],
            'churn': [1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        duplicates = quality_checker.find_duplicates(df, subset=['customer_id'])
        
        assert len(duplicates) > 0
        assert 'CUST001' in duplicates['customer_id'].values
    
    def test_value_range_validation(self, quality_checker: DataQualityChecker):
        """Test value range validation."""
        data = {
            'age': [25, 35, -5, 150],  # -5 and 150 are invalid ages
            'churn': [0, 1, 0, 2]  # 2 is invalid for binary churn
        }
        df = pd.DataFrame(data)
        
        ranges = {
            'age': (0, 120),
            'churn': (0, 1)
        }
        
        result = quality_checker.validate_value_ranges(df, ranges)
        
        assert not result.is_valid
        assert 'age' in result.range_violations
        assert 'churn' in result.range_violations
    
    def test_categorical_value_validation(self, quality_checker: DataQualityChecker):
        """Test categorical value validation."""
        data = {
            'contract': ['Month-to-month', 'One year', 'Invalid contract', 'Two year'],
            'payment_method': ['Credit card', 'Bank transfer', 'Invalid method', 'Electronic check']
        }
        df = pd.DataFrame(data)
        
        valid_values = {
            'contract': ['Month-to-month', 'One year', 'Two year'],
            'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check']
        }
        
        result = quality_checker.validate_categorical_values(df, valid_values)
        
        assert not result.is_valid
        assert 'contract' in result.invalid_categories
        assert 'payment_method' in result.invalid_categories


class TestSchemaValidator:
    """Test cases for schema validation."""
    
    @pytest.fixture
    def schema_validator(self):
        """Create SchemaValidator instance."""
        return SchemaValidator()
    
    @pytest.fixture
    def expected_schema(self):
        """Define expected schema for customer data."""
        return {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "pattern": "^CUST\\d{6}$"},
                "age": {"type": "integer", "minimum": 18, "maximum": 120},
                "tenure": {"type": "integer", "minimum": 0, "maximum": 120},
                "monthly_charges": {"type": "number", "minimum": 0},
                "total_charges": {"type": "number", "minimum": 0},
                "contract": {
                    "type": "string",
                    "enum": ["Month-to-month", "One year", "Two year"]
                },
                "payment_method": {
                    "type": "string",
                    "enum": ["Credit card", "Bank transfer", "Electronic check", "Mailed check"]
                },
                "churn": {"type": "integer", "enum": [0, 1]}
            },
            "required": [
                "customer_id", "age", "tenure", "monthly_charges",
                "total_charges", "contract", "payment_method", "churn"
            ],
            "additionalProperties": False
        }
    
    def test_valid_data_schema(self, schema_validator: SchemaValidator, expected_schema: Dict, sample_csv_data: Dict[str, Any]):
        """Test schema validation with valid data."""
        df = pd.DataFrame(sample_csv_data)
        
        # Convert DataFrame to records for schema validation
        records = df.to_dict('records')
        
        for record in records:
            result = schema_validator.validate_record(record, expected_schema)
            assert result.is_valid, f"Record validation failed: {result.errors}"
    
    def test_invalid_data_schema(self, schema_validator: SchemaValidator, expected_schema: Dict):
        """Test schema validation with invalid data."""
        invalid_record = {
            "customer_id": "INVALID_ID",  # Wrong pattern
            "age": 150,  # Too old
            "tenure": -1,  # Negative value
            "monthly_charges": -50.0,  # Negative value
            "total_charges": "invalid",  # Wrong type
            "contract": "Invalid contract",  # Not in enum
            "payment_method": "Invalid method",  # Not in enum
            "churn": 2  # Not in enum
        }
        
        result = schema_validator.validate_record(invalid_record, expected_schema)
        
        assert not result.is_valid
        assert len(result.errors) >= 6  # Multiple validation errors
    
    def test_missing_required_fields(self, schema_validator: SchemaValidator, expected_schema: Dict):
        """Test schema validation with missing required fields."""
        incomplete_record = {
            "customer_id": "CUST001234",
            "age": 25
            # Missing required fields
        }
        
        result = schema_validator.validate_record(incomplete_record, expected_schema)
        
        assert not result.is_valid
        assert any('required' in error.lower() for error in result.errors)
    
    def test_additional_properties(self, schema_validator: SchemaValidator, expected_schema: Dict):
        """Test schema validation with additional properties."""
        record_with_extra = {
            "customer_id": "CUST001234",
            "age": 25,
            "tenure": 12,
            "monthly_charges": 50.0,
            "total_charges": 600.0,
            "contract": "Month-to-month",
            "payment_method": "Credit card",
            "churn": 0,
            "extra_field": "should not be allowed"  # Extra field
        }
        
        result = schema_validator.validate_record(record_with_extra, expected_schema)
        
        assert not result.is_valid
        assert any('additional' in error.lower() for error in result.errors)


class TestValidationResult:
    """Test cases for validation result objects."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult object creation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=['Warning message'],
            metadata={'row_count': 100}
        )
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.metadata['row_count'] == 100
    
    def test_validation_result_serialization(self):
        """Test ValidationResult serialization to dict."""
        result = ValidationResult(
            is_valid=False,
            errors=['Error 1', 'Error 2'],
            warnings=['Warning 1'],
            metadata={'source': 'test'}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['is_valid'] is False
        assert len(result_dict['errors']) == 2
        assert len(result_dict['warnings']) == 1
        assert result_dict['metadata']['source'] == 'test'


@pytest.mark.performance
class TestDataValidationPerformance:
    """Performance tests for data validation."""
    
    def test_large_file_validation_performance(self, large_csv_file: Path, performance_test_data: Dict):
        """Test validation performance with large files."""
        import time
        
        validator = CSVValidator()
        start_time = time.time()
        
        result = validator.validate_file(large_csv_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < performance_test_data['response_time_threshold']
        assert result.is_valid
        assert result.row_count >= 10000
    
    def test_memory_usage_during_validation(self, large_csv_file: Path, performance_test_data: Dict):
        """Test memory usage during validation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        validator = CSVValidator()
        result = validator.validate_file(large_csv_file)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        assert memory_used < performance_test_data['memory_threshold']
        assert result.is_valid


@pytest.mark.integration
class TestDataValidationIntegration:
    """Integration tests for data validation components."""
    
    def test_end_to_end_validation_pipeline(self, valid_csv_file: Path):
        """Test complete validation pipeline."""
        # Initialize all validators
        csv_validator = CSVValidator()
        quality_checker = DataQualityChecker()
        schema_validator = SchemaValidator()
        
        # Step 1: Validate CSV file structure
        csv_result = csv_validator.validate_file(valid_csv_file)
        assert csv_result.is_valid
        
        # Step 2: Load and validate data quality
        df = pd.read_csv(valid_csv_file)
        quality_result = quality_checker.check_missing_values(df)
        assert quality_result.is_valid
        
        # Step 3: Validate schema for each record
        records = df.to_dict('records')
        schema = {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "age": {"type": "integer"},
                "churn": {"type": "integer"}
            },
            "required": ["customer_id", "age", "churn"]
        }
        
        for record in records:
            schema_result = schema_validator.validate_record(record, schema)
            assert schema_result.is_valid