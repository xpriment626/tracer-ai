"""
CSV test fixtures for data validation and processing tests.

Provides various CSV file fixtures with different data quality scenarios
for comprehensive testing of the customer churn detection pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv
import io
import json
from datetime import datetime, timedelta

class CSVFixtureGenerator:
    """Generator for various CSV test fixtures."""
    
    @staticmethod
    def create_valid_customer_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Generate valid customer data following expected schema."""
        np.random.seed(seed)
        
        # Customer IDs
        customer_ids = [f"CUST{i:06d}" for i in range(1, n_rows + 1)]
        
        # Demographics
        ages = np.random.randint(18, 80, n_rows)
        
        # Service data
        tenure_months = np.random.randint(1, 72, n_rows)
        monthly_charges = np.random.uniform(20.0, 120.0, n_rows)
        
        # Calculate total charges based on tenure and monthly charges with some variation
        base_total = tenure_months * monthly_charges
        variation = np.random.uniform(0.8, 1.2, n_rows)  # ±20% variation
        total_charges = base_total * variation
        
        # Contract types
        contract_types = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            n_rows, 
            p=[0.5, 0.3, 0.2]
        )
        
        # Payment methods
        payment_methods = np.random.choice(
            ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'],
            n_rows,
            p=[0.3, 0.25, 0.25, 0.2]
        )
        
        # Internet services
        internet_service = np.random.choice(
            ['DSL', 'Fiber optic', 'No'], 
            n_rows, 
            p=[0.4, 0.4, 0.2]
        )
        
        # Online security (correlated with internet service)
        online_security = []
        for internet in internet_service:
            if internet == 'No':
                online_security.append('No internet service')
            else:
                online_security.append(np.random.choice(['Yes', 'No'], p=[0.3, 0.7]))
        
        # Phone service
        phone_service = np.random.choice(['Yes', 'No'], n_rows, p=[0.9, 0.1])
        
        # Multiple lines (only relevant if phone service = Yes)
        multiple_lines = []
        for phone in phone_service:
            if phone == 'No':
                multiple_lines.append('No phone service')
            else:
                multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
        
        # Generate churn with realistic correlations
        churn_probability = np.zeros(n_rows)
        
        # Higher churn for month-to-month contracts
        churn_probability += np.where(contract_types == 'Month-to-month', 0.3, 0.1)
        
        # Higher churn for newer customers
        churn_probability += np.where(tenure_months < 12, 0.2, 0)
        
        # Higher churn for higher monthly charges
        churn_probability += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.1
        
        # Higher churn for electronic check payment
        churn_probability += np.where(payment_methods == 'Electronic check', 0.15, 0)
        
        # Add some randomness
        churn_probability += np.random.uniform(-0.1, 0.1, n_rows)
        
        # Ensure probabilities are in valid range
        churn_probability = np.clip(churn_probability, 0, 1)
        
        # Generate binary churn based on probabilities
        churn = np.random.binomial(1, churn_probability, n_rows)
        
        return pd.DataFrame({
            'customer_id': customer_ids,
            'gender': np.random.choice(['Male', 'Female'], n_rows),
            'senior_citizen': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
            'partner': np.random.choice(['Yes', 'No'], n_rows, p=[0.5, 0.5]),
            'dependents': np.random.choice(['Yes', 'No'], n_rows, p=[0.3, 0.7]),
            'tenure': tenure_months,
            'phone_service': phone_service,
            'multiple_lines': multiple_lines,
            'internet_service': internet_service,
            'online_security': online_security,
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_rows, p=[0.3, 0.5, 0.2]),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_rows, p=[0.3, 0.5, 0.2]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_rows, p=[0.3, 0.5, 0.2]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_rows, p=[0.4, 0.4, 0.2]),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_rows, p=[0.4, 0.4, 0.2]),
            'contract': contract_types,
            'paperless_billing': np.random.choice(['Yes', 'No'], n_rows, p=[0.6, 0.4]),
            'payment_method': payment_methods,
            'monthly_charges': np.round(monthly_charges, 2),
            'total_charges': np.round(total_charges, 2),
            'churn': churn
        })
    
    @staticmethod
    def create_csv_with_missing_values(base_data: pd.DataFrame, missing_rate: float = 0.1) -> pd.DataFrame:
        """Create CSV data with missing values."""
        data = base_data.copy()
        
        # Introduce missing values randomly
        for column in data.columns:
            if column not in ['customer_id', 'churn']:  # Don't make key columns missing
                n_missing = int(len(data) * missing_rate)
                missing_indices = np.random.choice(data.index, n_missing, replace=False)
                data.loc[missing_indices, column] = np.nan
        
        return data
    
    @staticmethod
    def create_csv_with_outliers(base_data: pd.DataFrame) -> pd.DataFrame:
        """Create CSV data with outliers in numerical columns."""
        data = base_data.copy()
        
        # Add outliers to numerical columns
        numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
        
        for column in numerical_columns:
            if column in data.columns:
                n_outliers = max(1, int(len(data) * 0.02))  # 2% outliers
                outlier_indices = np.random.choice(data.index, n_outliers, replace=False)
                
                if column == 'tenure':
                    # Negative tenure or extremely high tenure
                    outlier_values = np.random.choice([-5, -10, 200, 300], n_outliers)
                elif column == 'monthly_charges':
                    # Negative charges or extremely high charges
                    outlier_values = np.random.choice([-50, -100, 500, 1000], n_outliers)
                elif column == 'total_charges':
                    # Negative total or extremely high total
                    outlier_values = np.random.choice([-1000, -5000, 50000, 100000], n_outliers)
                
                data.loc[outlier_indices, column] = outlier_values
        
        return data
    
    @staticmethod
    def create_csv_with_invalid_categories(base_data: pd.DataFrame) -> pd.DataFrame:
        """Create CSV data with invalid categorical values."""
        data = base_data.copy()
        
        # Define invalid values for categorical columns
        invalid_values = {
            'gender': ['Unknown', 'Other', 'N/A'],
            'contract': ['Invalid Contract', '6 months', 'Lifetime'],
            'payment_method': ['Cash', 'Bitcoin', 'Invalid Method'],
            'internet_service': ['Satellite', 'Cable', 'Invalid Service'],
            'phone_service': ['Maybe', 'Unknown', '1'],
            'partner': ['Single', 'Married', '2'],
            'dependents': ['None', 'Many', 'Unknown']
        }
        
        # Introduce invalid values
        for column, invalid_list in invalid_values.items():
            if column in data.columns:
                n_invalid = max(1, int(len(data) * 0.05))  # 5% invalid values
                invalid_indices = np.random.choice(data.index, n_invalid, replace=False)
                invalid_choice = np.random.choice(invalid_list, n_invalid)
                data.loc[invalid_indices, column] = invalid_choice
        
        return data
    
    @staticmethod
    def create_csv_with_duplicates(base_data: pd.DataFrame, duplicate_rate: float = 0.1) -> pd.DataFrame:
        """Create CSV data with duplicate rows."""
        data = base_data.copy()
        
        # Create duplicates
        n_duplicates = int(len(data) * duplicate_rate)
        duplicate_indices = np.random.choice(data.index, n_duplicates, replace=True)
        
        # Duplicate selected rows (with slight modifications to avoid exact duplicates)
        for idx in duplicate_indices:
            duplicate_row = data.iloc[idx].copy()
            # Modify customer_id to make it a true duplicate
            original_id = duplicate_row['customer_id']
            duplicate_row['customer_id'] = original_id  # Keep same ID for duplicate detection
            
            # Append the duplicate row
            data = pd.concat([data, duplicate_row.to_frame().T], ignore_index=True)
        
        return data
    
    @staticmethod
    def create_csv_with_encoding_issues(base_data: pd.DataFrame) -> pd.DataFrame:
        """Create CSV data with encoding issues."""
        data = base_data.copy()
        
        # Add special characters that might cause encoding issues
        special_names = [
            'José García', 'François Müller', 'Ñoño López', 'Øyvind Hansen',
            'Владимир', 'محمد عبدالله', '山田太郎', 'Smith & Jones'
        ]
        
        # Replace some customer IDs with problematic names for testing
        if 'customer_name' not in data.columns:
            data['customer_name'] = [f"Customer {i}" for i in range(len(data))]
        
        n_special = min(len(special_names), len(data))
        special_indices = np.random.choice(data.index, n_special, replace=False)
        
        for i, idx in enumerate(special_indices):
            data.loc[idx, 'customer_name'] = special_names[i]
        
        return data


@pytest.fixture(scope="session")
def csv_fixture_generator():
    """Provide CSVFixtureGenerator instance."""
    return CSVFixtureGenerator()


@pytest.fixture(scope="function")
def perfect_csv_data(csv_fixture_generator: CSVFixtureGenerator) -> pd.DataFrame:
    """Perfect CSV data with no quality issues."""
    return csv_fixture_generator.create_valid_customer_data(n_rows=100, seed=42)


@pytest.fixture(scope="function")
def perfect_csv_file(temp_dir: Path, perfect_csv_data: pd.DataFrame) -> Path:
    """Perfect CSV file with no quality issues."""
    csv_path = temp_dir / "perfect_customer_data.csv"
    perfect_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def missing_values_csv_data(csv_fixture_generator: CSVFixtureGenerator, perfect_csv_data: pd.DataFrame) -> pd.DataFrame:
    """CSV data with missing values."""
    return csv_fixture_generator.create_csv_with_missing_values(perfect_csv_data, missing_rate=0.15)


@pytest.fixture(scope="function") 
def missing_values_csv_file(temp_dir: Path, missing_values_csv_data: pd.DataFrame) -> Path:
    """CSV file with missing values."""
    csv_path = temp_dir / "missing_values_data.csv"
    missing_values_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def outliers_csv_data(csv_fixture_generator: CSVFixtureGenerator, perfect_csv_data: pd.DataFrame) -> pd.DataFrame:
    """CSV data with outliers."""
    return csv_fixture_generator.create_csv_with_outliers(perfect_csv_data)


@pytest.fixture(scope="function")
def outliers_csv_file(temp_dir: Path, outliers_csv_data: pd.DataFrame) -> Path:
    """CSV file with outliers."""
    csv_path = temp_dir / "outliers_data.csv"
    outliers_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def invalid_categories_csv_data(csv_fixture_generator: CSVFixtureGenerator, perfect_csv_data: pd.DataFrame) -> pd.DataFrame:
    """CSV data with invalid categorical values."""
    return csv_fixture_generator.create_csv_with_invalid_categories(perfect_csv_data)


@pytest.fixture(scope="function")
def invalid_categories_csv_file(temp_dir: Path, invalid_categories_csv_data: pd.DataFrame) -> Path:
    """CSV file with invalid categorical values."""
    csv_path = temp_dir / "invalid_categories_data.csv"
    invalid_categories_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def duplicates_csv_data(csv_fixture_generator: CSVFixtureGenerator, perfect_csv_data: pd.DataFrame) -> pd.DataFrame:
    """CSV data with duplicate rows."""
    return csv_fixture_generator.create_csv_with_duplicates(perfect_csv_data, duplicate_rate=0.2)


@pytest.fixture(scope="function")
def duplicates_csv_file(temp_dir: Path, duplicates_csv_data: pd.DataFrame) -> Path:
    """CSV file with duplicate rows."""
    csv_path = temp_dir / "duplicates_data.csv"
    duplicates_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def encoding_issues_csv_data(csv_fixture_generator: CSVFixtureGenerator, perfect_csv_data: pd.DataFrame) -> pd.DataFrame:
    """CSV data with encoding issues."""
    return csv_fixture_generator.create_csv_with_encoding_issues(perfect_csv_data)


@pytest.fixture(scope="function")
def encoding_issues_csv_file(temp_dir: Path, encoding_issues_csv_data: pd.DataFrame) -> Path:
    """CSV file with encoding issues."""
    csv_path = temp_dir / "encoding_issues_data.csv"
    # Save with different encoding to simulate real-world scenario
    encoding_issues_csv_data.to_csv(csv_path, index=False, encoding='latin-1')
    return csv_path


@pytest.fixture(scope="function")
def comprehensive_issues_csv_data(csv_fixture_generator: CSVFixtureGenerator) -> pd.DataFrame:
    """CSV data with multiple quality issues combined."""
    # Start with valid data
    base_data = csv_fixture_generator.create_valid_customer_data(n_rows=200, seed=123)
    
    # Apply multiple issues
    data = csv_fixture_generator.create_csv_with_missing_values(base_data, missing_rate=0.1)
    data = csv_fixture_generator.create_csv_with_outliers(data)
    data = csv_fixture_generator.create_csv_with_invalid_categories(data)
    data = csv_fixture_generator.create_csv_with_duplicates(data, duplicate_rate=0.05)
    data = csv_fixture_generator.create_csv_with_encoding_issues(data)
    
    return data


@pytest.fixture(scope="function")
def comprehensive_issues_csv_file(temp_dir: Path, comprehensive_issues_csv_data: pd.DataFrame) -> Path:
    """CSV file with comprehensive quality issues."""
    csv_path = temp_dir / "comprehensive_issues_data.csv"
    comprehensive_issues_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def minimal_csv_data() -> pd.DataFrame:
    """Minimal CSV data with required columns only."""
    return pd.DataFrame({
        'customer_id': ['CUST000001', 'CUST000002', 'CUST000003'],
        'age': [25, 35, 45],
        'tenure': [12, 24, 36],
        'monthly_charges': [50.0, 75.5, 89.99],
        'total_charges': [600.0, 1812.0, 3239.64],
        'churn': [1, 0, 0]
    })


@pytest.fixture(scope="function")
def minimal_csv_file(temp_dir: Path, minimal_csv_data: pd.DataFrame) -> Path:
    """Minimal CSV file with required columns only."""
    csv_path = temp_dir / "minimal_data.csv"
    minimal_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def extra_columns_csv_data(minimal_csv_data: pd.DataFrame) -> pd.DataFrame:
    """CSV data with extra columns not in schema."""
    data = minimal_csv_data.copy()
    data['extra_column1'] = ['value1', 'value2', 'value3']
    data['extra_column2'] = [1.1, 2.2, 3.3]
    data['unexpected_column'] = [True, False, True]
    return data


@pytest.fixture(scope="function")
def extra_columns_csv_file(temp_dir: Path, extra_columns_csv_data: pd.DataFrame) -> Path:
    """CSV file with extra columns not in schema."""
    csv_path = temp_dir / "extra_columns_data.csv"
    extra_columns_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function") 
def wrong_data_types_csv_file(temp_dir: Path) -> Path:
    """CSV file with wrong data types."""
    csv_path = temp_dir / "wrong_types_data.csv"
    
    # Create CSV with string values where numbers expected
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['customer_id', 'age', 'tenure', 'monthly_charges', 'total_charges', 'churn'])
        writer.writerow(['CUST000001', 'twenty-five', '12', '50.0', '600.0', '1'])
        writer.writerow(['CUST000002', '35', 'two years', '75.5', 'expensive', '0'])
        writer.writerow(['CUST000003', '45.5', '36', 'free', '3239.64', 'yes'])
    
    return csv_path


@pytest.fixture(scope="function")
def inconsistent_delimiter_csv_file(temp_dir: Path) -> Path:
    """CSV file with inconsistent delimiters."""
    csv_path = temp_dir / "inconsistent_delimiter_data.csv"
    
    with open(csv_path, 'w') as f:
        # Mix of comma and semicolon delimiters
        f.write("customer_id,age,tenure,monthly_charges\n")
        f.write("CUST000001,25,12,50.0\n")
        f.write("CUST000002;35;24;75.5\n")  # Different delimiter
        f.write("CUST000003,45,36,89.99\n")
    
    return csv_path


@pytest.fixture(scope="function")
def malformed_csv_file(temp_dir: Path) -> Path:
    """CSV file with malformed structure."""
    csv_path = temp_dir / "malformed_data.csv"
    
    with open(csv_path, 'w') as f:
        f.write("customer_id,age,tenure,monthly_charges,churn\n")
        f.write("CUST000001,25,12,50.0,1\n")  # Valid row
        f.write("CUST000002,35,24\n")  # Missing columns
        f.write("CUST000003,45,36,89.99,0,extra_value\n")  # Extra columns
        f.write("CUST000004,30,\"quoted,value\",65.25,1\n")  # Quoted value with comma
        f.write(",28,15,55.0,0\n")  # Missing customer_id
    
    return csv_path


class SpecialCaseFixtures:
    """Special case CSV fixtures for edge case testing."""
    
    @staticmethod
    def create_empty_csv_file(temp_dir: Path) -> Path:
        """Create completely empty CSV file."""
        csv_path = temp_dir / "empty.csv"
        csv_path.touch()
        return csv_path
    
    @staticmethod
    def create_headers_only_csv_file(temp_dir: Path) -> Path:
        """Create CSV file with headers but no data."""
        csv_path = temp_dir / "headers_only.csv"
        csv_path.write_text("customer_id,age,tenure,monthly_charges,total_charges,churn\n")
        return csv_path
    
    @staticmethod
    def create_single_row_csv_file(temp_dir: Path) -> Path:
        """Create CSV file with single data row."""
        csv_path = temp_dir / "single_row.csv"
        csv_path.write_text(
            "customer_id,age,tenure,monthly_charges,total_charges,churn\n"
            "CUST000001,25,12,50.0,600.0,1\n"
        )
        return csv_path
    
    @staticmethod
    def create_very_wide_csv_file(temp_dir: Path, n_columns: int = 100) -> Path:
        """Create CSV file with many columns."""
        csv_path = temp_dir / "very_wide.csv"
        
        # Generate headers
        headers = ['customer_id', 'age', 'tenure', 'monthly_charges', 'total_charges', 'churn']
        headers.extend([f'extra_col_{i}' for i in range(n_columns - len(headers))])
        
        # Generate data
        data = ['CUST000001', '25', '12', '50.0', '600.0', '1']
        data.extend(['0'] * (n_columns - len(data)))
        
        with open(csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
            f.write(','.join(data) + '\n')
        
        return csv_path
    
    @staticmethod
    def create_binary_file_with_csv_extension(temp_dir: Path) -> Path:
        """Create binary file with .csv extension."""
        csv_path = temp_dir / "binary_fake.csv"
        
        # Write binary data
        binary_data = b'\x00\x01\x02\x03\x04\x05\xFF\xFE\xFD\xFC'
        csv_path.write_bytes(binary_data)
        
        return csv_path


@pytest.fixture(scope="function")
def special_case_fixtures(temp_dir: Path):
    """Provide SpecialCaseFixtures with temp directory."""
    return SpecialCaseFixtures()


# Performance test fixtures
@pytest.fixture(scope="function")
def large_csv_data(csv_fixture_generator: CSVFixtureGenerator) -> pd.DataFrame:
    """Large CSV data for performance testing."""
    return csv_fixture_generator.create_valid_customer_data(n_rows=10000, seed=42)


@pytest.fixture(scope="function")
def large_csv_file(temp_dir: Path, large_csv_data: pd.DataFrame) -> Path:
    """Large CSV file for performance testing."""
    csv_path = temp_dir / "large_customer_data.csv"
    large_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def very_large_csv_data(csv_fixture_generator: CSVFixtureGenerator) -> pd.DataFrame:
    """Very large CSV data for stress testing."""
    return csv_fixture_generator.create_valid_customer_data(n_rows=100000, seed=42)


@pytest.fixture(scope="function")
def very_large_csv_file(temp_dir: Path, very_large_csv_data: pd.DataFrame) -> Path:
    """Very large CSV file for stress testing."""
    csv_path = temp_dir / "very_large_customer_data.csv"
    very_large_csv_data.to_csv(csv_path, index=False)
    return csv_path


# Schema validation fixtures
@pytest.fixture(scope="function")
def expected_customer_schema() -> Dict[str, Any]:
    """Expected schema for customer data validation."""
    return {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "pattern": "^CUST\\d{6}$",
                "description": "Unique customer identifier"
            },
            "gender": {
                "type": "string",
                "enum": ["Male", "Female"],
                "description": "Customer gender"
            },
            "senior_citizen": {
                "type": "integer",
                "enum": [0, 1],
                "description": "Whether customer is senior citizen (1) or not (0)"
            },
            "partner": {
                "type": "string",
                "enum": ["Yes", "No"],
                "description": "Whether customer has a partner"
            },
            "dependents": {
                "type": "string", 
                "enum": ["Yes", "No"],
                "description": "Whether customer has dependents"
            },
            "tenure": {
                "type": "integer",
                "minimum": 0,
                "maximum": 120,
                "description": "Number of months customer has stayed"
            },
            "phone_service": {
                "type": "string",
                "enum": ["Yes", "No"],
                "description": "Whether customer has phone service"
            },
            "multiple_lines": {
                "type": "string",
                "enum": ["Yes", "No", "No phone service"],
                "description": "Whether customer has multiple lines"
            },
            "internet_service": {
                "type": "string",
                "enum": ["DSL", "Fiber optic", "No"],
                "description": "Type of internet service"
            },
            "online_security": {
                "type": "string", 
                "enum": ["Yes", "No", "No internet service"],
                "description": "Whether customer has online security"
            },
            "online_backup": {
                "type": "string",
                "enum": ["Yes", "No", "No internet service"],
                "description": "Whether customer has online backup"
            },
            "device_protection": {
                "type": "string",
                "enum": ["Yes", "No", "No internet service"], 
                "description": "Whether customer has device protection"
            },
            "tech_support": {
                "type": "string",
                "enum": ["Yes", "No", "No internet service"],
                "description": "Whether customer has tech support"
            },
            "streaming_tv": {
                "type": "string",
                "enum": ["Yes", "No", "No internet service"],
                "description": "Whether customer has streaming TV"
            },
            "streaming_movies": {
                "type": "string",
                "enum": ["Yes", "No", "No internet service"], 
                "description": "Whether customer has streaming movies"
            },
            "contract": {
                "type": "string",
                "enum": ["Month-to-month", "One year", "Two year"],
                "description": "Contract type"
            },
            "paperless_billing": {
                "type": "string",
                "enum": ["Yes", "No"],
                "description": "Whether customer uses paperless billing"
            },
            "payment_method": {
                "type": "string", 
                "enum": ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
                "description": "Payment method used"
            },
            "monthly_charges": {
                "type": "number",
                "minimum": 0,
                "maximum": 200,
                "description": "Monthly charges amount"
            },
            "total_charges": {
                "type": "number",
                "minimum": 0,
                "description": "Total charges amount"
            },
            "churn": {
                "type": "integer",
                "enum": [0, 1],
                "description": "Whether customer churned (1) or not (0)"
            }
        },
        "required": [
            "customer_id", "gender", "senior_citizen", "partner", "dependents",
            "tenure", "phone_service", "multiple_lines", "internet_service",
            "online_security", "online_backup", "device_protection", "tech_support",
            "streaming_tv", "streaming_movies", "contract", "paperless_billing",
            "payment_method", "monthly_charges", "total_charges", "churn"
        ],
        "additionalProperties": False
    }


@pytest.fixture(scope="function")
def minimal_schema() -> Dict[str, Any]:
    """Minimal schema for basic validation testing."""
    return {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "age": {"type": "integer", "minimum": 18, "maximum": 120},
            "tenure": {"type": "integer", "minimum": 0},
            "monthly_charges": {"type": "number", "minimum": 0},
            "total_charges": {"type": "number", "minimum": 0},
            "churn": {"type": "integer", "enum": [0, 1]}
        },
        "required": ["customer_id", "age", "tenure", "monthly_charges", "total_charges", "churn"]
    }