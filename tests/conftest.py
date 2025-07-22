"""
Test configuration and fixtures for the Customer Churn Detection MVP pipeline.
Provides shared fixtures, configuration, and utilities for all test modules.
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any
import pytest
import httpx
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"

# Import after environment setup
from tracer.core.config import Settings
from tracer.api.main import create_app

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test-specific configuration settings."""
    return Settings(
        environment="test",
        log_level="DEBUG",
        database_url="sqlite:///:memory:",
        enable_cors=True,
        debug=True
    )

@pytest.fixture(scope="session")
def app(test_settings: Settings):
    """Create FastAPI application instance for testing."""
    return create_app(settings=test_settings)

@pytest.fixture(scope="session")
def client(app) -> TestClient:
    """Create test client for synchronous API testing."""
    return TestClient(app)

@pytest.fixture(scope="session")
async def async_client(app) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async test client for async API testing."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="function")
def sample_csv_data() -> Dict[str, Any]:
    """Sample customer data for testing CSV processing."""
    return {
        'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
        'age': [25, 35, 45, 30, 28],
        'tenure': [12, 24, 36, 18, 15],
        'monthly_charges': [50.0, 75.5, 89.99, 65.25, 55.0],
        'total_charges': [600.0, 1812.0, 3239.64, 1174.5, 825.0],
        'contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
        'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check', 'Credit card'],
        'churn': [1, 0, 0, 1, 0]
    }

@pytest.fixture(scope="function")
def valid_csv_file(temp_dir: Path, sample_csv_data: Dict[str, Any]) -> Path:
    """Create a valid CSV file for testing."""
    df = pd.DataFrame(sample_csv_data)
    csv_path = temp_dir / "valid_customer_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture(scope="function")
def invalid_csv_file(temp_dir: Path) -> Path:
    """Create an invalid CSV file for testing error handling."""
    csv_path = temp_dir / "invalid_customer_data.csv"
    with open(csv_path, 'w') as f:
        f.write("invalid,csv,content\n")
        f.write("missing,columns\n")  # Inconsistent column count
        f.write("bad,data,here,extra\n")
    return csv_path

@pytest.fixture(scope="function")
def large_csv_file(temp_dir: Path) -> Path:
    """Create a large CSV file for performance testing."""
    np.random.seed(42)  # For reproducible data
    
    # Generate 10,000 rows of test data
    n_rows = 10000
    data = {
        'customer_id': [f'CUST{i:06d}' for i in range(n_rows)],
        'age': np.random.randint(18, 80, n_rows),
        'tenure': np.random.randint(1, 72, n_rows),
        'monthly_charges': np.random.uniform(20.0, 120.0, n_rows),
        'total_charges': np.random.uniform(20.0, 8000.0, n_rows),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n_rows),
        'churn': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])  # 30% churn rate
    }
    
    df = pd.DataFrame(data)
    csv_path = temp_dir / "large_customer_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture(scope="function")
def malformed_csv_file(temp_dir: Path) -> Path:
    """Create a CSV file with various data quality issues."""
    csv_path = temp_dir / "malformed_customer_data.csv"
    with open(csv_path, 'w') as f:
        f.write("customer_id,age,tenure,monthly_charges,total_charges,contract,payment_method,churn\n")
        f.write("CUST001,25,12,50.0,600.0,Month-to-month,Credit card,1\n")  # Valid row
        f.write("CUST002,,24,75.5,1812.0,One year,Bank transfer,0\n")  # Missing age
        f.write("CUST003,45,36,invalid_amount,3239.64,Two year,Electronic check,0\n")  # Invalid amount
        f.write("CUST004,30,18,65.25,,Month-to-month,Mailed check,1\n")  # Missing total_charges
        f.write("CUST005,28,15,55.0,825.0,Invalid contract,Credit card,2\n")  # Invalid contract and churn value
    return csv_path

@pytest.fixture(scope="function")
def prediction_request_payload() -> Dict[str, Any]:
    """Sample prediction request payload."""
    return {
        "customer_data": {
            "age": 35,
            "tenure": 24,
            "monthly_charges": 75.5,
            "total_charges": 1812.0,
            "contract": "One year",
            "payment_method": "Bank transfer"
        }
    }

@pytest.fixture(scope="function")
def batch_prediction_payload() -> Dict[str, Any]:
    """Sample batch prediction request payload."""
    return {
        "customers": [
            {
                "customer_id": "CUST001",
                "age": 25,
                "tenure": 12,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "contract": "Month-to-month",
                "payment_method": "Credit card"
            },
            {
                "customer_id": "CUST002",
                "age": 35,
                "tenure": 24,
                "monthly_charges": 75.5,
                "total_charges": 1812.0,
                "contract": "One year",
                "payment_method": "Bank transfer"
            }
        ]
    }

@pytest.fixture(scope="function")
def mock_trained_model():
    """Mock trained model for testing predictions."""
    from unittest.mock import Mock
    
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.25])  # 25% churn probability
    mock_model.predict_proba.return_value = np.array([[0.75, 0.25]])  # [no_churn, churn]
    return mock_model

@pytest.fixture(scope="function")
def performance_test_data():
    """Generate data for performance testing."""
    return {
        'response_time_threshold': 2.0,  # seconds
        'memory_threshold': 100,  # MB
        'concurrent_requests': 10,
        'request_timeout': 30.0
    }

# Test markers
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_customer_data(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic customer data for testing."""
        np.random.seed(seed)
        
        data = {
            'customer_id': [f'CUST{i:06d}' for i in range(n_rows)],
            'age': np.random.randint(18, 80, n_rows),
            'tenure': np.random.randint(1, 72, n_rows),
            'monthly_charges': np.random.uniform(20.0, 120.0, n_rows),
            'total_charges': np.random.uniform(20.0, 8000.0, n_rows),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
            'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n_rows),
            'churn': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_csv_with_issues(file_path: Path, issue_type: str):
        """Create CSV files with specific data quality issues."""
        if issue_type == "encoding":
            # Create file with encoding issues
            with open(file_path, 'w', encoding='latin-1') as f:
                f.write("customer_id,name,age\n")
                f.write("CUST001,José García,25\n")
                f.write("CUST002,François Müller,35\n")
        elif issue_type == "delimiter":
            # Create file with inconsistent delimiters
            with open(file_path, 'w') as f:
                f.write("customer_id|age|tenure\n")
                f.write("CUST001,25,12\n")  # Wrong delimiter in data
                f.write("CUST002|35|24\n")
        elif issue_type == "quotes":
            # Create file with quote issues
            with open(file_path, 'w') as f:
                f.write('customer_id,description,age\n')
                f.write('CUST001,"Premium customer with "special" status",25\n')
                f.write('CUST002,Regular customer,35\n')

# Export test utilities
__all__ = [
    "TestDataGenerator"
]