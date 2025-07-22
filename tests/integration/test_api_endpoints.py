"""
Integration tests for API endpoints.

Tests complete request-response cycles, authentication, validation,
and error handling for all customer churn detection API endpoints.
"""
import pytest
import httpx
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, Mock
import pandas as pd

from fastapi.testclient import TestClient
from tracer.api.main import create_app

class TestHealthCheckEndpoints:
    """Test health check and status endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()
        assert "version" in response.json()
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness check endpoint."""
        response = client.get("/ready")
        
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert "services" in response.json()
    
    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint for monitoring."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus-style metrics
        assert "http_requests_total" in response.text


class TestFileUploadEndpoints:
    """Test file upload and processing endpoints."""
    
    def test_upload_valid_csv_file(self, client: TestClient, valid_csv_file: Path):
        """Test uploading a valid CSV file."""
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            response = client.post("/api/v1/upload/csv", files=files)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "upload_id" in response_data
        assert "row_count" in response_data["metadata"]
        assert "column_count" in response_data["metadata"]
        assert response_data["metadata"]["row_count"] > 0
    
    def test_upload_invalid_file_format(self, client: TestClient, temp_dir: Path):
        """Test uploading invalid file format."""
        # Create a text file instead of CSV
        text_file = temp_dir / "not_csv.txt"
        text_file.write_text("This is not a CSV file")
        
        with open(text_file, 'rb') as f:
            files = {"file": ("not_csv.txt", f, "text/plain")}
            response = client.post("/api/v1/upload/csv", files=files)
        
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]
    
    def test_upload_empty_file(self, client: TestClient, temp_dir: Path):
        """Test uploading empty file."""
        empty_file = temp_dir / "empty.csv"
        empty_file.touch()
        
        with open(empty_file, 'rb') as f:
            files = {"file": ("empty.csv", f, "text/csv")}
            response = client.post("/api/v1/upload/csv", files=files)
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_upload_file_too_large(self, client: TestClient):
        """Test uploading file that exceeds size limit."""
        # Mock a large file by patching file size check
        with patch('tracer.api.endpoints.upload.get_file_size') as mock_size:
            mock_size.return_value = 100 * 1024 * 1024  # 100MB
            
            files = {"file": ("large.csv", b"fake,csv,content", "text/csv")}
            response = client.post("/api/v1/upload/csv", files=files)
            
            assert response.status_code == 413
            assert "too large" in response.json()["detail"].lower()
    
    def test_upload_malformed_csv(self, client: TestClient, malformed_csv_file: Path):
        """Test uploading malformed CSV file."""
        with open(malformed_csv_file, 'rb') as f:
            files = {"file": ("malformed.csv", f, "text/csv")}
            response = client.post("/api/v1/upload/csv", files=files)
        
        assert response.status_code == 200  # Should accept but warn
        response_data = response.json()
        assert response_data["success"] is True
        assert len(response_data["warnings"]) > 0
        assert any("data quality" in warning.lower() for warning in response_data["warnings"])
    
    def test_get_upload_status(self, client: TestClient, valid_csv_file: Path):
        """Test getting upload status."""
        # First upload a file
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Get status
        status_response = client.get(f"/api/v1/upload/{upload_id}/status")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["upload_id"] == upload_id
        assert "status" in status_data
        assert "created_at" in status_data
    
    def test_get_nonexistent_upload_status(self, client: TestClient):
        """Test getting status for non-existent upload."""
        fake_id = "non-existent-upload-id"
        response = client.get(f"/api/v1/upload/{fake_id}/status")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDataValidationEndpoints:
    """Test data validation endpoints."""
    
    def test_validate_csv_schema(self, client: TestClient, valid_csv_file: Path):
        """Test CSV schema validation."""
        # First upload file
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Validate schema
        validation_response = client.post(f"/api/v1/validate/{upload_id}/schema")
        
        assert validation_response.status_code == 200
        validation_data = validation_response.json()
        assert validation_data["valid"] is True
        assert "schema_errors" in validation_data
        assert len(validation_data["schema_errors"]) == 0
    
    def test_validate_data_quality(self, client: TestClient, valid_csv_file: Path):
        """Test data quality validation."""
        # Upload file first
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Validate data quality
        quality_response = client.post(f"/api/v1/validate/{upload_id}/quality")
        
        assert quality_response.status_code == 200
        quality_data = quality_response.json()
        assert "missing_values" in quality_data
        assert "duplicates" in quality_data
        assert "outliers" in quality_data
        assert "data_types" in quality_data
    
    def test_custom_validation_rules(self, client: TestClient, valid_csv_file: Path):
        """Test custom validation rules."""
        # Upload file first
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Define custom validation rules
        custom_rules = {
            "age": {"min": 18, "max": 120},
            "tenure": {"min": 0, "max": 120},
            "monthly_charges": {"min": 0},
            "churn": {"enum": [0, 1]}
        }
        
        validation_response = client.post(
            f"/api/v1/validate/{upload_id}/custom",
            json={"rules": custom_rules}
        )
        
        assert validation_response.status_code == 200
        validation_data = validation_response.json()
        assert "validation_results" in validation_data
        assert "rules_applied" in validation_data


class TestPredictionEndpoints:
    """Test machine learning prediction endpoints."""
    
    @pytest.fixture
    def mock_model_predictor(self):
        """Mock the model predictor."""
        with patch('tracer.core.ml_models.ChurnPredictor') as mock_predictor:
            mock_instance = Mock()
            mock_instance.predict_single.return_value = {
                "churn_probability": 0.35,
                "churn_prediction": 0,
                "risk_category": "Medium",
                "confidence": 0.82
            }
            mock_instance.predict_batch.return_value = [
                {
                    "customer_id": "CUST001",
                    "churn_probability": 0.25,
                    "churn_prediction": 0,
                    "risk_category": "Low",
                    "confidence": 0.85
                },
                {
                    "customer_id": "CUST002", 
                    "churn_probability": 0.75,
                    "churn_prediction": 1,
                    "risk_category": "High",
                    "confidence": 0.88
                }
            ]
            mock_predictor.return_value = mock_instance
            yield mock_predictor
    
    def test_single_prediction(self, client: TestClient, mock_model_predictor, prediction_request_payload: Dict[str, Any]):
        """Test single customer churn prediction."""
        response = client.post("/api/v1/predict/single", json=prediction_request_payload)
        
        assert response.status_code == 200
        prediction_data = response.json()
        assert "churn_probability" in prediction_data
        assert "churn_prediction" in prediction_data
        assert "risk_category" in prediction_data
        assert "confidence" in prediction_data
        assert 0 <= prediction_data["churn_probability"] <= 1
        assert prediction_data["churn_prediction"] in [0, 1]
        assert prediction_data["risk_category"] in ["Low", "Medium", "High", "Critical"]
    
    def test_batch_prediction(self, client: TestClient, mock_model_predictor, batch_prediction_payload: Dict[str, Any]):
        """Test batch customer churn prediction."""
        response = client.post("/api/v1/predict/batch", json=batch_prediction_payload)
        
        assert response.status_code == 200
        prediction_data = response.json()
        assert "predictions" in prediction_data
        assert len(prediction_data["predictions"]) == 2
        
        for pred in prediction_data["predictions"]:
            assert "customer_id" in pred
            assert "churn_probability" in pred
            assert "churn_prediction" in pred
            assert "risk_category" in pred
            assert "confidence" in pred
    
    def test_prediction_with_invalid_data(self, client: TestClient, mock_model_predictor):
        """Test prediction with invalid input data."""
        invalid_payload = {
            "customer_data": {
                "age": -5,  # Invalid age
                "tenure": "invalid",  # Invalid type
                # Missing required fields
            }
        }
        
        response = client.post("/api/v1/predict/single", json=invalid_payload)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
        assert len(error_data["detail"]) > 0
    
    def test_prediction_with_missing_model(self, client: TestClient):
        """Test prediction when model is not loaded."""
        with patch('tracer.core.ml_models.ChurnPredictor') as mock_predictor:
            mock_instance = Mock()
            mock_instance.predict_single.side_effect = ValueError("Model not loaded")
            mock_predictor.return_value = mock_instance
            
            prediction_payload = {
                "customer_data": {
                    "age": 35,
                    "tenure": 24,
                    "monthly_charges": 75.5,
                    "total_charges": 1812.0,
                    "contract": "One year",
                    "payment_method": "Bank transfer"
                }
            }
            
            response = client.post("/api/v1/predict/single", json=prediction_payload)
            
            assert response.status_code == 503  # Service unavailable
            assert "model" in response.json()["detail"].lower()
    
    def test_file_based_prediction(self, client: TestClient, mock_model_predictor, valid_csv_file: Path):
        """Test prediction on uploaded file."""
        # First upload file
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Predict on file
        prediction_response = client.post(f"/api/v1/predict/file/{upload_id}")
        
        assert prediction_response.status_code == 200
        prediction_data = prediction_response.json()
        assert "job_id" in prediction_data
        assert "status" in prediction_data
        assert prediction_data["status"] in ["queued", "processing", "completed"]
    
    def test_get_prediction_results(self, client: TestClient, mock_model_predictor, valid_csv_file: Path):
        """Test getting prediction results."""
        # Upload and predict
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        prediction_response = client.post(f"/api/v1/predict/file/{upload_id}")
        job_id = prediction_response.json()["job_id"]
        
        # Get results
        results_response = client.get(f"/api/v1/predict/results/{job_id}")
        
        # Should return status (might be processing or completed)
        assert results_response.status_code in [200, 202]
        results_data = results_response.json()
        assert "status" in results_data
        assert "job_id" in results_data


class TestModelManagementEndpoints:
    """Test model management endpoints."""
    
    def test_list_available_models(self, client: TestClient):
        """Test listing available models."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        models_data = response.json()
        assert "models" in models_data
        assert isinstance(models_data["models"], list)
    
    def test_get_model_info(self, client: TestClient):
        """Test getting model information."""
        # Mock model info
        with patch('tracer.core.ml_models.ModelPersistence.load_model_metadata') as mock_metadata:
            mock_metadata.return_value = {
                "name": "churn_model_v1",
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z",
                "metrics": {"accuracy": 0.85, "f1_score": 0.82},
                "features": ["age", "tenure", "monthly_charges"]
            }
            
            response = client.get("/api/v1/models/churn_model_v1/1.0.0")
            
            assert response.status_code == 200
            model_info = response.json()
            assert model_info["name"] == "churn_model_v1"
            assert model_info["version"] == "1.0.0"
            assert "metrics" in model_info
            assert "features" in model_info
    
    def test_get_nonexistent_model_info(self, client: TestClient):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/models/nonexistent/1.0.0")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_model_training_trigger(self, client: TestClient, valid_csv_file: Path):
        """Test triggering model training."""
        # Upload training data first
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("training_data.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Trigger training
        training_config = {
            "model_type": "xgboost",
            "hyperparameters": {
                "max_depth": 5,
                "learning_rate": 0.1,
                "n_estimators": 100
            },
            "validation_split": 0.2
        }
        
        response = client.post(
            f"/api/v1/models/train/{upload_id}",
            json=training_config
        )
        
        assert response.status_code == 202  # Accepted for processing
        training_data = response.json()
        assert "training_job_id" in training_data
        assert "status" in training_data
        assert training_data["status"] == "queued"
    
    def test_get_training_status(self, client: TestClient):
        """Test getting training job status."""
        # Mock training job
        fake_job_id = "train_job_123"
        
        with patch('tracer.core.training.get_training_status') as mock_status:
            mock_status.return_value = {
                "job_id": fake_job_id,
                "status": "training",
                "progress": 0.65,
                "current_metrics": {"accuracy": 0.82},
                "estimated_completion": "2024-01-01T01:00:00Z"
            }
            
            response = client.get(f"/api/v1/models/training/{fake_job_id}/status")
            
            assert response.status_code == 200
            status_data = response.json()
            assert status_data["job_id"] == fake_job_id
            assert status_data["status"] == "training"
            assert "progress" in status_data


class TestErrorHandlingAndAuthentication:
    """Test error handling and authentication."""
    
    def test_404_error_handling(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent/endpoint")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
        assert "not found" in error_data["detail"].lower()
    
    def test_405_method_not_allowed(self, client: TestClient):
        """Test 405 error for unsupported methods."""
        response = client.patch("/api/v1/health")  # PATCH not supported
        
        assert response.status_code == 405
        error_data = response.json()
        assert "method not allowed" in error_data["detail"].lower()
    
    def test_rate_limiting(self, client: TestClient):
        """Test rate limiting functionality."""
        # Make many requests quickly
        endpoint = "/health"
        responses = []
        
        for _ in range(100):  # Exceed rate limit
            response = client.get(endpoint)
            responses.append(response.status_code)
        
        # Should eventually get rate limited (429)
        assert any(status == 429 for status in responses[-10:])  # Check last 10 requests
    
    def test_request_timeout_handling(self, client: TestClient):
        """Test request timeout handling."""
        # Mock slow endpoint
        with patch('tracer.api.endpoints.upload.process_csv_file') as mock_process:
            import time
            mock_process.side_effect = lambda *args: time.sleep(10)  # Simulate slow processing
            
            files = {"file": ("test.csv", b"col1,col2\nval1,val2", "text/csv")}
            
            # This should timeout (depending on timeout configuration)
            response = client.post("/api/v1/upload/csv", files=files)
            
            # Might be 408 (timeout) or 200 if timeout is high
            assert response.status_code in [200, 408, 500]
    
    def test_invalid_json_payload(self, client: TestClient):
        """Test handling of invalid JSON payloads."""
        response = client.post(
            "/api/v1/predict/single",
            content="invalid json{",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "json" in error_data["detail"][0]["msg"].lower()
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health")
        
        assert response.status_code == 200
        # Should have CORS headers in test environment
        assert "access-control-allow-origin" in response.headers


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test asynchronous endpoint functionality."""
    
    async def test_async_file_upload(self, async_client: httpx.AsyncClient, valid_csv_file: Path):
        """Test asynchronous file upload."""
        with open(valid_csv_file, 'rb') as f:
            files = {"file": ("customer_data.csv", f, "text/csv")}
            response = await async_client.post("/api/v1/upload/csv", files=files)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] is True
        assert "upload_id" in response_data
    
    async def test_async_prediction(self, async_client: httpx.AsyncClient, prediction_request_payload: Dict[str, Any]):
        """Test asynchronous prediction."""
        with patch('tracer.core.ml_models.ChurnPredictor') as mock_predictor:
            mock_instance = Mock()
            mock_instance.predict_single.return_value = {
                "churn_probability": 0.35,
                "churn_prediction": 0,
                "risk_category": "Medium"
            }
            mock_predictor.return_value = mock_instance
            
            response = await async_client.post("/api/v1/predict/single", json=prediction_request_payload)
            
            assert response.status_code == 200
            prediction_data = response.json()
            assert "churn_probability" in prediction_data
    
    async def test_concurrent_requests(self, async_client: httpx.AsyncClient):
        """Test handling concurrent requests."""
        import asyncio
        
        # Make multiple concurrent requests
        tasks = []
        for i in range(10):
            task = async_client.get("/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)
        assert all(response.json()["status"] == "healthy" for response in responses)


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    def test_health_check_performance(self, client: TestClient, performance_test_data: Dict):
        """Test health check endpoint performance."""
        import time
        
        start_time = time.time()
        
        # Make multiple requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        end_time = time.time()
        avg_response_time = (end_time - start_time) / 100
        
        assert avg_response_time < 0.1  # Less than 100ms average
    
    def test_upload_performance(self, client: TestClient, large_csv_file: Path, performance_test_data: Dict):
        """Test upload endpoint performance with large files."""
        import time
        
        start_time = time.time()
        
        with open(large_csv_file, 'rb') as f:
            files = {"file": ("large_data.csv", f, "text/csv")}
            response = client.post("/api/v1/upload/csv", files=files)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert response.status_code == 200
        assert processing_time < performance_test_data['response_time_threshold']
    
    def test_prediction_throughput(self, client: TestClient, prediction_request_payload: Dict[str, Any]):
        """Test prediction endpoint throughput."""
        with patch('tracer.core.ml_models.ChurnPredictor') as mock_predictor:
            mock_instance = Mock()
            mock_instance.predict_single.return_value = {
                "churn_probability": 0.35,
                "churn_prediction": 0,
                "risk_category": "Medium"
            }
            mock_predictor.return_value = mock_instance
            
            import time
            start_time = time.time()
            
            # Make multiple prediction requests
            successful_requests = 0
            for _ in range(performance_test_data['concurrent_requests']):
                response = client.post("/api/v1/predict/single", json=prediction_request_payload)
                if response.status_code == 200:
                    successful_requests += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = successful_requests / total_time
            
            # Should handle at least 10 requests per second
            assert throughput >= 10
            assert successful_requests == performance_test_data['concurrent_requests']