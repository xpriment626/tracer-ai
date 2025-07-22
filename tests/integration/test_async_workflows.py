"""
Integration tests for asynchronous file upload and processing workflows.

Tests async file processing, background job management, webhook notifications,
and long-running task handling for the customer churn detection pipeline.
"""
import pytest
import asyncio
import httpx
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import time
import json

from tracer.core.async_processing import (
    FileProcessingQueue,
    BackgroundTaskManager,
    WebhookNotifier,
    AsyncFileProcessor
)

class TestAsyncFileProcessing:
    """Test asynchronous file processing workflows."""
    
    @pytest.fixture
    def file_processor(self):
        """Create AsyncFileProcessor instance."""
        return AsyncFileProcessor()
    
    @pytest.fixture
    def processing_queue(self):
        """Create FileProcessingQueue instance."""
        return FileProcessingQueue(max_workers=3, queue_size=100)
    
    @pytest.mark.asyncio
    async def test_async_csv_processing(self, file_processor: AsyncFileProcessor, valid_csv_file: Path):
        """Test asynchronous CSV file processing."""
        # Mock the processing steps
        with patch.object(file_processor, '_validate_csv') as mock_validate:
            with patch.object(file_processor, '_process_data') as mock_process:
                mock_validate.return_value = {"valid": True, "row_count": 100}
                mock_process.return_value = {"processed": True, "features_extracted": 50}
                
                result = await file_processor.process_csv_async(valid_csv_file)
                
                assert result["success"] is True
                assert result["validation"]["valid"] is True
                assert result["processing"]["processed"] is True
                mock_validate.assert_called_once()
                mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_processing_with_progress_callback(self, file_processor: AsyncFileProcessor, valid_csv_file: Path):
        """Test async processing with progress tracking."""
        progress_updates = []
        
        async def progress_callback(progress: Dict[str, Any]):
            progress_updates.append(progress)
        
        with patch.object(file_processor, '_validate_csv') as mock_validate:
            with patch.object(file_processor, '_process_data') as mock_process:
                # Mock progressive updates
                mock_validate.return_value = {"valid": True, "row_count": 100}
                mock_process.return_value = {"processed": True}
                
                result = await file_processor.process_csv_async(
                    valid_csv_file, 
                    progress_callback=progress_callback
                )
                
                assert result["success"] is True
                assert len(progress_updates) >= 2  # At least start and complete
                assert progress_updates[0]["stage"] in ["validation", "processing"]
                assert progress_updates[-1]["stage"] == "completed"
    
    @pytest.mark.asyncio
    async def test_async_processing_error_handling(self, file_processor: AsyncFileProcessor, invalid_csv_file: Path):
        """Test async processing error handling."""
        with patch.object(file_processor, '_validate_csv') as mock_validate:
            mock_validate.side_effect = Exception("Validation failed")
            
            result = await file_processor.process_csv_async(invalid_csv_file)
            
            assert result["success"] is False
            assert "error" in result
            assert "Validation failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self, file_processor: AsyncFileProcessor, temp_dir: Path):
        """Test processing multiple files concurrently."""
        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = temp_dir / f"test_file_{i}.csv"
            test_file.write_text("customer_id,age,churn\nCUST001,25,0\nCUST002,35,1\n")
            test_files.append(test_file)
        
        # Mock processing methods
        with patch.object(file_processor, '_validate_csv') as mock_validate:
            with patch.object(file_processor, '_process_data') as mock_process:
                mock_validate.return_value = {"valid": True, "row_count": 2}
                mock_process.return_value = {"processed": True}
                
                # Process files concurrently
                tasks = []
                for file_path in test_files:
                    task = file_processor.process_csv_async(file_path)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # All files should be processed successfully
                assert len(results) == 5
                assert all(isinstance(r, dict) and r["success"] for r in results)
    
    @pytest.mark.asyncio
    async def test_processing_queue_management(self, processing_queue: FileProcessingQueue):
        """Test file processing queue management."""
        # Create mock jobs
        async def mock_processing_job(file_path: Path) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"file": str(file_path), "processed": True}
        
        # Submit multiple jobs
        job_ids = []
        for i in range(10):
            file_path = Path(f"/mock/file_{i}.csv")
            job_id = await processing_queue.submit_job(file_path, mock_processing_job)
            job_ids.append(job_id)
        
        # Wait for all jobs to complete
        results = []
        for job_id in job_ids:
            result = await processing_queue.get_result(job_id, timeout=5.0)
            results.append(result)
        
        assert len(results) == 10
        assert all(r["processed"] for r in results)
    
    @pytest.mark.asyncio
    async def test_processing_timeout_handling(self, processing_queue: FileProcessingQueue):
        """Test handling of processing timeouts."""
        # Create a slow job that will timeout
        async def slow_job(file_path: Path) -> Dict[str, Any]:
            await asyncio.sleep(10)  # Longer than timeout
            return {"processed": True}
        
        file_path = Path("/mock/slow_file.csv")
        job_id = await processing_queue.submit_job(file_path, slow_job)
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await processing_queue.get_result(job_id, timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_queue_capacity_limits(self, processing_queue: FileProcessingQueue):
        """Test queue capacity limits."""
        # Fill the queue to capacity
        async def dummy_job(file_path: Path) -> Dict[str, Any]:
            await asyncio.sleep(1)
            return {"processed": True}
        
        # Submit jobs up to queue capacity
        job_ids = []
        for i in range(processing_queue.queue_size):
            file_path = Path(f"/mock/file_{i}.csv")
            job_id = await processing_queue.submit_job(file_path, dummy_job)
            job_ids.append(job_id)
        
        # Next submission should raise an exception or be rejected
        with pytest.raises(Exception):  # Queue full exception
            file_path = Path("/mock/overflow_file.csv")
            await processing_queue.submit_job(file_path, dummy_job)


class TestBackgroundTaskManagement:
    """Test background task management."""
    
    @pytest.fixture
    def task_manager(self):
        """Create BackgroundTaskManager instance."""
        return BackgroundTaskManager()
    
    @pytest.mark.asyncio
    async def test_background_task_creation(self, task_manager: BackgroundTaskManager):
        """Test creating and managing background tasks."""
        task_results = []
        
        async def background_task(task_id: str, data: Dict[str, Any]):
            await asyncio.sleep(0.1)
            task_results.append({"task_id": task_id, "data": data})
            return {"completed": True}
        
        # Create background task
        task_id = await task_manager.create_task(
            "test_task",
            background_task,
            {"file_path": "/mock/file.csv"}
        )
        
        # Wait for task completion
        result = await task_manager.wait_for_task(task_id, timeout=5.0)
        
        assert result["completed"] is True
        assert len(task_results) == 1
        assert task_results[0]["task_id"] == task_id
    
    @pytest.mark.asyncio
    async def test_task_status_tracking(self, task_manager: BackgroundTaskManager):
        """Test task status tracking."""
        async def tracked_task(task_id: str, data: Dict[str, Any]):
            await task_manager.update_task_status(task_id, "processing", {"progress": 0.5})
            await asyncio.sleep(0.1)
            await task_manager.update_task_status(task_id, "finalizing", {"progress": 0.9})
            return {"completed": True}
        
        task_id = await task_manager.create_task("tracked_task", tracked_task, {})
        
        # Check status updates
        await asyncio.sleep(0.05)  # Wait for task to start
        status = await task_manager.get_task_status(task_id)
        
        assert status["status"] in ["processing", "finalizing", "completed"]
        if "progress" in status:
            assert 0 <= status["progress"] <= 1
        
        # Wait for completion
        result = await task_manager.wait_for_task(task_id, timeout=5.0)
        assert result["completed"] is True
        
        # Final status should be completed
        final_status = await task_manager.get_task_status(task_id)
        assert final_status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_manager: BackgroundTaskManager):
        """Test task cancellation."""
        async def long_running_task(task_id: str, data: Dict[str, Any]):
            try:
                await asyncio.sleep(10)  # Long running task
                return {"completed": True}
            except asyncio.CancelledError:
                await task_manager.update_task_status(task_id, "cancelled")
                raise
        
        task_id = await task_manager.create_task("cancellable_task", long_running_task, {})
        
        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        cancelled = await task_manager.cancel_task(task_id)
        
        assert cancelled is True
        
        # Check final status
        status = await task_manager.get_task_status(task_id)
        assert status["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self, task_manager: BackgroundTaskManager):
        """Test handling of task failures."""
        async def failing_task(task_id: str, data: Dict[str, Any]):
            raise ValueError("Task failed intentionally")
        
        task_id = await task_manager.create_task("failing_task", failing_task, {})
        
        # Task should fail
        with pytest.raises(ValueError):
            await task_manager.wait_for_task(task_id, timeout=5.0)
        
        # Status should reflect failure
        status = await task_manager.get_task_status(task_id)
        assert status["status"] == "failed"
        assert "error" in status
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self, task_manager: BackgroundTaskManager):
        """Test managing multiple concurrent tasks."""
        async def concurrent_task(task_id: str, data: Dict[str, Any]):
            await asyncio.sleep(0.1)
            return {"task_id": task_id, "result": data["value"] * 2}
        
        # Create multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await task_manager.create_task(
                f"concurrent_task_{i}",
                concurrent_task,
                {"value": i}
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            result = await task_manager.wait_for_task(task_id, timeout=5.0)
            results.append(result)
        
        assert len(results) == 5
        assert all(r["result"] == i * 2 for i, r in enumerate(results))


class TestWebhookNotifications:
    """Test webhook notification system."""
    
    @pytest.fixture
    def webhook_notifier(self):
        """Create WebhookNotifier instance."""
        return WebhookNotifier()
    
    @pytest.mark.asyncio
    async def test_successful_webhook_notification(self, webhook_notifier: WebhookNotifier):
        """Test successful webhook notification."""
        webhook_url = "https://example.com/webhook"
        payload = {
            "event_type": "file_processed",
            "file_id": "file_123",
            "status": "completed",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Mock successful HTTP response
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"received": True}
            mock_post.return_value = mock_response
            
            result = await webhook_notifier.send_notification(webhook_url, payload)
            
            assert result["success"] is True
            assert result["status_code"] == 200
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_webhook_retry_mechanism(self, webhook_notifier: WebhookNotifier):
        """Test webhook retry mechanism on failure."""
        webhook_url = "https://example.com/webhook"
        payload = {"event_type": "test"}
        
        # Mock failing then succeeding responses
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response_fail = Mock()
            mock_response_fail.status_code = 500
            mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error", request=Mock(), response=mock_response_fail
            )
            
            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {"received": True}
            
            # First call fails, second succeeds
            mock_post.side_effect = [mock_response_fail, mock_response_success]
            
            result = await webhook_notifier.send_notification(
                webhook_url, 
                payload, 
                max_retries=2,
                retry_delay=0.1
            )
            
            assert result["success"] is True
            assert result["attempts"] == 2
            assert mock_post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_webhook_timeout_handling(self, webhook_notifier: WebhookNotifier):
        """Test webhook timeout handling."""
        webhook_url = "https://example.com/webhook"
        payload = {"event_type": "test"}
        
        # Mock timeout
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")
            
            result = await webhook_notifier.send_notification(
                webhook_url, 
                payload, 
                timeout=1.0,
                max_retries=1
            )
            
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_multiple_webhook_destinations(self, webhook_notifier: WebhookNotifier):
        """Test sending notifications to multiple webhook URLs."""
        webhook_urls = [
            "https://webhook1.example.com",
            "https://webhook2.example.com",
            "https://webhook3.example.com"
        ]
        payload = {"event_type": "file_processed", "status": "completed"}
        
        # Mock successful responses for all URLs
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"received": True}
            mock_post.return_value = mock_response
            
            results = await webhook_notifier.send_to_multiple(webhook_urls, payload)
            
            assert len(results) == 3
            assert all(r["success"] for r in results)
            assert mock_post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_webhook_authentication(self, webhook_notifier: WebhookNotifier):
        """Test webhook with authentication headers."""
        webhook_url = "https://secure.example.com/webhook"
        payload = {"event_type": "test"}
        auth_headers = {
            "Authorization": "Bearer secret_token",
            "X-API-Key": "api_key_123"
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            await webhook_notifier.send_notification(
                webhook_url, 
                payload, 
                headers=auth_headers
            )
            
            # Verify headers were included
            call_args = mock_post.call_args
            assert "Authorization" in call_args[1]["headers"]
            assert "X-API-Key" in call_args[1]["headers"]
            assert call_args[1]["headers"]["Authorization"] == "Bearer secret_token"


class TestEndToEndAsyncWorkflows:
    """Test complete end-to-end async workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_file_processing_workflow(self, async_client: httpx.AsyncClient, valid_csv_file: Path):
        """Test complete async file processing workflow from upload to notification."""
        # Mock external dependencies
        with patch('tracer.core.ml_models.ChurnPredictor') as mock_predictor:
            with patch('tracer.core.async_processing.WebhookNotifier') as mock_notifier:
                # Setup mocks
                mock_predictor_instance = Mock()
                mock_predictor_instance.predict_batch.return_value = [
                    {"customer_id": "CUST001", "churn_probability": 0.25, "risk_category": "Low"},
                    {"customer_id": "CUST002", "churn_probability": 0.75, "risk_category": "High"}
                ]
                mock_predictor.return_value = mock_predictor_instance
                
                mock_notifier_instance = AsyncMock()
                mock_notifier_instance.send_notification.return_value = {"success": True}
                mock_notifier.return_value = mock_notifier_instance
                
                # Step 1: Upload file
                with open(valid_csv_file, 'rb') as f:
                    files = {"file": ("customer_data.csv", f, "text/csv")}
                    upload_response = await async_client.post("/api/v1/upload/csv", files=files)
                
                assert upload_response.status_code == 200
                upload_data = upload_response.json()
                upload_id = upload_data["upload_id"]
                
                # Step 2: Trigger async processing
                processing_config = {
                    "webhook_url": "https://example.com/webhook",
                    "include_predictions": True,
                    "notify_on_completion": True
                }
                
                process_response = await async_client.post(
                    f"/api/v1/process/async/{upload_id}",
                    json=processing_config
                )
                
                assert process_response.status_code == 202
                process_data = process_response.json()
                job_id = process_data["job_id"]
                
                # Step 3: Check processing status (simulate polling)
                max_attempts = 10
                for attempt in range(max_attempts):
                    status_response = await async_client.get(f"/api/v1/process/status/{job_id}")
                    assert status_response.status_code == 200
                    
                    status_data = status_response.json()
                    if status_data["status"] == "completed":
                        break
                    elif status_data["status"] == "failed":
                        pytest.fail(f"Processing failed: {status_data.get('error')}")
                    
                    await asyncio.sleep(0.1)  # Wait before next check
                
                # Should have completed within max attempts
                assert status_data["status"] == "completed"
                
                # Step 4: Verify results
                results_response = await async_client.get(f"/api/v1/process/results/{job_id}")
                assert results_response.status_code == 200
                
                results_data = results_response.json()
                assert "predictions" in results_data
                assert len(results_data["predictions"]) > 0
                
                # Verify webhook was called
                mock_notifier_instance.send_notification.assert_called()
    
    @pytest.mark.asyncio
    async def test_batch_file_processing(self, async_client: httpx.AsyncClient, temp_dir: Path):
        """Test processing multiple files in batch."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = temp_dir / f"batch_file_{i}.csv"
            test_file.write_text(
                "customer_id,age,tenure,monthly_charges,churn\n"
                f"CUST{i:03d},25,12,50.0,0\n"
                f"CUST{i+100:03d},35,24,75.0,1\n"
            )
            test_files.append(test_file)
        
        # Upload all files
        upload_ids = []
        for test_file in test_files:
            with open(test_file, 'rb') as f:
                files = {"file": (test_file.name, f, "text/csv")}
                response = await async_client.post("/api/v1/upload/csv", files=files)
                assert response.status_code == 200
                upload_ids.append(response.json()["upload_id"])
        
        # Start batch processing
        batch_config = {
            "upload_ids": upload_ids,
            "processing_options": {
                "include_validation": True,
                "include_predictions": True
            }
        }
        
        batch_response = await async_client.post("/api/v1/process/batch", json=batch_config)
        assert batch_response.status_code == 202
        
        batch_job_id = batch_response.json()["batch_job_id"]
        
        # Wait for batch completion
        max_attempts = 20
        for attempt in range(max_attempts):
            status_response = await async_client.get(f"/api/v1/process/batch/{batch_job_id}/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            if status_data["status"] == "completed":
                break
            
            await asyncio.sleep(0.1)
        
        assert status_data["status"] == "completed"
        assert status_data["total_files"] == 3
        assert status_data["completed_files"] == 3
    
    @pytest.mark.asyncio
    async def test_processing_with_error_recovery(self, async_client: httpx.AsyncClient, malformed_csv_file: Path):
        """Test error recovery in async processing."""
        # Upload malformed file
        with open(malformed_csv_file, 'rb') as f:
            files = {"file": ("malformed.csv", f, "text/csv")}
            upload_response = await async_client.post("/api/v1/upload/csv", files=files)
        
        upload_id = upload_response.json()["upload_id"]
        
        # Configure processing with error handling
        processing_config = {
            "error_handling": "continue",  # Continue on errors
            "validation_level": "strict",
            "cleanup_data": True
        }
        
        process_response = await async_client.post(
            f"/api/v1/process/async/{upload_id}",
            json=processing_config
        )
        
        assert process_response.status_code == 202
        job_id = process_response.json()["job_id"]
        
        # Wait for processing to complete
        max_attempts = 10
        for attempt in range(max_attempts):
            status_response = await async_client.get(f"/api/v1/process/status/{job_id}")
            status_data = status_response.json()
            
            if status_data["status"] in ["completed", "completed_with_warnings"]:
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Processing failed: {status_data.get('error')}")
            
            await asyncio.sleep(0.1)
        
        # Should complete with warnings due to data quality issues
        assert status_data["status"] in ["completed", "completed_with_warnings"]
        if "warnings" in status_data:
            assert len(status_data["warnings"]) > 0


@pytest.mark.performance
class TestAsyncPerformance:
    """Performance tests for async workflows."""
    
    @pytest.mark.asyncio
    async def test_concurrent_file_uploads(self, async_client: httpx.AsyncClient, temp_dir: Path, performance_test_data: Dict):
        """Test concurrent file uploads performance."""
        # Create test files
        test_files = []
        for i in range(performance_test_data['concurrent_requests']):
            test_file = temp_dir / f"concurrent_file_{i}.csv"
            test_file.write_text("customer_id,age,churn\nCUST001,25,0\nCUST002,35,1\n")
            test_files.append(test_file)
        
        start_time = time.time()
        
        # Upload files concurrently
        tasks = []
        for test_file in test_files:
            async def upload_file(file_path):
                with open(file_path, 'rb') as f:
                    files = {"file": (file_path.name, f, "text/csv")}
                    return await async_client.post("/api/v1/upload/csv", files=files)
            
            tasks.append(upload_file(test_file))
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All uploads should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Should complete within reasonable time
        assert total_time < performance_test_data['response_time_threshold']
        
        # Calculate throughput
        throughput = len(test_files) / total_time
        assert throughput >= 5  # At least 5 uploads per second
    
    @pytest.mark.asyncio
    async def test_async_processing_scalability(self, processing_queue: FileProcessingQueue, temp_dir: Path):
        """Test async processing scalability with many files."""
        # Create many small files
        num_files = 50
        test_files = []
        
        for i in range(num_files):
            test_file = temp_dir / f"scale_test_{i}.csv"
            test_file.write_text("customer_id,age\nCUST001,25\nCUST002,35\n")
            test_files.append(test_file)
        
        # Process all files
        async def simple_processor(file_path: Path) -> Dict[str, Any]:
            # Simulate minimal processing
            await asyncio.sleep(0.01)  # 10ms processing time
            return {"file": str(file_path), "processed": True}
        
        start_time = time.time()
        
        # Submit all jobs
        job_ids = []
        for file_path in test_files:
            job_id = await processing_queue.submit_job(file_path, simple_processor)
            job_ids.append(job_id)
        
        # Wait for all to complete
        results = []
        for job_id in job_ids:
            result = await processing_queue.get_result(job_id, timeout=30.0)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(results) == num_files
        assert all(r["processed"] for r in results)
        
        # Should scale efficiently (not linear with file count due to concurrency)
        max_expected_time = (num_files * 0.01) / processing_queue.max_workers * 2  # 2x buffer
        assert total_time < max_expected_time