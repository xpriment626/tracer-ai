"""
Main application entry point for Tracer AI
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from tracer import initialize_tracer, settings
from tracer.api.endpoints import app
from tracer.core.logging_config import setup_logging
import structlog

# Initialize the application
initialize_tracer()
logger = structlog.get_logger(__name__)


def run_server():
    """Run the FastAPI server"""
    logger.info("Starting Tracer AI server",
               host=settings.api.host,
               port=settings.api.port,
               environment=settings.environment)
    
    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": settings.api.host,
        "port": settings.api.port,
        "log_level": settings.logging.log_level.lower(),
        "access_log": settings.logging.enable_request_logging,
        "reload": settings.api.reload and not settings.is_production(),
        "workers": settings.api.workers if settings.is_production() else 1
    }
    
    # Run server
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    run_server()