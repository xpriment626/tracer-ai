#!/usr/bin/env python3
"""
Quick server startup script for development
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run
from tracer.main import run_server

if __name__ == "__main__":
    run_server()