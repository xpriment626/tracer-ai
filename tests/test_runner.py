#!/usr/bin/env python3
"""
Test runner script for Customer Churn Detection MVP testing framework.

Provides convenient commands for running different test suites with proper
configuration and reporting. Supports various test categories and environments.
"""
import sys
import argparse
import subprocess
import os
from pathlib import Path
from typing import List, Optional
import time

class TestRunner:
    """Main test runner class."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n🚀 {description}")
        print(f"Command: {' '.join(command)}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                capture_output=False
            )
            
            elapsed = time.time() - start_time
            print(f"\n✅ {description} completed successfully in {elapsed:.2f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {description} failed after {elapsed:.2f}s")
            print(f"Exit code: {e.returncode}")
            return False
    
    def run_unit_tests(self, coverage: bool = True, verbose: bool = False) -> bool:
        """Run unit tests."""
        command = ["python", "-m", "pytest", "tests/unit/"]
        
        if coverage:
            command.extend(["--cov=src/tracer", "--cov-report=term-missing"])
        
        if verbose:
            command.append("-v")
        
        command.extend(["-m", "unit"])
        
        return self.run_command(command, "Running Unit Tests")
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        command = ["python", "-m", "pytest", "tests/integration/"]
        
        if verbose:
            command.append("-v")
        
        command.extend(["-m", "integration"])
        
        return self.run_command(command, "Running Integration Tests")
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests."""
        command = ["python", "-m", "pytest"]
        
        if verbose:
            command.append("-v")
        
        command.extend(["-m", "performance", "--durations=0"])
        
        return self.run_command(command, "Running Performance Tests")
    
    def run_security_tests(self, verbose: bool = False) -> bool:
        """Run security tests."""
        command = ["python", "-m", "pytest"]
        
        if verbose:
            command.append("-v")
        
        command.extend(["-m", "security"])
        
        return self.run_command(command, "Running Security Tests")
    
    def run_data_quality_tests(self, verbose: bool = False) -> bool:
        """Run data quality validation tests."""
        command = ["python", "-m", "pytest"]
        
        if verbose:
            command.append("-v")
        
        command.extend(["-m", "data_quality"])
        
        return self.run_command(command, "Running Data Quality Tests")
    
    def run_all_tests(self, skip_slow: bool = False, coverage: bool = True) -> bool:
        """Run all test suites."""
        command = ["python", "-m", "pytest"]
        
        if skip_slow:
            command.extend(["-m", "not slow"])
        
        if coverage:
            command.extend([
                "--cov=src/tracer", 
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml"
            ])
        
        return self.run_command(command, "Running All Tests")
    
    def run_fast_tests(self) -> bool:
        """Run only fast tests (unit + quick integration)."""
        command = [
            "python", "-m", "pytest", 
            "tests/unit/", 
            "-m", "not slow and not performance",
            "--maxfail=5"  # Stop after 5 failures for quick feedback
        ]
        
        return self.run_command(command, "Running Fast Tests")
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests for basic functionality."""
        command = [
            "python", "-m", "pytest",
            "-k", "test_health or test_upload_valid or test_basic_prediction",
            "--maxfail=1"
        ]
        
        return self.run_command(command, "Running Smoke Tests")
    
    def run_regression_tests(self) -> bool:
        """Run regression test suite."""
        # Run critical path tests to catch regressions
        command = [
            "python", "-m", "pytest",
            "-m", "not performance and not slow",
            "--tb=short"
        ]
        
        return self.run_command(command, "Running Regression Tests")
    
    def lint_and_format_check(self) -> bool:
        """Run linting and format checks."""
        success = True
        
        # Black format check
        if not self.run_command(
            ["python", "-m", "black", "--check", "src/", "tests/"],
            "Checking Code Formatting (Black)"
        ):
            success = False
        
        # Isort import order check
        if not self.run_command(
            ["python", "-m", "isort", "--check-only", "src/", "tests/"],
            "Checking Import Order (Isort)"
        ):
            success = False
        
        # Flake8 linting
        if not self.run_command(
            ["python", "-m", "flake8", "src/", "tests/"],
            "Running Linter (Flake8)"
        ):
            success = False
        
        # MyPy type checking
        if not self.run_command(
            ["python", "-m", "mypy", "src/tracer/"],
            "Type Checking (MyPy)"
        ):
            success = False
        
        return success
    
    def generate_coverage_report(self) -> bool:
        """Generate detailed coverage report."""
        # Run tests with coverage
        if not self.run_command(
            [
                "python", "-m", "pytest", 
                "--cov=src/tracer",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-report=json:coverage.json",
                "-q"  # Quiet mode for coverage generation
            ],
            "Generating Coverage Report"
        ):
            return False
        
        print(f"\n📊 Coverage reports generated:")
        print(f"  • HTML: {self.project_root}/htmlcov/index.html")
        print(f"  • XML:  {self.project_root}/coverage.xml")
        print(f"  • JSON: {self.project_root}/coverage.json")
        
        return True
    
    def run_test_discovery(self) -> bool:
        """Run test discovery to validate test structure."""
        command = [
            "python", "-m", "pytest", 
            "--collect-only", 
            "--quiet"
        ]
        
        return self.run_command(command, "Test Discovery")
    
    def validate_test_environment(self) -> bool:
        """Validate test environment setup."""
        print("\n🔍 Validating Test Environment")
        print("=" * 60)
        
        # Check Python version
        python_version = sys.version_info
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        required_packages = [
            "pytest", "pytest-asyncio", "pytest-cov", "httpx", 
            "pandas", "numpy", "fastapi", "uvicorn"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✓ {package}")
            except ImportError:
                print(f"✗ {package} (missing)")
                missing_packages.append(package)
        
        # Check test directories
        test_dirs = ["unit", "integration", "fixtures"]
        for dir_name in test_dirs:
            test_dir = self.tests_dir / dir_name
            if test_dir.exists():
                print(f"✓ tests/{dir_name}/")
            else:
                print(f"✗ tests/{dir_name}/ (missing)")
        
        # Check configuration files
        config_files = ["pytest.ini", "tests/conftest.py"]
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                print(f"✓ {config_file}")
            else:
                print(f"✗ {config_file} (missing)")
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("Run: pip install -r requirements.txt")
            return False
        
        print("\n✅ Test environment validation completed")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for Customer Churn Detection MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=[
            "unit", "integration", "performance", "security", 
            "data-quality", "all", "fast", "smoke", "regression",
            "lint", "coverage", "discover", "validate", "help"
        ],
        help="Test command to run"
    )
    
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="Skip coverage reporting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow running tests"
    )
    
    # Set up project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(project_root)
    
    # Handle help command
    if args.command == "help":
        print_usage_guide()
        return 0
    
    # Run selected command
    success = False
    
    if args.command == "unit":
        success = runner.run_unit_tests(
            coverage=not args.no_coverage, 
            verbose=args.verbose
        )
    elif args.command == "integration":
        success = runner.run_integration_tests(verbose=args.verbose)
    elif args.command == "performance":
        success = runner.run_performance_tests(verbose=args.verbose)
    elif args.command == "security":
        success = runner.run_security_tests(verbose=args.verbose)
    elif args.command == "data-quality":
        success = runner.run_data_quality_tests(verbose=args.verbose)
    elif args.command == "all":
        success = runner.run_all_tests(
            skip_slow=args.skip_slow,
            coverage=not args.no_coverage
        )
    elif args.command == "fast":
        success = runner.run_fast_tests()
    elif args.command == "smoke":
        success = runner.run_smoke_tests()
    elif args.command == "regression":
        success = runner.run_regression_tests()
    elif args.command == "lint":
        success = runner.lint_and_format_check()
    elif args.command == "coverage":
        success = runner.generate_coverage_report()
    elif args.command == "discover":
        success = runner.run_test_discovery()
    elif args.command == "validate":
        success = runner.validate_test_environment()
    
    return 0 if success else 1


def print_usage_guide():
    """Print detailed usage guide."""
    print("""
Customer Churn Detection MVP Test Runner
======================================

USAGE:
    python tests/test_runner.py [COMMAND] [OPTIONS]

COMMANDS:
    unit           Run unit tests only
    integration    Run integration tests only
    performance    Run performance tests only
    security       Run security tests only
    data-quality   Run data quality validation tests
    all            Run all test suites
    fast           Run only fast tests (unit + quick integration)
    smoke          Run smoke tests for basic functionality
    regression     Run regression test suite
    lint           Run linting and format checks
    coverage       Generate detailed coverage report
    discover       Run test discovery validation
    validate       Validate test environment setup
    help           Show this help message

OPTIONS:
    --no-coverage     Skip coverage reporting (faster execution)
    --verbose, -v     Enable verbose output
    --skip-slow       Skip slow running tests

EXAMPLES:
    # Quick feedback during development
    python tests/test_runner.py fast
    
    # Full test suite with coverage
    python tests/test_runner.py all
    
    # Unit tests with verbose output
    python tests/test_runner.py unit --verbose
    
    # Performance testing
    python tests/test_runner.py performance
    
    # Pre-commit validation
    python tests/test_runner.py lint
    python tests/test_runner.py fast
    
    # CI/CD pipeline
    python tests/test_runner.py validate
    python tests/test_runner.py all --skip-slow

WORKFLOW RECOMMENDATIONS:
    1. Development: python tests/test_runner.py fast
    2. Pre-commit: python tests/test_runner.py lint && python tests/test_runner.py unit
    3. Pre-push: python tests/test_runner.py all --skip-slow
    4. CI/CD: python tests/test_runner.py all
    5. Release: python tests/test_runner.py all && python tests/test_runner.py performance
""")


if __name__ == "__main__":
    sys.exit(main())