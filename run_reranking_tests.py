#!/usr/bin/env python3
"""
Comprehensive test runner for reranking functionality.

This script provides different test execution modes for reranking:
1. Unit tests only (fast, no external dependencies)
2. Integration tests (requires some services)
3. Full test suite (comprehensive coverage)
4. Performance benchmarks
5. Regression tests

Usage:
    python run_reranking_tests.py --mode unit
    python run_reranking_tests.py --mode integration
    python run_reranking_tests.py --mode full
    python run_reranking_tests.py --mode performance
    python run_reranking_tests.py --mode regression
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully in {elapsed_time:.1f}s")
        return True
    else:
        print(f"\n❌ {description} failed after {elapsed_time:.1f}s")
        return False


def run_unit_tests() -> bool:
    """Run unit tests for reranking providers."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_providers.py",
        "-v",
        "-m", "reranking_unit",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=5"
    ]
    return run_command(cmd, "Reranking Unit Tests")


def run_integration_tests() -> bool:
    """Run integration tests for RAG pipeline with reranking."""
    cmd = [
        "python", "-m", "pytest", 
        "tests/test_reranking_integration.py",
        "-v",
        "-m", "reranking_integration",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=5"
    ]
    return run_command(cmd, "Reranking Integration Tests")


def run_configuration_tests() -> bool:
    """Run configuration and environment tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_integration.py::TestRerankingConfigurationValidation",
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    return run_command(cmd, "Reranking Configuration Tests")


def run_error_handling_tests() -> bool:
    """Run error handling and fallback tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_integration.py::TestRerankingErrorHandlingAndFallbacks",
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    return run_command(cmd, "Reranking Error Handling Tests")


def run_regression_tests() -> bool:
    """Run regression tests to ensure existing functionality works."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_integration.py::TestRerankingRegressionTests",
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    return run_command(cmd, "Reranking Regression Tests")


def run_performance_tests() -> bool:
    """Run performance benchmark tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_integration.py::TestRerankingPerformanceBenchmarking",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--durations=10"
    ]
    return run_command(cmd, "Reranking Performance Tests")


def run_all_reranking_tests() -> bool:
    """Run all reranking tests."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_providers.py",
        "tests/test_reranking_integration.py",
        "-v",
        "-m", "reranking",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=10",
        "--durations=15"
    ]
    return run_command(cmd, "All Reranking Tests")


def run_coverage_analysis() -> bool:
    """Run tests with coverage analysis."""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_reranking_providers.py",
        "tests/test_reranking_integration.py",
        "-v",
        "--cov=src/ai_providers",
        "--cov-report=html:htmlcov_reranking",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "--tb=short",
        "--disable-warnings"
    ]
    return run_command(cmd, "Reranking Tests with Coverage")


def setup_test_environment() -> Dict[str, str]:
    """Setup test environment variables."""
    test_env = {
        "PYTHONPATH": "src",
        "PYTEST_CURRENT_TEST": "true",
        "USE_MOCK_PROVIDERS": "true",
        "OPENAI_API_KEY": "sk-test-mock-key-for-reranking-tests",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "HF_DEVICE": "cpu",
        "RERANKING_TIMEOUT": "30",
        "LOG_LEVEL": "INFO"
    }
    
    # Update environment
    for key, value in test_env.items():
        os.environ[key] = value
    
    return test_env


def validate_environment() -> bool:
    """Validate that the test environment is properly set up."""
    print("\n🔍 Validating test environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check required packages
    required_packages = ["pytest", "asyncio"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            return False
    
    # Check test files exist
    test_files = [
        "tests/test_reranking_providers.py",
        "tests/test_reranking_integration.py",
        "tests/conftest.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"✅ {test_file} exists")
        else:
            print(f"❌ {test_file} missing")
            return False
    
    print("✅ Environment validation passed")
    return True


def generate_test_report(results: Dict[str, bool]) -> None:
    """Generate a summary test report."""
    print(f"\n{'='*60}")
    print("RERANKING TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    failed_tests = total_tests - passed_tests
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<40} {status}")
    
    print(f"\n{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("🎉 All reranking tests passed!")
    else:
        print(f"⚠️  {failed_tests} test suite(s) failed")
    
    print(f"{'='*60}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run reranking functionality tests")
    parser.add_argument(
        "--mode",
        choices=["unit", "integration", "config", "errors", "regression", "performance", "full", "coverage"],
        default="unit",
        help="Test mode to run"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🧪 Reranking Test Runner")
    print(f"Mode: {args.mode}")
    
    # Setup environment
    if not validate_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    setup_test_environment()
    
    # Track results
    results = {}
    
    try:
        if args.mode == "unit":
            results["Unit Tests"] = run_unit_tests()
            
        elif args.mode == "integration":
            results["Integration Tests"] = run_integration_tests()
            
        elif args.mode == "config":
            results["Configuration Tests"] = run_configuration_tests()
            
        elif args.mode == "errors":
            results["Error Handling Tests"] = run_error_handling_tests()
            
        elif args.mode == "regression":
            results["Regression Tests"] = run_regression_tests()
            
        elif args.mode == "performance":
            results["Performance Tests"] = run_performance_tests()
            
        elif args.mode == "coverage":
            results["Coverage Analysis"] = run_coverage_analysis()
            
        elif args.mode == "full":
            results["Unit Tests"] = run_unit_tests()
            results["Integration Tests"] = run_integration_tests()
            results["Configuration Tests"] = run_configuration_tests()
            results["Error Handling Tests"] = run_error_handling_tests()
            results["Regression Tests"] = run_regression_tests()
            results["Performance Tests"] = run_performance_tests()
    
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    
    # Generate summary report
    generate_test_report(results)
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 All selected test suites passed!")
        sys.exit(0)
    else:
        print("\n❌ Some test suites failed")
        sys.exit(1)


if __name__ == "__main__":
    main()