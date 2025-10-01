"""
Test runner script for the Ollama AI Provider Support Epic testing framework.

This script provides:
1. Comprehensive test execution with configurable test suites
2. Test result reporting and validation
3. CI/CD integration support
4. Performance benchmarking and comparison
5. Epic validation and acceptance criteria checking
"""

import os
import sys
import subprocess
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestSuiteConfig:
    """Configuration for a test suite."""
    name: str
    files: List[str]
    markers: List[str]
    env_vars: Dict[str, str]
    timeout: int
    parallel: bool
    required_services: List[str]


@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration: float
    success: bool
    failures: List[str]
    coverage: Optional[float] = None


class TestRunner:
    """Main test runner for the epic testing framework."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load test configuration."""
        default_config = {
            "test_suites": {
                "unit": {
                    "name": "Unit Tests",
                    "files": ["test_ai_providers.py"],
                    "markers": ["unit"],
                    "env_vars": {"ENABLE_UNIT_TESTS": "true"},
                    "timeout": 300,
                    "parallel": True,
                    "required_services": []
                },
                "integration": {
                    "name": "Integration Tests", 
                    "files": ["test_ollama_integration.py"],
                    "markers": ["integration"],
                    "env_vars": {
                        "ENABLE_INTEGRATION_TESTS": "true",
                        "OLLAMA_BASE_URL": "http://localhost:11434"
                    },
                    "timeout": 900,
                    "parallel": False,
                    "required_services": ["ollama"]
                },
                "performance": {
                    "name": "Performance Tests",
                    "files": ["test_performance_comparison.py"],
                    "markers": ["performance"],
                    "env_vars": {"ENABLE_PERFORMANCE_TESTS": "true"},
                    "timeout": 1800,
                    "parallel": False,
                    "required_services": []
                },
                "migration": {
                    "name": "Migration Tests",
                    "files": ["test_migration_integrity.py"],
                    "markers": ["migration"],
                    "env_vars": {"ENABLE_MIGRATION_TESTS": "true"},
                    "timeout": 600,
                    "parallel": True,
                    "required_services": []
                },
                "docker": {
                    "name": "Docker Tests",
                    "files": ["test_docker_orchestration.py"],
                    "markers": ["docker"],
                    "env_vars": {"ENABLE_DOCKER_TESTS": "true"},
                    "timeout": 1200,
                    "parallel": False,
                    "required_services": ["docker"]
                },
                "stories": {
                    "name": "Epic Story Validation",
                    "files": ["test_epic_stories.py"],
                    "markers": ["story"],
                    "env_vars": {"ENABLE_STORY_VALIDATION": "true"},
                    "timeout": 600,
                    "parallel": True,
                    "required_services": []
                }
            },
            "reporting": {
                "formats": ["console", "json", "html"],
                "output_dir": "test_reports",
                "coverage_threshold": 80.0
            },
            "ci_integration": {
                "fail_fast": False,
                "retry_failed": True,
                "max_retries": 2
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    custom_config = yaml.safe_load(f)
                else:
                    custom_config = json.load(f)
                
                # Merge configurations
                default_config.update(custom_config)
        
        return default_config
    
    def check_prerequisites(self, suite_config: TestSuiteConfig) -> bool:
        """Check if prerequisites for a test suite are met."""
        # Check required services
        for service in suite_config.required_services:
            if not self._check_service_availability(service):
                print(f"❌ Required service '{service}' is not available")
                return False
        
        # Check environment variables
        missing_vars = []
        for var, value in suite_config.env_vars.items():
            if var not in os.environ:
                os.environ[var] = value
            elif os.environ[var] != value:
                print(f"⚠️  Environment variable {var} has different value than expected")
        
        return True
    
    def _check_service_availability(self, service: str) -> bool:
        """Check if a required service is available."""
        if service == "ollama":
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        
        elif service == "docker":
            try:
                result = subprocess.run(
                    ["docker", "version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                return result.returncode == 0
            except Exception:
                return False
        
        return True
    
    def run_test_suite(self, suite_name: str, suite_config: TestSuiteConfig) -> TestResult:
        """Run a specific test suite."""
        print(f"\n🧪 Running {suite_config.name}...")
        
        # Check prerequisites
        if not self.check_prerequisites(suite_config):
            return TestResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                duration=0.0,
                success=False,
                failures=[f"Prerequisites not met for {suite_name}"]
            )
        
        # Prepare pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test files
        test_files = [f"tests/{file}" for file in suite_config.files]
        cmd.extend(test_files)
        
        # Add markers
        if suite_config.markers:
            marker_expr = " or ".join(suite_config.markers)
            cmd.extend(["-m", marker_expr])
        
        # Add reporting options
        cmd.extend([
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            f"--timeout={suite_config.timeout}",  # Test timeout
            "--junit-xml=test_reports/junit.xml",  # JUnit XML output
        ])
        
        # Add parallel execution if enabled
        if suite_config.parallel:
            cmd.extend(["-n", "auto"])  # Requires pytest-xdist
        
        # Add coverage if requested
        if suite_name in ["unit", "integration"]:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:test_reports/coverage_html",
                "--cov-report=json:test_reports/coverage.json"
            ])
        
        # Set environment variables
        env = os.environ.copy()
        env.update(suite_config.env_vars)
        
        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=suite_config.timeout + 60  # Add buffer for pytest overhead
            )
            
            duration = time.time() - start_time
            
            # Parse results
            return self._parse_test_results(
                suite_name, 
                result, 
                duration
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                duration=duration,
                success=False,
                failures=[f"Test suite {suite_name} timed out after {suite_config.timeout}s"]
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                duration=duration,
                success=False,
                failures=[f"Error running test suite {suite_name}: {str(e)}"]
            )
    
    def _parse_test_results(self, suite_name: str, result: subprocess.CompletedProcess, duration: float) -> TestResult:
        """Parse pytest results."""
        output = result.stdout + result.stderr
        
        # Parse pytest output for test counts
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        failures = []
        
        lines = output.split('\n')
        for line in lines:
            # Look for pytest summary line
            if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                # Example: "5 passed, 2 failed, 1 skipped in 10.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        passed_tests = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        failed_tests = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        skipped_tests = int(parts[i-1])
                
                total_tests = passed_tests + failed_tests + skipped_tests
                break
        
        # Extract failure messages
        if result.returncode != 0:
            failure_lines = []
            in_failure = False
            for line in lines:
                if "FAILED" in line or "ERROR" in line:
                    in_failure = True
                    failure_lines.append(line)
                elif in_failure and line.startswith("="):
                    break
                elif in_failure:
                    failure_lines.append(line)
            
            failures = failure_lines[:10]  # Limit to first 10 failure lines
        
        # Parse coverage if available
        coverage = None
        try:
            coverage_file = "test_reports/coverage.json"
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    coverage = coverage_data.get("totals", {}).get("percent_covered")
        except Exception:
            pass
        
        return TestResult(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            duration=duration,
            success=result.returncode == 0,
            failures=failures,
            coverage=coverage
        )
    
    def run_all_suites(self, suites: Optional[List[str]] = None) -> bool:
        """Run all or specified test suites."""
        if suites is None:
            suites = list(self.config["test_suites"].keys())
        
        print(f"🚀 Starting test execution for suites: {', '.join(suites)}")
        
        # Create output directory
        os.makedirs("test_reports", exist_ok=True)
        
        # Run each test suite
        overall_success = True
        for suite_name in suites:
            if suite_name not in self.config["test_suites"]:
                print(f"❌ Unknown test suite: {suite_name}")
                continue
            
            suite_config_dict = self.config["test_suites"][suite_name]
            suite_config = TestSuiteConfig(
                name=suite_config_dict["name"],
                files=suite_config_dict["files"],
                markers=suite_config_dict["markers"],
                env_vars=suite_config_dict["env_vars"],
                timeout=suite_config_dict["timeout"],
                parallel=suite_config_dict["parallel"],
                required_services=suite_config_dict["required_services"]
            )
            
            result = self.run_test_suite(suite_name, suite_config)
            self.results.append(result)
            
            if not result.success:
                overall_success = False
                if self.config["ci_integration"]["fail_fast"]:
                    break
        
        # Generate reports
        self.generate_reports()
        
        return overall_success
    
    def generate_reports(self):
        """Generate test reports in various formats."""
        total_duration = time.time() - self.start_time
        
        # Console report
        print("\n" + "="*80)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*80)
        
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed_tests for r in self.results)
        total_failed = sum(r.failed_tests for r in self.results)
        total_skipped = sum(r.skipped_tests for r in self.results)
        
        print(f"Total Test Suites: {len(self.results)}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed} ✅")
        print(f"Failed: {total_failed} ❌")
        print(f"Skipped: {total_skipped} ⏭️")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Individual suite results
        print("\n📋 SUITE RESULTS:")
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            coverage_str = f" (Coverage: {result.coverage:.1f}%)" if result.coverage else ""
            print(f"  {status} {result.suite_name}: {result.passed_tests}/{result.total_tests} passed in {result.duration:.2f}s{coverage_str}")
            
            if not result.success and result.failures:
                print(f"    Failures: {len(result.failures)} issues")
        
        # Epic validation summary
        story_results = [r for r in self.results if r.suite_name == "stories"]
        if story_results:
            story_result = story_results[0]
            if story_result.success:
                print("\n🎉 EPIC VALIDATION: ALL STORIES PASSED")
            else:
                print("\n⚠️  EPIC VALIDATION: SOME STORIES FAILED")
        
        # JSON report
        if "json" in self.config["reporting"]["formats"]:
            self._generate_json_report()
        
        # HTML report
        if "html" in self.config["reporting"]["formats"]:
            self._generate_html_report()
        
        print(f"\n📁 Reports saved to: {self.config['reporting']['output_dir']}/")
    
    def _generate_json_report(self):
        """Generate JSON test report."""
        report_data = {
            "execution_summary": {
                "start_time": self.start_time,
                "total_duration": time.time() - self.start_time,
                "total_suites": len(self.results),
                "total_tests": sum(r.total_tests for r in self.results),
                "total_passed": sum(r.passed_tests for r in self.results),
                "total_failed": sum(r.failed_tests for r in self.results),
                "total_skipped": sum(r.skipped_tests for r in self.results),
                "overall_success": all(r.success for r in self.results)
            },
            "suite_results": [
                {
                    "suite_name": r.suite_name,
                    "total_tests": r.total_tests,
                    "passed_tests": r.passed_tests,
                    "failed_tests": r.failed_tests,
                    "skipped_tests": r.skipped_tests,
                    "duration": r.duration,
                    "success": r.success,
                    "coverage": r.coverage,
                    "failures": r.failures
                }
                for r in self.results
            ]
        }
        
        output_file = os.path.join(self.config["reporting"]["output_dir"], "test_results.json")
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_html_report(self):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ollama AI Provider Epic Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .suite {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .pass {{ border-color: #4CAF50; }}
        .fail {{ border-color: #f44336; }}
        .coverage {{ font-size: 0.9em; color: #666; }}
        .failures {{ background-color: #fff5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Ollama AI Provider Support Epic - Test Results</h1>
        <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Execution Summary</h2>
        <p><strong>Total Suites:</strong> {len(self.results)}</p>
        <p><strong>Total Tests:</strong> {sum(r.total_tests for r in self.results)}</p>
        <p><strong>Passed:</strong> {sum(r.passed_tests for r in self.results)} ✅</p>
        <p><strong>Failed:</strong> {sum(r.failed_tests for r in self.results)} ❌</p>
        <p><strong>Skipped:</strong> {sum(r.skipped_tests for r in self.results)} ⏭️</p>
        <p><strong>Duration:</strong> {time.time() - self.start_time:.2f}s</p>
    </div>
    
    <div class="results">
        <h2>📋 Suite Results</h2>
"""
        
        for result in self.results:
            status_class = "pass" if result.success else "fail"
            status_text = "PASS ✅" if result.success else "FAIL ❌"
            coverage_text = f" (Coverage: {result.coverage:.1f}%)" if result.coverage else ""
            
            html_content += f"""
        <div class="suite {status_class}">
            <h3>{result.suite_name} - {status_text}</h3>
            <p><strong>Tests:</strong> {result.passed_tests}/{result.total_tests} passed in {result.duration:.2f}s{coverage_text}</p>
"""
            
            if not result.success and result.failures:
                html_content += f"""
            <div class="failures">
                <h4>Failures:</h4>
                <pre>{''.join(result.failures[:5])}</pre>
            </div>
"""
            
            html_content += "        </div>\n"
        
        html_content += """
    </div>
</body>
</html>
"""
        
        output_file = os.path.join(self.config["reporting"]["output_dir"], "test_results.html")
        with open(output_file, 'w') as f:
            f.write(html_content)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Ollama AI Provider Epic Test Runner")
    parser.add_argument(
        "--suites", 
        nargs="+", 
        choices=["unit", "integration", "performance", "migration", "docker", "stories", "all"],
        default=["all"],
        help="Test suites to run"
    )
    parser.add_argument(
        "--config", 
        help="Path to test configuration file"
    )
    parser.add_argument(
        "--fail-fast", 
        action="store_true",
        help="Stop on first test suite failure"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Enable parallel test execution where supported"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(args.config)
    
    # Override config with CLI arguments
    if args.fail_fast:
        runner.config["ci_integration"]["fail_fast"] = True
    
    if args.parallel:
        for suite_config in runner.config["test_suites"].values():
            if suite_config["name"] not in ["Integration Tests", "Docker Tests"]:
                suite_config["parallel"] = True
    
    # Determine suites to run
    suites_to_run = None if "all" in args.suites else args.suites
    
    # Run tests
    try:
        success = runner.run_all_suites(suites_to_run)
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()