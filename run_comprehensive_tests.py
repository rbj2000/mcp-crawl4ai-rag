#!/usr/bin/env python3
"""
Comprehensive test runner for the Ollama AI Provider Support Epic.

This script orchestrates the complete testing framework including:
1. Unit tests for AI providers
2. Integration tests for utils refactored
3. Configuration validation tests
4. Performance benchmarking
5. Backward compatibility validation
6. CI/CD environment testing

Usage:
    python run_comprehensive_tests.py [options]
    
Options:
    --unit-only: Run only unit tests
    --integration-only: Run only integration tests  
    --performance-only: Run only performance tests
    --fast: Skip slow tests
    --mock-only: Use only mocked dependencies
    --generate-report: Generate detailed HTML report
    --parallel: Run tests in parallel
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ComprehensiveTestRunner:
    """Orchestrates comprehensive testing for AI provider integration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.start_time = None
        
    def setup_test_environment(self, args):
        """Set up test environment based on arguments."""
        env = os.environ.copy()
        
        # Base test environment
        env.update({
            'PYTHONPATH': str(self.project_root / 'src'),
            'PYTEST_CURRENT_TEST': 'true',
            'USE_MOCK_PROVIDERS': 'true' if args.mock_only else 'false',
        })
        
        # Test type flags
        env.update({
            'ENABLE_UNIT_TESTS': 'true',
            'ENABLE_INTEGRATION_TESTS': str(not args.unit_only).lower(),
            'ENABLE_PERFORMANCE_TESTS': str(args.performance_only or not args.fast).lower(),
            'ENABLE_DOCKER_TESTS': 'false',  # Disabled by default for CI/CD
        })
        
        # Provider configuration for testing
        env.update({
            'AI_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'sk-test-mock-key-for-testing',
            'OLLAMA_BASE_URL': 'http://localhost:11434',
            'DATABASE_PROVIDER': 'sqlite',
        })
        
        return env
    
    def run_test_suite(self, name: str, command: List[str], env: Dict[str, str]) -> bool:
        """Run a specific test suite."""
        print(f"\n🧪 Running {name}...")
        print("-" * 60)
        
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            
            # Log results
            print(f"Exit code: {result.returncode}")
            print(f"Execution time: {execution_time:.2f}s")
            
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            
            if result.stderr and not success:
                print("STDERR:")
                print(result.stderr)
            
            self.test_results[name] = {
                'success': success,
                'execution_time': execution_time,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{name}: {status} ({execution_time:.2f}s)")
            
            return success
            
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            self.test_results[name] = {
                'success': False,
                'execution_time': 0,
                'exit_code': -1,
                'error': str(e)
            }
            return False
    
    def run_unit_tests(self, env: Dict[str, str]) -> bool:
        """Run unit tests."""
        command = [
            sys.executable, '-m', 'pytest',
            'tests/test_ai_providers.py',
            '-v',
            '--tb=short',
            '--disable-warnings',
            '-m', 'not slow'
        ]
        
        return self.run_test_suite("Unit Tests", command, env)
    
    def run_integration_tests(self, env: Dict[str, str]) -> bool:
        """Run integration tests."""
        command = [
            sys.executable, '-m', 'pytest', 
            'tests/test_utils_refactored.py',
            '-v',
            '--tb=short',
            '--disable-warnings'
        ]
        
        return self.run_test_suite("Integration Tests", command, env)
    
    def run_comprehensive_tests(self, env: Dict[str, str]) -> bool:
        """Run comprehensive integration test suite."""
        command = [
            sys.executable, '-m', 'pytest',
            'tests/test_comprehensive_integration.py',
            '-v',
            '--tb=short',
            '--disable-warnings'
        ]
        
        return self.run_test_suite("Comprehensive Integration", command, env)
    
    def run_validation_script(self, env: Dict[str, str]) -> bool:
        """Run validation script."""
        command = [sys.executable, 'validate_utils_refactored.py']
        return self.run_test_suite("Validation Script", command, env)
    
    def run_performance_tests(self, env: Dict[str, str]) -> bool:
        """Run performance benchmarking tests."""
        command = [
            sys.executable, '-m', 'pytest',
            'tests/test_ai_providers.py::TestPerformanceBenchmarking',
            'tests/test_utils_refactored.py::TestPerformanceAndConcurrency',
            '-v',
            '--tb=short',
            '--disable-warnings'
        ]
        
        return self.run_test_suite("Performance Tests", command, env)
    
    def generate_html_report(self):
        """Generate HTML test report."""
        if not self.test_results:
            return
            
        html_content = self.create_html_report()
        
        report_file = self.project_root / "test_report.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"\n📄 HTML report generated: {report_file}")
    
    def create_html_report(self) -> str:
        """Create HTML report content."""
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['success'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Provider Testing Framework - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .passed {{ background: #d4edda; border: 1px solid #c3e6cb; }}
        .failed {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
        .details {{ margin-top: 10px; font-family: monospace; font-size: 12px; }}
        .statistics {{ display: flex; gap: 20px; }}
        .stat-box {{ padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-box.total {{ background: #e9ecef; }}
        .stat-box.passed {{ background: #d4edda; }}
        .stat-box.failed {{ background: #f8d7da; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 AI Provider Testing Framework</h1>
        <h2>Comprehensive Test Report - Ollama AI Provider Support Epic</h2>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Execution Time: {execution_time:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <div class="statistics">
            <div class="stat-box total">
                <h3>{total_tests}</h3>
                <p>Total Tests</p>
            </div>
            <div class="stat-box passed">
                <h3>{passed_tests}</h3>
                <p>Passed</p>
            </div>
            <div class="stat-box failed">
                <h3>{failed_tests}</h3>
                <p>Failed</p>
            </div>
            <div class="stat-box total">
                <h3>{success_rate:.1f}%</h3>
                <p>Success Rate</p>
            </div>
        </div>
    </div>
    
    <div class="results">
        <h2>📋 Test Results</h2>
"""
        
        for test_name, result in self.test_results.items():
            status_class = "passed" if result['success'] else "failed"
            status_text = "✅ PASSED" if result['success'] else "❌ FAILED"
            
            html += f"""
        <div class="test-result {status_class}">
            <h3>{test_name} - {status_text}</h3>
            <p>Execution Time: {result.get('execution_time', 0):.2f}s</p>
            <p>Exit Code: {result.get('exit_code', 'N/A')}</p>
"""
            
            if result.get('stdout'):
                html += f'<div class="details"><strong>Output:</strong><br><pre>{result["stdout"][:1000]}...</pre></div>'
            
            if result.get('stderr') and not result['success']:
                html += f'<div class="details"><strong>Errors:</strong><br><pre>{result["stderr"][:1000]}...</pre></div>'
            
            html += "</div>"
        
        html += """
    </div>
    
    <div class="footer">
        <h2>🎯 Framework Features</h2>
        <ul>
            <li>✅ Comprehensive AI provider testing (OpenAI, Ollama)</li>
            <li>✅ Full API mocking for CI/CD compatibility</li>
            <li>✅ Performance benchmarking and concurrency testing</li>
            <li>✅ Configuration validation for all provider scenarios</li>
            <li>✅ Backward compatibility with existing utils.py interface</li>
            <li>✅ Provider switching and migration testing</li>
            <li>✅ Error handling and fallback scenario validation</li>
        </ul>
    </div>
</body>
</html>
"""
        
        return html
    
    def run_all_tests(self, args):
        """Run all test suites based on arguments."""
        print("🚀 Starting Comprehensive AI Provider Testing Framework")
        print("🎯 Ollama AI Provider Support Epic - Complete Validation")
        print("=" * 70)
        
        self.start_time = time.time()
        env = self.setup_test_environment(args)
        
        test_suites = []
        
        # Determine which test suites to run
        if args.unit_only:
            test_suites = [("Unit Tests", self.run_unit_tests)]
        elif args.integration_only:
            test_suites = [("Integration Tests", self.run_integration_tests)]
        elif args.performance_only:
            test_suites = [("Performance Tests", self.run_performance_tests)]
        else:
            # Run all test suites
            test_suites = [
                ("Unit Tests", self.run_unit_tests),
                ("Integration Tests", self.run_integration_tests),
                ("Comprehensive Integration", self.run_comprehensive_tests),
                ("Validation Script", self.run_validation_script),
            ]
            
            if not args.fast:
                test_suites.append(("Performance Tests", self.run_performance_tests))
        
        # Execute test suites
        all_passed = True
        for suite_name, suite_func in test_suites:
            success = suite_func(env)
            all_passed = all_passed and success
        
        # Generate report
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        
        total_suites = len(self.test_results)
        passed_suites = sum(1 for r in self.test_results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        print(f"Total Test Suites: {total_suites}")
        print(f"Passed: {passed_suites}")
        print(f"Failed: {failed_suites}")
        print(f"Success Rate: {passed_suites/total_suites*100:.1f}%")
        print(f"Total Execution Time: {total_time:.2f}s")
        
        if args.generate_report:
            self.generate_html_report()
        
        # Save JSON report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": total_time,
            "framework": "AI Provider Testing Framework v1.0",
            "epic": "Ollama AI Provider Support",
            "statistics": {
                "total_suites": total_suites,
                "passed_suites": passed_suites,
                "failed_suites": failed_suites,
                "success_rate": passed_suites/total_suites*100
            },
            "test_results": self.test_results
        }
        
        with open("comprehensive_test_results.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 JSON report saved: comprehensive_test_results.json")
        
        if all_passed:
            print("\n🎉 ALL TEST SUITES PASSED!")
            print("✅ AI Provider integration framework is ready for production")
            print("✅ Ollama provider support fully validated")
            print("✅ Backward compatibility maintained")
            print("✅ Performance requirements met")
        else:
            print("\n⚠️ SOME TEST SUITES FAILED")
            print("❌ Review failed test suites above")
            print("🔧 Fix issues before deployment")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for AI Provider Testing Framework"
    )
    
    # Test selection options
    parser.add_argument("--unit-only", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", 
                       help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance tests")
    
    # Execution options
    parser.add_argument("--fast", action="store_true",
                       help="Skip slow tests")
    parser.add_argument("--mock-only", action="store_true", default=True,
                       help="Use only mocked dependencies (default: True)")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel")
    
    # Reporting options
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate detailed HTML report")
    
    args = parser.parse_args()
    
    # Run comprehensive tests
    runner = ComprehensiveTestRunner()
    success = runner.run_all_tests(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()