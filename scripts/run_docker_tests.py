#!/usr/bin/env python3
"""
Python-based Docker testing runner for MCP Crawl4AI RAG
Alternative to bash scripts with better error handling and reporting
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

class DockerTestRunner:
    """Python-based Docker test runner with comprehensive reporting"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results: Dict[str, Dict] = {}
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], capture_output: bool = True, timeout: int = 300) -> Tuple[int, str, str]:
        """Run shell command with error handling"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return 1, "", str(e)
    
    def wait_for_service(self, url: str, timeout: int = 60, service_name: str = "service") -> bool:
        """Wait for service to become healthy"""
        console.print(f"⏳ Waiting for {service_name} at {url}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    console.print(f"✅ {service_name} is ready")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        
        console.print(f"❌ {service_name} failed to become ready after {timeout}s")
        return False
    
    def test_docker_builds(self) -> bool:
        """Test all Docker image builds"""
        console.print("\n🏗️  Testing Docker Builds")
        
        providers = ["supabase", "sqlite", "pinecone", "all"]
        build_results = {}
        
        for provider in providers:
            console.print(f"Building image for {provider} provider...")
            
            cmd = [
                "docker", "build",
                "--build-arg", f"VECTOR_DB_PROVIDER={provider}",
                "-t", f"mcp-crawl4ai-rag:{provider}",
                "."
            ]
            
            returncode, stdout, stderr = self.run_command(cmd, timeout=600)
            
            build_results[provider] = {
                "success": returncode == 0,
                "stdout": stdout,
                "stderr": stderr
            }
            
            if returncode == 0:
                console.print(f"✅ {provider} build successful")
            else:
                console.print(f"❌ {provider} build failed")
                console.print(f"Error: {stderr}")
        
        self.test_results["docker_builds"] = build_results
        return all(result["success"] for result in build_results.values())
    
    def test_sqlite_provider(self) -> bool:
        """Test SQLite provider functionality"""
        console.print("\n🗄️  Testing SQLite Provider")
        
        # Start SQLite container
        env = os.environ.copy()
        env.update({
            "VECTOR_DB_PROVIDER": "sqlite",
            "SQLITE_DB_PATH": "/tmp/test.sqlite",
            "OPENAI_API_KEY": "test_key"
        })
        
        cmd = ["docker-compose", "--profile", "sqlite", "up", "-d"]
        returncode, stdout, stderr = self.run_command(cmd)
        
        if returncode != 0:
            console.print(f"❌ Failed to start SQLite container: {stderr}")
            return False
        
        try:
            # Wait for service
            if not self.wait_for_service("http://localhost:8052/health", service_name="SQLite MCP"):
                return False
            
            # Test functionality
            success = self._test_mcp_functionality("http://localhost:8052", "SQLite")
            
            self.test_results["sqlite_provider"] = {
                "success": success,
                "service_ready": True
            }
            
            return success
            
        finally:
            # Cleanup
            self.run_command(["docker-compose", "--profile", "sqlite", "down", "-v"])
    
    def test_neo4j_provider(self) -> bool:
        """Test Neo4j provider functionality"""
        console.print("\n🕸️  Testing Neo4j Provider")
        
        # Start Neo4j services
        env = os.environ.copy()
        env.update({
            "VECTOR_DB_PROVIDER": "neo4j_vector",
            "NEO4J_PASSWORD": "test_password",
            "OPENAI_API_KEY": "test_key"
        })
        
        cmd = ["docker-compose", "--profile", "neo4j", "up", "-d"]
        returncode, stdout, stderr = self.run_command(cmd)
        
        if returncode != 0:
            console.print(f"❌ Failed to start Neo4j services: {stderr}")
            return False
        
        try:
            # Wait for Neo4j database
            if not self.wait_for_service("http://localhost:7474", timeout=120, service_name="Neo4j Database"):
                return False
            
            # Wait for MCP server
            if not self.wait_for_service("http://localhost:8055/health", service_name="Neo4j MCP"):
                return False
            
            # Test functionality
            success = self._test_mcp_functionality("http://localhost:8055", "Neo4j")
            
            self.test_results["neo4j_provider"] = {
                "success": success,
                "database_ready": True,
                "mcp_ready": True
            }
            
            return success
            
        finally:
            # Cleanup
            self.run_command(["docker-compose", "--profile", "neo4j", "down", "-v"])
    
    def _test_mcp_functionality(self, base_url: str, provider_name: str) -> bool:
        """Test MCP server functionality"""
        try:
            # Test health endpoint
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code != 200:
                console.print(f"❌ {provider_name}: Health check failed")
                return False
            
            health_data = response.json()
            if health_data.get("status") != "healthy":
                console.print(f"❌ {provider_name}: Service not healthy")
                return False
            
            # Test tools endpoint
            response = requests.get(f"{base_url}/tools", timeout=10)
            if response.status_code != 200:
                console.print(f"❌ {provider_name}: Tools endpoint failed")
                return False
            
            tools = response.json()
            tool_names = [tool["name"] for tool in tools.get("tools", [])]
            
            required_tools = ["crawl_single_page", "perform_rag_query", "get_available_sources"]
            missing_tools = [tool for tool in required_tools if tool not in tool_names]
            
            if missing_tools:
                console.print(f"❌ {provider_name}: Missing tools: {missing_tools}")
                return False
            
            console.print(f"✅ {provider_name}: All functionality tests passed")
            return True
            
        except Exception as e:
            console.print(f"❌ {provider_name}: Test failed with error: {e}")
            return False
    
    def run_pytest_tests(self) -> bool:
        """Run pytest-based Docker tests"""
        console.print("\n🧪 Running pytest Docker tests")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/test_docker_integration.py",
            "tests/test_testcontainers.py",
            "-v", "--tb=short"
        ]
        
        returncode, stdout, stderr = self.run_command(cmd, timeout=900)
        
        self.test_results["pytest_tests"] = {
            "success": returncode == 0,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if returncode == 0:
            console.print("✅ pytest tests passed")
        else:
            console.print("❌ pytest tests failed")
            console.print(f"Output: {stdout}")
            console.print(f"Errors: {stderr}")
        
        return returncode == 0
    
    def generate_report(self) -> None:
        """Generate comprehensive test report"""
        duration = time.time() - self.start_time
        
        # Create summary table
        table = Table(title="Docker Test Results Summary")
        table.add_column("Test Category", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="white")
        
        total_tests = 0
        passed_tests = 0
        
        for category, result in self.test_results.items():
            total_tests += 1
            if result.get("success", False):
                passed_tests += 1
                status = "✅ PASS"
                details = "All checks passed"
            else:
                status = "❌ FAIL"
                if "stderr" in result and result["stderr"]:
                    details = result["stderr"][:50] + "..."
                else:
                    details = "Check failed"
            
            table.add_row(category.replace("_", " ").title(), status, details)
        
        console.print(table)
        
        # Overall summary
        if passed_tests == total_tests:
            panel_style = "green"
            status_emoji = "🎉"
            status_text = "ALL TESTS PASSED"
        else:
            panel_style = "red"
            status_emoji = "💥"
            status_text = "SOME TESTS FAILED"
        
        summary = f"""
{status_emoji} {status_text}

📊 Results: {passed_tests}/{total_tests} tests passed
⏱️  Duration: {duration:.2f}s
📁 Report saved to: docker_test_results.json
        """
        
        console.print(Panel(summary.strip(), style=panel_style, title="Test Summary"))
        
        # Save detailed results to JSON
        with open("docker_test_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "duration": duration,
                    "timestamp": time.time()
                },
                "results": self.test_results
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Python Docker test runner for MCP Crawl4AI RAG")
    parser.add_argument("--provider", choices=["sqlite", "neo4j", "all"], 
                       help="Test specific provider only")
    parser.add_argument("--build-only", action="store_true", 
                       help="Only test Docker builds")
    parser.add_argument("--pytest-only", action="store_true", 
                       help="Only run pytest tests")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip cleanup after tests")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    runner = DockerTestRunner(project_root)
    
    console.print("🚀 Starting Python Docker Tests for MCP Crawl4AI RAG")
    
    success = True
    
    try:
        if args.build_only:
            success = runner.test_docker_builds()
        elif args.pytest_only:
            success = runner.run_pytest_tests()
        elif args.provider:
            if args.provider == "sqlite":
                success = runner.test_sqlite_provider()
            elif args.provider == "neo4j":
                success = runner.test_neo4j_provider()
        else:
            # Run all tests
            success &= runner.test_docker_builds()
            success &= runner.test_sqlite_provider()
            success &= runner.test_neo4j_provider()
            success &= runner.run_pytest_tests()
        
        runner.generate_report()
        
        if success:
            console.print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            console.print("\n💥 Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        if not args.no_cleanup:
            console.print("🧹 Cleaning up...")
            runner.run_command(["docker-compose", "down", "-v", "--remove-orphans"])

if __name__ == "__main__":
    main()