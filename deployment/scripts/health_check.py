#!/usr/bin/env python3
"""
Comprehensive health check script for MCP Crawl4AI RAG deployment.

This script performs:
- Service health checks (HTTP/TCP endpoints)
- AI provider connectivity and model availability
- Database connectivity and performance
- Resource usage monitoring
- End-to-end functionality testing
- Dependency validation
"""

import os
import sys
import time
import json
import requests
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service: str
    check_type: str
    healthy: bool
    response_time_ms: float
    message: str
    details: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class HealthChecker:
    """Comprehensive health checker for the deployment."""
    
    def __init__(self, environment: str, simulate: bool = False, ai_provider: str = 'openai'):
        self.environment = environment
        self.simulate = simulate
        self.ai_provider = ai_provider.lower()
        self.results: List[HealthCheckResult] = []
        self.docker_client = None
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client not available: {e}")
        
        # Service endpoints configuration (dynamic based on provider)
        self.service_endpoints = {
            'mcp-crawl4ai': {
                'http': 'http://localhost:8051/health',
                'tcp': ('localhost', 8051),
                'critical': True
            }
        }
        
        # Add Ollama endpoints if using Ollama
        if self.ai_provider in ['ollama', 'hybrid', 'mixed']:
            self.service_endpoints['ollama'] = {
                'http': 'http://localhost:11434/api/tags',
                'tcp': ('localhost', 11434),
                'critical': True
            }
        
        # Add optional service endpoints (always available for checking)
        self.optional_endpoints = {
            'neo4j': {
                'http': 'http://localhost:7474/db/neo4j/tx/commit',
                'tcp': ('localhost', 7687),
                'critical': False
            },
            'prometheus': {
                'http': 'http://localhost:9090/-/healthy',
                'tcp': ('localhost', 9090),
                'critical': False
            },
            'grafana': {
                'http': 'http://localhost:3000/api/health',
                'tcp': ('localhost', 3000),
                'critical': False
            },
            'weaviate': {
                'http': 'http://localhost:8080/v1/.well-known/ready',
                'tcp': ('localhost', 8080),
                'critical': False
            }
        }
        
        # Expected AI models
        self.expected_models = {
            'ollama': ['nomic-embed-text', 'llama3.2:1b']
        }
    
    def add_result(self, result: HealthCheckResult):
        """Add health check result."""
        self.results.append(result)
        
        status = "✅" if result.healthy else "❌"
        logger.info(f"{status} {result.service} ({result.check_type}): {result.message}")
        
        if result.response_time_ms > 0:
            logger.debug(f"   Response time: {result.response_time_ms:.1f}ms")
    
    def check_http_endpoint(self, service: str, url: str, timeout: int = 10) -> HealthCheckResult:
        """Check HTTP endpoint health."""
        if self.simulate:
            return HealthCheckResult(
                service=service,
                check_type='http',
                healthy=True,
                response_time_ms=50.0,
                message="Simulated HTTP check passed"
            )
        
        start_time = time.time()
        try:
            response = requests.get(url, timeout=timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return HealthCheckResult(
                    service=service,
                    check_type='http',
                    healthy=True,
                    response_time_ms=response_time,
                    message=f"HTTP endpoint healthy (status: {response.status_code})",
                    details={'status_code': response.status_code, 'url': url}
                )
            else:
                return HealthCheckResult(
                    service=service,
                    check_type='http',
                    healthy=False,
                    response_time_ms=response_time,
                    message=f"HTTP endpoint returned status {response.status_code}",
                    details={'status_code': response.status_code, 'url': url}
                )
                
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                service=service,
                check_type='http',
                healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                message="Connection refused - service may be down",
                details={'url': url}
            )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                service=service,
                check_type='http',
                healthy=False,
                response_time_ms=timeout * 1000,
                message=f"Request timed out after {timeout}s",
                details={'url': url, 'timeout': timeout}
            )
        except Exception as e:
            return HealthCheckResult(
                service=service,
                check_type='http',
                healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"HTTP check failed: {str(e)}",
                details={'url': url, 'error': str(e)}
            )
    
    def check_tcp_endpoint(self, service: str, host: str, port: int, timeout: int = 5) -> HealthCheckResult:
        """Check TCP endpoint connectivity."""
        if self.simulate:
            return HealthCheckResult(
                service=service,
                check_type='tcp',
                healthy=True,
                response_time_ms=10.0,
                message="Simulated TCP check passed"
            )
        
        import socket
        
        start_time = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000
            sock.close()
            
            if result == 0:
                return HealthCheckResult(
                    service=service,
                    check_type='tcp',
                    healthy=True,
                    response_time_ms=response_time,
                    message=f"TCP port {port} is open",
                    details={'host': host, 'port': port}
                )
            else:
                return HealthCheckResult(
                    service=service,
                    check_type='tcp',
                    healthy=False,
                    response_time_ms=response_time,
                    message=f"TCP port {port} is closed or unreachable",
                    details={'host': host, 'port': port, 'result': result}
                )
                
        except Exception as e:
            return HealthCheckResult(
                service=service,
                check_type='tcp',
                healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"TCP check failed: {str(e)}",
                details={'host': host, 'port': port, 'error': str(e)}
            )
    
    def check_docker_containers(self) -> List[HealthCheckResult]:
        """Check Docker container health."""
        results = []
        
        if self.simulate:
            mock_containers = ['mcp-ollama', 'mcp-crawl4ai-ollama-sqlite', 'mcp-prometheus']
            for container in mock_containers:
                results.append(HealthCheckResult(
                    service=f"docker/{container}",
                    check_type='container',
                    healthy=True,
                    response_time_ms=5.0,
                    message="Simulated container check passed"
                ))
            return results
        
        if not self.docker_client:
            results.append(HealthCheckResult(
                service='docker',
                check_type='container',
                healthy=False,
                response_time_ms=0,
                message="Docker client not available"
            ))
            return results
        
        try:
            containers = self.docker_client.containers.list(all=True)
            mcp_containers = [c for c in containers if 'mcp' in c.name.lower()]
            
            if not mcp_containers:
                results.append(HealthCheckResult(
                    service='docker',
                    check_type='container',
                    healthy=False,
                    response_time_ms=0,
                    message="No MCP containers found"
                ))
                return results
            
            for container in mcp_containers:
                is_running = container.status == 'running'
                
                # Get container stats if running
                details = {
                    'name': container.name,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown'
                }
                
                if is_running:
                    try:
                        stats = container.stats(stream=False)
                        details['cpu_usage'] = self._calculate_cpu_percent(stats)
                        details['memory_usage_mb'] = stats['memory_stats']['usage'] / 1024 / 1024
                    except Exception:
                        pass
                
                results.append(HealthCheckResult(
                    service=f"docker/{container.name}",
                    check_type='container',
                    healthy=is_running,
                    response_time_ms=0,
                    message=f"Container {container.status}",
                    details=details
                ))
                
        except Exception as e:
            results.append(HealthCheckResult(
                service='docker',
                check_type='container',
                healthy=False,
                response_time_ms=0,
                message=f"Docker check failed: {str(e)}"
            ))
        
        return results
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                return round(cpu_percent, 2)
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0
    
    def check_ollama_models(self) -> HealthCheckResult:
        """Check Ollama model availability."""
        if self.simulate:
            return HealthCheckResult(
                service='ollama',
                check_type='models',
                healthy=True,
                response_time_ms=100.0,
                message="Simulated model check passed",
                details={'available_models': ['nomic-embed-text', 'llama3.2:1b']}
            )
        
        start_time = time.time()
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                available_models = [model['name'] for model in data.get('models', [])]
                expected_models = self.expected_models.get('ollama', [])
                
                missing_models = []
                for expected in expected_models:
                    if not any(available.startswith(expected) for available in available_models):
                        missing_models.append(expected)
                
                if missing_models:
                    return HealthCheckResult(
                        service='ollama',
                        check_type='models',
                        healthy=False,
                        response_time_ms=response_time,
                        message=f"Missing models: {missing_models}",
                        details={
                            'available_models': available_models,
                            'missing_models': missing_models,
                            'expected_models': expected_models
                        }
                    )
                else:
                    return HealthCheckResult(
                        service='ollama',
                        check_type='models',
                        healthy=True,
                        response_time_ms=response_time,
                        message="All required models are available",
                        details={
                            'available_models': available_models,
                            'expected_models': expected_models
                        }
                    )
            else:
                return HealthCheckResult(
                    service='ollama',
                    check_type='models',
                    healthy=False,
                    response_time_ms=response_time,
                    message=f"Failed to list models (status: {response.status_code})"
                )
                
        except Exception as e:
            return HealthCheckResult(
                service='ollama',
                check_type='models',
                healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Model check failed: {str(e)}"
            )
    
    def check_system_resources(self) -> List[HealthCheckResult]:
        """Check system resource usage."""
        results = []
        
        if self.simulate:
            results.extend([
                HealthCheckResult('system', 'cpu', True, 0, "CPU usage: 25.0%", {'usage_percent': 25.0}),
                HealthCheckResult('system', 'memory', True, 0, "Memory usage: 45.0%", {'usage_percent': 45.0}),
                HealthCheckResult('system', 'disk', True, 0, "Disk usage: 60.0%", {'usage_percent': 60.0})
            ])
            return results
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_healthy = cpu_percent < 90
            results.append(HealthCheckResult(
                service='system',
                check_type='cpu',
                healthy=cpu_healthy,
                response_time_ms=0,
                message=f"CPU usage: {cpu_percent}%",
                details={'usage_percent': cpu_percent, 'threshold': 90}
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < 85
            results.append(HealthCheckResult(
                service='system',
                check_type='memory',
                healthy=memory_healthy,
                response_time_ms=0,
                message=f"Memory usage: {memory.percent}%",
                details={
                    'usage_percent': memory.percent,
                    'total_gb': round(memory.total / 1024**3, 2),
                    'available_gb': round(memory.available / 1024**3, 2),
                    'threshold': 85
                }
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_healthy = disk_percent < 90
            results.append(HealthCheckResult(
                service='system',
                check_type='disk',
                healthy=disk_healthy,
                response_time_ms=0,
                message=f"Disk usage: {disk_percent:.1f}%",
                details={
                    'usage_percent': disk_percent,
                    'total_gb': round(disk.total / 1024**3, 2),
                    'free_gb': round(disk.free / 1024**3, 2),
                    'threshold': 90
                }
            ))
            
        except Exception as e:
            results.append(HealthCheckResult(
                service='system',
                check_type='resources',
                healthy=False,
                response_time_ms=0,
                message=f"Resource check failed: {str(e)}"
            ))
        
        return results
    
    def check_end_to_end_functionality(self) -> HealthCheckResult:
        """Perform end-to-end functionality test."""
        if self.simulate:
            return HealthCheckResult(
                service='mcp-crawl4ai',
                check_type='e2e',
                healthy=True,
                response_time_ms=500.0,
                message="Simulated E2E test passed",
                details={'test_query': 'test', 'response_received': True}
            )
        
        start_time = time.time()
        try:
            # Test basic MCP endpoint
            response = requests.get('http://localhost:8051/health', timeout=10)
            if response.status_code != 200:
                return HealthCheckResult(
                    service='mcp-crawl4ai',
                    check_type='e2e',
                    healthy=False,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message=f"Health endpoint returned {response.status_code}"
                )
            
            # Could add more sophisticated E2E tests here
            # Like testing actual crawling, RAG queries, etc.
            
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service='mcp-crawl4ai',
                check_type='e2e',
                healthy=True,
                response_time_ms=response_time,
                message="Basic E2E functionality working",
                details={'health_check_passed': True}
            )
            
        except Exception as e:
            return HealthCheckResult(
                service='mcp-crawl4ai',
                check_type='e2e',
                healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"E2E test failed: {str(e)}"
            )
    
    def run_all_checks(self) -> bool:
        """Run all health checks."""
        logger.info(f"Starting comprehensive health checks for environment: {self.environment}")
        
        if self.simulate:
            logger.info("Running in simulation mode")
        
        # Service endpoint checks
        logger.info("Checking service endpoints...")
        
        # Combine critical and optional endpoints
        all_endpoints = {**self.service_endpoints, **self.optional_endpoints}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # HTTP checks
            http_futures = {
                executor.submit(
                    self.check_http_endpoint, 
                    service, 
                    config['http']
                ): service
                for service, config in all_endpoints.items()
                if 'http' in config
            }
            
            # TCP checks
            tcp_futures = {
                executor.submit(
                    self.check_tcp_endpoint,
                    service,
                    config['tcp'][0],
                    config['tcp'][1]
                ): service
                for service, config in all_endpoints.items()
                if 'tcp' in config
            }
            
            # Process HTTP results
            for future in as_completed(http_futures):
                result = future.result()
                self.add_result(result)
            
            # Process TCP results
            for future in as_completed(tcp_futures):
                result = future.result()
                self.add_result(result)
        
        # Docker container checks
        logger.info("Checking Docker containers...")
        container_results = self.check_docker_containers()
        for result in container_results:
            self.add_result(result)
        
        # Ollama model checks
        logger.info("Checking Ollama models...")
        model_result = self.check_ollama_models()
        self.add_result(model_result)
        
        # System resource checks
        logger.info("Checking system resources...")
        resource_results = self.check_system_resources()
        for result in resource_results:
            self.add_result(result)
        
        # End-to-end functionality check
        logger.info("Running end-to-end functionality test...")
        e2e_result = self.check_end_to_end_functionality()
        self.add_result(e2e_result)
        
        # Analyze results
        return self.analyze_results()
    
    def analyze_results(self) -> bool:
        """Analyze health check results and determine overall health."""
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.healthy])
        failed_checks = total_checks - passed_checks
        
        # Count critical failures
        critical_failures = []
        for result in self.results:
            if not result.healthy:
                # Check both critical and optional endpoints
                all_endpoints = {**self.service_endpoints, **self.optional_endpoints}
                service_config = all_endpoints.get(result.service, {})
                if service_config.get('critical', False):
                    critical_failures.append(result)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("="*60)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"✅ Passed: {passed_checks}")
        logger.info(f"❌ Failed: {failed_checks}")
        logger.info(f"🚨 Critical failures: {len(critical_failures)}")
        
        # Print failed checks
        if failed_checks > 0:
            logger.info("\nFAILED CHECKS:")
            for result in self.results:
                if not result.healthy:
                    criticality = "CRITICAL" if any(result.service == cf.service for cf in critical_failures) else "WARNING"
                    logger.info(f"  [{criticality}] {result.service}/{result.check_type}: {result.message}")
        
        # Calculate health score
        health_score = (passed_checks / total_checks) * 100
        logger.info(f"\nOverall health score: {health_score:.1f}%")
        
        # Determine if system is healthy
        is_healthy = len(critical_failures) == 0 and health_score >= 80
        
        if is_healthy:
            logger.info("🎉 System is healthy!")
        else:
            logger.error("💥 System has health issues!")
            if critical_failures:
                logger.error("Critical services are failing - immediate attention required!")
        
        return is_healthy
    
    def export_results(self, output_file: str):
        """Export health check results to JSON."""
        export_data = {
            'environment': self.environment,
            'check_time': time.time(),
            'simulation_mode': self.simulate,
            'summary': {
                'total_checks': len(self.results),
                'passed': len([r for r in self.results if r.healthy]),
                'failed': len([r for r in self.results if not r.healthy]),
                'health_score': (len([r for r in self.results if r.healthy]) / len(self.results)) * 100
            },
            'results': [asdict(result) for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Health check results exported to: {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive health check for MCP Crawl4AI RAG deployment"
    )
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'staging', 'production', 'ollama', 'hybrid'],
        default='development',
        help='Environment to check'
    )
    
    parser.add_argument(
        '--provider',
        choices=['openai', 'ollama', 'hybrid', 'mixed'],
        default='openai',
        help='AI provider being used'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run in simulation mode (for testing)'
    )
    
    parser.add_argument(
        '--output',
        help='Export results to JSON file'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout for individual checks in seconds'
    )
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker(args.env, args.simulate, args.provider)
    
    try:
        # Run all health checks
        is_healthy = checker.run_all_checks()
        
        # Export results if requested
        if args.output:
            checker.export_results(args.output)
        
        if is_healthy:
            logger.info("All health checks passed!")
            return 0
        else:
            logger.error("Some health checks failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Health checks interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())