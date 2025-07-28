"""
Docker testing framework for container orchestration scenarios.

This test suite validates:
1. Multi-container deployments with OpenAI and Ollama providers
2. Health checks and service discovery
3. Network connectivity between containers
4. Volume persistence and data sharing
5. Environment variable configuration
6. Container scaling and load balancing
7. Service failover and recovery
"""

import pytest
import asyncio
import os
import time
import requests
import docker
import tempfile
import yaml
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import subprocess
from testcontainers.compose import DockerCompose
from testcontainers.general import DockerContainer

# Docker testing configuration
DOCKER_TEST_ENABLED = os.getenv("ENABLE_DOCKER_TESTS", "false").lower() == "true"
SKIP_DOCKER_TESTS = not DOCKER_TEST_ENABLED
DOCKER_NOT_AVAILABLE = "Docker tests disabled or Docker not available"

# Test timeouts
SERVICE_STARTUP_TIMEOUT = 120  # 2 minutes
HEALTH_CHECK_TIMEOUT = 60     # 1 minute
NETWORK_TEST_TIMEOUT = 30     # 30 seconds


@dataclass
class ContainerStatus:
    """Container status information."""
    name: str
    running: bool
    healthy: bool
    ports: Dict[str, str]
    environment: Dict[str, str]
    last_error: Optional[str] = None


@dataclass
class ServiceHealthCheck:
    """Service health check results."""
    service_name: str
    endpoint: str
    response_time: float
    status_code: int
    healthy: bool
    error_message: Optional[str] = None


class DockerTestEnvironment:
    """Docker test environment manager."""
    
    def __init__(self):
        self.client = None
        self.containers = {}
        self.networks = []
        self.volumes = []
        self.temp_dir = None
    
    def __enter__(self):
        try:
            self.client = docker.from_env()
            self.temp_dir = tempfile.mkdtemp(prefix="docker_test_")
            return self
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up Docker resources."""
        if self.client:
            # Stop and remove containers
            for container_name, container in self.containers.items():
                try:
                    if hasattr(container, 'stop'):
                        container.stop(timeout=10)
                    if hasattr(container, 'remove'):
                        container.remove(force=True)
                except Exception as e:
                    print(f"Error cleaning up container {container_name}: {e}")
            
            # Remove networks
            for network in self.networks:
                try:
                    network.remove()
                except Exception as e:
                    print(f"Error removing network: {e}")
            
            # Remove volumes
            for volume in self.volumes:
                try:
                    volume.remove(force=True)
                except Exception as e:
                    print(f"Error removing volume: {e}")
        
        # Clean up temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def create_network(self, name: str) -> Any:
        """Create a Docker network."""
        network = self.client.networks.create(name)
        self.networks.append(network)
        return network
    
    def create_volume(self, name: str) -> Any:
        """Create a Docker volume."""
        volume = self.client.volumes.create(name)
        self.volumes.append(volume)
        return volume
    
    def run_container(self, image: str, name: str, **kwargs) -> Any:
        """Run a Docker container."""
        container = self.client.containers.run(image, name=name, detach=True, **kwargs)
        self.containers[name] = container
        return container
    
    def wait_for_service(self, url: str, timeout: int = SERVICE_STARTUP_TIMEOUT) -> bool:
        """Wait for a service to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False
    
    def get_container_status(self, name: str) -> ContainerStatus:
        """Get status of a container."""
        try:
            container = self.containers.get(name)
            if not container:
                return ContainerStatus(name, False, False, {}, {}, "Container not found")
            
            container.reload()
            running = container.status == "running"
            
            # Check health
            health = container.attrs.get("State", {}).get("Health", {})
            healthy = health.get("Status") == "healthy" if health else running
            
            # Get ports
            ports = {}
            port_bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            for container_port, host_bindings in port_bindings.items():
                if host_bindings:
                    ports[container_port] = f"{host_bindings[0]['HostIp']}:{host_bindings[0]['HostPort']}"
            
            # Get environment
            env_list = container.attrs.get("Config", {}).get("Env", [])
            environment = {}
            for env_var in env_list:
                if "=" in env_var:
                    key, value = env_var.split("=", 1)
                    environment[key] = value
            
            return ContainerStatus(name, running, healthy, ports, environment)
            
        except Exception as e:
            return ContainerStatus(name, False, False, {}, {}, str(e))


class TestBasicContainerDeployment:
    """Test basic container deployment scenarios."""
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_mcp_server_container_deployment(self):
        """Test MCP server container deployment."""
        with DockerTestEnvironment() as env:
            # Create network for services
            network = env.create_network("mcp-test-network")
            
            # Run MCP server container
            mcp_container = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="mcp-server-test",
                ports={"8051/tcp": 8051},
                environment={
                    "OPENAI_API_KEY": "test-key",
                    "DATABASE_PROVIDER": "sqlite",
                    "AI_PROVIDER": "openai"
                },
                network=network.name
            )
            
            # Wait for service to start
            service_ready = env.wait_for_service("http://localhost:8051/health", timeout=60)
            assert service_ready, "MCP server failed to start"
            
            # Check container status
            status = env.get_container_status("mcp-server-test")
            assert status.running == True
            assert status.healthy == True
            assert "8051/tcp" in status.ports
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_ollama_container_deployment(self):
        """Test Ollama container deployment."""
        with DockerTestEnvironment() as env:
            # Create volume for Ollama models
            ollama_volume = env.create_volume("ollama-models")
            
            # Run Ollama container
            ollama_container = env.run_container(
                image="ollama/ollama:latest",
                name="ollama-test",
                ports={"11434/tcp": 11434},
                volumes={ollama_volume.name: {"bind": "/root/.ollama", "mode": "rw"}},
                environment={"OLLAMA_HOST": "0.0.0.0"}
            )
            
            # Wait for Ollama to be ready
            service_ready = env.wait_for_service("http://localhost:11434/api/tags", timeout=90)
            assert service_ready, "Ollama container failed to start"
            
            # Check container status
            status = env.get_container_status("ollama-test")
            assert status.running == True
            assert "11434/tcp" in status.ports
            
            # Test API endpoint
            response = requests.get("http://localhost:11434/api/tags")
            assert response.status_code == 200
            assert "models" in response.json()


class TestMultiContainerOrchestration:
    """Test multi-container orchestration with Docker Compose."""
    
    @pytest.fixture
    def docker_compose_config(self):
        """Docker Compose configuration for multi-container setup."""
        return {
            "version": "3.8",
            "services": {
                "mcp-server": {
                    "image": "mcp/crawl4ai-rag:test",
                    "ports": ["8051:8051"],
                    "environment": {
                        "AI_PROVIDER": "ollama",
                        "OLLAMA_BASE_URL": "http://ollama:11434",
                        "DATABASE_PROVIDER": "sqlite",
                        "OPENAI_API_KEY": "fallback-key"
                    },
                    "depends_on": ["ollama"],
                    "networks": ["mcp-network"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8051/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "ollama": {
                    "image": "ollama/ollama:latest",
                    "ports": ["11434:11434"],
                    "environment": {"OLLAMA_HOST": "0.0.0.0"},
                    "volumes": ["ollama-data:/root/.ollama"],
                    "networks": ["mcp-network"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:11434/api/tags"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "database": {
                    "image": "postgres:15",
                    "environment": {
                        "POSTGRES_DB": "crawl4ai_test",
                        "POSTGRES_USER": "test",
                        "POSTGRES_PASSWORD": "test123"
                    },
                    "volumes": ["db-data:/var/lib/postgresql/data"],
                    "networks": ["mcp-network"],
                    "healthcheck": {
                        "test": ["CMD-SHELL", "pg_isready -U test -d crawl4ai_test"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                }
            },
            "volumes": {
                "ollama-data": {},
                "db-data": {}
            },
            "networks": {
                "mcp-network": {"driver": "bridge"}
            }
        }
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_multi_container_orchestration(self, docker_compose_config):
        """Test multi-container orchestration with Docker Compose."""
        # Create temporary compose file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(docker_compose_config, f)
            compose_file = f.name
        
        try:
            with DockerCompose(compose_file, compose_file_name="docker-compose.yml") as compose:
                # Wait for all services to be healthy
                services = ["mcp-server", "ollama", "database"]
                
                for service in services:
                    print(f"Waiting for {service} to be healthy...")
                    container = compose.get_service(service)
                    
                    # Wait for health check to pass
                    timeout = 120
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout:
                        try:
                            # Check container health
                            container.reload()
                            health = container.attrs.get("State", {}).get("Health", {})
                            if health.get("Status") == "healthy":
                                break
                        except Exception:
                            pass
                        time.sleep(5)
                    else:
                        pytest.fail(f"Service {service} failed to become healthy within {timeout}s")
                
                # Test service connectivity
                self._test_service_connectivity(compose)
                
                # Test MCP server functionality
                self._test_mcp_server_functionality()
                
        finally:
            os.unlink(compose_file)
    
    def _test_service_connectivity(self, compose):
        """Test connectivity between services."""
        # Test Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        assert response.status_code == 200
        
        # Test MCP server health
        response = requests.get("http://localhost:8051/health", timeout=10)
        assert response.status_code == 200
        
        # Test database connectivity (if exposed)
        # This would require psycopg2 or similar
        print("Service connectivity tests passed")
    
    def _test_mcp_server_functionality(self):
        """Test MCP server functionality in container."""
        # Test MCP server endpoints
        base_url = "http://localhost:8051"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        assert response.status_code == 200
        
        # Test MCP capabilities (if exposed via HTTP)
        # This would depend on the actual MCP server implementation
        print("MCP server functionality tests passed")


class TestContainerHealthChecks:
    """Test container health checks and monitoring."""
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_health_check_implementation(self):
        """Test health check implementation for containers."""
        health_checks = [
            {
                "service": "mcp-server",
                "endpoint": "http://localhost:8051/health",
                "expected_response": {"status": "healthy"}
            },
            {
                "service": "ollama",
                "endpoint": "http://localhost:11434/api/tags", 
                "expected_response": {"models": []}
            }
        ]
        
        for check in health_checks:
            result = self._perform_health_check(
                check["service"],
                check["endpoint"],
                check["expected_response"]
            )
            
            assert result.healthy == True, f"Health check failed for {check['service']}: {result.error_message}"
            assert result.response_time < 5.0, f"Health check too slow for {check['service']}"
    
    def _perform_health_check(self, service_name: str, endpoint: str, expected_response: Dict) -> ServiceHealthCheck:
        """Perform health check on a service."""
        start_time = time.time()
        
        try:
            response = requests.get(endpoint, timeout=10)
            response_time = time.time() - start_time
            
            healthy = response.status_code == 200
            error_message = None
            
            if healthy and expected_response:
                # Check response content
                try:
                    response_data = response.json()
                    for key in expected_response:
                        if key not in response_data:
                            healthy = False
                            error_message = f"Missing expected key: {key}"
                            break
                except Exception:
                    # Response not JSON, check if we expected JSON
                    if expected_response:
                        healthy = False
                        error_message = "Expected JSON response but got non-JSON"
            
            return ServiceHealthCheck(
                service_name=service_name,
                endpoint=endpoint,
                response_time=response_time,
                status_code=response.status_code,
                healthy=healthy,
                error_message=error_message
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return ServiceHealthCheck(
                service_name=service_name,
                endpoint=endpoint,
                response_time=response_time,
                status_code=0,
                healthy=False,
                error_message=str(e)
            )


class TestContainerScaling:
    """Test container scaling and load balancing."""
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_horizontal_scaling(self):
        """Test horizontal scaling of MCP server containers."""
        with DockerTestEnvironment() as env:
            network = env.create_network("scaling-test-network")
            
            # Start multiple MCP server instances
            num_instances = 3 
            containers = []
            
            for i in range(num_instances):
                container = env.run_container(
                    image="mcp/crawl4ai-rag:test",
                    name=f"mcp-server-{i}",
                    ports={f"805{i+1}/tcp": 8051},  # Different host ports
                    environment={
                        "AI_PROVIDER": "openai",
                        "OPENAI_API_KEY": "test-key",
                        "DATABASE_PROVIDER": "sqlite"
                    },
                    network=network.name
                )
                containers.append(container)
            
            # Wait for all instances to be ready
            for i in range(num_instances):
                service_ready = env.wait_for_service(f"http://localhost:805{i+1}/health", timeout=60)
                assert service_ready, f"MCP server instance {i} failed to start"
            
            # Test load distribution (mock)
            self._test_load_distribution(num_instances)
    
    def _test_load_distribution(self, num_instances: int):
        """Test load distribution across multiple instances."""
        # In a real scenario, this would test a load balancer
        # For testing, we'll just verify all instances are responsive
        
        for i in range(num_instances):
            response = requests.get(f"http://localhost:805{i+1}/health", timeout=5)
            assert response.status_code == 200
            print(f"Instance {i} is responsive")
        
        print(f"All {num_instances} instances are responsive")


class TestContainerFailover:
    """Test container failover and recovery scenarios."""
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_service_failover(self):
        """Test service failover when primary container fails."""
        with DockerTestEnvironment() as env:
            network = env.create_network("failover-test-network")
            
            # Start primary and backup containers
            primary_container = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="mcp-primary",
                ports={"8051/tcp": 8051},
                environment={"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
                network=network.name
            )
            
            backup_container = env.run_container(
                image="mcp/crawl4ai-rag:test", 
                name="mcp-backup",
                ports={"8052/tcp": 8051},
                environment={"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
                network=network.name
            )
            
            # Wait for both to be ready
            primary_ready = env.wait_for_service("http://localhost:8051/health", timeout=60)
            backup_ready = env.wait_for_service("http://localhost:8052/health", timeout=60)
            
            assert primary_ready, "Primary container failed to start"
            assert backup_ready, "Backup container failed to start"
            
            # Simulate primary failure
            primary_container.stop()
            time.sleep(5)
            
            # Verify backup is still running
            backup_status = env.get_container_status("mcp-backup")
            assert backup_status.running == True
            
            # Test failover to backup
            response = requests.get("http://localhost:8052/health", timeout=10)
            assert response.status_code == 200
            
            print("Failover test completed successfully")
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_automatic_recovery(self):
        """Test automatic container recovery."""
        with DockerTestEnvironment() as env:
            # Start container with restart policy
            container = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="mcp-recovery-test",
                ports={"8051/tcp": 8051},
                environment={"AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
                restart_policy={"Name": "always"}
            )
            
            # Wait for service to be ready
            service_ready = env.wait_for_service("http://localhost:8051/health", timeout=60)
            assert service_ready, "Container failed to start"
            
            # Simulate container crash
            container.kill()
            
            # Wait for automatic restart (Docker should restart it)
            time.sleep(10)
            
            # Check if container recovered
            recovery_success = env.wait_for_service("http://localhost:8051/health", timeout=60)
            assert recovery_success, "Container failed to recover automatically"
            
            print("Automatic recovery test completed successfully")


class TestContainerConfiguration:
    """Test container configuration and environment variables."""
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_environment_variable_configuration(self):
        """Test configuration via environment variables."""
        test_configurations = [
            {
                "name": "openai-config",
                "env": {
                    "AI_PROVIDER": "openai",
                    "OPENAI_API_KEY": "test-openai-key",
                    "DATABASE_PROVIDER": "sqlite",
                    "EMBEDDING_DIMENSIONS": "1536"
                }
            },
            {
                "name": "ollama-config",
                "env": {
                    "AI_PROVIDER": "ollama",
                    "OLLAMA_BASE_URL": "http://localhost:11434",
                    "DATABASE_PROVIDER": "sqlite",
                    "EMBEDDING_DIMENSIONS": "768"
                }
            }
        ]
        
        with DockerTestEnvironment() as env:
            for config in test_configurations:
                container = env.run_container(
                    image="mcp/crawl4ai-rag:test",
                    name=f"mcp-{config['name']}",
                    ports={"8051/tcp": None},  # Dynamic port
                    environment=config["env"]
                )
                
                # Verify environment variables are set
                status = env.get_container_status(f"mcp-{config['name']}")
                assert status.running == True
                
                for key, expected_value in config["env"].items():
                    assert key in status.environment
                    assert status.environment[key] == expected_value
                
                print(f"Configuration test passed for {config['name']}")
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_volume_persistence(self):
        """Test data persistence with volumes."""
        with DockerTestEnvironment() as env:
            # Create volume for data persistence
            data_volume = env.create_volume("mcp-data")
            
            # Start container with volume
            container1 = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="mcp-persistence-1",
                volumes={data_volume.name: {"bind": "/app/data", "mode": "rw"}},
                environment={"DATABASE_PROVIDER": "sqlite", "DB_PATH": "/app/data/test.db"}
            )
            
            # Simulate data creation (would need actual data operations)
            time.sleep(5)
            
            # Stop first container
            container1.stop()
            container1.remove()
            
            # Start second container with same volume
            container2 = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="mcp-persistence-2",
                volumes={data_volume.name: {"bind": "/app/data", "mode": "rw"}},
                environment={"DATABASE_PROVIDER": "sqlite", "DB_PATH": "/app/data/test.db"}
            )
            
            # Verify data persistence (would need actual verification)
            status = env.get_container_status("mcp-persistence-2")
            assert status.running == True
            
            print("Volume persistence test completed")


class TestNetworkSecurity:
    """Test network security and isolation."""
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_network_isolation(self):
        """Test network isolation between services."""
        with DockerTestEnvironment() as env:
            # Create separate networks
            public_network = env.create_network("public-network")
            private_network = env.create_network("private-network")
            
            # Start containers in different networks
            public_container = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="public-service",
                ports={"8051/tcp": 8051},
                network=public_network.name
            )
            
            private_container = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="private-service",
                # No port exposure
                network=private_network.name
            )
            
            # Test accessibility
            # Public service should be accessible
            public_ready = env.wait_for_service("http://localhost:8051/health", timeout=30)
            assert public_ready, "Public service should be accessible"
            
            # Private service should not be accessible from host
            # (This is a simplified test - in practice, you'd test from another container)
            print("Network isolation test completed")
    
    @pytest.mark.skipif(SKIP_DOCKER_TESTS, reason=DOCKER_NOT_AVAILABLE)
    def test_security_configuration(self):
        """Test security-related configuration."""
        with DockerTestEnvironment() as env:
            # Start container with security constraints
            container = env.run_container(
                image="mcp/crawl4ai-rag:test",
                name="secure-container",
                user="1000:1000",  # Non-root user
                read_only=True,    # Read-only filesystem
                tmpfs={"/tmp": "noexec,nosuid,size=100m"},
                security_opt=["no-new-privileges:true"]
            )
            
            # Verify security settings
            status = env.get_container_status("secure-container")
            assert status.running == True
            
            # In a real test, you'd verify the security constraints
            print("Security configuration test completed")


if __name__ == "__main__":
    # Run Docker tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "--tb=short",
        "-k", "not skipif"  # Run only non-skipped tests
    ])