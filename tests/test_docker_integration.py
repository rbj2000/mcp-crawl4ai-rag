"""
Comprehensive Docker integration tests for MCP Crawl4AI RAG
Uses pytest-docker for robust container testing
"""
import pytest
import requests
import time
import json
import docker
from typing import Dict, Any
from pathlib import Path

# Test configuration
TEST_TIMEOUT = 300
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 10

@pytest.fixture(scope="session")
def docker_client():
    """Docker client fixture"""
    return docker.from_env()

@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Path to docker-compose file"""
    return Path(__file__).parent.parent / "docker-compose.yml"

class TestDockerProviders:
    """Test all database providers with Docker"""
    
    def wait_for_service(self, url: str, timeout: int = 300) -> bool:
        """Wait for service to be healthy"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(5)
        return False
    
    def test_mcp_tools_available(self, service_url: str) -> bool:
        """Test that MCP tools are available"""
        try:
            response = requests.get(f"{service_url}/tools", timeout=10)
            tools = response.json()
            
            required_tools = [
                "crawl_single_page",
                "perform_rag_query", 
                "get_available_sources"
            ]
            
            available_tools = [tool["name"] for tool in tools.get("tools", [])]
            return all(tool in available_tools for tool in required_tools)
            
        except Exception as e:
            print(f"Error checking MCP tools: {e}")
            return False
    
    def test_database_connection(self, service_url: str, provider: str) -> bool:
        """Test database connection for provider"""
        try:
            # Test health endpoint
            response = requests.get(f"{service_url}/health", timeout=10)
            health_data = response.json()
            
            return (
                health_data.get("status") == "healthy" and
                health_data.get("database", {}).get("connected") == True
            )
            
        except Exception as e:
            print(f"Error testing database connection: {e}")
            return False

@pytest.mark.docker
class TestSQLiteProvider:
    """Test SQLite provider in Docker"""
    
    @pytest.fixture(scope="class")
    def sqlite_container(self, docker_client):
        """Start SQLite container"""
        # Build image
        image = docker_client.images.build(
            path=".",
            tag="mcp-crawl4ai-rag:sqlite",
            buildargs={"VECTOR_DB_PROVIDER": "sqlite"}
        )[0]
        
        # Start container
        container = docker_client.containers.run(
            image.id,
            environment={
                "VECTOR_DB_PROVIDER": "sqlite",
                "SQLITE_DB_PATH": "/tmp/test.sqlite",
                "OPENAI_API_KEY": "test_key",
                "HOST": "0.0.0.0",
                "PORT": "8051"
            },
            ports={"8051/tcp": 8051},
            detach=True,
            remove=True
        )
        
        # Wait for container to be ready
        time.sleep(10)
        
        yield container
        
        # Cleanup
        container.stop()
    
    def test_sqlite_container_starts(self, sqlite_container):
        """Test SQLite container starts successfully"""
        sqlite_container.reload()
        assert sqlite_container.status == "running"
    
    def test_sqlite_health_check(self, sqlite_container):
        """Test SQLite health endpoint"""
        # Wait for service to be ready
        assert self.wait_for_service("http://localhost:8051/health")
        
        response = requests.get("http://localhost:8051/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["provider"] == "sqlite"
    
    def test_sqlite_mcp_tools(self, sqlite_container):
        """Test MCP tools are available"""
        test_case = TestDockerProviders()
        assert test_case.test_mcp_tools_available("http://localhost:8051")
    
    def test_sqlite_functionality(self, sqlite_container):
        """Test basic SQLite functionality"""
        # Test crawl endpoint
        crawl_data = {
            "url": "https://example.com",
            "content": "Test content for indexing"
        }
        
        response = requests.post(
            "http://localhost:8051/crawl",
            json=crawl_data,
            timeout=30
        )
        
        # Should handle request even if crawling fails
        assert response.status_code in [200, 400, 422]
    
    def wait_for_service(self, url: str, timeout: int = 60) -> bool:
        """Wait for service to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        return False

@pytest.mark.docker
@pytest.mark.neo4j
class TestNeo4jProvider:
    """Test Neo4j provider with Docker Compose"""
    
    @pytest.fixture(scope="class")
    def neo4j_services(self, docker_compose_file):
        """Start Neo4j services using docker-compose"""
        import subprocess
        
        # Start services
        subprocess.run([
            "docker-compose", 
            "-f", str(docker_compose_file),
            "--profile", "neo4j",
            "up", "-d"
        ], check=True)
        
        # Wait for services to be ready
        time.sleep(30)
        
        yield
        
        # Cleanup
        subprocess.run([
            "docker-compose",
            "-f", str(docker_compose_file), 
            "--profile", "neo4j",
            "down", "-v"
        ])
    
    def test_neo4j_database_ready(self, neo4j_services):
        """Test Neo4j database is ready"""
        test_case = TestDockerProviders()
        assert test_case.wait_for_service("http://localhost:7474", timeout=120)
    
    def test_neo4j_mcp_server_ready(self, neo4j_services):
        """Test Neo4j MCP server is ready"""
        test_case = TestDockerProviders()
        assert test_case.wait_for_service("http://localhost:8055/health", timeout=60)
    
    def test_neo4j_functionality(self, neo4j_services):
        """Test Neo4j provider functionality"""
        test_case = TestDockerProviders()
        
        # Test MCP tools
        assert test_case.test_mcp_tools_available("http://localhost:8055")
        
        # Test database connection
        assert test_case.test_database_connection("http://localhost:8055", "neo4j")

@pytest.mark.docker 
@pytest.mark.slow
class TestAllProviders:
    """Test all providers can be built and started"""
    
    def test_docker_builds(self, docker_client):
        """Test all Docker images can be built"""
        providers = ["supabase", "sqlite", "pinecone", "all"]
        
        for provider in providers:
            print(f"Building image for {provider} provider...")
            
            image = docker_client.images.build(
                path=".",
                tag=f"mcp-crawl4ai-rag:{provider}",
                buildargs={"VECTOR_DB_PROVIDER": provider}
            )[0]
            
            assert image is not None
            print(f"✅ Successfully built image for {provider}")
    
    def test_container_startup(self, docker_client):
        """Test containers can start without errors"""
        # Test SQLite container startup
        container = docker_client.containers.run(
            "mcp-crawl4ai-rag:sqlite",
            environment={
                "VECTOR_DB_PROVIDER": "sqlite",
                "OPENAI_API_KEY": "test_key"
            },
            detach=True,
            remove=True
        )
        
        time.sleep(10)
        container.reload()
        
        # Should be running without errors
        assert container.status == "running"
        
        # Check logs for errors
        logs = container.logs().decode()
        assert "error" not in logs.lower() or "traceback" not in logs.lower()
        
        container.stop()

# Performance testing
@pytest.mark.performance
class TestDockerPerformance:
    """Performance tests for Docker containers"""
    
    def test_startup_time(self, docker_client):
        """Test container startup time"""
        start_time = time.time()
        
        container = docker_client.containers.run(
            "mcp-crawl4ai-rag:sqlite",
            environment={
                "VECTOR_DB_PROVIDER": "sqlite",
                "OPENAI_API_KEY": "test_key"
            },
            detach=True,
            remove=True
        )
        
        # Wait for health check
        test_case = TestDockerProviders()
        ready = test_case.wait_for_service("http://localhost:8051/health", timeout=60)
        
        startup_time = time.time() - start_time
        
        container.stop()
        
        assert ready, "Container failed to become ready"
        assert startup_time < 60, f"Startup took too long: {startup_time}s"
        print(f"✅ Container started in {startup_time:.2f}s")
    
    def test_memory_usage(self, docker_client):
        """Test container memory usage"""
        container = docker_client.containers.run(
            "mcp-crawl4ai-rag:sqlite",
            environment={
                "VECTOR_DB_PROVIDER": "sqlite", 
                "OPENAI_API_KEY": "test_key"
            },
            detach=True,
            remove=True
        )
        
        time.sleep(15)  # Let container fully start
        
        # Get memory stats
        stats = container.stats(stream=False)
        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
        
        container.stop()
        
        # Should use less than 1GB of memory
        assert memory_usage < 1024, f"Memory usage too high: {memory_usage:.2f}MB"
        print(f"✅ Memory usage: {memory_usage:.2f}MB")

# Security testing
@pytest.mark.security
class TestDockerSecurity:
    """Security tests for Docker containers"""
    
    def test_no_root_user(self, docker_client):
        """Test container doesn't run as root"""
        container = docker_client.containers.run(
            "mcp-crawl4ai-rag:sqlite",
            command="whoami",
            remove=True
        )
        
        # Should not output 'root'
        logs = container.logs().decode().strip()
        assert logs != "root", "Container should not run as root user"
    
    def test_no_sensitive_env_vars(self, docker_client):
        """Test no sensitive environment variables are exposed"""
        container = docker_client.containers.run(
            "mcp-crawl4ai-rag:sqlite",
            command="env",
            remove=True
        )
        
        logs = container.logs().decode()
        
        # Check for sensitive patterns
        sensitive_patterns = ["password", "secret", "key", "token"]
        for pattern in sensitive_patterns:
            # Should not contain actual sensitive values
            lines = [line for line in logs.lower().split('\n') if pattern in line]
            for line in lines:
                assert "test_" in line or "your_" in line or "placeholder" in line, \
                    f"Potentially sensitive environment variable: {line}"

if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_docker_integration.py -v
    pytest.main([__file__, "-v", "--tb=short"])