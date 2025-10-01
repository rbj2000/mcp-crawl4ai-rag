"""
Advanced Docker testing using testcontainers-python
Provides real container instances for integration testing
"""
import pytest
import requests
import time
from testcontainers.compose import DockerCompose
from testcontainers.generic import GenericContainer
from testcontainers.neo4j import Neo4jContainer
from pathlib import Path

class TestMCPWithTestcontainers:
    """Test MCP server using testcontainers"""
    
    @pytest.fixture(scope="class")
    def sqlite_container(self):
        """SQLite MCP container"""
        with GenericContainer(
            image="mcp-crawl4ai-rag:sqlite"
        ).with_env("VECTOR_DB_PROVIDER", "sqlite") \
         .with_env("OPENAI_API_KEY", "test_key") \
         .with_env("SQLITE_DB_PATH", "/tmp/test.sqlite") \
         .with_exposed_ports(8051) as container:
            
            # Wait for service to be ready
            self._wait_for_service(
                f"http://localhost:{container.get_exposed_port(8051)}/health"
            )
            
            yield container
    
    @pytest.fixture(scope="class")
    def neo4j_with_mcp(self):
        """Neo4j + MCP server using docker-compose"""
        compose_path = Path(__file__).parent.parent
        
        with DockerCompose(
            compose_file_name="docker-compose.yml",
            compose_file_path=str(compose_path),
            compose_env_file=".env.test"
        ).with_profile("neo4j") as compose:
            
            # Wait for services
            self._wait_for_service("http://localhost:7474", timeout=120)  # Neo4j
            self._wait_for_service("http://localhost:8055/health", timeout=60)  # MCP
            
            yield compose
    
    def _wait_for_service(self, url: str, timeout: int = 60) -> bool:
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
        raise TimeoutError(f"Service {url} not ready after {timeout}s")
    
    def test_sqlite_container_functionality(self, sqlite_container):
        """Test SQLite container functionality"""
        port = sqlite_container.get_exposed_port(8051)
        base_url = f"http://localhost:{port}"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["provider"] == "sqlite"
        
        # Test MCP tools endpoint
        response = requests.get(f"{base_url}/tools")
        assert response.status_code == 200
        
        tools = response.json()
        tool_names = [tool["name"] for tool in tools.get("tools", [])]
        assert "crawl_single_page" in tool_names
        assert "perform_rag_query" in tool_names
    
    def test_neo4j_integration(self, neo4j_with_mcp):
        """Test Neo4j + MCP integration"""
        # Test Neo4j is accessible
        response = requests.get("http://localhost:7474")
        assert response.status_code == 200
        
        # Test MCP server with Neo4j backend
        response = requests.get("http://localhost:8055/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["provider"] == "neo4j_vector"
        
        # Test knowledge graph tools are available
        response = requests.get("http://localhost:8055/tools")
        tools = response.json()
        tool_names = [tool["name"] for tool in tools.get("tools", [])]
        
        # Should have knowledge graph tools
        assert "parse_github_repository" in tool_names
        assert "query_knowledge_graph" in tool_names

class TestProviderIsolation:
    """Test that providers are properly isolated"""
    
    def test_multiple_providers_parallel(self):
        """Test multiple providers can run in parallel"""
        containers = []
        
        try:
            # Start SQLite container
            sqlite_container = GenericContainer(
                "mcp-crawl4ai-rag:sqlite"
            ).with_env("VECTOR_DB_PROVIDER", "sqlite") \
             .with_env("OPENAI_API_KEY", "test_key") \
             .with_exposed_ports(8051)
            
            sqlite_container.start()
            containers.append(sqlite_container)
            
            # Start another SQLite container on different port
            sqlite_container2 = GenericContainer(
                "mcp-crawl4ai-rag:sqlite"  
            ).with_env("VECTOR_DB_PROVIDER", "sqlite") \
             .with_env("OPENAI_API_KEY", "test_key") \
             .with_env("PORT", "8052") \
             .with_exposed_ports(8052)
            
            sqlite_container2.start()
            containers.append(sqlite_container2)
            
            # Wait for both to be ready
            time.sleep(15)
            
            # Test both are accessible
            port1 = sqlite_container.get_exposed_port(8051)
            port2 = sqlite_container2.get_exposed_port(8052)
            
            response1 = requests.get(f"http://localhost:{port1}/health")
            response2 = requests.get(f"http://localhost:{port2}/health")
            
            assert response1.status_code == 200
            assert response2.status_code == 200
            
            # Both should be healthy and isolated
            health1 = response1.json()
            health2 = response2.json()
            
            assert health1["status"] == "healthy"
            assert health2["status"] == "healthy"
            
        finally:
            # Cleanup
            for container in containers:
                try:
                    container.stop()
                except:
                    pass

class TestDockerComposeFunctionality:
    """Test docker-compose configurations"""
    
    def test_docker_compose_profiles(self):
        """Test all docker-compose profiles work"""
        compose_path = Path(__file__).parent.parent
        profiles = ["sqlite", "neo4j"]  # Skip external deps for CI
        
        for profile in profiles:
            print(f"Testing docker-compose profile: {profile}")
            
            with DockerCompose(
                compose_file_name="docker-compose.yml",
                compose_file_path=str(compose_path)
            ).with_profile(profile) as compose:
                
                # Give services time to start
                time.sleep(20)
                
                # Test appropriate service is running
                if profile == "sqlite":
                    response = requests.get("http://localhost:8052/health")
                    assert response.status_code == 200
                    
                elif profile == "neo4j":
                    # Test Neo4j is running
                    response = requests.get("http://localhost:7474")
                    assert response.status_code == 200
                    
                    # Test MCP server is running
                    response = requests.get("http://localhost:8055/health")
                    assert response.status_code == 200

if __name__ == "__main__":
    # Install with: pip install testcontainers pytest-docker
    # Run with: python -m pytest tests/test_testcontainers.py -v -s
    pytest.main([__file__, "-v", "-s", "--tb=short"])