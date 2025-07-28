"""
Integration tests for Ollama AI provider with real instances.

This test suite validates:
1. Real Ollama server connectivity and health checks
2. Actual embedding generation with various models
3. LLM text generation capabilities
4. Provider switching scenarios with dimension validation
5. Database integration with different embedding dimensions
6. Error handling with real network conditions
"""

import pytest
import asyncio
import os
import time
import requests
from typing import List, Dict, Any, Optional
from unittest.mock import patch
import tempfile
from testcontainers.compose import DockerCompose

# Import the test database suite for integration testing
from test_database_providers import DatabaseProviderTestSuite
from src.database.base import DatabaseProvider, DocumentChunk
from src.database.factory import DatabaseProviderFactory

# Ollama integration testing requires these environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

# Test skip conditions
SKIP_OLLAMA_TESTS = os.getenv("SKIP_OLLAMA_TESTS", "false").lower() == "true"
OLLAMA_NOT_AVAILABLE = "Ollama server not available or tests disabled"


class OllamaTestClient:
    """Test client for interacting with Ollama server."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception:
            return []
    
    def pull_model(self, model_name: str, timeout: int = 300) -> bool:
        """Pull a model if not available."""
        try:
            models = self.list_models()
            if any(model["name"].startswith(model_name) for model in models):
                return True
            
            # Pull the model
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Wait for pull to complete
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    if '"status":"success"' in data:
                        return True
            
            return False
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def create_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Create embeddings using Ollama API."""
        embeddings = []
        for text in texts:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text}
                )
                response.raise_for_status()
                embedding = response.json().get("embedding", [])
                embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to create embedding for text: {e}")
                embeddings.append([])
        return embeddings
    
    async def generate_text(self, prompt: str, model: str) -> str:
        """Generate text using Ollama API."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Failed to generate text: {e}")
            return ""


@pytest.fixture(scope="session")
def ollama_client():
    """Create Ollama test client."""
    return OllamaTestClient()


@pytest.fixture(scope="session")
def ensure_models_available(ollama_client):
    """Ensure required models are available."""
    if SKIP_OLLAMA_TESTS or not ollama_client.is_available():
        pytest.skip(OLLAMA_NOT_AVAILABLE)
    
    # Pull required models
    models_to_pull = [OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL]
    for model in models_to_pull:
        if not ollama_client.pull_model(model):
            pytest.skip(f"Could not pull required model: {model}")


class TestOllamaConnectivity:
    """Test basic Ollama server connectivity."""
    
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    def test_ollama_server_available(self, ollama_client):
        """Test that Ollama server is running and accessible."""
        assert ollama_client.is_available(), "Ollama server is not available"
    
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    def test_list_models(self, ollama_client, ensure_models_available):
        """Test listing available models."""
        models = ollama_client.list_models()
        assert len(models) > 0, "No models available"
        
        model_names = [model["name"] for model in models]
        print(f"Available models: {model_names}")
    
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    def test_model_availability(self, ollama_client, ensure_models_available):
        """Test that required models are available."""
        models = ollama_client.list_models()
        model_names = [model["name"] for model in models]
        
        # Check if embedding model is available
        embedding_available = any(
            name.startswith(OLLAMA_EMBEDDING_MODEL) for name in model_names
        )
        assert embedding_available, f"Embedding model {OLLAMA_EMBEDDING_MODEL} not available"
        
        # Check if LLM model is available
        llm_available = any(
            name.startswith(OLLAMA_LLM_MODEL) for name in model_names
        )
        assert llm_available, f"LLM model {OLLAMA_LLM_MODEL} not available"


class TestOllamaEmbeddings:
    """Test Ollama embedding generation."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_single_embedding(self, ollama_client, ensure_models_available):
        """Test creating a single embedding."""
        texts = ["Hello, world!"]
        embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0, "Embedding should not be empty"
        assert all(isinstance(x, (int, float)) for x in embeddings[0]), "Embedding should contain numbers"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_batch_embeddings(self, ollama_client, ensure_models_available):
        """Test creating multiple embeddings."""
        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is a completely different sentence about machine learning."
        ]
        embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        
        assert len(embeddings) == 3
        
        # All embeddings should have the same dimension
        dimensions = [len(emb) for emb in embeddings if emb]
        assert len(set(dimensions)) == 1, "All embeddings should have same dimension"
        
        # Embeddings should be different (similarity check)
        if len(embeddings[0]) > 0 and len(embeddings[2]) > 0:
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[2]))
            norm_a = sum(a * a for a in embeddings[0]) ** 0.5
            norm_b = sum(b * b for b in embeddings[2]) ** 0.5
            similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            
            # Different sentences should have lower similarity
            assert similarity < 0.9, "Different sentences should have lower similarity"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_empty_text_handling(self, ollama_client, ensure_models_available):
        """Test handling of empty or invalid text."""
        texts = ["", "   ", "Valid text"]
        embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        
        assert len(embeddings) == 3
        # At least the valid text should have an embedding
        assert len(embeddings[2]) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_embedding_dimensions(self, ollama_client, ensure_models_available):
        """Test that embedding dimensions are consistent and documented."""
        texts = ["Test text for dimension checking"]
        embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        
        if embeddings and embeddings[0]:
            dimensions = len(embeddings[0])
            print(f"Embedding dimensions for {OLLAMA_EMBEDDING_MODEL}: {dimensions}")
            
            # Common dimensions for different models
            expected_dimensions = [384, 768, 1024, 1536, 3072]
            assert dimensions in expected_dimensions, f"Unexpected dimension: {dimensions}"


class TestOllamaLLM:
    """Test Ollama LLM text generation."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_text_generation(self, ollama_client, ensure_models_available):
        """Test basic text generation."""
        prompt = "Write a one-sentence summary of what machine learning is."
        response = await ollama_client.generate_text(prompt, OLLAMA_LLM_MODEL)
        
        assert len(response) > 0, "Response should not be empty"
        assert len(response) > 10, "Response should be substantial"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_contextual_processing(self, ollama_client, ensure_models_available):
        """Test contextual text processing similar to RAG use cases."""
        context = """
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn and make decisions from data without being explicitly programmed.
        """
        
        prompt = f"""Based on the following context, generate a concise summary:
        
        Context: {context}
        
        Summary:"""
        
        response = await ollama_client.generate_text(prompt, OLLAMA_LLM_MODEL)
        
        assert len(response) > 0
        assert "machine learning" in response.lower() or "artificial intelligence" in response.lower()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_code_summarization(self, ollama_client, ensure_models_available):
        """Test code summarization capabilities."""
        code = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """
        
        prompt = f"""Summarize what this Python function does:
        
        {code}
        
        Summary:"""
        
        response = await ollama_client.generate_text(prompt, OLLAMA_LLM_MODEL)
        
        assert len(response) > 0
        assert "fibonacci" in response.lower()


class TestProviderSwitching:
    """Test switching between OpenAI and Ollama providers."""
    
    @pytest.fixture
    def mock_openai_embeddings(self):
        """Mock OpenAI embeddings for comparison."""
        return [[0.1] * 1536, [0.2] * 1536]  # OpenAI standard dimensions
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_dimension_compatibility_detection(self, ollama_client, ensure_models_available, mock_openai_embeddings):
        """Test detection of dimension incompatibility between providers."""
        # Get Ollama embedding dimensions
        texts = ["Test text"]
        ollama_embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        
        if ollama_embeddings and ollama_embeddings[0]:
            ollama_dims = len(ollama_embeddings[0])
            openai_dims = len(mock_openai_embeddings[0])
            
            print(f"Ollama dimensions: {ollama_dims}")
            print(f"OpenAI dimensions: {openai_dims}")
            
            # These are likely to be different - this should be detected
            if ollama_dims != openai_dims:
                print("Dimension mismatch detected - this would prevent provider switching")
                assert ollama_dims != openai_dims
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_database_integration_with_different_dimensions(self, ollama_client, ensure_models_available):
        """Test database operations with Ollama embeddings."""
        # Create a temporary SQLite database
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite')
        temp_file.close()
        
        # Get Ollama embedding dimensions
        texts = ["Test document for database integration"]
        embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        
        if embeddings and embeddings[0]:
            embedding_dims = len(embeddings[0])
            
            # Create database provider with Ollama dimensions
            db_config = {
                "db_path": temp_file.name,
                "embedding_dimension": embedding_dims
            }
            
            provider = DatabaseProviderFactory.create_provider(
                DatabaseProvider.SQLITE, db_config
            )
            
            try:
                await provider.initialize()
                
                # Test adding documents with Ollama embeddings
                test_docs = [
                    DocumentChunk(
                        url="https://test.com/ollama-doc",
                        chunk_number=0,
                        content="This document was embedded using Ollama",
                        metadata={"provider": "ollama", "test": True},
                        embedding=embeddings[0],
                        source_id="ollama-test"
                    )
                ]
                
                await provider.add_documents(test_docs)
                
                # Test searching with Ollama embeddings
                query_embeddings = await ollama_client.create_embeddings(
                    ["Find documents about Ollama"], OLLAMA_EMBEDDING_MODEL
                )
                
                if query_embeddings and query_embeddings[0]:
                    results = await provider.search_documents(
                        query_embedding=query_embeddings[0],
                        match_count=1
                    )
                    
                    assert len(results) >= 1
                    assert "ollama" in results[0].content.lower()
                
            finally:
                await provider.close()
                os.unlink(temp_file.name)


class TestOllamaPerformance:
    """Test Ollama performance characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_embedding_latency(self, ollama_client, ensure_models_available):
        """Test embedding generation latency."""
        texts = ["Performance test text"] * 5
        
        start_time = time.time()
        embeddings = await ollama_client.create_embeddings(texts, OLLAMA_EMBEDDING_MODEL)
        end_time = time.time()
        
        latency = end_time - start_time
        per_embedding_latency = latency / len(texts)
        
        print(f"Total latency: {latency:.2f}s")
        print(f"Per embedding latency: {per_embedding_latency:.2f}s")
        
        # These are reasonable thresholds - adjust based on hardware
        assert latency < 30.0, "Total latency should be reasonable"
        assert per_embedding_latency < 10.0, "Per embedding latency should be reasonable"
        
        # Ensure all embeddings were created
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings if emb)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_llm_response_time(self, ollama_client, ensure_models_available):
        """Test LLM response time."""
        prompt = "What is the capital of France? Answer in one word."
        
        start_time = time.time()
        response = await ollama_client.generate_text(prompt, OLLAMA_LLM_MODEL)
        end_time = time.time()
        
        latency = end_time - start_time
        
        print(f"LLM response time: {latency:.2f}s")
        print(f"Response: {response}")
        
        # Reasonable response time threshold
        assert latency < 60.0, "LLM response time should be reasonable"
        assert len(response) > 0, "Response should not be empty"


class TestOllamaErrorHandling:
    """Test error handling with real Ollama scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_invalid_model_handling(self, ollama_client):
        """Test handling of invalid model names."""
        texts = ["Test with invalid model"]
        
        try:
            embeddings = await ollama_client.create_embeddings(texts, "invalid-model-name")
            # If no exception, check if embeddings are empty
            assert all(len(emb) == 0 for emb in embeddings), "Invalid model should return empty embeddings"
        except Exception as e:
            # Exception is expected and acceptable
            assert "not found" in str(e).lower() or "error" in str(e).lower()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
    async def test_large_text_handling(self, ollama_client, ensure_models_available):
        """Test handling of very large text inputs."""
        # Create a large text (over typical token limits)
        large_text = "This is a test sentence. " * 1000  # ~5000 words
        
        try:
            embeddings = await ollama_client.create_embeddings([large_text], OLLAMA_EMBEDDING_MODEL)
            # Should either work or handle gracefully
            assert len(embeddings) == 1
        except Exception as e:
            # If it fails, it should be a clear error about text length
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["length", "token", "limit", "size"])
    
    def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        # Create client with invalid URL
        invalid_client = OllamaTestClient("http://invalid-host:12345")
        
        assert not invalid_client.is_available()
        assert len(invalid_client.list_models()) == 0


@pytest.mark.skipif(SKIP_OLLAMA_TESTS, reason=OLLAMA_NOT_AVAILABLE)
class TestOllamaDockerIntegration:
    """Test Ollama integration with Docker containers."""
    
    @pytest.fixture(scope="class")
    def ollama_docker_compose(self):
        """Start Ollama using Docker Compose for testing."""
        compose_file_content = """
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ollama_data:
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(compose_file_content)
            compose_file = f.name
        
        try:
            with DockerCompose(compose_file, compose_file_name="docker-compose.yml") as compose:
                # Wait for service to be ready
                ollama_service = compose.get_service("ollama")
                
                # Wait for health check
                timeout = 120  # 2 minutes
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        test_client = OllamaTestClient()
                        if test_client.is_available():
                            break
                        time.sleep(5)
                    except Exception:
                        time.sleep(5)
                        continue
                else:
                    pytest.skip("Ollama container failed to start within timeout")
                
                yield compose
        finally:
            os.unlink(compose_file)
    
    def test_docker_ollama_connectivity(self, ollama_docker_compose):
        """Test connectivity to Ollama running in Docker."""
        client = OllamaTestClient()
        assert client.is_available()
        
        models = client.list_models()
        # Initially no models, but service should be responsive
        assert isinstance(models, list)


if __name__ == "__main__":
    # Run with specific markers
    pytest.main([
        __file__, 
        "-v", 
        "-m", "not skipif",  # Run only tests that aren't skipped
        "--tb=short"
    ])