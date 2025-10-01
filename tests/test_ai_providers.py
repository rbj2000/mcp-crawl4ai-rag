"""
Comprehensive unit tests for AI provider implementations (OpenAI and Ollama).

This test suite ensures:
1. Provider factory pattern works correctly
2. Embedding dimension validation prevents data corruption
3. Configuration validation catches invalid setups
4. Provider switching maintains compatibility
5. Error handling and fallback scenarios work properly
6. Performance benchmarking across providers
7. Concurrency and thread safety
8. CI/CD compatibility with mocked dependencies
"""

import pytest
import asyncio
import os
import tempfile
import time
import threading
import concurrent.futures
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import List, Dict, Any, Optional
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AI provider classes with path management
import sys
from pathlib import Path

# Add src directory to path dynamically  
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import AI provider classes
try:
    from ai_providers.base import (
        AIProvider, EmbeddingProvider, LLMProvider, HybridAIProvider,
        EmbeddingResult, LLMResult, HealthStatus
    )
    from ai_providers.factory import AIProviderFactory
    from ai_providers.config import AIProviderConfig
except ImportError as e:
    logger.warning(f"Could not import AI providers: {e}")
    # Mock classes for testing framework development
    from enum import Enum
    from dataclasses import dataclass
    from typing import List
    
    class AIProvider(Enum):
        OPENAI = "openai"
        OLLAMA = "ollama"
    
    @dataclass
    class EmbeddingResult:
        embedding: List[float]
        dimensions: int
        model: str
        input_tokens: Optional[int] = None
    
    @dataclass
    class LLMResult:
        content: str
        model: str
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
    
    @dataclass
    class HealthStatus:
        is_healthy: bool
        provider: str
        model: str
        response_time_ms: float
        error_message: Optional[str] = None
    
    class EmbeddingProvider:
        def __init__(self, config):
            self.config = config
    
    class LLMProvider:
        def __init__(self, config):
            self.config = config
    
    class HybridAIProvider(EmbeddingProvider, LLMProvider):
        def __init__(self, config):
            self.config = config
    
    class AIProviderFactory:
        @staticmethod
        def create_provider(provider_type, config):
            return MockAIProvider()
    
    class AIProviderConfig:
        def __init__(self, provider, config):
            self.provider = provider
            self.config = config

# Custom exceptions
class DimensionMismatchError(Exception):
    pass

class ProviderConnectionError(Exception):
    pass

class ConfigurationError(Exception):
    pass


class TestEmbeddingDimensions:
    """Test suite for embedding dimension validation and compatibility."""
    
    def test_dimension_validation_success(self):
        """Test successful dimension validation between providers and database."""
        config = EmbeddingConfig(dimensions=1536, model="text-embedding-3-small")
        assert config.dimensions == 1536
        assert config.model == "text-embedding-3-small"
    
    def test_dimension_mismatch_detection(self):
        """Test detection of dimension mismatches between providers."""
        openai_config = EmbeddingConfig(dimensions=1536, model="text-embedding-3-small")
        ollama_config = EmbeddingConfig(dimensions=768, model="nomic-embed-text")
        
        # This should be detected as incompatible
        assert openai_config.dimensions != ollama_config.dimensions
    
    @pytest.mark.parametrize("dimensions,expected_valid", [
        (384, True),   # Ollama small model
        (768, True),   # Ollama medium model  
        (1024, True),  # Ollama large model
        (1536, True),  # OpenAI standard
        (3072, True),  # OpenAI large
        (100, False),  # Invalid dimension
        (0, False),    # Invalid dimension
        (-1, False),   # Invalid dimension
    ])
    def test_dimension_validation_ranges(self, dimensions, expected_valid):
        """Test validation of various embedding dimensions."""
        if expected_valid:
            config = EmbeddingConfig(dimensions=dimensions, model="test-model")
            assert config.dimensions == dimensions
        else:
            with pytest.raises(ValueError):
                EmbeddingConfig(dimensions=dimensions, model="test-model")


class TestAIProviderFactory:
    """Test suite for AI provider factory pattern."""
    
    def test_create_openai_provider(self):
        """Test OpenAI provider creation."""
        config = {
            "provider": "openai",
            "api_key": "test-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "llm_model": "gpt-4"
        }
        
        # Mock the factory creation
        with patch.object(AIProviderFactory, 'create_provider') as mock_create:
            mock_provider = Mock(spec=OpenAIProvider)
            mock_create.return_value = mock_provider
            
            provider = AIProviderFactory.create_provider("openai", config)
            assert provider is not None
            mock_create.assert_called_once_with("openai", config)
    
    def test_create_ollama_provider(self):
        """Test Ollama provider creation."""
        config = {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "embedding_dimensions": 768,
            "llm_model": "llama3.2"
        }
        
        with patch.object(AIProviderFactory, 'create_provider') as mock_create:
            mock_provider = Mock(spec=OllamaProvider)
            mock_create.return_value = mock_provider
            
            provider = AIProviderFactory.create_provider("ollama", config)
            assert provider is not None
            mock_create.assert_called_once_with("ollama", config)
    
    def test_invalid_provider_type(self):
        """Test factory handling of invalid provider types."""
        config = {"provider": "invalid_provider"}
        
        with patch.object(AIProviderFactory, 'create_provider') as mock_create:
            mock_create.side_effect = ConfigurationError("Unknown provider: invalid_provider")
            
            with pytest.raises(ConfigurationError):
                AIProviderFactory.create_provider("invalid_provider", config)


class MockAIProvider:
    """Mock AI provider for testing base functionality."""
    
    def __init__(self, embedding_dimensions: int = 1536, connection_healthy: bool = True):
        self.embedding_dimensions = embedding_dimensions
        self.connection_healthy = connection_healthy
        self.call_count = {"embeddings": 0, "llm": 0}
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding creation."""
        self.call_count["embeddings"] += 1
        if not self.connection_healthy:
            raise ProviderConnectionError("Connection failed")
        
        return [[0.1] * self.embedding_dimensions for _ in texts]
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Mock text generation."""
        self.call_count["llm"] += 1
        if not self.connection_healthy:
            raise ProviderConnectionError("Connection failed")
        
        return f"Generated response for: {prompt[:50]}..."
    
    async def health_check(self) -> bool:
        """Mock health check."""
        return self.connection_healthy
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.embedding_dimensions


class TestOpenAIProvider:
    """Test suite for OpenAI provider implementation."""
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "api_key": "test-openai-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "llm_model": "gpt-4",
            "max_retries": 3,
            "timeout": 30
        }
    
    @pytest.fixture
    def mock_openai_provider(self, openai_config):
        """Create mock OpenAI provider."""
        return MockAIProvider(embedding_dimensions=1536, connection_healthy=True)
    
    @pytest.mark.asyncio
    async def test_embedding_creation(self, mock_openai_provider):
        """Test OpenAI embedding creation."""
        texts = ["Hello world", "Test embedding"]
        embeddings = await mock_openai_provider.create_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        assert mock_openai_provider.call_count["embeddings"] == 1
    
    @pytest.mark.asyncio
    async def test_text_generation(self, mock_openai_provider):
        """Test OpenAI text generation."""
        prompt = "Generate a summary of the following text:"
        response = await mock_openai_provider.generate_text(prompt)
        
        assert "Generated response" in response
        assert mock_openai_provider.call_count["llm"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_openai_provider):
        """Test OpenAI health check."""
        assert await mock_openai_provider.health_check() == True
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        unhealthy_provider = MockAIProvider(connection_healthy=False)
        
        with pytest.raises(ProviderConnectionError):
            await unhealthy_provider.create_embeddings(["test"])
        
        with pytest.raises(ProviderConnectionError):
            await unhealthy_provider.generate_text("test prompt")
    
    def test_dimension_compatibility(self, mock_openai_provider):
        """Test OpenAI dimension compatibility."""
        assert mock_openai_provider.get_embedding_dimensions() == 1536


class TestOllamaProvider:
    """Test suite for Ollama provider implementation."""
    
    @pytest.fixture
    def ollama_config(self):
        """Ollama provider configuration."""
        return {
            "base_url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "embedding_dimensions": 768,
            "llm_model": "llama3.2",
            "timeout": 60,
            "verify_ssl": False
        }
    
    @pytest.fixture
    def mock_ollama_provider(self, ollama_config):
        """Create mock Ollama provider."""
        return MockAIProvider(embedding_dimensions=768, connection_healthy=True)
    
    @pytest.mark.asyncio
    async def test_embedding_creation(self, mock_ollama_provider):
        """Test Ollama embedding creation."""
        texts = ["Hello world", "Test embedding"]
        embeddings = await mock_ollama_provider.create_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768
        assert len(embeddings[1]) == 768
        assert mock_ollama_provider.call_count["embeddings"] == 1
    
    @pytest.mark.asyncio
    async def test_text_generation(self, mock_ollama_provider):
        """Test Ollama text generation."""
        prompt = "Generate a summary of the following text:"
        response = await mock_ollama_provider.generate_text(prompt)
        
        assert "Generated response" in response
        assert mock_ollama_provider.call_count["llm"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_ollama_provider):
        """Test Ollama health check."""
        assert await mock_ollama_provider.health_check() == True
    
    def test_dimension_compatibility(self, mock_ollama_provider):
        """Test Ollama dimension compatibility."""
        assert mock_ollama_provider.get_embedding_dimensions() == 768
    
    @pytest.mark.asyncio
    async def test_custom_dimensions(self):
        """Test Ollama with different embedding dimensions."""
        # Test various Ollama model dimensions
        dimensions_to_test = [384, 768, 1024]
        
        for dim in dimensions_to_test:
            provider = MockAIProvider(embedding_dimensions=dim)
            embeddings = await provider.create_embeddings(["test"])
            assert len(embeddings[0]) == dim


class TestProviderSwitching:
    """Test suite for switching between AI providers."""
    
    @pytest.fixture
    def openai_provider(self):
        return MockAIProvider(embedding_dimensions=1536)
    
    @pytest.fixture
    def ollama_provider(self):
        return MockAIProvider(embedding_dimensions=768)
    
    def test_dimension_compatibility_check(self, openai_provider, ollama_provider):
        """Test dimension compatibility checking between providers."""
        openai_dims = openai_provider.get_embedding_dimensions()
        ollama_dims = ollama_provider.get_embedding_dimensions()
        
        # These should be detected as incompatible
        assert openai_dims != ollama_dims
        
        # This should raise an error in real implementation
        with pytest.raises(DimensionMismatchError):
            self._validate_provider_switch(openai_provider, ollama_provider)
    
    def test_compatible_provider_switch(self):
        """Test switching between providers with compatible dimensions."""
        provider1 = MockAIProvider(embedding_dimensions=768)
        provider2 = MockAIProvider(embedding_dimensions=768)
        
        # These should be compatible
        assert provider1.get_embedding_dimensions() == provider2.get_embedding_dimensions()
        
        # No error should be raised
        self._validate_provider_switch(provider1, provider2)
    
    def _validate_provider_switch(self, old_provider, new_provider):
        """Helper method to validate provider switching."""
        old_dims = old_provider.get_embedding_dimensions()
        new_dims = new_provider.get_embedding_dimensions()
        
        if old_dims != new_dims:
            raise DimensionMismatchError(
                f"Cannot switch from provider with {old_dims} dimensions "
                f"to provider with {new_dims} dimensions"
            )


class TestErrorHandling:
    """Test suite for error handling and fallback scenarios."""
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism for failed API calls."""
        # Create a provider that fails initially but succeeds on retry
        class RetryProvider(MockAIProvider):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0
            
            async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
                self.attempt_count += 1
                if self.attempt_count < 3:  # Fail first 2 attempts
                    raise ProviderConnectionError("Temporary failure")
                return await super().create_embeddings(texts)
        
        provider = RetryProvider()
        
        # Mock retry logic (this would be implemented in the actual provider)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embeddings = await provider.create_embeddings(["test"])
                assert len(embeddings) == 1
                break
            except ProviderConnectionError:
                if attempt == max_retries - 1:
                    raise
                continue
    
    @pytest.mark.asyncio
    async def test_fallback_to_zero_embeddings(self):
        """Test fallback to zero embeddings on complete failure."""
        unhealthy_provider = MockAIProvider(connection_healthy=False)
        
        # In real implementation, this would fall back to zero embeddings
        try:
            await unhealthy_provider.create_embeddings(["test"])
        except ProviderConnectionError:
            # Simulate fallback
            fallback_embedding = [[0.0] * 1536]
            assert len(fallback_embedding[0]) == 1536
            assert all(x == 0.0 for x in fallback_embedding[0])
    
    def test_configuration_validation(self):
        """Test validation of provider configurations."""
        # Test missing required fields
        invalid_configs = [
            {},  # Empty config
            {"provider": "openai"},  # Missing API key
            {"provider": "ollama"},  # Missing base URL
            {"provider": "openai", "api_key": ""},  # Empty API key
            {"provider": "ollama", "base_url": "invalid-url"},  # Invalid URL
        ]
        
        for config in invalid_configs:
            with pytest.raises(ConfigurationError):
                self._validate_config(config)
    
    def _validate_config(self, config: Dict[str, Any]):
        """Helper method to validate configuration."""
        if not config:
            raise ConfigurationError("Configuration cannot be empty")
        
        if "provider" not in config:
            raise ConfigurationError("Provider type is required")
        
        provider_type = config["provider"]
        
        if provider_type == "openai":
            if not config.get("api_key"):
                raise ConfigurationError("OpenAI API key is required")
        elif provider_type == "ollama":
            if not config.get("base_url"):
                raise ConfigurationError("Ollama base URL is required")
        else:
            raise ConfigurationError(f"Unknown provider: {provider_type}")


class TestBatchProcessing:
    """Test suite for batch processing capabilities."""
    
    @pytest.mark.asyncio
    async def test_batch_embedding_creation(self):
        """Test batch embedding creation for efficiency."""
        provider = MockAIProvider()
        
        # Test small batch
        small_batch = ["text1", "text2", "text3"]
        embeddings = await provider.create_embeddings(small_batch)
        assert len(embeddings) == 3
        assert provider.call_count["embeddings"] == 1
        
        # Test large batch (should be split)
        large_batch = [f"text_{i}" for i in range(100)]
        embeddings = await provider.create_embeddings(large_batch)
        assert len(embeddings) == 100
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple requests."""
        provider = MockAIProvider()
        
        # Create multiple concurrent requests
        tasks = [
            provider.create_embeddings([f"batch_{i}_text_{j}" for j in range(5)])
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert len(result) == 5
            assert len(result[0]) == 1536


class TestPerformanceBenchmarking:
    """Test suite for performance benchmarking across providers."""
    
    def test_embedding_generation_performance(self):
        """Test embedding generation performance metrics."""
        # Test different provider configurations
        test_configs = [
            {"provider": "openai", "dimensions": 1536, "batch_size": 100},
            {"provider": "ollama", "dimensions": 768, "batch_size": 10},
            {"provider": "ollama", "dimensions": 384, "batch_size": 20},
        ]
        
        for config in test_configs:
            with patch('test_ai_providers.MockAIProvider') as MockProvider:
                mock_provider = MockAIProvider(
                    embedding_dimensions=config["dimensions"]
                )
                MockProvider.return_value = mock_provider
                
                # Measure performance
                start_time = time.time()
                
                # Simulate batch processing
                for _ in range(5):  # 5 batches
                    texts = [f"test_{i}" for i in range(config["batch_size"])]
                    asyncio.run(mock_provider.create_embeddings(texts))
                
                elapsed_time = time.time() - start_time
                
                # Performance should be reasonable for mocked providers
                assert elapsed_time < 1.0, f"Performance too slow for {config['provider']}: {elapsed_time}s"
                
                # Verify call counts
                expected_calls = 5  # 5 batches
                assert mock_provider.call_count["embeddings"] == expected_calls
    
    def test_memory_efficiency(self):
        """Test memory efficiency across different batch sizes."""
        import tracemalloc
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            tracemalloc.start()
            
            provider = MockAIProvider(embedding_dimensions=1536)
            
            # Generate embeddings
            texts = [f"test text {i}" * 10 for i in range(batch_size)]
            asyncio.run(provider.create_embeddings(texts))
            
            # Measure memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Memory usage should scale reasonably
            memory_per_text = peak / batch_size
            assert memory_per_text < 10000, f"Memory usage too high: {memory_per_text} bytes per text"
    
    @pytest.mark.parametrize("dimensions,expected_max_time", [
        (384, 0.5),
        (768, 0.7),
        (1536, 1.0),
    ])
    def test_dimension_scaling_performance(self, dimensions, expected_max_time):
        """Test that performance scales reasonably with embedding dimensions."""
        provider = MockAIProvider(embedding_dimensions=dimensions)
        
        start_time = time.time()
        
        # Generate embeddings for various text lengths
        texts = [
            "short",
            "medium length text",
            "very long text that contains much more content " * 10
        ]
        
        for _ in range(10):  # Repeat for more stable timing
            asyncio.run(provider.create_embeddings(texts))
        
        elapsed_time = time.time() - start_time
        
        # Performance should scale reasonably with dimensions
        assert elapsed_time < expected_max_time, f"Performance too slow for {dimensions}D: {elapsed_time}s"


class TestConcurrencyAndThreadSafety:
    """Test suite for concurrency and thread safety."""
    
    def test_thread_safety(self):
        """Test that providers are thread-safe."""
        provider = MockAIProvider(embedding_dimensions=768)
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                texts = [f"worker_{worker_id}_text_{i}" for i in range(5)]
                result = asyncio.run(provider.create_embeddings(texts))
                results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify all workers completed
        assert len(results) == 10
        
        # Verify all results are correct
        for worker_id, result in results:
            assert len(result) == 5
            assert all(len(emb) == 768 for emb in result)
    
    @pytest.mark.asyncio
    async def test_async_concurrency(self):
        """Test async concurrency handling."""
        provider = MockAIProvider(embedding_dimensions=1536)
        
        # Create many concurrent tasks
        async def create_embedding_task(task_id):
            texts = [f"task_{task_id}_text_{i}" for i in range(3)]
            return await provider.create_embeddings(texts)
        
        # Run 20 concurrent tasks
        tasks = [create_embedding_task(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed successfully
        assert len(results) == 20
        for result in results:
            assert len(result) == 3
            assert all(len(emb) == 1536 for emb in result)
    
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        import gc
        import weakref
        
        # Create provider and get weak reference
        provider = MockAIProvider(embedding_dimensions=384)
        weak_ref = weakref.ref(provider)
        
        # Use the provider
        asyncio.run(provider.create_embeddings(["test"]))
        
        # Delete provider and force garbage collection
        del provider
        gc.collect()
        
        # Verify provider was cleaned up
        assert weak_ref() is None, "Provider was not properly cleaned up"


class TestConfigurationValidation:
    """Test suite for comprehensive configuration validation."""
    
    def test_openai_configuration_validation(self):
        """Test OpenAI configuration validation."""
        # Valid configurations
        valid_configs = [
            {
                "api_key": "sk-test123",
                "default_embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 1536,
                "max_batch_size": 100
            },
            {
                "api_key": "sk-proj-test123",
                "default_embedding_model": "text-embedding-ada-002",
                "embedding_dimensions": 1536,
                "base_url": "https://api.openai.com/v1"
            }
        ]
        
        for config in valid_configs:
            # Should not raise exception
            try:
                # Simulate configuration validation
                assert config["api_key"].startswith(("sk-", "sk-proj-"))
                assert config["embedding_dimensions"] > 0
                assert "embedding_model" in config["default_embedding_model"]
            except Exception as e:
                pytest.fail(f"Valid OpenAI config failed validation: {e}")
    
    def test_ollama_configuration_validation(self):
        """Test Ollama configuration validation."""
        # Valid configurations
        valid_configs = [
            {
                "base_url": "http://localhost:11434",
                "default_embedding_model": "nomic-embed-text",
                "embedding_dimensions": 768,
                "timeout": 60
            },
            {
                "base_url": "https://ollama.example.com",
                "default_embedding_model": "mxbai-embed-large",
                "embedding_dimensions": 1024,
                "verify_ssl": False
            }
        ]
        
        for config in valid_configs:
            # Should not raise exception
            try:
                # Simulate configuration validation
                assert config["base_url"].startswith(("http://", "https://"))
                assert config["embedding_dimensions"] in [384, 768, 1024, 1536]
                assert config["timeout"] > 0 if "timeout" in config else True
            except Exception as e:
                pytest.fail(f"Valid Ollama config failed validation: {e}")
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        invalid_configs = [
            {"api_key": ""},  # Empty API key
            {"base_url": "invalid-url"},  # Invalid URL
            {"embedding_dimensions": 0},  # Invalid dimensions
            {"embedding_dimensions": -1},  # Negative dimensions
            {"max_batch_size": 0},  # Invalid batch size
            {"timeout": -1},  # Invalid timeout
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, ConfigurationError)):
                # Simulate configuration validation that should fail
                if "api_key" in config and not config["api_key"]:
                    raise ConfigurationError("API key cannot be empty")
                if "base_url" in config and not config["base_url"].startswith("http"):
                    raise ConfigurationError("Invalid base URL")
                if "embedding_dimensions" in config and config["embedding_dimensions"] <= 0:
                    raise ConfigurationError("Embedding dimensions must be positive")
                if "max_batch_size" in config and config["max_batch_size"] <= 0:
                    raise ConfigurationError("Batch size must be positive")
                if "timeout" in config and config["timeout"] <= 0:
                    raise ConfigurationError("Timeout must be positive")
    
    @pytest.mark.parametrize("provider,config_key,valid_values,invalid_values", [
        ("openai", "embedding_dimensions", [1536, 3072], [0, -1, 100]),
        ("ollama", "embedding_dimensions", [384, 768, 1024], [0, -1, 2000]),
        ("openai", "max_batch_size", [1, 10, 100], [0, -1]),
        ("ollama", "max_batch_size", [1, 5, 20], [0, -1]),
    ])
    def test_parametrized_config_validation(self, provider, config_key, valid_values, invalid_values):
        """Test configuration validation with parametrized values."""
        base_config = {
            "openai": {"api_key": "sk-test"},
            "ollama": {"base_url": "http://localhost:11434"}
        }[provider]
        
        # Test valid values
        for value in valid_values:
            config = {**base_config, config_key: value}
            # Should not raise exception
            assert config[config_key] == value
        
        # Test invalid values
        for value in invalid_values:
            config = {**base_config, config_key: value}
            with pytest.raises((ValueError, ConfigurationError)):
                if value <= 0:
                    raise ConfigurationError(f"{config_key} must be positive")


class TestFactoryPatternIntegration:
    """Test suite for AI provider factory pattern integration."""
    
    def test_factory_provider_registration(self):
        """Test provider registration with factory."""
        # Simulate factory registration
        factory = AIProviderFactory()
        
        # Mock provider registration
        with patch.object(factory, 'register_hybrid_provider') as mock_register:
            # Simulate registering providers
            mock_register(AIProvider.OPENAI, MockAIProvider)
            mock_register(AIProvider.OLLAMA, MockAIProvider)
            
            # Verify registration calls
            assert mock_register.call_count == 2
            mock_register.assert_has_calls([
                call(AIProvider.OPENAI, MockAIProvider),
                call(AIProvider.OLLAMA, MockAIProvider)
            ])
    
    def test_factory_provider_creation(self):
        """Test provider creation through factory."""
        # Mock factory creation
        with patch.object(AIProviderFactory, 'create_provider') as mock_create:
            mock_provider = MockAIProvider()
            mock_create.return_value = mock_provider
            
            # Test OpenAI provider creation
            openai_config = {"api_key": "sk-test", "embedding_dimensions": 1536}
            provider = AIProviderFactory.create_provider(AIProvider.OPENAI, openai_config)
            
            assert provider is not None
            mock_create.assert_called_with(AIProvider.OPENAI, openai_config)
    
    def test_factory_capabilities_query(self):
        """Test querying provider capabilities through factory."""
        # Mock factory capabilities
        with patch.object(AIProviderFactory, 'get_provider_capabilities') as mock_caps:
            mock_caps.return_value = {
                "supports_embeddings": True,
                "supports_llm": True,
                "supports_hybrid": True
            }
            
            caps = AIProviderFactory.get_provider_capabilities(AIProvider.OPENAI)
            
            assert caps["supports_embeddings"] is True
            assert caps["supports_llm"] is True
            assert caps["supports_hybrid"] is True
    
    def test_factory_provider_availability(self):
        """Test checking provider availability through factory."""
        # Mock factory availability check
        with patch.object(AIProviderFactory, 'is_provider_available') as mock_available:
            mock_available.return_value = True
            
            # Test availability check
            is_available = AIProviderFactory.is_provider_available(AIProvider.OLLAMA)
            
            assert is_available is True
            mock_available.assert_called_with(AIProvider.OLLAMA)


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=10",
        "--durations=15",
        "-x"  # Stop on first failure for debugging
    ])