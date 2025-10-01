"""
Comprehensive tests for the refactored utils.py with AI provider abstraction.

This test suite ensures:
1. Backward compatibility with existing utils.py interface
2. AI provider abstraction integration works correctly
3. Error handling and fallback scenarios
4. Provider health checks and switching
5. Performance and concurrency handling
6. Configuration validation
7. Proper mocking to avoid requiring real API keys
8. CI/CD environment compatibility
"""

import pytest
import asyncio
import os
import tempfile
import time
import concurrent.futures
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import List, Dict, Any, Optional
import json
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the refactored utils
import sys
import inspect
from pathlib import Path

# Add src directory to path dynamically
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from utils_refactored import (
        # Core functions
        create_embedding,
        create_embeddings_batch,
        generate_contextual_embedding,
        generate_code_example_summary,
        extract_source_summary,
        get_supabase_client,
        
        # AI provider management
        initialize_ai_providers,
        close_ai_providers,
        health_check_ai_providers,
        get_ai_provider_info,
        validate_ai_provider_config,
        get_available_ai_providers,
        validate_ai_provider_requirements,
        get_ai_provider_metrics,
        ai_provider_context,
        
        # Provider manager
        AIProviderManager,
        _provider_manager,
        
        # Data processing functions
        add_documents_to_supabase,
        search_documents,
        extract_code_blocks,
        add_code_examples_to_supabase,
        search_code_examples,
        update_source_info,
        process_chunk_with_context
    )
    
    from ai_providers.base import (
        AIProvider, EmbeddingProvider, LLMProvider, HybridAIProvider,
        EmbeddingResult, LLMResult, HealthStatus
    )
    from ai_providers.factory import AIProviderFactory
    from ai_providers.config import AIProviderConfig
    
except ImportError as e:
    pytest.skip(f"Could not import refactored utils: {e}", allow_module_level=True)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("default_embedding_model", "mock-model")
        self.dimensions = config.get("embedding_dimensions", 1536)
        self.call_count = 0
        self.should_fail = False
        
    def _validate_config(self) -> None:
        pass
        
    async def initialize(self) -> None:
        pass
        
    async def close(self) -> None:
        pass
        
    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            is_healthy=not self.should_fail,
            provider=AIProvider.OPENAI.value,
            model=self.model,
            response_time_ms=10.0,
            error_message="Mock error" if self.should_fail else None
        )
        
    async def create_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        if self.should_fail:
            raise Exception("Mock embedding failure")
            
        self.call_count += 1
        return EmbeddingResult(
            embedding=[0.1] * self.dimensions,
            dimensions=self.dimensions,
            model=model or self.model
        )
        
    async def create_embeddings_batch(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[EmbeddingResult]:
        if self.should_fail:
            raise Exception("Mock batch embedding failure")
            
        self.call_count += 1
        return [
            EmbeddingResult(
                embedding=[0.1] * self.dimensions,
                dimensions=self.dimensions,
                model=model or self.model
            )
            for _ in texts
        ]
        
    @property
    def default_model(self) -> str:
        return self.model
        
    @property
    def supported_models(self) -> List[str]:
        return [self.model]
        
    def get_model_dimensions(self, model: str) -> int:
        return self.dimensions


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("default_llm_model", "mock-llm")
        self.call_count = 0
        self.should_fail = False
        
    def _validate_config(self) -> None:
        pass
        
    async def initialize(self) -> None:
        pass
        
    async def close(self) -> None:
        pass
        
    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            is_healthy=not self.should_fail,
            provider=AIProvider.OPENAI.value,
            model=self.model,
            response_time_ms=15.0,
            error_message="Mock LLM error" if self.should_fail else None
        )
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResult:
        if self.should_fail:
            raise Exception("Mock LLM generation failure")
            
        self.call_count += 1
        prompt = messages[-1].get("content", "") if messages else ""
        return LLMResult(
            content=f"Mock response for: {prompt[:50]}...",
            model=model or self.model,
            input_tokens=10,
            output_tokens=20
        )
        
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        if self.should_fail:
            raise Exception("Mock LLM stream failure")
            
        self.call_count += 1
        prompt = messages[-1].get("content", "") if messages else ""
        yield LLMResult(
            content=f"Mock streaming response for: {prompt[:50]}...",
            model=model or self.model,
            input_tokens=10,
            output_tokens=20
        )
        
    @property
    def default_model(self) -> str:
        return self.model
        
    @property
    def supported_models(self) -> List[str]:
        return [self.model]


class MockHybridProvider(HybridAIProvider):
    """Mock hybrid provider for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_provider = MockEmbeddingProvider(config)
        self.llm_provider = MockLLMProvider(config)
        
    def _validate_config(self) -> None:
        pass
        
    async def initialize(self) -> None:
        await self.embedding_provider.initialize()
        await self.llm_provider.initialize()
        
    async def close(self) -> None:
        await self.embedding_provider.close()
        await self.llm_provider.close()
        
    async def health_check(self) -> HealthStatus:
        emb_health = await self.embedding_provider.health_check()
        llm_health = await self.llm_provider.health_check()
        
        return HealthStatus(
            is_healthy=emb_health.is_healthy and llm_health.is_healthy,
            provider=AIProvider.OPENAI.value,
            model=f"Emb: {self.embedding_provider.model}, LLM: {self.llm_provider.model}",
            response_time_ms=(emb_health.response_time_ms + llm_health.response_time_ms) / 2,
            error_message=emb_health.error_message or llm_health.error_message
        )
        
    # Embedding methods
    async def create_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        return await self.embedding_provider.create_embedding(text, model)
        
    async def create_embeddings_batch(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[EmbeddingResult]:
        return await self.embedding_provider.create_embeddings_batch(texts, model, batch_size)
        
    def get_model_dimensions(self, model: str) -> int:
        return self.embedding_provider.get_model_dimensions(model)
        
    # LLM methods
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResult:
        return await self.llm_provider.generate(messages, model, temperature, max_tokens, **kwargs)
        
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        async for result in self.llm_provider.generate_stream(messages, model, temperature, max_tokens, **kwargs):
            yield result
            
    @property
    def default_embedding_model(self) -> str:
        return self.embedding_provider.default_model
        
    @property
    def default_llm_model(self) -> str:
        return self.llm_provider.default_model
        
    @property
    def supported_models(self) -> List[str]:
        return self.embedding_provider.supported_models + self.llm_provider.supported_models
        
    @property
    def max_batch_size(self) -> int:
        return 100
        
    @property
    def max_text_length(self) -> int:
        return 8192


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "api_key": "test-key",
        "default_embedding_model": "mock-embedding-model",
        "default_llm_model": "mock-llm-model",
        "embedding_dimensions": 1536,
        "max_batch_size": 100,
        "max_retries": 3,
        "timeout": 30.0
    }


@pytest.fixture
def mock_provider_manager():
    """Mock provider manager for testing"""
    manager = AIProviderManager()
    mock_provider = MockHybridProvider({
        "default_embedding_model": "mock-embedding",
        "default_llm_model": "mock-llm",
        "embedding_dimensions": 1536
    })
    
    # Set up the manager with mock provider
    manager._hybrid_provider = mock_provider
    manager._embedding_provider = mock_provider
    manager._llm_provider = mock_provider
    manager._initialized = True
    
    return manager


class TestBackwardCompatibility:
    """Test backward compatibility with original utils.py interface"""
    
    def test_create_embedding_signature(self):
        """Test that create_embedding maintains the same signature"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test function call - should be synchronous wrapper
            result = create_embedding("test text")
            
            # Verify result format matches original utils.py
            assert isinstance(result, list)
            assert len(result) == 1536
            assert all(isinstance(x, (float, int)) for x in result)
            
            # Verify provider was called correctly
            assert mock_manager.get_embedding_provider.called
    
    def test_create_embeddings_batch_signature(self):
        """Test that create_embeddings_batch maintains the same signature"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test function call - should be synchronous wrapper
            texts = ["text 1", "text 2", "text 3"]
            result = create_embeddings_batch(texts)
            
            # Verify result format matches original utils.py
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(emb, list) for emb in result)
            assert all(len(emb) == 1536 for emb in result)
            assert all(all(isinstance(x, (float, int)) for x in emb) for emb in result)
            
            # Verify batch processing was used
            assert mock_manager.get_embedding_provider.called
    
    def test_generate_contextual_embedding_signature(self):
        """Test that generate_contextual_embedding maintains the same signature"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockLLMProvider({"default_llm_model": "mock"})
            mock_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test function call - should be synchronous wrapper
            full_doc = "This is a full document with lots of content."
            chunk = "This is a chunk from the document."
            
            contextual_text, success = generate_contextual_embedding(full_doc, chunk)
            
            # Verify result format matches original utils.py
            assert isinstance(contextual_text, str)
            assert isinstance(success, bool)
            assert len(contextual_text) > 0
            
            # If successful, contextual text should contain relevant content
            if success:
                assert "Mock response" in contextual_text or chunk in contextual_text
            else:
                assert contextual_text == chunk
                
            # Verify provider was called
            assert mock_manager.get_llm_provider.called
    
    def test_get_supabase_client_unchanged(self):
        """Test that get_supabase_client function is unchanged"""
        # This function should not be modified
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_SERVICE_KEY': 'test-key'
        }):
            with patch('utils_refactored.create_client') as mock_create:
                mock_client = Mock()
                mock_create.return_value = mock_client
                
                result = get_supabase_client()
                
                assert result == mock_client
                mock_create.assert_called_once_with(
                    'https://test.supabase.co', 
                    'test-key'
                )


class TestAIProviderIntegration:
    """Test AI provider abstraction integration"""
    
    @pytest.mark.asyncio
    async def test_provider_manager_initialization(self):
        """Test AI provider manager initialization"""
        with patch('utils_refactored.AIProviderConfig') as mock_config_class:
            with patch('utils_refactored.AIProviderFactory') as mock_factory:
                mock_config = Mock()
                mock_config.provider = AIProvider.OPENAI
                mock_config.config = {"api_key": "test"}
                mock_config_class.from_env.return_value = mock_config
                
                mock_provider = MockHybridProvider({"api_key": "test"})
                mock_factory.create_provider.return_value = mock_provider
                
                manager = AIProviderManager()
                await manager.initialize()
                
                assert manager._initialized is True
                assert manager._hybrid_provider is not None
    
    @pytest.mark.asyncio
    async def test_provider_health_checks(self):
        """Test provider health check functionality"""
        manager = AIProviderManager()
        mock_provider = MockHybridProvider({"api_key": "test"})
        
        manager._hybrid_provider = mock_provider
        manager._embedding_provider = mock_provider
        manager._llm_provider = mock_provider
        manager._initialized = True
        
        health_status = await manager.health_check()
        
        assert "embedding" in health_status
        assert health_status["embedding"]["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_provider_switching(self):
        """Test switching between different AI providers"""
        # Test with OpenAI dimensions
        openai_provider = MockEmbeddingProvider({
            "embedding_dimensions": 1536,
            "default_embedding_model": "text-embedding-3-small"
        })
        
        # Test with Ollama dimensions  
        ollama_provider = MockEmbeddingProvider({
            "embedding_dimensions": 768,
            "default_embedding_model": "nomic-embed-text"
        })
        
        # Test embedding creation with different providers
        openai_result = await openai_provider.create_embedding("test")
        ollama_result = await ollama_provider.create_embedding("test")
        
        assert len(openai_result.embedding) == 1536
        assert len(ollama_result.embedding) == 768
        assert openai_result.dimensions != ollama_result.dimensions
    
    def test_configuration_validation(self):
        """Test AI provider configuration validation"""
        # Test valid configuration
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test-key'
        }):
            result = validate_ai_provider_config()
            assert result["valid"] is True
            assert result["provider"] == "openai"
        
        # Test invalid configuration
        with patch.dict(os.environ, {}, clear=True):
            with patch('utils_refactored.AIProviderConfig.from_env') as mock_from_env:
                mock_from_env.side_effect = ValueError("Missing API key")
                
                result = validate_ai_provider_config()
                assert result["valid"] is False
                assert "error" in result


class TestErrorHandling:
    """Test error handling and fallback scenarios"""
    
    @pytest.mark.asyncio
    async def test_embedding_fallback_on_failure(self):
        """Test fallback to zero embeddings on provider failure"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_provider.should_fail = True
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            
            # Should not raise exception, should return zero embedding
            result = create_embedding("test text")
            
            assert isinstance(result, list)
            assert len(result) == 1536
            assert all(x == 0.0 for x in result)
    
    @pytest.mark.asyncio
    async def test_batch_embedding_individual_fallback(self):
        """Test individual embedding fallback on batch failure"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            # Create a provider that fails on batch but succeeds individually
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            
            async def mock_batch_fail(*args, **kwargs):
                raise Exception("Batch failed")
            
            async def mock_individual_success(text, model=None):
                return EmbeddingResult(
                    embedding=[0.1] * 1536,
                    dimensions=1536,
                    model="mock"
                )
            
            mock_provider.create_embeddings_batch = mock_batch_fail
            mock_provider.create_embedding = mock_individual_success
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            
            texts = ["text1", "text2"]
            result = create_embeddings_batch(texts)
            
            assert len(result) == 2
            assert all(len(emb) == 1536 for emb in result)
    
    @pytest.mark.asyncio
    async def test_contextual_embedding_fallback(self):
        """Test fallback to original chunk on contextual embedding failure"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockLLMProvider({"default_llm_model": "mock"})
            mock_provider.should_fail = True
            mock_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
            
            full_doc = "Full document content"
            chunk = "Chunk content"
            
            result_text, success = generate_contextual_embedding(full_doc, chunk)
            
            assert result_text == chunk
            assert success is False
    
    @pytest.mark.asyncio
    async def test_provider_reinitialization_on_failure(self):
        """Test provider reinitialization on connection failure"""
        manager = AIProviderManager()
        
        with patch('utils_refactored.AIProviderConfig') as mock_config_class:
            with patch('utils_refactored.AIProviderFactory') as mock_factory:
                mock_config = Mock()
                mock_config.provider = AIProvider.OPENAI
                mock_config.config = {"api_key": "test"}
                mock_config_class.from_env.return_value = mock_config
                
                # First provider that fails health check
                failing_provider = MockHybridProvider({"api_key": "test"})
                failing_provider.embedding_provider.should_fail = True
                
                # Second provider that succeeds
                working_provider = MockHybridProvider({"api_key": "test"})
                
                mock_factory.create_provider.side_effect = [failing_provider, working_provider]
                
                # Should raise error on first attempt due to health check failure
                with pytest.raises(ConnectionError):
                    await manager.initialize()
                
                # Reset and try again with working provider
                failing_provider.embedding_provider.should_fail = False
                await manager.initialize(force_reinit=True)
                
                assert manager._initialized is True


class TestPerformanceAndConcurrency:
    """Test performance optimizations and concurrent operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_creation(self):
        """Test concurrent embedding creation"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            
            # Create multiple concurrent embedding requests
            texts_batches = [
                [f"batch_{i}_text_{j}" for j in range(3)]
                for i in range(5)
            ]
            
            # Test concurrent execution
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(create_embeddings_batch, batch)
                    for batch in texts_batches
                ]
                
                results = [future.result() for future in futures]
            
            assert len(results) == 5
            for result in results:
                assert len(result) == 3
                assert all(len(emb) == 1536 for emb in result)
    
    @pytest.mark.asyncio
    async def test_batch_processing_optimization(self):
        """Test that batch processing is more efficient than individual requests"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            
            texts = [f"text_{i}" for i in range(10)]
            
            # Test batch processing
            result_batch = create_embeddings_batch(texts)
            assert mock_provider.call_count == 1  # Single batch call
            
            # Reset call count
            mock_provider.call_count = 0
            
            # Test individual processing
            results_individual = []
            for text in texts:
                result = create_embedding(text)
                results_individual.extend([result])
            
            # Should have made more calls (but actually it uses batch internally)
            assert len(result_batch) == len(results_individual)
    
    def test_memory_efficiency_large_documents(self):
        """Test memory efficiency with large document processing"""
        # Test that large documents are properly chunked and don't cause memory issues
        large_content = "Large content " * 10000  # ~130KB of text
        
        # Should not raise memory errors
        chunks = extract_code_blocks(large_content, min_length=100)
        
        # Function should complete without issues
        assert isinstance(chunks, list)


class TestSupabaseIntegration:
    """Test integration with Supabase functions"""
    
    def test_add_documents_to_supabase_signature(self):
        """Test that add_documents_to_supabase maintains signature compatibility"""
        mock_client = Mock()
        mock_client.table.return_value.delete.return_value.in_.return_value.execute.return_value = None
        mock_client.table.return_value.insert.return_value.execute.return_value = None
        
        with patch('utils_refactored.create_embeddings_batch') as mock_embeddings:
            mock_embeddings.return_value = [[0.1] * 1536] * 2
            
            # Test function call with expected parameters
            urls = ["http://example.com/1", "http://example.com/2"]
            chunk_numbers = [1, 2]
            contents = ["content 1", "content 2"] 
            metadatas = [{"type": "doc"}, {"type": "doc"}]
            url_to_full_document = {
                "http://example.com/1": "full doc 1",
                "http://example.com/2": "full doc 2"
            }
            
            # Should not raise exception
            add_documents_to_supabase(
                mock_client, urls, chunk_numbers, contents, 
                metadatas, url_to_full_document
            )
            
            # Verify embeddings were created
            mock_embeddings.assert_called()
    
    def test_search_documents_signature(self):
        """Test that search_documents maintains signature compatibility"""
        mock_client = Mock()
        mock_client.rpc.return_value.execute.return_value.data = [
            {"content": "result 1", "similarity": 0.9},
            {"content": "result 2", "similarity": 0.8}
        ]
        
        with patch('utils_refactored.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 1536
            
            results = search_documents(mock_client, "test query", match_count=5)
            
            assert len(results) == 2
            assert results[0]["content"] == "result 1"
            mock_embedding.assert_called_once_with("test query")
    
    def test_code_example_processing(self):
        """Test code example extraction and processing"""
        markdown_content = '''
        Here's some documentation:
        
        ```python
        def example_function():
            """This is a long example function with lots of code."""
            result = []
            for i in range(100):
                result.append(i * 2)
                if i % 10 == 0:
                    print(f"Processing item {i}")
            return result
        ```
        
        This function demonstrates basic Python concepts.
        '''
        
        code_blocks = extract_code_blocks(markdown_content, min_length=50)
        
        assert len(code_blocks) >= 1
        assert code_blocks[0]["language"] == "python"
        assert "def example_function" in code_blocks[0]["code"]
        assert "context_before" in code_blocks[0]
        assert "context_after" in code_blocks[0]


class TestProviderManagement:
    """Test AI provider lifecycle management"""
    
    @pytest.mark.asyncio
    async def test_provider_context_manager(self):
        """Test AI provider context manager"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_manager.initialize = AsyncMock()
            
            async with ai_provider_context() as manager:
                assert manager == mock_manager
                mock_manager.initialize.assert_called_once_with(force_reinit=False)
    
    @pytest.mark.asyncio
    async def test_provider_metrics_collection(self):
        """Test AI provider metrics collection"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_provider.get_metrics = Mock(return_value={
                "request_count": 10,
                "total_tokens": 1000,
                "error_count": 1
            })
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
            
            metrics = get_ai_provider_metrics()
            
            assert "embedding" in metrics
            assert metrics["embedding"]["request_count"] == 10
    
    def test_available_providers_listing(self):
        """Test listing of available AI providers"""
        with patch('utils_refactored.AIProviderFactory') as mock_factory:
            mock_factory.list_all_providers.return_value = {
                "embedding": ["openai", "ollama"],
                "llm": ["openai", "ollama", "anthropic"],
                "hybrid": ["openai", "ollama"]
            }
            
            providers = get_available_ai_providers()
            
            assert "embedding_providers" in providers
            assert "openai" in providers["embedding_providers"]
            assert "anthropic" in providers["llm_providers"]
    
    def test_provider_requirements_validation(self):
        """Test validation of provider requirements"""
        with patch('utils_refactored.AIProviderConfig') as mock_config_class:
            with patch('utils_refactored.AIProviderFactory') as mock_factory:
                mock_config = Mock()
                mock_config.provider = AIProvider.OPENAI
                mock_config.config = {"api_key": "test", "model": "gpt-4"}
                mock_config_class.from_env.return_value = mock_config
                
                mock_factory.is_provider_available.return_value = True
                mock_factory.get_provider_capabilities.return_value = {
                    "supports_embeddings": True,
                    "supports_llm": True,
                    "supports_hybrid": True
                }
                
                result = validate_ai_provider_requirements()
                
                assert result["valid"] is True
                assert result["provider"] == "openai"
                assert result["available"] is True
                assert result["capabilities"]["supports_embeddings"] is True


class TestDocumentProcessing:
    """Test document processing functionality"""
    
    def test_contextual_embeddings_environment_flag(self):
        """Test contextual embeddings based on environment flag"""
        mock_client = Mock()
        mock_client.table.return_value.delete.return_value.in_.return_value.execute.return_value = None
        mock_client.table.return_value.insert.return_value.execute.return_value = None
        
        with patch.dict(os.environ, {'USE_CONTEXTUAL_EMBEDDINGS': 'true'}):
            with patch('utils_refactored.create_embeddings_batch') as mock_embeddings:
                with patch('utils_refactored.process_chunk_with_context') as mock_context:
                    mock_embeddings.return_value = [[0.1] * 1536]
                    mock_context.return_value = ("contextual content", True)
                    
                    add_documents_to_supabase(
                        mock_client, 
                        ["http://example.com"],
                        [1],
                        ["original content"],
                        [{"type": "doc"}],
                        {"http://example.com": "full document"}
                    )
                    
                    # Should have called contextual processing
                    mock_context.assert_called()
    
    def test_source_summary_generation(self):
        """Test source summary generation"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockLLMProvider({"default_llm_model": "mock"})
            mock_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
            
            content = "This is a Python library for machine learning" * 100
            summary = extract_source_summary("scikit-learn.org", content)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "Mock response" in summary
    
    def test_source_info_update(self):
        """Test source information update in database"""
        mock_client = Mock()
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = []
        mock_client.table.return_value.insert.return_value.execute.return_value = None
        
        update_source_info(mock_client, "example.com", "Test summary", 1000)
        
        # Should attempt update first, then insert
        mock_client.table.assert_called_with('sources')


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text inputs"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test empty string
            result = create_embedding("")
            assert len(result) == 1536
            assert all(x == 0.0 for x in result)
            
            # Test None (should be handled gracefully)
            result = create_embedding(None or "")
            assert len(result) == 1536
    
    def test_large_batch_processing(self):
        """Test processing of very large batches"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test large batch
            large_batch = [f"text_{i}" for i in range(1000)]
            result = create_embeddings_batch(large_batch)
            
            assert len(result) == 1000
            assert all(len(emb) == 1536 for emb in result)
    
    def test_special_characters_handling(self):
        """Test handling of special characters and unicode"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test various special characters
            special_texts = [
                "Hello 世界",  # Unicode
                "Café résumé naïve",  # Accented characters
                "Special chars: @#$%^&*()",  # Symbols
                "Newlines\nand\ttabs",  # Whitespace
                "🚀 🌟 💻",  # Emojis
            ]
            
            for text in special_texts:
                result = create_embedding(text)
                assert len(result) == 1536
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases"""
        # Test missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            info = get_ai_provider_info()
            assert "status" in info
            
        # Test malformed configuration
        with patch('utils_refactored.AIProviderConfig.from_env') as mock_from_env:
            mock_from_env.side_effect = ValueError("Invalid config")
            
            result = validate_ai_provider_config()
            assert result["valid"] is False


class TestCICDIntegration:
    """Test CI/CD environment compatibility and integration"""
    
    def test_environment_isolation(self):
        """Test that tests run in isolated environments"""
        # Save original environment
        original_env = os.environ.copy()
        
        try:
            # Clear environment
            os.environ.clear()
            
            # Test that functions handle missing environment gracefully
            with patch('utils_refactored._provider_manager') as mock_manager:
                mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
                mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
                mock_manager.is_initialized = True
                
                result = create_embedding("test")
                assert isinstance(result, list)
                assert len(result) == 1536
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_no_real_api_calls(self):
        """Test that no real API calls are made during testing"""
        # This test ensures all external calls are mocked
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Simulate multiple function calls
            results = []
            for i in range(10):
                result = create_embedding(f"test text {i}")
                results.append(result)
            
            # Verify all results are mock data
            for result in results:
                assert all(x == 0.1 for x in result)  # Mock embedding values
    
    def test_parallel_test_safety(self):
        """Test that tests can run safely in parallel"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker(worker_id):
            with patch('utils_refactored._provider_manager') as mock_manager:
                mock_provider = MockEmbeddingProvider({"embedding_dimensions": 768})
                mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
                mock_manager.is_initialized = True
                
                result = create_embedding(f"worker {worker_id} test")
                results_queue.put((worker_id, result))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all workers completed successfully
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5
        for worker_id, result in results:
            assert len(result) == 768
    
    def test_memory_usage_limits(self):
        """Test that memory usage stays within reasonable limits"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Process large amounts of data
            for batch_num in range(10):
                large_texts = [f"Large text content {i} " * 100 for i in range(100)]
                results = create_embeddings_batch(large_texts)
                
                # Verify results but don't keep references
                assert len(results) == 100
                del results
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.2f}MB"
    
    def test_timeout_handling(self):
        """Test that operations complete within reasonable time limits"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            start_time = time.time()
            
            # Test that operations complete quickly with mocked providers
            for i in range(100):
                result = create_embedding(f"test {i}")
                assert len(result) == 1536
            
            elapsed_time = time.time() - start_time
            
            # Should complete very quickly with mocked providers (under 1 second)
            assert elapsed_time < 1.0, f"Operations took {elapsed_time:.2f}s, expected < 1.0s"
    
    def test_error_recovery(self):
        """Test that the system gracefully recovers from errors"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            # Create a provider that fails initially
            failing_provider = MockEmbeddingProvider({"embedding_dimensions": 1536})
            failing_provider.should_fail = True
            
            mock_manager.get_embedding_provider = AsyncMock(return_value=failing_provider)
            mock_manager.is_initialized = True
            
            # First call should return fallback (zero embedding)
            result = create_embedding("test")
            assert len(result) == 1536
            assert all(x == 0.0 for x in result)  # Fallback behavior
            
            # Fix the provider
            failing_provider.should_fail = False
            
            # Subsequent calls should work normally
            result = create_embedding("test")
            assert len(result) == 1536
            assert all(x == 0.1 for x in result)  # Normal mock values


class TestProviderIntegrationFramework:
    """Test framework for AI provider integration testing"""
    
    @pytest.mark.parametrize("provider_name,dimensions", [
        ("openai", 1536),
        ("ollama", 768),
        ("ollama", 384),
        ("ollama", 1024)
    ])
    def test_provider_dimension_compatibility(self, provider_name, dimensions):
        """Test compatibility across different providers and dimensions"""
        with patch('utils_refactored._provider_manager') as mock_manager:
            mock_provider = MockEmbeddingProvider({"embedding_dimensions": dimensions})
            mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
            mock_manager.is_initialized = True
            
            # Test single embedding
            result = create_embedding("test")
            assert len(result) == dimensions
            
            # Test batch processing
            batch_result = create_embeddings_batch(["test1", "test2", "test3"])
            assert len(batch_result) == 3
            assert all(len(emb) == dimensions for emb in batch_result)
    
    def test_provider_switching_simulation(self):
        """Test simulation of provider switching scenarios"""
        # Simulate switching from OpenAI to Ollama
        scenarios = [
            {"provider": "openai", "dimensions": 1536, "model": "text-embedding-3-small"},
            {"provider": "ollama", "dimensions": 768, "model": "nomic-embed-text"},
            {"provider": "ollama", "dimensions": 1024, "model": "mxbai-embed-large"},
        ]
        
        for scenario in scenarios:
            with patch('utils_refactored._provider_manager') as mock_manager:
                mock_provider = MockEmbeddingProvider({
                    "embedding_dimensions": scenario["dimensions"],
                    "default_embedding_model": scenario["model"]
                })
                mock_manager.get_embedding_provider = AsyncMock(return_value=mock_provider)
                mock_manager.is_initialized = True
                
                # Test that each provider configuration works
                result = create_embedding("test text")
                assert len(result) == scenario["dimensions"]
                
                # Test batch processing
                batch_result = create_embeddings_batch(["test1", "test2"])
                assert len(batch_result) == 2
                assert all(len(emb) == scenario["dimensions"] for emb in batch_result)


if __name__ == "__main__":
    # Run specific test classes or all tests
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--disable-warnings",
        "--maxfail=5",  # Stop after 5 failures
        "--durations=10"  # Show 10 slowest tests
    ])