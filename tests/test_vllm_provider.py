"""
Comprehensive unit tests for vLLM AI Provider implementation.

This test suite ensures:
1. Text embedding generation works correctly
2. LLM text generation produces valid outputs
3. Vision model integration (image captions, OCR)
4. Configuration validation catches invalid setups
5. Health checks validate endpoint connectivity
6. Error handling and retry logic work properly
7. Provider factory integration works correctly
8. Graceful fallback scenarios
9. OpenAI-compatible API format compliance
10. Backward compatibility maintained

Testing Standards:
- pytest with asyncio support
- aioresponses for HTTP mocking
- 90%+ code coverage target
- No live API calls (all mocked)
"""

import pytest
import pytest_asyncio
import asyncio
import os
import json
import base64
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any, Optional
import logging
from aioresponses import aioresponses

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import vLLM provider and dependencies
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from ai_providers.base import (
        AIProvider, EmbeddingProvider, LLMProvider, VisionProvider,
        EmbeddingResult, LLMResult, VisionResult, HealthStatus
    )
    from ai_providers.factory import AIProviderFactory
    from ai_providers.config import AIProviderConfig
    from ai_providers.providers.vllm_provider import VLLMProvider
    VLLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import vLLM provider: {e}")
    VLLM_AVAILABLE = False


# Test Fixtures

@pytest.fixture
def vllm_provider_config():
    """vLLM provider configuration fixture."""
    return {
        "base_url": "http://test-vllm.example.com/v1",
        "api_key": "test-vllm-api-key",
        "text_model": "meta-llama/Llama-3.1-8B-Instruct",
        "default_embedding_model": "BAAI/bge-large-en-v1.5",
        "default_llm_model": "meta-llama/Llama-3.1-8B-Instruct",
        "vision_model": "llava-hf/llava-v1.6-mistral-7b-hf",
        "embedding_dimensions": 1024,
        "max_batch_size": 32,
        "max_retries": 3,
        "timeout": 120.0,
        "vision_enabled": True
    }


@pytest.fixture
def vllm_minimal_config():
    """Minimal vLLM configuration with defaults."""
    return {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY"
    }


@pytest.fixture
def mock_embedding_response():
    """Mock embedding API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * 1024,  # 1024-dimensional embedding
                "index": 0
            }
        ],
        "model": "BAAI/bge-large-en-v1.5",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }


@pytest.fixture
def mock_embeddings_batch_response():
    """Mock batch embeddings API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1 + i * 0.01] * 1024,
                "index": i
            }
            for i in range(5)
        ],
        "model": "BAAI/bge-large-en-v1.5",
        "usage": {
            "prompt_tokens": 50,
            "total_tokens": 50
        }
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM generation API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from vLLM."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30
        }
    }


@pytest.fixture
def mock_vision_caption_response():
    """Mock vision model caption response."""
    return {
        "id": "chatcmpl-vision-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llava-hf/llava-v1.6-mistral-7b-hf",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "A test image showing example content for unit testing."
                },
                "finish_reason": "stop"
            }
        ]
    }


@pytest.fixture
def mock_vision_ocr_response():
    """Mock vision model OCR response."""
    return {
        "id": "chatcmpl-ocr-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llava-hf/llava-v1.6-mistral-7b-hf",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sample text extracted from image: 'Hello World Test'"
                },
                "finish_reason": "stop"
            }
        ]
    }


@pytest.fixture
def test_image_data():
    """Test image data as bytes."""
    # Create minimal valid JPEG header
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00' + b'\x00' * 100


@pytest_asyncio.fixture
async def initialized_vllm_provider(vllm_provider_config):
    """Create and initialize vLLM provider for testing."""
    from aiohttp import ClientSession, ClientTimeout

    provider = VLLMProvider(vllm_provider_config)

    # Manually create session without calling initialize (to avoid health check)
    timeout = ClientTimeout(total=provider.config.get("timeout", 120.0), connect=30.0)
    provider.session = ClientSession(
        timeout=timeout,
        headers={
            "Authorization": f"Bearer {provider.config['api_key']}",
            "Content-Type": "application/json"
        }
    )

    yield provider

    # Cleanup
    if provider.session:
        await provider.session.close()


# Unit Tests - Configuration


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_initialization_valid_config(vllm_provider_config):
    """Test vLLM provider initialization with valid configuration."""
    provider = VLLMProvider(vllm_provider_config)

    assert provider is not None
    assert provider.config["base_url"] == "http://test-vllm.example.com/v1"
    assert provider.config["api_key"] == "test-vllm-api-key"
    assert provider.config["embedding_dimensions"] == 1024
    assert provider.config["vision_enabled"] is True


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_initialization_minimal_config(vllm_minimal_config):
    """Test vLLM provider initialization with minimal configuration."""
    provider = VLLMProvider(vllm_minimal_config)

    assert provider is not None
    assert provider.config["base_url"] == "http://localhost:8000/v1"
    assert provider.config["api_key"] == "EMPTY"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_invalid_base_url():
    """Test vLLM provider rejects invalid base URL."""
    invalid_config = {
        "base_url": "not-a-valid-url",
        "api_key": "test-key"
    }

    # Should raise ValueError for invalid URL
    with pytest.raises(ValueError):
        provider = VLLMProvider(invalid_config)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_default_models(vllm_provider_config):
    """Test vLLM provider default model properties."""
    provider = VLLMProvider(vllm_provider_config)

    assert provider.default_embedding_model == "BAAI/bge-large-en-v1.5"
    assert provider.default_llm_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert provider.default_vision_model == "llava-hf/llava-v1.6-mistral-7b-hf"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_supported_models(vllm_provider_config):
    """Test vLLM provider lists supported models."""
    provider = VLLMProvider(vllm_provider_config)

    embedding_models = provider.supported_models
    assert len(embedding_models) > 0
    assert "BAAI/bge-large-en-v1.5" in embedding_models

    vision_models = provider.supported_vision_models
    assert len(vision_models) > 0
    assert "llava-hf/llava-v1.6-mistral-7b-hf" in vision_models


# Unit Tests - Text Embedding Generation


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_create_embedding_success(initialized_vllm_provider, mock_embedding_response):
    """Test successful text embedding generation."""
    provider = initialized_vllm_provider

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=200,
            payload=mock_embedding_response
        )

        result = await provider.create_embedding("Test text for embedding")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 1024
        assert result.dimensions == 1024
        assert result.model == "BAAI/bge-large-en-v1.5"
        assert result.input_tokens == 10


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_create_embeddings_batch_success(initialized_vllm_provider, mock_embeddings_batch_response):
    """Test successful batch embeddings generation."""
    provider = initialized_vllm_provider

    texts = [f"Test text {i}" for i in range(5)]

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=200,
            payload=mock_embeddings_batch_response
        )

        results = await provider.create_embeddings_batch(texts)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) == 1024
            assert result.dimensions == 1024
            assert result.embedding[0] == pytest.approx(0.1 + i * 0.01, abs=0.001)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_create_embedding_empty_text(vllm_provider_config):
    """Test embedding generation with empty text raises ValueError."""
    provider = VLLMProvider(vllm_provider_config)

    with pytest.raises(ValueError, match="empty"):
        await provider.create_embedding("")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_create_embedding_retry_on_429(initialized_vllm_provider, mock_embedding_response):
    """Test embedding generation retries on 429 rate limit."""
    provider = initialized_vllm_provider

    with aioresponses() as mocked:
        # First request returns 429, second succeeds
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=429,
            payload={"error": "Rate limit exceeded"}
        )
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=200,
            payload=mock_embedding_response
        )

        result = await provider.create_embedding("Test text")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 1024


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_create_embedding_max_retries_exceeded(vllm_provider_config):
    """Test embedding generation fails after max retries."""
    provider = VLLMProvider(vllm_provider_config)
    provider.config["max_retries"] = 2

    with aioresponses() as mocked:
        # All requests return 500
        for _ in range(3):
            mocked.post(
                "http://test-vllm.example.com/v1/embeddings",
                status=500,
                payload={"error": "Internal server error"}
            )

        with pytest.raises(Exception):
            await provider.create_embedding("Test text")


# Unit Tests - LLM Generation


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_generate_success(initialized_vllm_provider, mock_llm_response):
    """Test successful LLM text generation."""
    provider = initialized_vllm_provider

    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/chat/completions",
            status=200,
            payload=mock_llm_response
        )

        result = await provider.generate(messages)

        assert isinstance(result, LLMResult)
        assert result.content == "This is a test response from vLLM."
        assert result.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.finish_reason == "stop"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_generate_with_parameters(initialized_vllm_provider, mock_llm_response):
    """Test LLM generation with custom parameters."""
    provider = initialized_vllm_provider

    messages = [{"role": "user", "content": "Test prompt"}]

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/chat/completions",
            status=200,
            payload=mock_llm_response
        )

        result = await provider.generate(
            messages,
            temperature=0.5,
            max_tokens=100,
            top_p=0.9
        )

        assert isinstance(result, LLMResult)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_generate_empty_messages(vllm_provider_config):
    """Test LLM generation with empty messages raises ValueError."""
    provider = VLLMProvider(vllm_provider_config)

    with pytest.raises(ValueError):
        await provider.generate([])


# Unit Tests - Vision Model Integration


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_generate_image_caption_success(
    initialized_vllm_provider,
    mock_vision_caption_response,
    test_image_data
):
    """Test successful image caption generation."""
    provider = initialized_vllm_provider

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/chat/completions",
            status=200,
            payload=mock_vision_caption_response
        )

        result = await provider.generate_image_caption(test_image_data)

        assert isinstance(result, VisionResult)
        assert result.content == "A test image showing example content for unit testing."
        assert result.model == "llava-hf/llava-v1.6-mistral-7b-hf"
        assert result.processing_time_ms > 0


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_generate_image_caption_custom_prompt(
    initialized_vllm_provider,
    mock_vision_caption_response,
    test_image_data
):
    """Test image caption generation with custom prompt."""
    provider = initialized_vllm_provider

    custom_prompt = "Describe this image in technical detail."

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/chat/completions",
            status=200,
            payload=mock_vision_caption_response
        )

        result = await provider.generate_image_caption(
            test_image_data,
            prompt_template=custom_prompt
        )

        assert isinstance(result, VisionResult)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_extract_text_from_image_success(
    initialized_vllm_provider,
    mock_vision_ocr_response,
    test_image_data
):
    """Test successful OCR text extraction from image."""
    provider = initialized_vllm_provider

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/chat/completions",
            status=200,
            payload=mock_vision_ocr_response
        )

        result = await provider.extract_text_from_image(test_image_data)

        assert isinstance(result, VisionResult)
        assert "Hello World Test" in result.content
        assert result.model == "llava-hf/llava-v1.6-mistral-7b-hf"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_generate_image_embedding_not_supported(
    vllm_provider_config,
    test_image_data
):
    """Test that direct image embeddings raise NotImplementedError."""
    provider = VLLMProvider(vllm_provider_config)

    with pytest.raises(NotImplementedError):
        await provider.generate_image_embedding(test_image_data)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_supports_vision(vllm_provider_config):
    """Test vision support detection."""
    provider = VLLMProvider(vllm_provider_config)

    assert provider.supports_vision() is True


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_vision_empty_image_data(vllm_provider_config):
    """Test vision operations with empty image data raise ValueError."""
    provider = VLLMProvider(vllm_provider_config)

    with pytest.raises(ValueError, match="empty"):
        await provider.generate_image_caption(b"")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_vision_oversized_image(vllm_provider_config):
    """Test vision operations with oversized image raise ValueError."""
    provider = VLLMProvider(vllm_provider_config)

    # Create image data > 20MB
    oversized_image = b'\x00' * (21 * 1024 * 1024)

    with pytest.raises(ValueError, match="exceeds maximum"):
        await provider.generate_image_caption(oversized_image)


# Unit Tests - Health Checks


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_health_check_success(vllm_provider_config, mock_embedding_response):
    """Test successful health check."""
    provider = VLLMProvider(vllm_provider_config)

    with aioresponses() as mocked:
        # Mock the GET request to /v1/models endpoint
        mocked.get(
            "http://test-vllm.example.com/v1/models",
            status=200,
            payload={"data": [{"id": "BAAI/bge-large-en-v1.5"}]}
        )
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=200,
            payload=mock_embedding_response
        )

        health = await provider.health_check()

        assert isinstance(health, HealthStatus)
        assert health.is_healthy is True
        assert health.provider == "vllm"
        assert health.response_time_ms > 0
        assert health.error_message is None


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_health_check_endpoint_down(vllm_provider_config):
    """Test health check with endpoint down."""
    provider = VLLMProvider(vllm_provider_config)

    with aioresponses() as mocked:
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=503,
            payload={"error": "Service unavailable"}
        )

        health = await provider.health_check()

        assert isinstance(health, HealthStatus)
        assert health.is_healthy is False
        assert health.error_message is not None


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_health_check_timeout(vllm_provider_config):
    """Test health check timeout handling."""
    provider = VLLMProvider(vllm_provider_config)
    provider.config["timeout"] = 0.1  # Very short timeout

    with aioresponses() as mocked:
        # Mock slow response
        async def slow_callback(url, **kwargs):
            await asyncio.sleep(1.0)
            return aioresponses.CallbackResult(status=200, body=json.dumps({}))

        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            callback=slow_callback
        )

        health = await provider.health_check()

        assert health.is_healthy is False


# Unit Tests - Provider Factory Integration


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_factory_creation(vllm_provider_config):
    """Test vLLM provider creation via factory."""
    provider = AIProviderFactory.create_provider(
        AIProvider.VLLM,
        vllm_provider_config
    )

    assert provider is not None
    assert isinstance(provider, VLLMProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, VisionProvider)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_capabilities():
    """Test vLLM provider capabilities detection."""
    capabilities = AIProviderFactory.get_provider_capabilities(AIProvider.VLLM)

    assert capabilities["supports_embeddings"] is True
    assert capabilities["supports_llm"] is True
    assert capabilities["supports_hybrid"] is True


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_available():
    """Test vLLM provider availability check."""
    is_available = AIProviderFactory.is_provider_available(AIProvider.VLLM)

    assert is_available is True


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_provider_in_registry():
    """Test vLLM provider is listed in registry."""
    all_providers = AIProviderFactory.list_all_providers()

    assert "vllm" in all_providers["hybrid"]
    assert "vllm" in all_providers["embedding"]
    assert "vllm" in all_providers["llm"]


# Unit Tests - Error Handling and Edge Cases


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_invalid_model_error(initialized_vllm_provider):
    """Test that invalid model returns clear error."""
    provider = initialized_vllm_provider

    with aioresponses() as mocked:
        # Request with invalid model fails with 404
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=404,
            payload={"error": "Model not found"}
        )

        # Should raise exception for invalid model (fail fast)
        with pytest.raises(Exception, match="404"):
            await provider.create_embedding("Test text", model="invalid-model")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_network_error_handling(vllm_provider_config):
    """Test network error handling with retries."""
    provider = VLLMProvider(vllm_provider_config)

    with aioresponses() as mocked:
        # Simulate network errors
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            exception=ConnectionError("Network unreachable")
        )

        with pytest.raises(Exception):
            await provider.create_embedding("Test text")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_malformed_response_handling(vllm_provider_config):
    """Test handling of malformed API responses."""
    provider = VLLMProvider(vllm_provider_config)

    with aioresponses() as mocked:
        # Return invalid JSON
        mocked.post(
            "http://test-vllm.example.com/v1/embeddings",
            status=200,
            payload={"invalid": "structure"}  # Missing required fields
        )

        with pytest.raises(Exception):
            await provider.create_embedding("Test text")


# Unit Tests - Model Dimension Handling


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_get_model_dimensions(vllm_provider_config):
    """Test model dimension lookup."""
    provider = VLLMProvider(vllm_provider_config)

    dimensions = provider.get_model_dimensions("BAAI/bge-large-en-v1.5")
    assert dimensions == 1024


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_unknown_model_dimensions(vllm_provider_config):
    """Test dimension lookup for unknown model uses config default."""
    provider = VLLMProvider(vllm_provider_config)

    dimensions = provider.get_model_dimensions("unknown/model")
    assert dimensions == 1024  # Falls back to config default


# Unit Tests - Concurrency and Performance


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_concurrent_requests(initialized_vllm_provider, mock_embedding_response):
    """Test concurrent embedding requests."""
    provider = initialized_vllm_provider

    with aioresponses() as mocked:
        # Mock 10 concurrent requests
        for _ in range(10):
            mocked.post(
                "http://test-vllm.example.com/v1/embeddings",
                status=200,
                payload=mock_embedding_response
            )

        tasks = [
            provider.create_embedding(f"Test text {i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert isinstance(result, EmbeddingResult)


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
@pytest.mark.asyncio
async def test_vllm_batch_size_respecting(initialized_vllm_provider):
    """Test batch processing respects max_batch_size."""
    provider = initialized_vllm_provider
    provider.config["max_batch_size"] = 3

    # Create batch larger than max_batch_size
    texts = [f"Text {i}" for i in range(10)]

    with aioresponses() as mocked:
        # Mock responses for ceil(10/3) = 4 batches
        # Each batch contains different number of results
        batch_responses = [
            {"object": "list", "data": [{"object": "embedding", "embedding": [0.1] * 1024, "index": i} for i in range(3)]},
            {"object": "list", "data": [{"object": "embedding", "embedding": [0.2] * 1024, "index": i} for i in range(3)]},
            {"object": "list", "data": [{"object": "embedding", "embedding": [0.3] * 1024, "index": i} for i in range(3)]},
            {"object": "list", "data": [{"object": "embedding", "embedding": [0.4] * 1024, "index": i} for i in range(1)]},  # Last batch with 1 item
        ]

        for response in batch_responses:
            mocked.post(
                "http://test-vllm.example.com/v1/embeddings",
                status=200,
                payload=response
            )

        results = await provider.create_embeddings_batch(texts, batch_size=3)

        assert len(results) == 10


# Integration marker for backward compatibility tests


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM provider not available")
@pytest.mark.unit
def test_vllm_backward_compatibility_interfaces(vllm_provider_config):
    """Test vLLM provider maintains backward compatibility with existing interfaces."""
    provider = VLLMProvider(vllm_provider_config)

    # Check all required interfaces are implemented
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, VisionProvider)

    # Check all required methods exist
    assert hasattr(provider, 'create_embedding')
    assert hasattr(provider, 'create_embeddings_batch')
    assert hasattr(provider, 'generate')
    assert hasattr(provider, 'generate_stream')
    assert hasattr(provider, 'generate_image_caption')
    assert hasattr(provider, 'extract_text_from_image')
    assert hasattr(provider, 'health_check')
    assert hasattr(provider, 'initialize')
    assert hasattr(provider, 'close')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
