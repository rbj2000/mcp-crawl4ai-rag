"""vLLM AI provider implementation with OpenAI-compatible API"""

import asyncio
import base64
import io
import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientSession

from ..base import (
    EmbeddingProvider,
    LLMProvider,
    HybridAIProvider,
    VisionProvider,
    EmbeddingResult,
    LLMResult,
    HealthStatus,
    VisionResult,
    AIProvider
)

logger = logging.getLogger(__name__)


class VLLMProvider(HybridAIProvider, VisionProvider):
    """vLLM provider with text, embedding, and vision model support via OpenAI-compatible API"""

    # Common vLLM text models
    TEXT_MODEL_CONFIGS = {
        "meta-llama/Llama-3.1-8B-Instruct": {"context_length": 8192},
        "meta-llama/Llama-3.1-70B-Instruct": {"context_length": 8192},
        "meta-llama/Llama-2-7b-chat-hf": {"context_length": 4096},
        "mistralai/Mistral-7B-Instruct-v0.2": {"context_length": 8192},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"context_length": 32768},
    }

    # Common vLLM vision models
    VISION_MODEL_CONFIGS = {
        "llava-hf/llava-v1.6-mistral-7b-hf": {"supports_vision": True},
        "llava-hf/llava-v1.6-vicuna-13b-hf": {"supports_vision": True},
        "Qwen/Qwen-VL-Chat": {"supports_vision": True},
        "microsoft/Phi-3-vision-128k-instruct": {"supports_vision": True},
        "THUDM/cogvlm-chat-hf": {"supports_vision": True},
    }

    # Common vLLM embedding models (dimension auto-detected)
    EMBEDDING_MODEL_CONFIGS = {
        "BAAI/bge-large-en-v1.5": {"dimensions": 1024},
        "BAAI/bge-base-en-v1.5": {"dimensions": 768},
        "nomic-ai/nomic-embed-text-v1": {"dimensions": 768},
        "intfloat/e5-large-v2": {"dimensions": 1024},
        "intfloat/e5-base-v2": {"dimensions": 768},
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize vLLM provider

        Args:
            config: Configuration dictionary with vLLM settings
        """
        self.config = config
        self._validate_config()

        self.session: Optional[ClientSession] = None
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.api_key = config.get("api_key", "EMPTY")  # vLLM default

        # Model configuration
        self.text_model = config.get("text_model", "meta-llama/Llama-3.1-8B-Instruct")
        self.embedding_model = config.get("embedding_model", "BAAI/bge-large-en-v1.5")
        self.vision_model = config.get("vision_model", "llava-hf/llava-v1.6-mistral-7b-hf")
        self.embedding_dimensions = config.get("embedding_dimensions", 1024)

        # Connection settings
        self.timeout = config.get("timeout", 120.0)
        self.connect_timeout = config.get("connect_timeout", 10.0)

        # Rate limiting and retry settings
        self.max_retries = config.get("max_retries", 3)
        self.initial_retry_delay = config.get("initial_retry_delay", 2.0)
        self.max_retry_delay = config.get("max_retry_delay", 60.0)
        self.retry_exponential_base = config.get("retry_exponential_base", 2.0)

        # Batch processing settings
        self._batch_size = config.get("batch_size", 32)

        # Performance monitoring
        self._request_count = 0
        self._total_tokens = 0
        self._error_count = 0

        # Vision settings
        self._vision_enabled = config.get("vision_enabled", True)

    def _validate_config(self) -> None:
        """Validate vLLM-specific configuration"""
        base_url = self.config.get("base_url", "http://localhost:8000/v1")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url '{base_url}'. Must start with http:// or https://")

        # Remove trailing slashes for consistency
        self.config["base_url"] = base_url.rstrip("/")

        # Validate timeout values
        timeout = self.config.get("timeout", 120.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Timeout must be a positive number")

    async def initialize(self) -> None:
        """Initialize the vLLM HTTP client"""
        try:
            timeout = ClientTimeout(
                total=self.timeout,
                connect=self.connect_timeout
            )

            self.session = ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )

            # Test connection with health check
            health_status = await self.health_check()
            if not health_status.is_healthy:
                raise ConnectionError(f"vLLM health check failed: {health_status.error_message}")

            logger.info(f"vLLM provider initialized successfully at {self.base_url}")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM provider: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise ConnectionError(f"Failed to initialize vLLM client: {e}")

    async def close(self) -> None:
        """Close the vLLM HTTP client"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("vLLM provider closed")

    async def health_check(self) -> HealthStatus:
        """Check if vLLM endpoint is healthy"""
        if not self.session:
            # Try to create a temporary session for health check
            try:
                timeout = ClientTimeout(total=10.0, connect=5.0)
                temp_session = ClientSession(timeout=timeout)

                start_time = time.time()
                async with temp_session.get(f"{self.base_url}/models") as response:
                    response_time_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        models_data = await response.json()
                        await temp_session.close()
                        return HealthStatus(
                            is_healthy=True,
                            provider=AIProvider.VLLM.value,
                            model=self.text_model,
                            response_time_ms=response_time_ms,
                            metadata={"available_models": len(models_data.get("data", []))}
                        )
                    else:
                        await temp_session.close()
                        return HealthStatus(
                            is_healthy=False,
                            provider=AIProvider.VLLM.value,
                            model=self.text_model,
                            response_time_ms=response_time_ms,
                            error_message=f"HTTP {response.status}"
                        )
            except Exception as e:
                if 'temp_session' in locals():
                    await temp_session.close()
                return HealthStatus(
                    is_healthy=False,
                    provider=AIProvider.VLLM.value,
                    model=self.text_model,
                    response_time_ms=0.0,
                    error_message=str(e)
                )

        # Session exists, use it for health check
        start_time = time.time()
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                response_time_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    models_data = await response.json()
                    return HealthStatus(
                        is_healthy=True,
                        provider=AIProvider.VLLM.value,
                        model=self.text_model,
                        response_time_ms=response_time_ms,
                        metadata={"available_models": len(models_data.get("data", []))}
                    )
                else:
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.VLLM.value,
                        model=self.text_model,
                        response_time_ms=response_time_ms,
                        error_message=f"HTTP {response.status}"
                    )
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.VLLM.value,
                model=self.text_model,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )

    async def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: Dict[str, Any],
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Make HTTP request with exponential backoff retry logic"""
        if not self.session:
            raise ConnectionError("vLLM session not initialized")

        url = f"{self.base_url}/{endpoint}"

        try:
            async with self.session.request(method, url, json=payload) as response:
                self._request_count += 1

                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limit
                    if retry_count < self.max_retries:
                        delay = min(
                            self.initial_retry_delay * (self.retry_exponential_base ** retry_count),
                            self.max_retry_delay
                        )
                        logger.warning(f"Rate limited, retrying in {delay}s (attempt {retry_count + 1}/{self.max_retries})")
                        await asyncio.sleep(delay)
                        return await self._make_request_with_retry(method, endpoint, payload, retry_count + 1)
                    else:
                        raise Exception(f"Rate limit exceeded after {self.max_retries} retries")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except ClientError as e:
            if retry_count < self.max_retries:
                delay = min(
                    self.initial_retry_delay * (self.retry_exponential_base ** retry_count),
                    self.max_retry_delay
                )
                logger.warning(f"Request failed: {e}, retrying in {delay}s (attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(delay)
                return await self._make_request_with_retry(method, endpoint, payload, retry_count + 1)
            else:
                self._error_count += 1
                raise

    # ===== EmbeddingProvider Implementation =====

    async def create_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """Create embedding for a single text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        model = model or self.embedding_model

        payload = {
            "model": model,
            "input": text
        }

        start_time = time.time()
        response = await self._make_request_with_retry("POST", "embeddings", payload)
        processing_time_ms = (time.time() - start_time) * 1000

        if "data" not in response or not response["data"]:
            raise Exception("Empty embedding response from vLLM")

        embedding = response["data"][0]["embedding"]

        # Track token usage if provided
        if "usage" in response:
            self._total_tokens += response["usage"].get("total_tokens", 0)

        return EmbeddingResult(
            embedding=embedding,
            dimensions=len(embedding),
            model=model,
            input_tokens=response.get("usage", {}).get("total_tokens")
        )

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[EmbeddingResult]:
        """Create embeddings for multiple texts in batches"""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        model = model or self.embedding_model
        batch_size = batch_size or self._batch_size

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            payload = {
                "model": model,
                "input": batch
            }

            response = await self._make_request_with_retry("POST", "embeddings", payload)

            if "data" not in response:
                raise Exception("Empty embedding response from vLLM")

            # Track token usage
            if "usage" in response:
                self._total_tokens += response["usage"].get("total_tokens", 0)

            for item in response["data"]:
                embedding = item["embedding"]
                results.append(
                    EmbeddingResult(
                        embedding=embedding,
                        dimensions=len(embedding),
                        model=model
                    )
                )

        return results

    @property
    def default_model(self) -> str:
        """Default embedding model for backwards compatibility"""
        return self.embedding_model

    @property
    def default_embedding_model(self) -> str:
        """Default embedding model"""
        return self.embedding_model

    @property
    def supported_models(self) -> List[str]:
        """List of supported embedding models"""
        return list(self.EMBEDDING_MODEL_CONFIGS.keys())

    def get_model_dimensions(self, model: str) -> int:
        """Get embedding dimensions for a model"""
        if model in self.EMBEDDING_MODEL_CONFIGS:
            return self.EMBEDDING_MODEL_CONFIGS[model]["dimensions"]
        return self.embedding_dimensions  # Use configured dimensions as fallback

    # ===== LLMProvider Implementation =====

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResult:
        """Generate text using vLLM"""
        self.validate_messages(messages)

        model = model or self.text_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        response = await self._make_request_with_retry("POST", "chat/completions", payload)

        if "choices" not in response or not response["choices"]:
            raise Exception("Empty completion response from vLLM")

        choice = response["choices"][0]
        content = choice.get("message", {}).get("content", "")

        # Track token usage
        usage = response.get("usage", {})
        if "total_tokens" in usage:
            self._total_tokens += usage["total_tokens"]

        return LLMResult(
            content=content,
            model=model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            metadata={"usage": usage}
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate text with streaming (not fully implemented for vLLM yet)"""
        # For now, fall back to non-streaming
        result = await self.generate(messages, model, temperature, max_tokens, **kwargs)
        yield result

    @property
    def default_llm_model(self) -> str:
        """Default LLM model"""
        return self.text_model

    # ===== VisionProvider Implementation =====

    def supports_vision(self) -> bool:
        """Check if vision is supported"""
        return self._vision_enabled and self.vision_model in self.VISION_MODEL_CONFIGS

    @property
    def default_vision_model(self) -> str:
        """Default vision model"""
        return self.vision_model

    @property
    def supported_vision_models(self) -> List[str]:
        """List of supported vision models"""
        return list(self.VISION_MODEL_CONFIGS.keys())

    async def generate_image_embedding(
        self,
        image_data: bytes,
        model: Optional[str] = None
    ) -> EmbeddingResult:
        """Generate embedding for an image"""
        self.validate_image_data(image_data)

        model = model or self.vision_model

        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # vLLM vision models may not support direct embedding
        # This is a placeholder - actual implementation depends on vLLM capabilities
        raise NotImplementedError(
            "Direct image embedding not yet supported in vLLM. "
            "Use generate_image_caption to get text description first."
        )

    async def generate_image_caption(
        self,
        image_data: bytes,
        prompt_template: Optional[str] = None,
        model: Optional[str] = None
    ) -> VisionResult:
        """Generate caption for an image"""
        self.validate_image_data(image_data)

        model = model or self.vision_model
        prompt = prompt_template or "Describe this image in detail."

        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # vLLM vision API uses chat completions with image content
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        start_time = time.time()
        response = await self._make_request_with_retry("POST", "chat/completions", payload)
        processing_time_ms = (time.time() - start_time) * 1000

        if "choices" not in response or not response["choices"]:
            raise Exception("Empty vision response from vLLM")

        content = response["choices"][0]["message"]["content"]

        return VisionResult(
            content=content,
            model=model,
            processing_time_ms=processing_time_ms,
            metadata=response.get("usage")
        )

    async def extract_text_from_image(
        self,
        image_data: bytes,
        model: Optional[str] = None
    ) -> VisionResult:
        """Extract text from image using OCR"""
        # Use vision model with OCR-focused prompt
        ocr_prompt = (
            "Extract all visible text from this image. "
            "Preserve formatting and structure. "
            "If no text is visible, respond with 'No text detected'."
        )

        return await self.generate_image_caption(image_data, ocr_prompt, model)
